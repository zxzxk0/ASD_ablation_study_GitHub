# -*- coding: utf-8 -*-
"""
train_mlp_v6_final.py
─────────────────────────────────────────────────────────────────────
최종 버전. 아래 세 파트를 한 번에 실행.

PART 1 — Method 비교 (EVA02-B-16 고정)
  image_mlp / pea_ridge / pea_svr / pea_gpr / pea_mlp / attention_mil

PART 2 — Backbone 비교 (pea_mlp 고정, 최적 방법)
  EVA02-B-16 / ViT-L-14(OpenAI) / DINOv2 ViT-S/14 / ResNet-50

PART 3 — Ensemble
  pea_mlp(EVA02) + attention_mil(EVA02) 평균

모두 LOPO + Participant-level 평가 (공정 비교).
"""

import argparse, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch, torch.nn as nn
import open_clip
import torchvision.models as tv_models
import torchvision.transforms as tv_tf
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# ===== 경로 =====
# Defaults are repository-relative; override with CLI arguments.
IMG_ROOT  = Path("data") / "Images" / "TSImages"
DATA_PATH = Path("data") / "all_scanpath_absolute.jsonl"
OUT_DIR   = Path("outputs") / "checkpoints_v6"
CACHE_DIR = Path("outputs") / "cache_v2"

META_CANDIDATES_CONT = {"age", "iq", "age_z"}
META_CANDIDATES_CAT  = {"gender", "class", "diagnosis"}
CARS_MIN, CARS_MAX   = 15.0, 60.0

SEVERITY_BINS = {
    "Mild (15-29)":     (15.0, 30.0),
    "Moderate (30-36)": (30.0, 37.0),
    "Severe (37-60)":   (37.0, 61.0),
}

_num_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# ===== 시드 고정 =====
FIXED_SEED = 42

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────
# 파라미터
# ─────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta",          action="store_true")
    ap.add_argument("--pca_dim",      type=int,   default=32)
    ap.add_argument("--epochs",       type=int,   default=30000)
    ap.add_argument("--patience",     type=int,   default=100)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--data_csv",     type=str,   default=str(DATA_PATH),
                    help="Input JSONL containing rendered scanpath records")
    ap.add_argument("--image-root",   type=Path,  default=IMG_ROOT,
                    help="Directory containing rendered scanpath images")
    ap.add_argument("--output-dir",   type=Path,  default=OUT_DIR,
                    help="Directory for model checkpoints and outputs")
    ap.add_argument("--cache-dir",    type=Path,  default=CACHE_DIR,
                    help="Directory for embedding caches")
    return ap.parse_args()


# ─────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────
def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "meta" in df.columns:
        all_keys = set()
        for m in df["meta"]:
            if isinstance(m, dict):
                all_keys.update(m.keys())
        for k in all_keys:
            if k not in df.columns:
                df[k] = df["meta"].apply(
                    lambda m: m.get(k) if isinstance(m, dict) else np.nan)
    return df


def get_path_col(df):
    for c in ["abs_path", "image_relpath", "image"]:
        if c in df.columns:
            return c
    raise ValueError("이미지 경로 컬럼 없음")


def get_targets(df) -> np.ndarray:
    vals = []
    for _, row in df.iterrows():
        d = row.to_dict(); v = None
        for col in ["cars", "label", "score", "target", "y"]:
            if col in d and d[col] is not None:
                try: v = float(d[col]); break
                except: pass
        if v is None:
            meta = d.get("meta")
            if isinstance(meta, dict):
                for col in ["cars", "label"]:
                    if col in meta:
                        try: v = float(meta[col]); break
                        except: pass
        vals.append(np.nan if v is None else v)
    arr = np.array(vals, dtype=np.float32)
    if np.isnan(arr).any():
        raise ValueError("레이블 파싱 실패")
    return arr


def get_groups(df) -> np.ndarray:
    ids = []
    for _, row in df.iterrows():
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        pid  = (meta or {}).get("participant_id") or str(row.name)
        ids.append(str(pid))
    return np.array(ids)


def get_meta(df):
    cols = set(df.columns)
    cont = [c for c in META_CANDIDATES_CONT if c in cols]
    cat  = [c for c in META_CANDIDATES_CAT  if c in cols]
    C = df[cont].astype(float).fillna(0.0) if cont else pd.DataFrame()
    K = (pd.concat([pd.get_dummies(df[c].astype(str).str.lower().str.strip(), prefix=c)
                    for c in cat], axis=1) if cat else pd.DataFrame())
    M = (pd.concat([C, K], axis=1).values.astype(np.float32)
         if not (C.empty and K.empty) else None)
    return M, len(cont)


# ─────────────────────────────────────────
# 임베딩 — CLIP / EVA02
# ─────────────────────────────────────────
def load_clip_model(name, pretrained, device):
    model, _, prep = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device=device)
    model.eval()
    return model, prep


@torch.no_grad()
def _encode_clip(im, model, prep, device):
    t = prep(im).unsqueeze(0).to(device)
    f = model.encode_image(t)
    return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()


def five_crop(img):
    w, h = img.size; s = min(w, h)
    return [img.crop((0,0,s,s)), img.crop((w-s,0,w,s)),
            img.crop((0,h-s,s,h)), img.crop((w-s,h-s,w,h)),
            img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))]


def embed_clip(df, img_root, name, pretrained, device, tta, cache_key):
    cache = CACHE_DIR / f"{cache_key}.npz"
    path_col = get_path_col(df)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"  [CACHE] {cache.name}  shape={z['X'].shape}")
        return {z["paths"].tolist()[i]: z["X"][i]
                for i in range(len(z["paths"]))}
    model, prep = load_clip_model(name, pretrained, device)
    embs, paths = [], []
    for rel in tqdm(df[path_col], desc=f"Embed {name}@{pretrained}"):
        p = Path(str(rel)); p = p if p.is_absolute() else img_root / p
        if not p.exists(): continue
        im = Image.open(p).convert("RGB")
        f  = (np.mean(np.stack([_encode_clip(c, model, prep, device)
                                 for c in five_crop(im)]), axis=0)
              if tta else _encode_clip(im, model, prep, device))
        embs.append(f); paths.append(str(p))
    X = np.stack(embs)
    np.savez(cache, X=X, paths=np.array(paths, dtype=object))
    return {paths[i]: X[i] for i in range(len(paths))}


# ─────────────────────────────────────────
# 임베딩 — DINOv2
# ─────────────────────────────────────────
def embed_dinov2(df, img_root, device, tta, cache_key):
    cache = CACHE_DIR / f"{cache_key}.npz"
    path_col = get_path_col(df)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"  [CACHE] {cache.name}  shape={z['X'].shape}")
        return {z["paths"].tolist()[i]: z["X"][i]
                for i in range(len(z["paths"]))}
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device).eval()
    prep  = tv_tf.Compose([
        tv_tf.Resize(224, interpolation=tv_tf.InterpolationMode.BICUBIC),
        tv_tf.CenterCrop(224),
        tv_tf.ToTensor(),
        tv_tf.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    @torch.no_grad()
    def _enc(im):
        t = prep(im).unsqueeze(0).to(device)
        return model(t).squeeze(0).cpu().numpy()

    embs, paths = [], []
    for rel in tqdm(df[path_col], desc="Embed DINOv2"):
        p = Path(str(rel)); p = p if p.is_absolute() else img_root / p
        if not p.exists(): continue
        im = Image.open(p).convert("RGB")
        f  = (np.mean(np.stack([_enc(c) for c in five_crop(im)]), axis=0)
              if tta else _enc(im))
        embs.append(f); paths.append(str(p))
    X = np.stack(embs)
    np.savez(cache, X=X, paths=np.array(paths, dtype=object))
    return {paths[i]: X[i] for i in range(len(paths))}


# ─────────────────────────────────────────
# 임베딩 — ResNet-50
# ─────────────────────────────────────────
def embed_resnet50(df, img_root, device, tta, cache_key):
    cache = CACHE_DIR / f"{cache_key}.npz"
    path_col = get_path_col(df)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"  [CACHE] {cache.name}  shape={z['X'].shape}")
        return {z["paths"].tolist()[i]: z["X"][i]
                for i in range(len(z["paths"]))}
    base  = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    model = nn.Sequential(*list(base.children())[:-1])   # avgpool 출력 (2048-dim)
    model = model.to(device).eval()
    prep  = tv_tf.Compose([
        tv_tf.Resize(224),
        tv_tf.CenterCrop(224),
        tv_tf.ToTensor(),
        tv_tf.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    @torch.no_grad()
    def _enc(im):
        t = prep(im).unsqueeze(0).to(device)
        return model(t).squeeze().cpu().numpy()

    embs, paths = [], []
    for rel in tqdm(df[path_col], desc="Embed ResNet-50"):
        p = Path(str(rel)); p = p if p.is_absolute() else img_root / p
        if not p.exists(): continue
        im = Image.open(p).convert("RGB")
        f  = (np.mean(np.stack([_enc(c) for c in five_crop(im)]), axis=0)
              if tta else _enc(im))
        embs.append(f); paths.append(str(p))
    X = np.stack(embs)
    np.savez(cache, X=X, paths=np.array(paths, dtype=object))
    return {paths[i]: X[i] for i in range(len(paths))}


# ─────────────────────────────────────────
# path_map → 정렬된 배열
# ─────────────────────────────────────────
def path_map_to_array(df, img_root, path_map):
    path_col = get_path_col(df)
    embs, ok = [], []
    for rel in df[path_col]:
        p = Path(str(rel)); p = p if p.is_absolute() else img_root / p
        key = str(p)
        if key in path_map:
            embs.append(path_map[key]); ok.append(True)
        else:
            ok.append(False)
    X = np.stack(embs)
    df_filt = df[ok].reset_index(drop=True)
    return X, df_filt


# ─────────────────────────────────────────
# Participant-level 집계
# ─────────────────────────────────────────
def aggregate(X_img, y_img, groups, M_img=None):
    pids = sorted(set(groups))
    Xa, ya, ga, Ma = [], [], [], []
    for pid in pids:
        m = groups == pid
        Xa.append(X_img[m].mean(axis=0))
        ya.append(float(y_img[m].mean()))
        ga.append(pid)
        if M_img is not None:
            Ma.append(M_img[m].mean(axis=0))
    Xa = np.stack(Xa); ya = np.array(ya, dtype=np.float32); ga = np.array(ga)
    if Ma:
        Xa = np.concatenate([Xa, np.stack(Ma)], axis=1)
    return Xa, ya, ga


# ─────────────────────────────────────────
# 평가
# ─────────────────────────────────────────
def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    sp   = spearmanr(y_true, y_pred).statistic
    return mae, rmse, r2, sp


def sev_table(y_true, y_pred):
    rows = []
    for lbl, (lo, hi) in SEVERITY_BINS.items():
        m = (y_true >= lo) & (y_true < hi); n = int(m.sum())
        rows.append({
            "Severity": lbl, "N": n,
            "MAE":  round(mean_absolute_error(y_true[m], y_pred[m]), 3) if n else float("nan"),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))), 3) if n else float("nan"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# MLP
# ─────────────────────────────────────────
class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(True), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.ReLU(True), nn.Dropout(p * 0.5),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_val, y_val, args, device,
              hidden=128, ckpt_path=None):
    y_mu = float(y_tr.mean()); y_sd = float(y_tr.std()) or 1.0
    y_tr_z = (y_tr - y_mu) / y_sd
    bs = min(args.batch_size, max(4, len(X_tr)))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(),
                      torch.from_numpy(y_tr_z).float()),
        batch_size=bs, shuffle=True)
    model = SmallMLP(X_tr.shape[1], hidden=hidden).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=1e-5)
    crit  = nn.HuberLoss(delta=1.0)
    best_mae, best_preds, pat = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
        preds   = np.clip(preds * y_sd + y_mu, CARS_MIN, CARS_MAX)
        val_mae = mean_absolute_error(y_val, preds)
        if val_mae < best_mae:
            best_mae, best_preds, pat = val_mae, preds.copy(), 0
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        else:
            pat += 1
        if pat >= args.patience:
            break
    return best_preds


# ─────────────────────────────────────────
# Attention MIL
# ─────────────────────────────────────────
class AttentionMIL(nn.Module):
    def __init__(self, in_dim, ah=64, rh=64, p=0.4):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, ah), nn.Tanh(), nn.Dropout(p), nn.Linear(ah, 1))
        self.reg  = nn.Sequential(
            nn.Linear(in_dim, rh), nn.ReLU(True), nn.Dropout(p), nn.Linear(rh, 1))
    def forward(self, bag):
        a = torch.softmax(self.attn(bag), dim=0)
        z = (a * bag).sum(0, keepdim=True)
        return self.reg(z).squeeze()


def train_attention_mil(X_img, y_img, groups,
                        tr_pids, val_pids, args, device, ckpt_path=None):
    def bags(pids):
        return (
            [torch.from_numpy(X_img[groups == pid]).float() for pid in pids],
            np.array([float(y_img[groups == pid].mean()) for pid in pids],
                     dtype=np.float32)
        )
    tr_bags, y_tr   = bags(tr_pids)
    val_bags, y_val = bags(val_pids)
    y_mu = float(y_tr.mean()); y_sd = float(y_tr.std()) or 1.0
    y_tr_z = (y_tr - y_mu) / y_sd
    model = AttentionMIL(X_img.shape[1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=1e-5)
    crit  = nn.HuberLoss(delta=1.0)
    best_mae, best_preds, pat = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); opt.zero_grad(); loss = torch.tensor(0., device=device)
        for bag, lz in zip(tr_bags, y_tr_z):
            loss = loss + crit(model(bag.to(device)),
                               torch.tensor(lz, device=device))
        (loss / len(tr_bags)).backward(); opt.step(); sch.step()
        model.eval(); preds = []
        with torch.no_grad():
            for bag in val_bags:
                preds.append(np.clip(
                    model(bag.to(device)).item() * y_sd + y_mu,
                    CARS_MIN, CARS_MAX))
        preds   = np.array(preds)
        val_mae = mean_absolute_error(y_val, preds)
        if val_mae < best_mae:
            best_mae, best_preds, pat = val_mae, preds.copy(), 0
            if ckpt_path: torch.save(model.state_dict(), ckpt_path)
        else:
            pat += 1
        if pat >= args.patience: break
    return best_preds, y_val


# ─────────────────────────────────────────
# LOPO 범용 실행
# ─────────────────────────────────────────
def run_lopo(method, X_img, y_img, groups, M_img, args, device, tag=""):
    pids  = sorted(set(groups))
    logo  = LeaveOneGroupOut()

    # pea_* 는 참가자 집계 데이터 사용
    X_agg, y_agg, grp_agg = aggregate(X_img, y_img, groups, M_img)

    if method == "image_mlp":
        split_X, split_y, split_g = X_img, y_img, groups
    else:
        split_X, split_y, split_g = X_agg, y_agg, grp_agg

    all_true, all_pred = [], []

    for fold, (tr_idx, val_idx) in enumerate(
            logo.split(split_X, split_y, groups=split_g), 1):
        val_pid    = split_g[val_idx][0]
        train_pids = sorted(set(split_g[tr_idx]))
        print(f"  [{tag or method}] fold {fold:02d}/26  pid={val_pid}", end="")

        # ── image_mlp ───────────────────────────────────────────────
        if method == "image_mlp":
            set_seed(FIXED_SEED * 100 + fold)
            X_tr, X_val = split_X[tr_idx], split_X[val_idx]
            y_tr, y_vi  = split_y[tr_idx], split_y[val_idx]
            if M_img is not None:
                M_tr, M_val = M_img[tr_idx].copy(), M_img[val_idx].copy()
                mu = M_tr.mean(0); sd = M_tr.std(0); sd[sd==0] = 1
                M_tr  = (M_tr  - mu) / sd
                M_val = (M_val - mu) / sd
                X_tr  = np.concatenate([X_tr,  M_tr],  axis=1)
                X_val = np.concatenate([X_val, M_val], axis=1)
            ckpt  = OUT_DIR / f"{tag}_{method}_f{fold:02d}.pt"
            preds = train_mlp(X_tr, y_tr, X_val, y_vi, args, device,
                               hidden=256, ckpt_path=ckpt)
            p_pred = float(preds.mean())
            p_true = float(y_vi.mean())

        # ── pea_* ────────────────────────────────────────────────────
        elif method.startswith("pea_"):
            X_tr, X_val = split_X[tr_idx], split_X[val_idx]
            y_tr, y_val = split_y[tr_idx], split_y[val_idx]
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr); X_val_s = sc.transform(X_val)
            if args.pca_dim > 0 and args.pca_dim < X_tr_s.shape[1]:
                pca = PCA(n_components=min(args.pca_dim, len(X_tr_s)-1),
                           random_state=FIXED_SEED)
                X_tr_s  = pca.fit_transform(X_tr_s)
                X_val_s = pca.transform(X_val_s)
            if method == "pea_ridge":
                reg = RidgeCV(alphas=[0.01,0.1,1,10,100,1000],
                               cv=min(5, len(X_tr_s)))
                reg.fit(X_tr_s, y_tr)
                p_pred = float(np.clip(reg.predict(X_val_s).ravel()[0],
                                        CARS_MIN, CARS_MAX))
            elif method == "pea_svr":
                reg = SVR(kernel="rbf", C=10., epsilon=0.5, gamma="scale")
                reg.fit(X_tr_s, y_tr)
                p_pred = float(np.clip(reg.predict(X_val_s).ravel()[0],
                                        CARS_MIN, CARS_MAX))
            elif method == "pea_gpr":
                kern = (ConstantKernel(1.) * RBF(1.) + WhiteKernel(1.))
                reg  = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=5,
                                                random_state=FIXED_SEED,
                                                normalize_y=True)
                reg.fit(X_tr_s, y_tr)
                p_pred = float(np.clip(
                    reg.predict(X_val_s, return_std=False).ravel()[0],
                    CARS_MIN, CARS_MAX))
            elif method == "pea_mlp":
                set_seed(FIXED_SEED * 100 + fold)   # ← 시드 고정
                ckpt  = OUT_DIR / f"{tag}_{method}_f{fold:02d}.pt"
                preds = train_mlp(X_tr_s.astype(np.float32),
                                   y_tr.astype(np.float32),
                                   X_val_s.astype(np.float32),
                                   y_val.astype(np.float32),
                                   args, device, hidden=64, ckpt_path=ckpt)
                p_pred = float(np.clip(preds.ravel()[0], CARS_MIN, CARS_MAX))
            p_true = float(y_val[0])

        # ── attention_mil ────────────────────────────────────────────
        elif method == "attention_mil":
            set_seed(FIXED_SEED * 100 + fold)       # ← 시드 고정
            ckpt  = OUT_DIR / f"{tag}_{method}_f{fold:02d}.pt"
            preds, y_val_arr = train_attention_mil(
                X_img, y_img, groups, train_pids, [val_pid],
                args, device, ckpt_path=ckpt)
            p_pred = float(preds[0])
            p_true = float(y_val_arr[0])

        print(f"  pred={p_pred:.2f}  true={p_true:.2f}  "
              f"err={abs(p_pred - p_true):.2f}")
        all_true.append(p_true); all_pred.append(p_pred)

    yt = np.array(all_true); yp = np.array(all_pred)
    mae, rmse, r2, sp = metrics(yt, yp)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Spearman": sp,
            "y_true": yt, "y_pred": yp}


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def main():
    global IMG_ROOT, OUT_DIR, CACHE_DIR
    args = parse_args()
    IMG_ROOT = Path(args.image_root)
    OUT_DIR = Path(args.output_dir)
    CACHE_DIR = Path(args.cache_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  epochs={args.epochs}  patience={args.patience}")
    print(f"[INFO] FIXED_SEED={FIXED_SEED}\n")

    df_all = load_jsonl(Path(args.data_csv))

    # ══════════════════════════════════════════════════════════════════
    # PART 1 — Method 비교  (EVA02-B-16 고정)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  PART 1 — Method Comparison  [EVA02-B-16]")
    print("="*65)

    pm = embed_clip(df_all, IMG_ROOT, "EVA02-B-16", "merged2b_s8b_b131k",
                    device, args.tta, "all_data_EVA02-B-16_merged2b_s8b_b131k")
    X_eva, df_eva = path_map_to_array(df_all, IMG_ROOT, pm)
    y_eva  = get_targets(df_eva)
    g_eva  = get_groups(df_eva)
    M_eva, _ = get_meta(df_eva)

    part1_methods = ["image_mlp", "pea_ridge", "pea_svr",
                     "pea_gpr", "pea_mlp", "attention_mil"]
    p1_results = {}
    for m in part1_methods:
        print(f"\n{'─'*55}\n  METHOD: {m}\n{'─'*55}")
        res = run_lopo(m, X_eva, y_eva, g_eva, M_eva, args, device, tag="p1")
        p1_results[m] = res
        print(f"  ✓ MAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}"
              f"  R²={res['R2']:.3f}  Spearman={res['Spearman']:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 2 — Backbone 비교  (pea_mlp 고정)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  PART 2 — Backbone Comparison  [pea_mlp method]")
    print("="*65)

    backbone_cfgs = [
        ("EVA02-B-16",   "clip",   "EVA02-B-16", "merged2b_s8b_b131k"),
        ("CLIP-ViT-L14", "clip",   "ViT-L-14",   "openai"),
        ("DINOv2",       "dinov2", None,          None),
        ("ResNet-50",    "resnet", None,          None),
    ]

    p2_results = {}
    for bb_name, bb_type, bb_model, bb_ckpt in backbone_cfgs:
        print(f"\n{'─'*55}\n  BACKBONE: {bb_name}\n{'─'*55}")

        cache_key = f"all_data_{bb_name.replace('-','_').replace('/','_')}"
        if bb_type == "clip":
            pm = embed_clip(df_all, IMG_ROOT, bb_model, bb_ckpt,
                            device, args.tta, cache_key)
        elif bb_type == "dinov2":
            pm = embed_dinov2(df_all, IMG_ROOT, device, args.tta, cache_key)
        elif bb_type == "resnet":
            pm = embed_resnet50(df_all, IMG_ROOT, device, args.tta, cache_key)

        X_bb, df_bb = path_map_to_array(df_all, IMG_ROOT, pm)
        y_bb = get_targets(df_bb)
        g_bb = get_groups(df_bb)
        M_bb, _ = get_meta(df_bb)

        res = run_lopo("pea_mlp", X_bb, y_bb, g_bb, M_bb,
                       args, device, tag=f"p2_{bb_name}")
        p2_results[bb_name] = res
        print(f"  ✓ MAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}"
              f"  R²={res['R2']:.3f}  Spearman={res['Spearman']:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 3 — Ensemble  (pea_mlp + attention_mil, EVA02)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  PART 3 — Ensemble  [pea_mlp + attention_mil, EVA02]")
    print("="*65)

    y_true_ens = p1_results["pea_mlp"]["y_true"]
    y_ens = (p1_results["pea_mlp"]["y_pred"] +
             p1_results["attention_mil"]["y_pred"]) / 2
    ens_mae, ens_rmse, ens_r2, ens_sp = metrics(y_true_ens, y_ens)
    ens_sev = sev_table(y_true_ens, y_ens)

    print(f"\n  Ensemble MAE={ens_mae:.3f}  RMSE={ens_rmse:.3f}"
          f"  R²={ens_r2:.3f}  Spearman={ens_sp:.3f}")
    print(ens_sev.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════
    # 최종 리포트
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "="*80)
    print("  TABLE A — Method Comparison  [EVA02-B-16 / LOPO / Participant-level]")
    print("="*80)
    rows = []
    for m, res in p1_results.items():
        rows.append({"Method": m, "_mae": res["MAE"],
                     "MAE": f"{res['MAE']:.3f}", "RMSE": f"{res['RMSE']:.3f}",
                     "R²":  f"{res['R2']:.3f}", "Spearman": f"{res['Spearman']:.3f}"})
    rows.append({"Method": "Ensemble (pea_mlp+attn_mil)", "_mae": ens_mae,
                 "MAE": f"{ens_mae:.3f}", "RMSE": f"{ens_rmse:.3f}",
                 "R²":  f"{ens_r2:.3f}", "Spearman": f"{ens_sp:.3f}"})
    tA = pd.DataFrame(rows).sort_values("_mae").drop(columns=["_mae"])
    print(tA.to_string(index=False))

    print("\n" + "="*80)
    print("  TABLE B — Backbone Comparison  [pea_mlp / LOPO / Participant-level]")
    print("="*80)
    rows2 = []
    for bb, res in p2_results.items():
        rows2.append({"Backbone": bb, "_mae": res["MAE"],
                      "MAE": f"{res['MAE']:.3f}", "RMSE": f"{res['RMSE']:.3f}",
                      "R²":  f"{res['R2']:.3f}", "Spearman": f"{res['Spearman']:.3f}"})
    tB = pd.DataFrame(rows2).sort_values("_mae").drop(columns=["_mae"])
    print(tB.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════
    # Severity 분석 (논문 Table 6 업데이트용)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("  SEVERITY — pea_mlp  [논문 Table 6 PEA-MLP row 업데이트용]")
    print("="*80)
    pea_res = p1_results["pea_mlp"]
    pea_sev = sev_table(pea_res["y_true"], pea_res["y_pred"])
    print(pea_sev.to_string(index=False))

    # 가중 평균 검증: 전체 MAE와 일치 확인
    w_avg = sum(
        row["N"] * row["MAE"]
        for _, row in pea_sev.iterrows()
        if not np.isnan(row["MAE"])
    ) / pea_sev["N"].sum()
    print(f"\n  [검증] PEA-MLP severity weighted MAE = {w_avg:.3f}"
          f"  (전체 MAE = {pea_res['MAE']:.3f})")
    if abs(w_avg - pea_res["MAE"]) < 0.01:
        print("  ✓ 일치 — 논문 Table 6에 위 수치를 사용하세요.")
    else:
        print("  ✗ 불일치 — 재확인 필요.")

    print("\n" + "="*80)
    print("  SEVERITY — attention_mil")
    print("="*80)
    attn_res = p1_results["attention_mil"]
    print(sev_table(attn_res["y_true"], attn_res["y_pred"]).to_string(index=False))

    print("\n" + "="*80)
    print("  SEVERITY — Ensemble")
    print("="*80)
    print(ens_sev.to_string(index=False))

    # 최종 best 선정
    best_row = tA.iloc[0]
    print(f"\n  🏆 전체 최적: {best_row['Method']}"
          f"  MAE={best_row['MAE']}  R²={best_row['R²']}")
    print("="*80)


if __name__ == "__main__":
    main()