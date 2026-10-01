# -*- coding: utf-8 -*-
r"""
run_audit_replay_E1_E4.py
=========================
Protocol-replay experiments for the literature audit table: the evaluation
protocols commonly used in prior work on the Carette/Cilia scanpath-image
dataset are re-run on the same images, with the SAME model under a leaky and a
participant-disjoint protocol, so that the only difference is the protocol.

Task: ASD (TS) vs TD (TC) classification from scanpath images.

  E1  split unit            image-level vs participant-level stratified 5-fold
                            (frozen backbones; nested logistic regression and a
                            1-NN "memoriser"); participant-label permutation null.
  E2  augmentation order    augment-then-split vs split-then-augment
                            (image-level and participant-level); same features,
                            same classifier; permutation null.
  E3  test-set early        MLP head on frozen embeddings. Epoch selection and
      stopping/selection    hyper-parameter (config) selection are separated:
                            epoch {final, inner-CV, test} with config fixed;
                            config {inner-CV, test} with epoch fixed at final;
                            joint (config+epoch) {inner-CV, test}. Participant-
                            and image-level splits; permutation null;
                            label-invariance test.
  E4  end-to-end CNN        ImageNet CNN fine-tuned (ResNet-50 default);
                            image- vs participant-level split; best-TEST-epoch
                            checkpoint vs validation-selected epoch vs final
                            epoch (one training run gives all three).

Metrics are reported at the image level (what most prior papers report) and
at the participant level (mean probability per child).

Debugging
---------
Every experiment runs explicit checks (split disjointness, no participant or
augmented-sibling overlap where the protocol forbids it, scaler fitted on
train only, labels constant within participant, permutation validity,
label-invariance of nested selection, embedding determinism, pretrained
weights actually loaded, ...). FAILs are printed immediately; a summary is
printed at the end and saved to debug_checks.csv. --strict stops at the first
critical failure. --self-test runs E1-E4 end-to-end on synthetic images with
NO class signal but strong participant identity and verifies that the pipeline
detects identity leakage.

Outputs (out_dir)
-----------------
  E1_metrics.csv / E2_metrics.csv / E3_metrics.csv / E4_metrics.csv
        one row per run x level x subset (real and permuted labels)
  summary_metrics.csv       mean, SD, 2.5/97.5 percentiles over repeats
  paired_differences.csv    leaky minus proper protocol, paired by repeat
  permutation_summary.csv   null mean/SD/97.5th pct and empirical p-values
  mechanism_summary.csv     share of test images whose participant / source
                            image is in train; nearest-neighbour same-participant rate
  E3_selection.csv, E4_curves.csv   chosen epochs / configs per fold
  data_images.csv, data_exclusions.csv, data_participants.csv
  paper_numbers.md, debug_checks.csv, run_config.json

Examples
--------
  python run_audit_replay_E1_E4.py --self-test
  python run_audit_replay_E1_E4.py --experiments E1 E2 E3 --strict
  python run_audit_replay_E1_E4.py --experiments E4 --strict
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import hashlib
import inspect
import json
import re
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\zxzxk\Downloads\7073087")
DEF_IMAGES = ROOT / "Images"
DEF_META = ROOT / "Metadata" / "Metadata" / "Metadata_Participants.csv"
DEF_OUT = ROOT / "outputs" / "audit_replay_E1_E4"
PID_REGEX = r"_(\d+)\.[^.]+$"          # TS001_39.png -> participant 39
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

E3_CONFIGS = [dict(h=h, p=p, lr=lr, wd=1e-2)
              for h in (64, 256) for p in (0.2, 0.5) for lr in (1e-3, 3e-4)]
E3_DEFAULT = 6   # h=256, p=0.5, lr=1e-3


# ─────────────────────────────────────────────────────────────────────
# Debug infrastructure
# ─────────────────────────────────────────────────────────────────────
class LocalChecks:
    """Collects checks inside worker processes; merged into CHK by main."""

    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail="", critical=False):
        self.rows.append(dict(check=name, ok=bool(ok), critical=bool(critical),
                              detail=str(detail)[:300]))


class Checker:
    def __init__(self):
        self.rows, self.seen, self.strict = [], set(), False

    def check(self, name, ok, detail="", critical=False):
        ok = bool(ok)
        self.rows.append(dict(check=name, ok=ok, critical=bool(critical), detail=str(detail)[:300]))
        if not ok:
            print(f"  [DEBUG {'CRITICAL FAIL' if critical else 'FAIL'}] {name}: {detail}")
            if critical and self.strict:
                raise SystemExit(f"[STOP] critical check failed: {name}")
        elif name not in self.seen:
            print(f"  [DEBUG PASS] {name}" + (f": {detail}" if detail != "" else ""))
        self.seen.add(name)

    def merge(self, rows):
        for r in rows:
            self.check(r["check"], r["ok"], r["detail"], r["critical"])

    @staticmethod
    def info(msg):
        print(f"  [DEBUG INFO] {msg}")

    def summary(self, path=None):
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        g = df.groupby("check").agg(n=("ok", "size"), n_fail=("ok", lambda s: int((~s).sum())),
                                    critical=("critical", "max")).reset_index()
        print("\n" + "=" * 86 + "\n  DEBUG SUMMARY\n" + "=" * 86)
        for _, r in g.iterrows():
            st = "PASS" if r.n_fail == 0 else ("CRITICAL FAIL" if r.critical else "FAIL")
            print(f"  {st:>13}  {r.check[:64]:<64} ({r.n - r.n_fail}/{r.n})")
        tot = int(g.n_fail.sum())
        print("=" * 86)
        print(f"  {'ALL CHECKS PASSED' if tot == 0 else f'{tot} CHECK(S) FAILED'}")
        print("=" * 86)
        if path is not None:
            df.to_csv(path, index=False)
        return df


CHK = Checker()


def src_hash(*objs):
    h = hashlib.sha1()
    for o in objs:
        h.update((inspect.getsource(o) if callable(o) else json.dumps(o, sort_keys=True, default=str)).encode())
    return h.hexdigest()[:10]


# ─────────────────────────────────────────────────────────────────────
# Metadata and images
# ─────────────────────────────────────────────────────────────────────
def normalize_pid(v):
    t = str(v).strip()
    try:
        f = float(t)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return t


def find_column(columns, candidates):
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    for c in candidates:
        k = re.sub(r"[^a-z0-9]", "", c.lower())
        if k in norm:
            return norm[k]
    return None


def load_metadata(path, asd_label, td_label):
    meta = pd.read_csv(path)
    meta.columns = [str(c).strip() for c in meta.columns]
    pid_c = find_column(meta.columns, ["ParticipantID", "Participant", "SubjectID", "Subject", "ID"])
    cls_c = find_column(meta.columns, ["Class", "Diagnosis", "Group", "Dx"])
    age_c = find_column(meta.columns, ["Age"])
    sex_c = find_column(meta.columns, ["Gender", "Sex"])
    if pid_c is None or cls_c is None:
        raise SystemExit(f"[FATAL] cannot find participant/class columns in {list(meta.columns)}")
    m = pd.DataFrame({"pid": meta[pid_c].map(normalize_pid),
                      "raw_class": meta[cls_c].astype(str).str.strip().str.upper()})
    m["age"] = pd.to_numeric(meta[age_c], errors="coerce") if age_c else np.nan
    m["sex"] = meta[sex_c].astype(str).str.strip().str.upper() if sex_c else ""
    asd = asd_label.upper()
    vals = sorted(m.raw_class.unique().tolist())
    if td_label:
        td = td_label.upper()
    else:
        others = [v for v in vals if v != asd]
        if len(others) != 1:
            raise SystemExit(f"[FATAL] cannot infer TD label from class values {vals}; use --td-label")
        td = others[0]
    CHK.check("meta: ASD and TD labels present", asd in vals and td in vals,
              f"class values {vals}; ASD={asd}, TD={td}", critical=True)
    m = m[m.raw_class.isin([asd, td])].copy()
    m["label"] = (m.raw_class == asd).astype(int)
    print(f"[META] columns pid={pid_c!r} class={cls_c!r} age={age_c!r} sex={sex_c!r}; "
          f"{(m.label == 1).sum()} ASD rows, {(m.label == 0).sum()} TD rows")
    dup = m.duplicated(["pid", "raw_class"]).sum()
    CHK.check("meta: no duplicated (participant, class) rows", dup == 0, dup, critical=True)
    return m, asd, td


def discover_images(images_dir, meta, pid_regex, asd, td):
    pat = re.compile(pid_regex)
    files = sorted(p for p in Path(images_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
    n_cls = meta.groupby("pid").raw_class.nunique()
    collide = set(n_cls[n_cls > 1].index)
    rows, excl = [], []
    for f in files:
        mt = pat.search(f.name)
        if mt is None:
            excl.append(dict(path=str(f), reason="unparsable_pid")); continue
        pid = normalize_pid(mt.group(1))
        hint = None
        up = f.name.upper()
        for lab in (asd, td):
            if up.startswith(lab):
                hint = lab
        if hint is None:
            for part in [p.upper() for p in f.parts[:-1]][::-1]:
                for lab in (asd, td):
                    if part.startswith(lab):
                        hint = lab
                if hint:
                    break
        cand = meta[meta.pid == pid]
        if hint is not None:
            if len(cand) and not (cand.raw_class == hint).any():
                excl.append(dict(path=str(f), reason=f"class_mismatch(file={hint},meta={cand.raw_class.tolist()})"))
                continue
            cand = cand[cand.raw_class == hint]
        if len(cand) == 0:
            excl.append(dict(path=str(f), reason="missing_metadata")); continue
        if len(cand) > 1:
            excl.append(dict(path=str(f), reason="ambiguous_class_for_pid")); continue
        r = cand.iloc[0]
        key = f"{r.raw_class}_{pid}" if pid in collide else pid
        rows.append(dict(path=str(f), fname=f.name, pid=key, raw_pid=pid, label=int(r.label),
                         raw_class=r.raw_class, class_hint=hint or "", age=r.age, sex=r.sex))
    df = pd.DataFrame(rows)
    ex = pd.DataFrame(excl, columns=["path", "reason"])
    CHK.check("data: images matched to metadata", len(df) > 0, f"{len(df)} matched, {len(ex)} excluded",
              critical=True)
    if len(ex):
        CHK.info(f"exclusions by reason: {ex.reason.str.split('(').str[0].value_counts().to_dict()}")
    CHK.check("data: no filename/folder class contradicting metadata",
              not ex.reason.str.startswith("class_mismatch").any(),
              ex[ex.reason.str.startswith("class_mismatch")].head(3).to_dict("records"))
    if collide:
        n_nohint = int((df.raw_pid.isin(collide) & (df.class_hint == "")).sum())
        CHK.check("data: participant IDs shared by ASD and TD resolved by class prefix",
                  n_nohint == 0, f"{len(collide)} colliding IDs; {n_nohint} images without class hint",
                  critical=True)
        CHK.info(f"{len(collide)} participant IDs occur in both classes -> keys like 'TS_12' / 'TC_12'")
    CHK.check("data: label constant within participant",
              (df.groupby("pid").label.nunique() == 1).all(), critical=True)
    CHK.check("data: no duplicated image paths", not df.path.duplicated().any(), critical=True)
    # exact-duplicate image files (same bytes)
    md5 = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in df.path]
    df["md5"] = md5
    dups = df[df.md5.duplicated(keep=False)]
    cross = dups.groupby("md5").pid.nunique()
    CHK.check("data: no byte-identical images across different participants",
              (cross <= 1).all(), f"{int((cross > 1).sum())} duplicated groups across participants")
    if len(dups):
        CHK.info(f"{len(dups)} images belong to byte-identical groups (within participant: "
                 f"{int((cross == 1).sum())} groups)")
    per = df.groupby(["pid", "label"]).size().reset_index(name="n_images")
    for lab, nm in [(1, "ASD"), (0, "TD")]:
        s = per[per.label == lab].n_images
        CHK.info(f"{nm}: {len(s)} participants, {int(s.sum())} images "
                 f"(per participant min {s.min()}, median {s.median():.0f}, max {s.max()})")
    return df.reset_index(drop=True), ex, per


# ─────────────────────────────────────────────────────────────────────
# Augmentations (deterministic, typical of prior work) and backbones
# ─────────────────────────────────────────────────────────────────────
def _fill(im):
    return im.getpixel((0, 0))


def _zoom(im, f=0.9):
    from PIL import Image
    w, h = im.size
    dw, dh = int(w * (1 - f) / 2), int(h * (1 - f) / 2)
    return im.crop((dw, dh, w - dw, h - dh)).resize((w, h), Image.BILINEAR)


def _shift(im, f=0.05):
    from PIL import Image
    w, h = im.size
    return im.transform(im.size, Image.AFFINE, (1, 0, -f * w, 0, 1, -f * h), fillcolor=_fill(im))


def aug_list():
    from PIL import Image, ImageOps
    return [("hflip", ImageOps.mirror),
            ("rot+10", lambda im: im.rotate(10, resample=Image.BILINEAR, fillcolor=_fill(im))),
            ("rot-10", lambda im: im.rotate(-10, resample=Image.BILINEAR, fillcolor=_fill(im))),
            ("zoom0.9", _zoom),
            ("shift5%", _shift),
            ("vflip", ImageOps.flip)]


def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def build_encoder(name, device, pretrained):
    if name == "pixels":   # self-test only
        def enc(imgs):
            return np.stack([np.asarray(im.convert("L").resize((32, 32)), np.float32).ravel() / 255.0
                             for im in imgs])
        return enc
    import torch
    import torchvision.transforms as T
    if name in ("resnet50", "resnet18"):
        import torchvision
        if name == "resnet50":
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = torchvision.models.resnet50(weights=w)
        else:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.resnet18(weights=w)
        m.fc = torch.nn.Identity()
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        fwd = m
    elif name in ("eva02_b16", "clip_vitl14"):
        import open_clip
        arch, tag = {"eva02_b16": ("EVA02-B-16", "merged2b_s8b_b131k"),
                     "clip_vitl14": ("ViT-L-14", "openai")}[name]
        m, _, tf = open_clip.create_model_and_transforms(arch, pretrained=tag if pretrained else None)
        fwd = m.encode_image
    elif name == "dinov2_vitb14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained)
        tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        fwd = m
    else:
        raise SystemExit(f"[FATAL] unknown backbone {name}")
    m = m.to(device).eval()

    def enc(imgs):
        with torch.no_grad():
            x = torch.stack([tf(im) for im in imgs]).to(device)
            return fwd(x).float().cpu().numpy()
    return enc


def get_embeddings(df, backbone, n_aug, device, pretrained, cache_dir, batch=32):
    """Returns X (n_items, d), src (index of source image), aug (0 = original)."""
    augs = aug_list()[:n_aug]
    sig = hashlib.sha1(json.dumps(dict(bb=backbone, aug=[a for a, _ in augs], pre=pretrained,
                                       files=[(p, Path(p).stat().st_size) for p in df.path]),
                                  sort_keys=True).encode()).hexdigest()[:10]
    cf = Path(cache_dir) / f"emb_{backbone}_aug{n_aug}_{sig}.npz"
    if cf.exists():
        z = np.load(cf)
        print(f"[EMB] {backbone} aug={n_aug}: loaded cache {cf.name} {z['X'].shape}")
        X, src, aug = z["X"], z["src"], z["aug"]
    else:
        t0 = time.time()
        enc = build_encoder(backbone, device, pretrained)
        Xs, srcs, augs_i = [], [], []
        idx = list(range(len(df)))
        for b in range(0, len(idx), batch):
            ims, s_, a_ = [], [], []
            for i in idx[b:b + batch]:
                im = load_rgb(df.path[i])
                ims.append(im); s_.append(i); a_.append(0)
                for k, (_, fn) in enumerate(augs, 1):
                    ims.append(fn(im)); s_.append(i); a_.append(k)
            Xs.append(enc(ims)); srcs += s_; augs_i += a_
        X = np.concatenate(Xs).astype(np.float32)
        src, aug = np.array(srcs), np.array(augs_i)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cf, X=X, src=src, aug=aug)
        print(f"[EMB] {backbone} aug={n_aug}: extracted {X.shape} in {(time.time()-t0)/60:.1f} min")
        # determinism: re-encode a few originals
        enc2 = enc
        re_ = enc2([load_rgb(df.path[i]) for i in range(min(4, len(df)))])
        orig_rows = np.where(aug == 0)[0][:len(re_)]
        # GPU convolutions (cuDNN/TF32) are not bit-identical across batch sizes,
        # so compare direction (cosine) rather than raw values
        A_, B_ = re_, X[orig_rows]
        cos = (A_ * B_).sum(1) / (np.linalg.norm(A_, axis=1) * np.linalg.norm(B_, axis=1) + 1e-12)
        diff = float(np.abs(A_ - B_).max() / (np.abs(B_).max() + 1e-8))
        CHK.check("emb: re-encoding originals is reproducible (cosine > 0.9999)", cos.min() > 0.9999,
                  f"{backbone}: min cosine {cos.min():.6f}, max rel diff {diff:.2e}")
    CHK.check("emb: finite values", np.isfinite(X).all(), backbone, critical=True)
    CHK.check("emb: one row per (image, augmentation)", len(X) == len(df) * (1 + n_aug),
              f"{backbone}: {len(X)} rows", critical=True)
    CHK.check("emb: source index covers every image", set(src.tolist()) == set(range(len(df))),
              critical=True)
    if n_aug:
        o = X[aug == 0]
        a = X[aug == 1]
        same = np.array([np.allclose(o[i], a[i]) for i in range(len(o))])
        CHK.check("aug: augmented embeddings differ from their source", not same.all(),
                  f"{backbone}: identical in {same.mean():.2f} of images")
    return X, src, aug


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────
def clf_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec == prec and sens == sens and prec + sens) else np.nan
    return dict(n=len(y), AUC=float(auc), acc=float((pred == y).mean()), bal_acc=float((sens + spec) / 2),
                sens=float(sens), spec=float(spec), F1=float(f1), brier=float(((p - y) ** 2).mean()),
                majority_rate=float(max(y.mean(), 1 - y.mean())))


def level_metrics(y, pids, p, mask=None):
    """Image-level and participant-level (mean probability per participant)."""
    if mask is None:
        mask = np.ones(len(y), bool)
    y, pids, p = np.asarray(y)[mask], np.asarray(pids)[mask], np.asarray(p)[mask]
    out = {"image": clf_metrics(y, p)}
    d = pd.DataFrame(dict(pid=pids, y=y, p=p)).groupby("pid").agg(y=("y", "first"), p=("p", "mean"))
    out["participant"] = clf_metrics(d.y.values, d.p.values)
    return out


def permute_participant_labels(y, pids, rng):
    u = pd.DataFrame(dict(pid=pids, y=y)).groupby("pid").y.first()
    perm = dict(zip(u.index, rng.permutation(u.values)))
    return np.array([perm[p] for p in pids], int)


# ─────────────────────────────────────────────────────────────────────
# Shared CV engine (logistic regression with nested C, 1-NN memoriser)
# ─────────────────────────────────────────────────────────────────────
def strat_group_split(y, keys, n_splits, seed):
    """Deterministic stratified group k-fold, independent of the sklearn version.
    Groups (keys) are shuffled within each class and dealt round-robin to folds,
    continuing across classes, so every fold receives groups of both classes
    whenever each class has >= n_splits groups. Returns [(train_idx, test_idx)]."""
    y = np.asarray(y); keys = np.asarray(keys).astype(str)
    g = pd.DataFrame(dict(k=keys, y=y)).groupby("k", sort=True).y.agg(["first", "nunique"])
    if (g["nunique"] > 1).any():
        raise ValueError("label not constant within split group")
    rng = np.random.default_rng(seed)
    fold_of, pos = {}, int(rng.integers(n_splits))
    for c in sorted(g["first"].unique()):
        ks = g.index[g["first"] == c].to_numpy()
        if len(ks) < n_splits:
            raise ValueError(f"class {c} has {len(ks)} groups < n_splits={n_splits}")
        for k in rng.permutation(ks):
            fold_of[k] = pos % n_splits; pos += 1
    f = np.array([fold_of[k] for k in keys])
    idx = np.arange(len(keys))
    return [(idx[f != i], idx[f == i]) for i in range(n_splits)]


def lr_fit_predict(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1], sc


def select_C(Xtr, ytr, keys_tr, inner_splits, seed, lc):
    sp = strat_group_split(ytr, keys_tr, inner_splits, seed)
    oof = np.full((len(C_GRID), len(ytr)), np.nan)
    for itr, iva in sp:
        lc.check("inner: keys disjoint between inner-train and inner-val",
                 not set(keys_tr[itr]) & set(keys_tr[iva]), critical=True)
        for ci, C in enumerate(C_GRID):
            oof[ci, iva], _ = lr_fit_predict(Xtr[itr], ytr[itr], Xtr[iva], C)
    aucs = [roc_auc_score(ytr, o) for o in oof]
    return C_GRID[int(np.argmax(aucs))], aucs


def nn1(Xtr, Xte):
    a = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    b = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
    S = b @ a.T
    j = S.argmax(1)
    return j, S[np.arange(len(j)), j]


def cv_task(task_id, X, y, keys, pids, src, model, kind, n_splits, inner_splits, seed):
    """kind: participant-disjoint kinds must never share participants; the
    split_then_aug kinds must never share source images."""
    lc = LocalChecks()
    n = len(y)
    oof = np.full(n, np.nan); fold = np.full(n, -1)
    st = dict(test_pid_in_train=[], test_src_in_train=[], nn_same_pid=[], nn_same_src=[],
              nn_sim=[], C=[])
    for f, (tr, te) in enumerate(strat_group_split(y, keys, n_splits, seed)):
        lc.check("split: outer keys disjoint", not set(keys[tr]) & set(keys[te]), critical=True)
        lc.check("split: both classes present in train and test",
                 len(set(y[tr])) == 2 and len(set(y[te])) == 2, f"fold {f}", critical=True)
        if kind in ("participant", "split_then_aug_participant"):
            lc.check("leak: participant-level protocol shares no participant",
                     not set(pids[tr]) & set(pids[te]), critical=True)
        if kind in ("split_then_aug_image", "split_then_aug_participant"):
            lc.check("leak: split-then-augment has no augmented sibling of a test image in train",
                     not set(src[tr]) & set(src[te]), critical=True)
        st["test_pid_in_train"].append(float(np.isin(pids[te], pids[tr]).mean()))
        st["test_src_in_train"].append(float(np.isin(src[te], src[tr]).mean()))
        j, sim = nn1(X[tr], X[te])
        st["nn_same_pid"].append(float((pids[tr][j] == pids[te]).mean()))
        st["nn_same_src"].append(float((src[tr][j] == src[te]).mean()))
        st["nn_sim"].append(float(np.median(sim)))
        if model == "lr":
            C, _ = select_C(X[tr], y[tr], keys[tr], inner_splits, seed * 100 + f + 1, lc)
            p, sc = lr_fit_predict(X[tr], y[tr], X[te], C)
            lc.check("leak: scaler fitted on outer-train only",
                     np.allclose(sc.mean_, X[tr].mean(0), atol=1e-4), critical=True)
            st["C"].append(C)
        else:
            p = y[tr][j].astype(float)
        oof[te] = p; fold[te] = f
    lc.check("cv: every item predicted exactly once", np.isfinite(oof).all() and (fold >= 0).all(),
             critical=True)
    return dict(task=task_id, oof=oof.tolist(),
                stats={k: (float(np.mean(v)) if len(v) else None) for k, v in st.items()},
                checks=lc.rows)


# ─────────────────────────────────────────────────────────────────────
# Task runner with fingerprinted cache
# ─────────────────────────────────────────────────────────────────────
def run_tasks(specs, fn, cache_dir, fp, n_jobs, label):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    done, todo = {}, []
    for s in specs:
        cf = cache_dir / f"{s['task_id']}_{fp}.json"
        if cf.exists():
            done[s["task_id"]] = json.load(open(cf))
        else:
            todo.append(s)
    print(f"[{label}] {len(specs)} runs: {len(done)} cached, {len(todo)} to run (n_jobs={n_jobs})")
    t0 = time.time()
    if todo:
        chunk = max(1, min(len(todo), 2 * abs(n_jobs) if n_jobs > 0 else 16))
        for b in range(0, len(todo), chunk):
            part = todo[b:b + chunk]
            res = Parallel(n_jobs=n_jobs)(delayed(fn)(**s) for s in part)
            for s, r in zip(part, res):
                json.dump(r, open(cache_dir / f"{s['task_id']}_{fp}.json", "w"))
                done[s["task_id"]] = r
            el = time.time() - t0
            k = b + len(part)
            print(f"   {k}/{len(todo)} done, {el/60:.1f} min elapsed, ~{el/k*(len(todo)-k)/60:.1f} min left")
    for s in specs:
        CHK.merge(done[s["task_id"]]["checks"])
    return done


# ─────────────────────────────────────────────────────────────────────
# E1 split unit & E2 augmentation order
# ─────────────────────────────────────────────────────────────────────
def rows_from_result(res, y, pids, subsets, **tags):
    out = []
    for sub_name, mask in subsets.items():
        lm = level_metrics(y, pids, np.array(res["oof"]), mask)
        for lev, m in lm.items():
            out.append(dict(**tags, level=lev, subset=sub_name, **m, **{f"mech_{k}": v for k, v in res["stats"].items()}))
    return out


def run_E1(ctx):
    a, df = ctx["args"], ctx["df"]
    y, pids = df.label.values, df.pid.values
    src = np.arange(len(df))
    rows = []
    for bb in a.backbones:
        X, _, _ = get_embeddings(df, bb, 0, ctx["device"], ctx["pretrained"], ctx["emb_cache"])
        fp = src_hash(cv_task, select_C, lr_fit_predict, nn1, strat_group_split, dict(C=C_GRID, o=a.outer_splits,
                      i=a.inner_splits, sig=ctx["data_sig"], bb=bb, pre=ctx["pretrained"]))
        specs = []
        for model in ("lr", "1nn"):
            for proto in ("image", "participant"):
                keys = src.astype(str) if proto == "image" else pids
                for r in range(a.repeats):
                    specs.append(dict(task_id=f"E1_{bb}_{model}_{proto}_real_{r}", X=X, y=y, keys=keys,
                                      pids=pids, src=src, model=model, kind=proto,
                                      n_splits=a.outer_splits, inner_splits=a.inner_splits,
                                      seed=a.seed + r))
        perm_y = {}
        if bb in a.perm_backbones and a.n_perm > 0:
            for i in range(a.n_perm):
                yp = permute_participant_labels(y, pids, np.random.default_rng(a.seed + 50000 + i))
                perm_y[i] = yp
                for model in ("lr", "1nn"):
                    for proto in ("image", "participant"):
                        keys = src.astype(str) if proto == "image" else pids
                        specs.append(dict(task_id=f"E1_{bb}_{model}_{proto}_perm_{i}", X=X, y=yp,
                                          keys=keys, pids=pids, src=src, model=model, kind=proto,
                                          n_splits=a.outer_splits, inner_splits=a.inner_splits,
                                          seed=a.seed + (i % a.repeats)))
            check_permutations(y, pids, perm_y)
        res = run_tasks(specs, cv_task, ctx["cache"] / "E1", fp, a.n_jobs, f"E1 {bb}")
        for s in specs:
            kind_ = "perm" if "_perm_" in s["task_id"] else "real"
            rep = int(s["task_id"].rsplit("_", 1)[1])
            rows += rows_from_result(res[s["task_id"]], s["y"], pids, {"all": None}, exp="E1",
                                     backbone=bb, model=s["model"], protocol=s["kind"], kind=kind_, rep=rep)
    return pd.DataFrame(rows)


def check_permutations(y, pids, perm_y):
    u = pd.DataFrame(dict(pid=pids, y=y)).groupby("pid").y.first()
    ok_const, ok_counts, n_changed = True, True, []
    for yp in perm_y.values():
        d = pd.DataFrame(dict(pid=pids, y=yp)).groupby("pid").y
        ok_const &= bool((d.nunique() == 1).all())
        up = d.first().loc[u.index]
        ok_counts &= bool(up.sum() == u.sum())
        n_changed.append(float((up.values != u.values).mean()))
    CHK.check("perm: permuted label constant within participant", ok_const, critical=True)
    CHK.check("perm: participant-level class counts preserved", ok_counts, critical=True)
    CHK.info(f"perm: mean share of participants whose label changed = {np.mean(n_changed):.2f}")


def run_E2(ctx):
    a, df = ctx["args"], ctx["df"]
    y0, pid0 = df.label.values, df.pid.values
    bb = a.e2_backbone
    X, src, aug = get_embeddings(df, bb, a.n_aug, ctx["device"], ctx["pretrained"], ctx["emb_cache"])
    y, pids = y0[src], pid0[src]
    orig = aug == 0
    CHK.check("aug: augmented copies inherit participant and label",
              all((pids[src == i] == pid0[i]).all() and (y[src == i] == y0[i]).all() for i in range(len(df))),
              critical=True)
    item = np.arange(len(X)).astype(str)
    protos = {  # name: (item mask, split key, kind)
        "noaug_image": (orig, src.astype(str), "image"),
        "noaug_participant": (orig, pids, "participant"),
        "aug_then_split": (np.ones(len(X), bool), item, "aug_then_split"),
        "split_then_aug_image": (np.ones(len(X), bool), src.astype(str), "split_then_aug_image"),
        "split_then_aug_participant": (np.ones(len(X), bool), pids, "split_then_aug_participant"),
    }
    fp = src_hash(cv_task, select_C, lr_fit_predict, nn1, strat_group_split, dict(C=C_GRID, o=a.outer_splits,
                  i=a.inner_splits, sig=ctx["data_sig"], bb=bb, aug=a.n_aug, pre=ctx["pretrained"]))
    specs, meta_ = [], {}
    for model in ("lr", "1nn"):
        for pn, (mask, keys, kind) in protos.items():
            for r in range(a.repeats):
                tid = f"E2_{bb}_{model}_{pn}_real_{r}"
                specs.append(dict(task_id=tid, X=X[mask], y=y[mask], keys=keys[mask], pids=pids[mask],
                                  src=src[mask], model=model, kind=kind, n_splits=a.outer_splits,
                                  inner_splits=a.inner_splits, seed=a.seed + r))
                meta_[tid] = (pn, mask, "real", r)
    perm_y = {}
    for i in range(a.e2_n_perm):
        yp0 = permute_participant_labels(y0, pid0, np.random.default_rng(a.seed + 60000 + i))
        perm_y[i] = yp0
        yp = yp0[src]
        for model in ("lr", "1nn"):
            for pn in ("aug_then_split", "split_then_aug_participant"):
                mask, keys, kind = protos[pn]
                tid = f"E2_{bb}_{model}_{pn}_perm_{i}"
                specs.append(dict(task_id=tid, X=X[mask], y=yp[mask], keys=keys[mask], pids=pids[mask],
                                  src=src[mask], model=model, kind=kind, n_splits=a.outer_splits,
                                  inner_splits=a.inner_splits, seed=a.seed + (i % a.repeats)))
                meta_[tid] = (pn, mask, "perm", i)
    if perm_y:
        check_permutations(y0, pid0, perm_y)
    res = run_tasks(specs, cv_task, ctx["cache"] / "E2", fp, a.n_jobs, f"E2 {bb}")
    rows = []
    for s in specs:
        pn, mask, kind_, rep = meta_[s["task_id"]]
        is_o = orig[mask]
        subsets = {"orig": is_o} if pn.startswith("noaug") else {"orig": is_o, "all": None}
        rows += rows_from_result(res[s["task_id"]], s["y"], s["pids"], subsets, exp="E2", backbone=bb,
                                 model=s["model"], protocol=pn, kind=kind_, rep=rep)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# E3 MLP head: test-set early stopping / model selection
# ─────────────────────────────────────────────────────────────────────
def mlp_traj(Xtr, ytr, Xev, cfg, epochs, seed, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    xe = torch.tensor((Xev - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], cfg["h"]), nn.ReLU(), nn.Dropout(cfg["p"]),
                      nn.Linear(cfg["h"], 1)).to(device)
    pw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    out = np.empty((epochs, len(Xev)), np.float32)
    for e in range(epochs):
        m.train(); opt.zero_grad()
        lossf(m(xt).squeeze(-1), yt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            out[e] = torch.sigmoid(m(xe).squeeze(-1)).cpu().numpy()
    return out


def _acc_loss(P, y):
    """P (..., n) probabilities; returns accuracy and mean BCE along last axis."""
    y = np.asarray(y, float)
    Pc = np.clip(P, 1e-6, 1 - 1e-6)
    acc = ((P >= 0.5) == (y > 0.5)).mean(-1)
    loss = -(y * np.log(Pc) + (1 - y) * np.log(1 - Pc)).mean(-1)
    return acc, loss


def _best(acc, loss):
    """Lexicographic: highest accuracy, then lowest loss, then earliest index."""
    flat_acc, flat_loss = acc.ravel(), loss.ravel()
    order = np.lexsort((np.arange(flat_acc.size), flat_loss, -flat_acc))
    return np.unravel_index(order[0], acc.shape)


def e3_fold(Xtr, ytr, keys_tr, Xte, yte, epochs, inner_splits, seed, device, lc):
    C = len(E3_CONFIGS)
    outer = np.stack([mlp_traj(Xtr, ytr, Xte, cfg, epochs, seed + ci, device)
                      for ci, cfg in enumerate(E3_CONFIGS)])                  # (C, E, n_te)
    in_acc = np.zeros((C, epochs)); in_loss = np.zeros((C, epochs)); n_in = 0
    sp = strat_group_split(ytr, keys_tr, inner_splits, seed + 7)
    for k, (itr, iva) in enumerate(sp):
        lc.check("E3 inner: keys disjoint", not set(keys_tr[itr]) & set(keys_tr[iva]), critical=True)
        for ci, cfg in enumerate(E3_CONFIGS):
            P = mlp_traj(Xtr[itr], ytr[itr], Xtr[iva], cfg, epochs, seed + 1000 * (k + 1) + ci, device)
            acc, loss = _acc_loss(P, ytr[iva])
            in_acc[ci] += acc * len(iva); in_loss[ci] += loss * len(iva)
        n_in += len(iva)
    lc.check("E3 inner: pooled count == outer-train size", n_in == len(ytr), critical=True)
    in_acc /= n_in; in_loss /= n_in
    cn, en = _best(in_acc, in_loss)                      # never sees yte
    t_acc, t_loss = _acc_loss(outer, yte)                 # (C, E)  uses yte (leaky)
    cl, el = _best(t_acc, t_loss)
    D = E3_DEFAULT
    ed = _best(t_acc[D:D + 1], t_loss[D:D + 1])[1]           # epoch on test, config fixed
    end = _best(in_acc[D:D + 1], in_loss[D:D + 1])[1]        # epoch by inner CV, config fixed
    cc = _best(t_acc[:, -1:], t_loss[:, -1:])[0]             # config on test, final epoch
    cnc = _best(in_acc[:, -1:], in_loss[:, -1:])[0]          # config by inner CV, final epoch
    # 3 x 3 decomposition: {epoch: final / inner-CV / test} x {config: default / inner-CV / test}
    preds = dict(fixed_final=outer[D, -1],
                 nested_es_default=outer[D, end], leaky_es=outer[D, ed],
                 nested_cfg_final=outer[cnc, -1], leaky_cfg_final=outer[cc, -1],
                 leaky_select=outer[cl, el], nested_select=outer[cn, en])
    info = dict(nested_cfg=int(cn), nested_epoch=int(en) + 1, leaky_cfg=int(cl), leaky_epoch=int(el) + 1,
                leaky_es_epoch=int(ed) + 1, nested_es_default_epoch=int(end) + 1,
                leaky_cfg_final=int(cc), nested_cfg_final=int(cnc), test_acc_leaky=float(t_acc[cl, el]),
                test_acc_nested=float(t_acc[cn, en]), inner_acc_nested=float(in_acc[cn, en]))
    lc.check("E3 select: leaky test accuracy >= nested test accuracy (by construction)",
             t_acc[cl, el] >= t_acc[cn, en] - 1e-12, info)
    return preds, info


def e3_task(task_id, X, y, keys, pids, kind, epochs, n_splits, inner_splits, seed, device):
    import torch
    torch.set_num_threads(1)
    lc = LocalChecks()
    n = len(y)
    oof = {p: np.full(n, np.nan) for p in ("fixed_final", "nested_es_default", "leaky_es", "nested_cfg_final",
                                           "leaky_cfg_final", "leaky_select", "nested_select")}
    sel = []
    st = dict(test_pid_in_train=[])
    for f, (tr, te) in enumerate(strat_group_split(y, keys, n_splits, seed)):
        lc.check("E3 split: outer keys disjoint", not set(keys[tr]) & set(keys[te]), critical=True)
        lc.check("split: both classes present in train and test",
                 len(set(y[tr])) == 2 and len(set(y[te])) == 2, f"E3 fold {f}")
        if kind == "participant":
            lc.check("leak: participant-level protocol shares no participant",
                     not set(pids[tr]) & set(pids[te]), critical=True)
        st["test_pid_in_train"].append(float(np.isin(pids[te], pids[tr]).mean()))
        preds, info = e3_fold(X[tr], y[tr], keys[tr], X[te], y[te], epochs, inner_splits,
                              seed * 100 + f, device, lc)
        for p, v in preds.items():
            oof[p][te] = v
        sel.append(dict(fold=f, **info))
    for p in oof:
        lc.check("E3 cv: every item predicted exactly once", np.isfinite(oof[p]).all(), critical=True)
    return dict(task=task_id, oof={k: v.tolist() for k, v in oof.items()}, selection=sel,
                stats={k: float(np.mean(v)) for k, v in st.items()}, checks=lc.rows)


def e3_invariance_test(X, y, keys, epochs, inner_splits, seed, device):
    """Flip ONLY the test labels of one fold: nested/fixed predictions must not move."""
    tr, te = strat_group_split(y, keys, 5, seed)[0]
    lc = LocalChecks()
    pa, ia = e3_fold(X[tr], y[tr], keys[tr], X[te], y[te], epochs, inner_splits, seed, device, lc)
    pb, ib = e3_fold(X[tr], y[tr], keys[tr], X[te], 1 - y[te], epochs, inner_splits, seed, device, lc)
    tol = 1e-5 if device == "cpu" else 1e-3
    for p in ("nested_select", "nested_es_default", "nested_cfg_final", "fixed_final"):
        d = float(np.abs(pa[p] - pb[p]).max())
        CHK.check(f"invariance E3: {p} unchanged when only test labels are flipped", d <= tol,
                  f"max |diff| {d:.2e}", critical=True)
    CHK.check("invariance E3: nested (config, epoch) unchanged",
              (ia["nested_cfg"], ia["nested_epoch"]) == (ib["nested_cfg"], ib["nested_epoch"]),
              critical=True)
    CHK.info(f"E3 leaky choice with true test labels: cfg {ia['leaky_cfg']} epoch {ia['leaky_epoch']}; "
             f"with flipped labels: cfg {ib['leaky_cfg']} epoch {ib['leaky_epoch']} "
             f"(leaky follows the test labels)")


def run_E3(ctx):
    a, df = ctx["args"], ctx["df"]
    y, pids = df.label.values, df.pid.values
    bb = a.e3_backbone
    X, _, _ = get_embeddings(df, bb, 0, ctx["device"], ctx["pretrained"], ctx["emb_cache"])
    src = np.arange(len(df)).astype(str)
    if a.check_invariance:
        print("[E3] label-invariance test ...")
        e3_invariance_test(X, y, pids, a.e3_epochs, a.inner_splits, a.seed, a.e3_device)
    fp = src_hash(e3_task, e3_fold, mlp_traj, _best, _acc_loss, strat_group_split,
                  dict(cfg=E3_CONFIGS, d=E3_DEFAULT, ep=a.e3_epochs, o=a.outer_splits,
                       i=a.inner_splits, sig=ctx["data_sig"], bb=bb, pre=ctx["pretrained"]))
    specs, meta_ = [], {}
    for kind in ("participant", "image"):
        keys = pids if kind == "participant" else src
        for r in range(a.e3_repeats):
            tid = f"E3_{bb}_{kind}_real_{r}"
            specs.append(dict(task_id=tid, X=X, y=y, keys=keys, pids=pids, kind=kind, epochs=a.e3_epochs,
                              n_splits=a.outer_splits, inner_splits=a.inner_splits, seed=a.seed + r,
                              device=a.e3_device))
            meta_[tid] = (kind, "real", r, y)
    perm_y = {}
    for i in range(a.e3_n_perm):
        yp = permute_participant_labels(y, pids, np.random.default_rng(a.seed + 70000 + i))
        perm_y[i] = yp
        for kind in ("participant", "image"):
            keys = pids if kind == "participant" else src
            tid = f"E3_{bb}_{kind}_perm_{i}"
            specs.append(dict(task_id=tid, X=X, y=yp, keys=keys, pids=pids, kind=kind, epochs=a.e3_epochs,
                              n_splits=a.outer_splits, inner_splits=a.inner_splits,
                              seed=a.seed + (i % a.e3_repeats), device=a.e3_device))
            meta_[tid] = (kind, "perm", i, yp)
    if perm_y:
        check_permutations(y, pids, perm_y)
    res = run_tasks(specs, e3_task, ctx["cache"] / "E3", fp, a.n_jobs, f"E3 {bb}")
    rows, sel_rows = [], []
    for s in specs:
        kind, kind_, rep, yy = meta_[s["task_id"]]
        r = res[s["task_id"]]
        for proto, oof in r["oof"].items():
            rr = dict(oof=oof, stats=r["stats"])
            rows += rows_from_result(rr, yy, pids, {"all": None}, exp="E3", backbone=bb, model="mlp",
                                     protocol=f"{kind}_split/{proto}", kind=kind_, rep=rep)
        if kind_ == "real":
            for sr in r["selection"]:
                sel_rows.append(dict(split=kind, rep=rep, **sr))
    pd.DataFrame(sel_rows).to_csv(ctx["out"] / "E3_selection.csv", index=False)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# E4 end-to-end CNN fine-tuning
# ─────────────────────────────────────────────────────────────────────
def build_cnn(arch, pretrained):
    import torch.nn as nn
    import torchvision
    if arch == "tiny":
        return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
                             nn.Flatten(), nn.Linear(128, 1))
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                                        if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                                        if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "vgg16":
        m = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
                                     if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else:
        raise SystemExit(f"[FATAL] unknown E4 arch {arch}")
    return m


def e4_run(imgs, y, pids, kind, seed, a, device, pretrained, lc, check_pretrained=False):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    n = len(y)
    keys = pids if kind == "participant" else np.arange(n).astype(str)
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tf_train = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(10), T.ToTensor(), norm])
    tf_eval = T.Compose([T.ToTensor(), norm])
    use_amp = device == "cuda"
    oof = {p: np.full(n, np.nan) for p in ("final_epoch", "best_test_epoch", "best_val_epoch")}
    curves = []
    st = dict(test_pid_in_train=[])

    def predict(model, idx):
        model.eval(); out = []
        with torch.no_grad():
            for b in range(0, len(idx), a.e4_batch):
                x = torch.stack([tf_eval(imgs[i]) for i in idx[b:b + a.e4_batch]]).to(device)
                with torch.autocast("cuda", enabled=use_amp):
                    out.append(torch.sigmoid(model(x).float().squeeze(-1)).cpu().numpy())
        return np.concatenate(out)

    for f, (tr, te) in enumerate(strat_group_split(y, keys, a.outer_splits, seed)):
        itr, iva = strat_group_split(y[tr], keys[tr], 5, seed + f + 1)[0]
        trn, val = tr[itr], tr[iva]
        lc.check("split: both classes present in train and test",
                 all(len(set(y[s_])) == 2 for s_ in (trn, val, te)), f"E4 fold {f}")
        lc.check("E4 split: train/val/test item sets pairwise disjoint",
                 not (set(trn) & set(val) or set(trn) & set(te) or set(val) & set(te)), critical=True)
        if kind == "participant":
            lc.check("leak: E4 participant protocol shares no participant across train/val/test",
                     not (set(pids[trn]) & set(pids[te]) or set(pids[val]) & set(pids[te])
                          or set(pids[trn]) & set(pids[val])), critical=True)
        st["test_pid_in_train"].append(float(np.isin(pids[te], pids[trn]).mean()))
        torch.manual_seed(seed * 100 + f)
        model = build_cnn(a.e4_arch, pretrained).to(device)
        if check_pretrained and f == 0 and a.e4_arch != "tiny":
            torch.manual_seed(0)
            fresh = build_cnn(a.e4_arch, False)
            w_p = next(model.parameters()).detach().cpu(); w_f = next(fresh.parameters()).detach()
            lc.check("E4: ImageNet weights actually loaded" if pretrained else "E4: random init as requested",
                     (not torch.allclose(w_p, w_f)) if pretrained else True,
                     f"first-layer weight mean {w_p.mean():.4f} vs random {w_f.mean():.4f}",
                     critical=pretrained)
        pw = float((y[trn] == 0).sum()) / max(float((y[trn] == 1).sum()), 1.0)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=a.e4_lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        g = np.random.default_rng(seed * 1000 + f)
        E = a.e4_epochs
        Pv = np.empty((E, len(val)), np.float32); Pt = np.empty((E, len(te)), np.float32)
        for e in range(E):
            model.train()
            order = g.permutation(trn)
            for b in range(0, len(order), a.e4_batch):
                idx = order[b:b + a.e4_batch]
                x = torch.stack([tf_train(imgs[i]) for i in idx]).to(device)
                t = torch.tensor(y[idx], dtype=torch.float32, device=device)
                opt.zero_grad()
                with torch.autocast("cuda", enabled=use_amp):
                    loss = lossf(model(x).float().squeeze(-1), t)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            Pv[e] = predict(model, val); Pt[e] = predict(model, te)
        if f == 0:
            again = predict(model, te)
            lc.check("E4: evaluation is deterministic (no augmentation at test time)",
                     np.abs(again - Pt[-1]).max() < (1e-3 if use_amp else 1e-5),
                     f"max diff {np.abs(again - Pt[-1]).max():.2e}")
        v_acc, v_loss = _acc_loss(Pv, y[val]); t_acc, t_loss = _acc_loss(Pt, y[te])
        e_val = int(_best(v_acc[None], v_loss[None])[1])
        e_test = int(_best(t_acc[None], t_loss[None])[1])
        # invariance: flip ONLY the test labels in the full label vector and
        # recompute the validation-based choice through the same indexing
        y_mod = y.copy(); y_mod[te] = 1 - y_mod[te]
        v_acc2, v_loss2 = _acc_loss(Pv, y_mod[val])
        lc.check("invariance E4: validation-selected epoch unchanged when test labels are flipped",
                 int(_best(v_acc2[None], v_loss2[None])[1]) == e_val, critical=True)
        oof["final_epoch"][te] = Pt[-1]; oof["best_test_epoch"][te] = Pt[e_test]
        oof["best_val_epoch"][te] = Pt[e_val]
        curves.append(dict(fold=f, epoch_val=e_val + 1, epoch_test=e_test + 1,
                           val_acc=v_acc.tolist(), test_acc=t_acc.tolist(),
                           test_acc_at_val_epoch=float(t_acc[e_val]), test_acc_at_test_epoch=float(t_acc[e_test])))
        lc.check("E4 select: best-test-epoch accuracy >= val-selected accuracy (by construction)",
                 t_acc[e_test] >= t_acc[e_val] - 1e-12)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    for p in oof:
        lc.check("E4 cv: every image predicted exactly once", np.isfinite(oof[p]).all(), critical=True)
    return oof, curves, {k: float(np.mean(v)) for k, v in st.items()}


def run_E4(ctx):
    from PIL import Image
    a, df = ctx["args"], ctx["df"]
    y, pids = df.label.values, df.pid.values
    t0 = time.time()
    imgs = [load_rgb(p).resize((a.e4_img_size, a.e4_img_size), Image.BILINEAR) for p in df.path]
    print(f"[E4] {len(imgs)} images preloaded at {a.e4_img_size}px ({time.time()-t0:.0f}s); "
          f"arch={a.e4_arch}, pretrained={ctx['pretrained']}, device={ctx['device']}")
    fp = src_hash(e4_run, build_cnn, _best, _acc_loss, strat_group_split,
                  dict(arch=a.e4_arch, ep=a.e4_epochs, lr=a.e4_lr, b=a.e4_batch, s=a.e4_img_size,
                       o=a.outer_splits, sig=ctx["data_sig"], pre=ctx["pretrained"]))
    cdir = ctx["cache"] / "E4"; cdir.mkdir(parents=True, exist_ok=True)
    runs = [(k, "real", s, y) for s in range(a.e4_seeds) for k in ("image", "participant")]
    for i in range(a.e4_n_perm):
        yp = permute_participant_labels(y, pids, np.random.default_rng(a.seed + 80000 + i))
        runs += [(k, "perm", i, yp) for k in ("image", "participant")]
    rows, curve_rows = [], []
    first = True
    for kind, kind_, rep, yy in runs:
        tid = f"E4_{a.e4_arch}_{kind}_{kind_}_{rep}"
        cf = cdir / f"{tid}_{fp}.json"
        if cf.exists():
            r = json.load(open(cf)); print(f"[E4] {tid}: cached")
        else:
            t1 = time.time()
            lc = LocalChecks()
            oof, curves, st = e4_run(imgs, yy, pids, kind, a.seed + rep, a, ctx["device"],
                                     ctx["pretrained"], lc, check_pretrained=first)
            first = False
            r = dict(oof={k: v.tolist() for k, v in oof.items()}, curves=curves, stats=st, checks=lc.rows)
            json.dump(r, open(cf, "w"))
            print(f"[E4] {tid}: {(time.time()-t1)/60:.1f} min")
        CHK.merge(r["checks"])
        for proto, oof in r["oof"].items():
            rows += rows_from_result(dict(oof=oof, stats=r["stats"]), yy, pids, {"all": None}, exp="E4",
                                     backbone=a.e4_arch, model="finetune", protocol=f"{kind}_split/{proto}",
                                     kind=kind_, rep=rep)
        if kind_ == "real":
            for c in r["curves"]:
                curve_rows.append(dict(split=kind, rep=rep, fold=c["fold"], epoch_val=c["epoch_val"],
                                       epoch_test=c["epoch_test"],
                                       test_acc_at_val_epoch=c["test_acc_at_val_epoch"],
                                       test_acc_at_test_epoch=c["test_acc_at_test_epoch"]))
    pd.DataFrame(curve_rows).to_csv(ctx["out"] / "E4_curves.csv", index=False)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────────────────────────────
KEYS = ["exp", "backbone", "model", "protocol", "level", "subset"]
METRICS = ["AUC", "bal_acc", "acc", "sens", "spec", "F1", "brier"]

PAIRS = {
    "E1": [("image", "participant")],
    "E2": [("noaug_image", "noaug_participant"), ("aug_then_split", "split_then_aug_participant"),
           ("aug_then_split", "split_then_aug_image"), ("split_then_aug_image", "split_then_aug_participant"),
           ("aug_then_split", "noaug_participant")],
    "E3": [("participant_split/leaky_select", "participant_split/nested_select"),
           # epoch-selection leak alone (config fixed at default)
           ("participant_split/leaky_es", "participant_split/nested_es_default"),
           # config-selection leak alone (epoch fixed at final)
           ("participant_split/leaky_cfg_final", "participant_split/nested_cfg_final"),
           ("participant_split/leaky_es", "participant_split/fixed_final"),
           ("image_split/leaky_select", "participant_split/nested_select"),
           ("image_split/nested_select", "participant_split/nested_select")],
    "E4": [("participant_split/best_test_epoch", "participant_split/best_val_epoch"),
           ("image_split/best_val_epoch", "participant_split/best_val_epoch"),
           ("image_split/best_test_epoch", "participant_split/best_val_epoch")],
}


def summarize(allm, out):
    real = allm[allm.kind == "real"]
    agg = []
    for k, d in real.groupby(KEYS):
        row = dict(zip(KEYS, k), n_repeats=len(d), n_units=int(d.n.iloc[0]),
                   majority_rate=float(d.majority_rate.iloc[0]))
        for m in METRICS:
            row[f"{m}_mean"] = d[m].mean(); row[f"{m}_sd"] = d[m].std(ddof=1) if len(d) > 1 else np.nan
            row[f"{m}_p2.5"] = d[m].quantile(0.025); row[f"{m}_p97.5"] = d[m].quantile(0.975)
        agg.append(row)
    summ = pd.DataFrame(agg); summ.to_csv(out / "summary_metrics.csv", index=False)

    pr = []
    for exp, pairs in PAIRS.items():
        e = real[real.exp == exp]
        for A, B in pairs:
            for k, d in e.groupby(["backbone", "model", "level", "subset"]):
                da, db = d[d.protocol == A].set_index("rep"), d[d.protocol == B].set_index("rep")
                common = da.index.intersection(db.index)
                if not len(common):
                    continue
                row = dict(exp=exp, backbone=k[0], model=k[1], level=k[2], subset=k[3], A=A, B=B,
                           n_pairs=len(common))
                for m in ("AUC", "bal_acc", "acc"):
                    diff = da.loc[common, m] - db.loc[common, m]
                    row[f"d{m}_mean"] = diff.mean(); row[f"d{m}_sd"] = diff.std(ddof=1) if len(diff) > 1 else np.nan
                    row[f"d{m}_p2.5"] = diff.quantile(0.025); row[f"d{m}_p97.5"] = diff.quantile(0.975)
                    row[f"d{m}_n_positive"] = int((diff > 0).sum())
                pr.append(row)
    pairs_df = pd.DataFrame(pr); pairs_df.to_csv(out / "paired_differences.csv", index=False)

    perm = allm[allm.kind == "perm"]
    ps = []
    real_groups = {k: d for k, d in real.groupby(KEYS)}
    for k, d in perm.groupby(KEYS):
        obs = real_groups.get(k)
        if obs is None:
            continue
        row = dict(zip(KEYS, k), n_perm=len(d))
        for m in ("AUC", "bal_acc"):
            null = d[m].dropna().values; o = float(obs[m].mean())
            row[f"{m}_observed_mean"] = o; row[f"{m}_null_mean"] = null.mean()
            row[f"{m}_null_sd"] = null.std(ddof=1) if len(null) > 1 else np.nan
            row[f"{m}_null_p97.5"] = np.quantile(null, 0.975)
            row[f"{m}_p_value"] = (1 + (null >= o).sum()) / (1 + len(null))
        ps.append(row)
    perm_df = pd.DataFrame(ps); perm_df.to_csv(out / "permutation_summary.csv", index=False)

    mcols = [c for c in real.columns if c.startswith("mech_")]
    mech = real[real.level == "image"].groupby(KEYS)[mcols].mean().reset_index()
    mech.to_csv(out / "mechanism_summary.csv", index=False)
    return summ, pairs_df, perm_df, mech


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def paper_numbers(summ, pairs_df, perm_df, mech, out, args, data_info):
    L = ["# Protocol-replay results (ASD vs TD)\n", data_info, ""]
    for exp in ("E1", "E2", "E3", "E4"):
        s = summ[summ.exp == exp]
        if s.empty:
            continue
        L.append(f"## {exp}")
        for _, r in s.sort_values(["backbone", "model", "level", "subset", "protocol"]).iterrows():
            L.append(f"- {r.backbone} / {r.model} / {r.level} / {r.subset} / **{r.protocol}**: "
                     f"AUC {fmt(r.AUC_mean)} ± {fmt(r.AUC_sd)}, bal.acc {fmt(r['bal_acc_mean'])} ± "
                     f"{fmt(r['bal_acc_sd'])}, acc {fmt(r.acc_mean)} (majority {fmt(r.majority_rate)}); "
                     f"n={r.n_units}, repeats={r.n_repeats}")
        p = pairs_df[pairs_df.exp == exp] if not pairs_df.empty else pairs_df
        for _, r in p.iterrows():
            L.append(f"  - Δ ({r.A} − {r.B}) [{r.backbone}/{r.model}/{r.level}/{r.subset}]: "
                     f"ΔAUC {fmt(r.dAUC_mean)} (2.5–97.5%: {fmt(r['dAUC_p2.5'])}–{fmt(r['dAUC_p97.5'])}), "
                     f"Δbal.acc {fmt(r.dbal_acc_mean)}, positive in {r.dAUC_n_positive}/{r.n_pairs}")
        q = perm_df[perm_df.exp == exp] if not perm_df.empty else perm_df
        for _, r in q.iterrows():
            L.append(f"  - permutation [{r.backbone}/{r.model}/{r.protocol}/{r.level}/{r.subset}]: "
                     f"observed AUC {fmt(r.AUC_observed_mean)} vs null {fmt(r.AUC_null_mean)} ± "
                     f"{fmt(r.AUC_null_sd)} (97.5th pct {fmt(r['AUC_null_p97.5'])}), p = {fmt(r.AUC_p_value)} "
                     f"(N = {r.n_perm})")
        m = mech[mech.exp == exp]
        for _, r in m.iterrows():
            vals = ", ".join(f"{c[5:]} {fmt(r[c], 2)}" for c in m.columns if c.startswith("mech_")
                             and r[c] == r[c])
            L.append(f"  - mechanism [{r.backbone}/{r.model}/{r.protocol}/{r.subset}]: {vals}")
        L.append("")
    (out / "paper_numbers.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────
# Synthetic self-test data
# ─────────────────────────────────────────────────────────────────────
def make_synthetic(root, n_per_class=12):
    """Participant-specific scanpath-like drawings, NO class signal; participant
    IDs deliberately collide between classes (TS 1..n and TC 1..n)."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(0)
    rows = []
    for cls, folder in (("TS", "TSImages"), ("TC", "TCImages")):
        d = Path(root) / "Images" / folder; d.mkdir(parents=True, exist_ok=True)
        k = 0
        for pid in range(1, n_per_class + 1):
            base = rng.uniform(8, 56, size=(6, 2)); col = tuple(int(c) for c in rng.integers(0, 255, 3))
            for _ in range(int(rng.integers(5, 9))):
                k += 1
                im = Image.new("RGB", (64, 64), (255, 255, 255)); dr = ImageDraw.Draw(im)
                pts = base + rng.normal(0, 1.5, base.shape)
                dr.line([tuple(p) for p in pts], fill=col, width=2)
                for p in pts:
                    dr.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], outline=col)
                im.save(d / f"{cls}{k:03d}_{pid}.png")
            rows.append(dict(ParticipantID=pid, Class=cls, Gender="M", Age=float(rng.uniform(3, 12)),
                             CARS=float(rng.uniform(20, 45)) if cls == "TS" else np.nan))
    md = Path(root) / "Metadata"; md.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(md / "Metadata_Participants.csv", index=False)
    return Path(root) / "Images", md / "Metadata_Participants.csv"


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["E1", "E2", "E3", "E4"],
                    choices=["E1", "E2", "E3", "E4"])
    ap.add_argument("--images-dir", type=Path, default=DEF_IMAGES)
    ap.add_argument("--metadata-csv", type=Path, default=DEF_META)
    ap.add_argument("--out-dir", type=Path, default=DEF_OUT)
    ap.add_argument("--pid-regex", default=PID_REGEX)
    ap.add_argument("--asd-label", default="TS")
    ap.add_argument("--td-label", default=None, help="default: the only other class value")
    ap.add_argument("--backbones", nargs="+", default=["resnet50", "eva02_b16"],
                    help="E1 frozen backbones: resnet50 resnet18 eva02_b16 clip_vitl14 dinov2_vitb14")
    ap.add_argument("--perm-backbones", nargs="+", default=None, help="default: first of --backbones")
    ap.add_argument("--e2-backbone", default="resnet50")
    ap.add_argument("--e3-backbone", default="eva02_b16")
    ap.add_argument("--n-aug", type=int, default=5, help="augmented copies per image in E2 (max 6)")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10, help="E1/E2 repeated splits")
    ap.add_argument("--n-perm", type=int, default=100, help="E1 permutations")
    ap.add_argument("--e2-n-perm", type=int, default=50)
    ap.add_argument("--e3-repeats", type=int, default=5)
    ap.add_argument("--e3-n-perm", type=int, default=50)
    ap.add_argument("--e3-epochs", type=int, default=200)
    ap.add_argument("--e3-device", default="cpu")
    ap.add_argument("--e4-arch", default="resnet50", choices=["resnet50", "resnet18", "vgg16", "tiny"])
    ap.add_argument("--e4-seeds", type=int, default=3)
    ap.add_argument("--e4-epochs", type=int, default=30)
    ap.add_argument("--e4-lr", type=float, default=1e-4)
    ap.add_argument("--e4-batch", type=int, default=32)
    ap.add_argument("--e4-img-size", type=int, default=224)
    ap.add_argument("--e4-n-perm", type=int, default=0, help="E4 permutations (each = 2 x 5-fold fine-tunes)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-invariance", dest="check_invariance", action="store_false")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic images (identity signal, no class signal), tiny settings")
    a = ap.parse_args()

    tmp = None
    if a.self_test:
        tmp = Path(tempfile.mkdtemp(prefix="audit_selftest_"))
        a.images_dir, a.metadata_csv = make_synthetic(tmp)
        a.out_dir = Path("selftest_out"); shutil.rmtree(a.out_dir, ignore_errors=True)
        a.backbones = ["pixels"]; a.e2_backbone = "pixels"; a.e3_backbone = "pixels"
        a.n_aug = 2; a.repeats = 3; a.n_perm = 4; a.e2_n_perm = 2; a.e3_repeats = 1; a.e3_n_perm = 2
        a.e3_epochs = 25; a.e4_arch = "tiny"; a.e4_seeds = 1; a.e4_epochs = 3; a.e4_img_size = 32
        a.e4_n_perm = 1; a.no_pretrained = True; a.n_jobs = 2
        print(f"[SELF-TEST] synthetic data in {tmp}")
    if a.perm_backbones is None:
        a.perm_backbones = a.backbones[:1]
    CHK.strict = a.strict

    import torch
    device = "cpu" if (a.cpu or not torch.cuda.is_available()) else "cuda"
    out = a.out_dir; out.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] experiments={a.experiments} device={device} out={out}")

    # optional backbones: skip unavailable ones instead of crashing mid-run
    def available(bb):
        if bb in ("eva02_b16", "clip_vitl14"):
            try:
                import open_clip  # noqa: F401
            except ImportError:
                return False
        return True
    for attr in ("backbones", "perm_backbones"):
        keep = [b for b in getattr(a, attr) if available(b)]
        if len(keep) < len(getattr(a, attr)):
            CHK.info(f"open_clip not installed -> dropping {set(getattr(a, attr)) - set(keep)} from --{attr} "
                     f"(pip install open_clip_torch to include them)")
        setattr(a, attr, keep)
    for attr in ("e2_backbone", "e3_backbone"):
        if not available(getattr(a, attr)):
            CHK.info(f"--{attr} {getattr(a, attr)} unavailable -> using resnet50")
            setattr(a, attr, "resnet50")

    print("\n[DEBUG] data checks")
    meta, asd, td = load_metadata(a.metadata_csv, a.asd_label, a.td_label)
    df, excl, per = discover_images(a.images_dir, meta, a.pid_regex, asd, td)
    df.to_csv(out / "data_images.csv", index=False); excl.to_csv(out / "data_exclusions.csv", index=False)
    per.to_csv(out / "data_participants.csv", index=False)
    n_asd, n_td = int((per.label == 1).sum()), int((per.label == 0).sum())
    data_info = (f"{n_asd} ASD and {n_td} TD participants; {int((df.label == 1).sum())} ASD and "
                 f"{int((df.label == 0).sum())} TD images; outer {a.outer_splits}-fold stratified, "
                 f"inner {a.inner_splits}-fold; seed {a.seed}")
    print(f"[DATA] {data_info}")
    data_sig = hashlib.sha1(json.dumps([df.path.tolist(), df.pid.tolist(), df.label.tolist()]).encode()).hexdigest()[:10]
    json.dump(dict(vars(a), device=device, data_sig=data_sig, torch=torch.__version__),
              open(out / "run_config.json", "w"), indent=2, default=str)

    ctx = dict(args=a, df=df, device=device, pretrained=not a.no_pretrained, out=out,
               cache=out / "cache", emb_cache=out / "cache" / "embeddings", data_sig=data_sig)
    frames = []
    t0 = time.time()
    for exp, fn in (("E1", run_E1), ("E2", run_E2), ("E3", run_E3), ("E4", run_E4)):
        if exp in a.experiments:
            print(f"\n{'=' * 30} {exp} {'=' * 30}")
            t1 = time.time()
            d = fn(ctx)
            d.to_csv(out / f"{exp}_metrics.csv", index=False)
            frames.append(d)
            print(f"[{exp}] done in {(time.time()-t1)/60:.1f} min")

    allm = pd.concat(frames, ignore_index=True)
    summ, pairs_df, perm_df, mech = summarize(allm, out)

    print("\n[DEBUG] post-run checks")
    for exp in allm.exp.unique():
        r = allm[(allm.exp == exp) & (allm.kind == "real")]
        CHK.check("post: metrics finite for real-label runs", np.isfinite(r.AUC).all(),
                  f"{exp}: {int((~np.isfinite(r.AUC)).sum())} NaN AUC")
    if "E1" in a.experiments:
        m = mech[(mech.exp == "E1") & (mech.protocol == "participant")]
        CHK.check("post: participant protocol has zero test participants in train",
                  (m.mech_test_pid_in_train == 0).all(), critical=True)
        mi = mech[(mech.exp == "E1") & (mech.protocol == "image")]
        CHK.info(f"E1 image protocol: share of test images whose participant is in train = "
                 f"{mi.mech_test_pid_in_train.mean():.2f}; nearest train image from same participant = "
                 f"{mi.mech_nn_same_pid.mean():.2f}")
    if not perm_df.empty:
        proper = {"participant", "split_then_aug_participant", "participant_split/nested_select",
                  "participant_split/nested_es_default", "participant_split/nested_cfg_final",
                  "participant_split/fixed_final", "participant_split/best_val_epoch",
                  "participant_split/final_epoch"}
        pp = perm_df[perm_df.protocol.isin(proper) & (perm_df.level == "participant") & (perm_df.n_perm >= 20)]
        for _, r in pp.iterrows():
            tol = max(0.05, 3 * r.AUC_null_sd / np.sqrt(r.n_perm))
            CHK.check("post: leak-free protocols have permutation-null AUC centred at 0.5",
                      abs(r.AUC_null_mean - 0.5) < tol,
                      f"{r.exp}/{r.backbone}/{r.model}/{r.protocol}: null mean {r.AUC_null_mean:.3f} (tol {tol:.3f})")
        lk = perm_df[perm_df.protocol.str.contains("leaky|best_test|image|aug_then", regex=True)
                     & (perm_df.level == "image")]
        for _, r in lk.iterrows():
            CHK.info(f"permutation-null AUC {r.exp}/{r.backbone}/{r.model}/{r.protocol}/{r.subset}: "
                     f"{r.AUC_null_mean:.3f} ± {r.AUC_null_sd:.3f}  (>0.5 = protocol manufactures signal)")
    if a.self_test and "E1" in a.experiments:
        s = summ[(summ.exp == "E1") & (summ.model == "1nn") & (summ.level == "image")].set_index("protocol")
        gap = s.loc["image", "AUC_mean"] - s.loc["participant", "AUC_mean"]
        CHK.check("self-test: identity leakage detected (1-NN image-split AUC >> participant-split AUC)",
                  gap > 0.2, f"image {s.loc['image', 'AUC_mean']:.2f} vs participant "
                             f"{s.loc['participant', 'AUC_mean']:.2f}", critical=True)
        sp = summ[(summ.exp == "E1") & (summ.level == "participant") & (summ.protocol == "participant")]
        CHK.check("self-test: no class signal -> participant-split AUC near chance",
                  (abs(sp.AUC_mean - 0.5) < 0.25).all(), sp[["model", "AUC_mean"]].round(2).to_dict("records"))

    txt = paper_numbers(summ, pairs_df, perm_df, mech, out, a, data_info)
    print("\n" + txt)
    CHK.summary(out / "debug_checks.csv")
    print(f"\n[DONE] {(time.time()-t0)/60:.1f} min -> {out}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
