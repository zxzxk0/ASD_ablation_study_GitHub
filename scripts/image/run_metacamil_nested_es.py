# -*- coding: utf-8 -*-
"""
run_metacamil_nested_es.py
==========================
MetaCA-MIL (image arm, EVA02-B/16 embeddings) under four epoch-selection
protocols, all computed from ONE training trajectory per outer fold so that
the only thing that differs between protocols is how the stopping epoch is
chosen.

Protocols (all participant-level LOPO, n = 26)
----------------------------------------------
  leaky_test_es       Early stopping on the HELD-OUT participant's MAE with
                      patience (exactly the rule in metacamil_stability.py /
                      mil_arch_search.py). Test labels leak into the
                      prediction. Reproduces the "non-nested" number.
  fixed_final_epoch   No early stopping; prediction after max_epochs.
  nested_refit        (PRIMARY) Inner K-fold CV over the 25 training
                      participants chooses the epoch E* that minimises
                      pooled inner-validation MAE; the outer model trained on
                      all 25 participants (same schedule) is read out at E*.
                      The test participant is never used for any decision.
  nested_inner_ens    Each inner-fold model is early-stopped on ITS OWN
                      inner-validation participants (same patience rule as
                      leaky), predicts the test participant at that epoch;
                      the K predictions are averaged.

The architecture/hyper-parameters are fixed to p3_best_cand (selected by the
earlier non-nested 86-configuration search). NOTE for the paper: because
that configuration was chosen with leaky LOPO scores, the nested result here
is, if anything, still optimistically biased with respect to architecture
selection. `--configs best_cand base` additionally runs the plain base
configuration (d_model 64, rh 64, 1 head, no LN / residual) that was not
selected on test scores.

Outputs (out_dir)
-----------------
  participant_predictions.csv  every seed x fold x protocol prediction, chosen epochs
  per_seed_metrics.csv         MAE, RMSE, R2, R2_oos, Spearman, Pearson, CCC,
                               bias, LoA, baseline MAE/R2, skill
  summary_metrics.csv          seed mean +- SD, min, max; seed-ensemble metrics
                               with participant bootstrap 95% CIs
  paired_differences.csv       protocol A - protocol B (dMAE, dR2) with
                               bootstrap CIs, Wilcoxon on |error|, per-seed deltas
  severity_metrics.csv         Mild / Moderate / Severe MAE, RMSE, bias
  epoch_selection.csv          per seed x fold chosen epochs (leaky, nested, inner)
  permutation_runs.csv         (if --n-perm > 0) full-pipeline label-permutation null
  permutation_summary.csv      p-values for every protocol
  paper_numbers.md             ready-to-paste numbers
  run_config.json

Resume: finished seeds / permutations are cached in out_dir/cache/*.json and
skipped on rerun.

Examples
--------
  python run_metacamil_nested_es.py                       # 5 seeds, best_cand
  python run_metacamil_nested_es.py --configs best_cand base
  python run_metacamil_nested_es.py --n-perm 100          # adds permutation null (slow)
  python run_metacamil_nested_es.py --smoke               # synthetic quick test
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────
# Paths (same as mil_arch_search.py / metacamil_stability.py)
# ─────────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\download\7073087")
DATA_PATH = ROOT / "all_scanpath_absolute.jsonl"
CACHE_DIR = ROOT / "outputs" / "cache_v2"
CACHE_FILE = "all_data_EVA02-B-16_merged2b_s8b_b131k.npz"
OUT_DIR = ROOT / "outputs" / "metacamil_nested_es"

CARS_MIN, CARS_MAX = 15.0, 60.0
SEVERITY_BINS = {
    "Mild (15-29)": (15.0, 30.0),
    "Moderate (30-36)": (30.0, 37.0),
    "Severe (37-60)": (37.0, 61.0),
}

CONFIGS = {
    # p3_best_cand in search_results.csv (MAE 1.499 / R2 0.833 under leaky LOPO)
    "best_cand": dict(in_dim=512, meta_dim=3, d_model=128, rh=128, n_heads=2,
                      use_ln=True, use_res=True, p=0.4),
    # base configuration of the p2/p3 grids (not selected on test scores)
    "base": dict(in_dim=512, meta_dim=3, d_model=64, rh=64, n_heads=1,
                 use_ln=False, use_res=False, p=0.4),
}
HP = {"lr": 1e-3, "wd": 0.05, "epochs": 1000, "patience": 100}
SEEDS = [42, 123, 456, 789, 2024]

PROTOCOLS = ["leaky_test_es", "fixed_final_epoch", "nested_refit", "nested_inner_ens"]
PAIRS = [("leaky_test_es", "nested_refit"),
         ("leaky_test_es", "fixed_final_epoch"),
         ("nested_refit", "fixed_final_epoch"),
         ("nested_inner_ens", "nested_refit")]


# ─────────────────────────────────────────────────────────────────────
# Model (identical to metacamil_stability.py)
# ─────────────────────────────────────────────────────────────────────
class MetaCAMIL(nn.Module):
    def __init__(self, in_dim, meta_dim=3, d_model=128, rh=128, n_heads=2,
                 use_ln=True, use_res=True, p=0.4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(meta_dim, d_model)
        self.k_proj = nn.Linear(in_dim, d_model)
        self.v_proj = nn.Linear(in_dim, in_dim)
        self.ln = nn.LayerNorm(in_dim) if use_ln else None
        self.res_prj = nn.Linear(meta_dim, in_dim) if use_res else None
        self.drop = nn.Dropout(p)
        self.reg = nn.Sequential(nn.Linear(in_dim, rh), nn.ReLU(),
                                 nn.Dropout(p * 0.5), nn.Linear(rh, 1))

    def forward(self, bags, mask, meta):
        B, N, D = bags.shape
        q = self.q_proj(meta).view(B, self.n_heads, 1, self.d_head)
        k = self.k_proj(bags).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        s = (q @ k.transpose(-2, -1)) * self.scale
        s = s.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e4)
        a = self.drop(torch.softmax(s, dim=-1))
        d_v = D // self.n_heads
        v_h = self.v_proj(bags).view(B, N, self.n_heads, d_v).permute(0, 2, 1, 3)
        z = (a @ v_h).squeeze(2).contiguous().view(B, -1)
        if self.ln is not None:
            z = self.ln(z)
        if self.res_prj is not None:
            z = z + self.res_prj(meta)
        return self.reg(z).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────
def load_data(data_path: Path, cache_path: Path):
    """Same parsing as metacamil_stability.load_data, then grouped by participant."""
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    z = np.load(cache_path, allow_pickle=True)
    paths = z["paths"].tolist()
    p2e = {paths[i]: z["X"][i] for i in range(len(paths))}
    X, y, g, M = [], [], [], []
    n_missing = 0
    for row in rows:
        path = row.get("image", "")
        if path not in p2e:
            n_missing += 1
            continue
        meta = row.get("meta") or {}
        age = float(meta.get("age", 0.0))
        gv = str(meta.get("gender", "")).upper()
        gm, gf = (1.0, 0.0) if gv == "M" else (0.0, 1.0)
        X.append(p2e[path]); y.append(float(row.get("cars", 0.0)))
        g.append(str(meta.get("participant_id", ""))); M.append([age, gm, gf])
    print(f"[DATA] {len(X)} images matched, {n_missing} jsonl rows without embedding")
    return (np.stack(X).astype(np.float32), np.array(y, np.float32),
            np.array(g), np.array(M, np.float32))


def make_synthetic(n_part=26, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g, M = [], [], [], []
    for i in range(n_part):
        n_img = rng.integers(4, 12)
        cars = float(rng.uniform(20, 50))
        age = float(rng.uniform(3, 12)); male = rng.random() < 0.8
        for _ in range(n_img):
            X.append(rng.normal(size=512).astype(np.float32) + 0.02 * cars)
            y.append(cars); g.append(f"P{i:02d}")
            M.append([age, 1.0 if male else 0.0, 0.0 if male else 1.0])
    return (np.stack(X).astype(np.float32), np.array(y, np.float32),
            np.array(g), np.array(M, np.float32))


class Participants:
    """Per-participant storage; bags are built with fold-specific age scaling."""

    def __init__(self, X, y, g, M):
        self.pids = sorted(set(g.tolist()))
        self.X = {p: X[g == p] for p in self.pids}
        self.M = {p: M[g == p] for p in self.pids}
        self.y = {}
        for p in self.pids:
            yp = y[g == p]
            if np.ptp(yp) > 0:
                print(f"[WARN] participant {p} has non-constant CARS across images; using mean")
            self.y[p] = float(yp.mean())

    def age_stats(self, pids):
        # image-weighted, as in the original (M_tr = M_img[tr_idx])
        a = np.concatenate([self.M[p][:, 0] for p in pids])
        sd = float(a.std())
        return float(a.mean()), (sd if sd > 0 else 1.0)

    def bags(self, pids, mu_a, sd_a, y_override=None):
        bags, labels, metas = [], [], []
        for p in pids:
            mi = self.M[p].copy(); mi[:, 0] = (mi[:, 0] - mu_a) / sd_a
            bags.append(self.X[p])
            labels.append(y_override[p] if y_override is not None else self.y[p])
            metas.append(mi.mean(axis=0))
        return bags, np.array(labels, np.float32), metas


def build_batch(bags, metas, device):
    tensors = [torch.from_numpy(b).float() for b in bags]
    padded = pad_sequence(tensors, batch_first=True)
    lengths = torch.tensor([len(b) for b in bags])
    N = padded.shape[1]
    mask = torch.arange(N).unsqueeze(0) < lengths.unsqueeze(1)
    meta_t = torch.from_numpy(np.stack(metas)).float()
    return padded.to(device), mask.to(device), meta_t.to(device)


# ─────────────────────────────────────────────────────────────────────
# Training: one full trajectory, predictions recorded at every epoch
# ─────────────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_trajectory(tr_bags, y_tr, tr_metas, ev_bags, ev_metas, model_kwargs,
                     hp, init_seed, device, use_amp):
    """Train for hp['epochs'] full-batch epochs (same optimiser, cosine schedule,
    Huber loss, y z-scoring as the original). Returns clipped predictions for
    the evaluation bags at every epoch: array (epochs, n_eval)."""
    set_seed(init_seed)
    model = MetaCAMIL(**model_kwargs).to(device)
    y_mu = float(y_tr.mean()); y_sd = float(y_tr.std()) or 1.0
    y_z = torch.from_numpy(((y_tr - y_mu) / y_sd).astype(np.float32)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"], eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    huber = nn.HuberLoss(delta=1.0)

    tr_pad, tr_mask, tr_meta = build_batch(tr_bags, tr_metas, device)
    ev_pad, ev_mask, ev_meta = build_batch(ev_bags, ev_metas, device)
    out = np.empty((hp["epochs"], len(ev_bags)), np.float32)

    for ep in range(hp["epochs"]):
        model.train(); opt.zero_grad()
        if use_amp:
            with torch.autocast("cuda"):
                loss = huber(model(tr_pad, tr_mask, tr_meta), y_z)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            huber(model(tr_pad, tr_mask, tr_meta), y_z).backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    vp = model(ev_pad, ev_mask, ev_meta)
            else:
                vp = model(ev_pad, ev_mask, ev_meta)
        out[ep] = np.clip(vp.float().cpu().numpy() * y_sd + y_mu, CARS_MIN, CARS_MAX)
    return out


def patience_select(curve, patience):
    """Exact emulation of the original loop: returns the epoch index whose
    prediction the original code would have returned."""
    best, best_e, pat = np.inf, 0, 0
    for e, v in enumerate(curve):
        if v < best:
            best, best_e, pat = v, e, 0
        else:
            pat += 1
        if pat >= patience:
            break
    return best_e


def inner_splitter(train_pids, ys, n_splits, seed):
    y_arr = np.array([ys[p] for p in train_pids])
    bins = np.digitize(y_arr, [30.0, 37.0])
    counts = np.bincount(bins, minlength=3)
    if (counts[counts > 0] >= n_splits).all():
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed)
                    .split(np.zeros(len(y_arr)), bins))
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y_arr))))


# ─────────────────────────────────────────────────────────────────────
# Debug / self-check infrastructure
# ─────────────────────────────────────────────────────────────────────
class Checker:
    """Collects PASS/FAIL checks; FAILs are always printed, PASSes only on
    first occurrence of each check name (to keep the log readable)."""

    def __init__(self):
        self.rows, self.seen = [], set()
        self.strict = False

    def check(self, name, ok, detail="", critical=False):
        ok = bool(ok)
        self.rows.append(dict(check=name, ok=ok, critical=critical, detail=str(detail)))
        if not ok:
            tag = "CRITICAL FAIL" if critical else "FAIL"
            print(f"  [DEBUG {tag}] {name}: {detail}")
            if critical and self.strict:
                raise SystemExit(f"[STOP] critical check failed: {name}")
        elif name not in self.seen:
            print(f"  [DEBUG PASS] {name}" + (f": {detail}" if detail else ""))
        self.seen.add(name)

    def info(self, msg):
        print(f"  [DEBUG INFO] {msg}")

    def summary(self, path=None):
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        g = df.groupby("check").agg(n=("ok", "size"), n_fail=("ok", lambda s: int((~s).sum())),
                                    critical=("critical", "max")).reset_index()
        print("\n" + "=" * 78 + "\n  DEBUG SUMMARY\n" + "=" * 78)
        for _, r in g.iterrows():
            status = "PASS" if r.n_fail == 0 else ("CRITICAL FAIL" if r.critical else "FAIL")
            print(f"  {status:>13}  {r.check:<58} ({r.n - r.n_fail}/{r.n})")
        tot_fail = int(g.n_fail.sum())
        print("=" * 78)
        print(f"  {'ALL CHECKS PASSED' if tot_fail == 0 else f'{tot_fail} CHECK(S) FAILED'}")
        print("=" * 78)
        if path is not None:
            df.to_csv(path, index=False)
        return df


CHK = Checker()


def debug_data_checks(parts, X, y, g, M):
    CHK.check("data: embedding dim == 512", X.shape[1] == 512, X.shape, critical=True)
    CHK.check("data: no NaN/inf in embeddings", np.isfinite(X).all(), critical=True)
    CHK.check("data: 26 participants", len(parts.pids) == 26, len(parts.pids))
    n_img = {p: len(parts.X[p]) for p in parts.pids}
    CHK.info(f"images/participant min={min(n_img.values())} median={np.median(list(n_img.values())):.0f} "
             f"max={max(n_img.values())} total={sum(n_img.values())}")
    const_y = all(np.ptp(y[g == p]) == 0 for p in parts.pids)
    CHK.check("data: CARS constant within participant", const_y, critical=True)
    const_m = all(np.ptp(M[g == p], axis=0).max() == 0 for p in parts.pids)
    CHK.check("data: age/sex constant within participant", const_m)
    yv = np.array([parts.y[p] for p in parts.pids])
    CHK.check("data: CARS in [15, 60]", (yv >= 15).all() and (yv <= 60).all(),
              f"min={yv.min()} max={yv.max()}")
    bins = {lbl: int(((yv >= lo) & (yv < hi)).sum()) for lbl, (lo, hi) in SEVERITY_BINS.items()}
    CHK.info(f"severity counts {bins} (original SEV_WEIGHTS imply 8/13/5)")
    CHK.check("data: no empty participant id", all(p != "" for p in parts.pids), critical=True)


def debug_metric_selftest():
    rng = np.random.default_rng(0)
    from sklearn.metrics import r2_score, mean_absolute_error
    yy = rng.uniform(15, 60, 26); pp = yy + rng.normal(0, 5, 26)
    CHK.check("metric: r2 == sklearn r2_score", abs(r2(yy, pp) - r2_score(yy, pp)) < 1e-10)
    CHK.check("metric: MAE == sklearn", abs(metrics(yy, pp)["MAE"] - mean_absolute_error(yy, pp)) < 1e-10)
    CHK.check("metric: CCC(y, y) == 1", abs(ccc(yy, yy) - 1) < 1e-10)
    # LOPO train-mean predictor has R2 = 1 - (n/(n-1))^2 exactly
    base = np.array([(yy.sum() - v) / (len(yy) - 1) for v in yy])
    n = len(yy)
    CHK.check("metric: LOPO mean predictor R2 identity",
              abs(r2(yy, base) - (1 - (n / (n - 1)) ** 2)) < 1e-10, f"{r2(yy, base):.4f}")
    CHK.check("select: patience rule on toy curve",
              patience_select(np.array([5, 4, 3, 3.5, 3.6, 3.7, 1.0]), 3) == 2)


def original_train_fold(model, tr_bags, y_tr, tr_metas, val_bags, y_val, val_metas,
                        hp, device, use_amp):
    """Verbatim logic of metacamil_stability.train_fold (test-participant ES),
    used only to verify that patience_select() emulates it exactly."""
    y_mu = float(y_tr.mean()); y_sd = float(y_tr.std()) or 1.0
    y_z = torch.from_numpy(((y_tr - y_mu) / y_sd).astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"], eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    huber = nn.HuberLoss(delta=1.0)
    tr_pad, tr_mask, tr_meta = build_batch(tr_bags, tr_metas, device)
    vl_pad, vl_mask, vl_meta = build_batch(val_bags, val_metas, device)
    best_mae, best_preds, best_ep, pat = float("inf"), None, -1, 0
    for epoch in range(hp["epochs"]):
        model.train(); opt.zero_grad()
        if use_amp:
            with torch.autocast("cuda"):
                loss = huber(model(tr_pad, tr_mask, tr_meta), y_z)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            huber(model(tr_pad, tr_mask, tr_meta), y_z).backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    vp = model(vl_pad, vl_mask, vl_meta)
            else:
                vp = model(vl_pad, vl_mask, vl_meta)
        vp_np = np.clip(vp.float().cpu().numpy() * y_sd + y_mu, CARS_MIN, CARS_MAX)
        val_mae = float(np.abs(y_val - vp_np).mean())
        if val_mae < best_mae:
            best_mae, best_preds, best_ep, pat = val_mae, vp_np.copy(), epoch, 0
        else:
            pat += 1
        if pat >= hp["patience"]:
            break
    return best_preds, best_ep


# ─────────────────────────────────────────────────────────────────────
# One outer fold
# ─────────────────────────────────────────────────────────────────────
def run_fold(parts, fold, test_pid, ys, model_kwargs, hp, seed, inner_splits, device,
             use_amp, y_override=None, debug=True):
    pids = parts.pids
    train_pids = [p for p in pids if p != test_pid]
    y_true = ys[test_pid]
    train_mean = float(np.mean([ys[p] for p in train_pids]))

    if debug:
        CHK.check("leak: test pid not in outer-train", test_pid not in train_pids, critical=True)
        CHK.check("leak: outer-train size == n-1", len(train_pids) == len(pids) - 1, critical=True)

    # ---- outer trajectory (all n-1 training participants) ----
    mu_a, sd_a = parts.age_stats(train_pids)
    tr_b, tr_y, tr_m = parts.bags(train_pids, mu_a, sd_a, y_override)
    te_b, _, te_m = parts.bags([test_pid], mu_a, sd_a, y_override)
    if debug:
        exp_y = np.array([ys[p] for p in train_pids], np.float32)
        CHK.check("leak: outer training labels == train participants' labels only",
                  np.array_equal(tr_y, exp_y), critical=True)
        mu_all, _ = parts.age_stats(pids)
        CHK.check("leak: age scaling uses outer-train stats (differs from all-26 stats)",
                  mu_a != mu_all or np.ptp([parts.M[p][0, 0] for p in pids]) == 0,
                  f"train mu={mu_a:.4f} all mu={mu_all:.4f}")
    outer = train_trajectory(tr_b, tr_y, tr_m, te_b, te_m, model_kwargs, hp,
                             seed + fold, device, use_amp)[:, 0]
    if debug:
        CHK.check("traj: outer trajectory shape == (epochs,)", outer.shape == (hp["epochs"],),
                  outer.shape, critical=True)
        CHK.check("traj: predictions finite and within [15, 60]",
                  np.isfinite(outer).all() and outer.min() >= CARS_MIN and outer.max() <= CARS_MAX)
    test_err_curve = np.abs(outer - y_true)
    e_leaky = patience_select(test_err_curve, hp["patience"])
    e_oracle = int(np.argmin(test_err_curve))

    # ---- inner CV on training participants only (test label never used) ----
    splits = inner_splitter(train_pids, ys, inner_splits, seed * 1000 + fold)
    if debug:
        all_val = np.concatenate([iva for _, iva in splits])
        CHK.check("inner: each outer-train participant in exactly one inner-val fold",
                  sorted(all_val.tolist()) == list(range(len(train_pids))), critical=True)
        CHK.check("inner: n folds", len(splits) == inner_splits, len(splits))
    pooled_abs = np.zeros(hp["epochs"], np.float64); n_pooled = 0
    inner_test_preds, inner_epochs = [], []
    for k, (itr, iva) in enumerate(splits):
        itr_p = [train_pids[i] for i in itr]; iva_p = [train_pids[i] for i in iva]
        if debug:
            CHK.check("leak: inner-train ∩ inner-val = ∅", not (set(itr_p) & set(iva_p)), critical=True)
            CHK.check("leak: test pid not in inner-train/inner-val",
                      test_pid not in itr_p and test_pid not in iva_p, critical=True)
            CHK.check("leak: inner-train ∪ inner-val == outer-train",
                      set(itr_p) | set(iva_p) == set(train_pids), critical=True)
        mu_i, sd_i = parts.age_stats(itr_p)
        b_tr, y_itr, m_tr = parts.bags(itr_p, mu_i, sd_i, y_override)
        b_ev, y_iva, m_ev = parts.bags(iva_p + [test_pid], mu_i, sd_i, y_override)
        if debug:
            CHK.check("leak: inner training labels == inner-train participants only",
                      np.array_equal(y_itr, np.array([ys[p] for p in itr_p], np.float32)),
                      critical=True)
            CHK.check("leak: inner age stats from inner-train only",
                      (mu_i, sd_i) == parts.age_stats(itr_p), critical=True)
        traj = train_trajectory(b_tr, y_itr, m_tr, b_ev, m_ev, model_kwargs, hp,
                                seed + fold + 10000 * (k + 1), device, use_amp)
        val_pred, test_pred = traj[:, :-1], traj[:, -1]
        # the last eval column is the test participant: its label is dropped here
        y_val_only = y_iva[:-1]
        if debug:
            CHK.check("leak: inner selection uses inner-val labels only (test column dropped)",
                      len(y_val_only) == len(iva_p) and val_pred.shape[1] == len(iva_p),
                      critical=True)
        abs_err = np.abs(val_pred - y_val_only[None, :])
        pooled_abs += abs_err.sum(axis=1); n_pooled += abs_err.shape[1]
        e_k = patience_select(abs_err.mean(axis=1), hp["patience"])
        inner_epochs.append(e_k); inner_test_preds.append(float(test_pred[e_k]))
    inner_curve = pooled_abs / n_pooled
    e_nested = int(np.argmin(inner_curve))
    if debug:
        CHK.check("inner: pooled count == n outer-train participants",
                  n_pooled == len(train_pids), n_pooled, critical=True)

    return dict(fold=fold, pid=test_pid, y_true=y_true, train_mean=train_mean,
                leaky_test_es=float(outer[e_leaky]),
                fixed_final_epoch=float(outer[-1]),
                nested_refit=float(outer[e_nested]),
                nested_inner_ens=float(np.mean(inner_test_preds)),
                epoch_leaky=e_leaky + 1, epoch_oracle=e_oracle + 1,
                epoch_nested=e_nested + 1,
                epoch_inner_mean=float(np.mean(inner_epochs) + 1),
                epoch_inner_list=";".join(str(e + 1) for e in inner_epochs),
                inner_val_mae_at_nested=float(inner_curve[e_nested]),
                test_mae_at_leaky_epoch=float(test_err_curve[e_leaky]),
                test_mae_at_nested_epoch=float(test_err_curve[e_nested]),
                pred_oracle=float(outer[e_oracle]))


def run_seed(parts: Participants, model_kwargs, hp, seed, inner_splits, device,
             use_amp, y_override=None, verbose=True, debug=True):
    """One complete LOPO pass for one seed; returns per-fold records."""
    ys = y_override if y_override is not None else parts.y
    recs = []
    for fold, test_pid in enumerate(parts.pids, 1):
        t0 = time.time()
        rec = run_fold(parts, fold, test_pid, ys, model_kwargs, hp, seed, inner_splits,
                       device, use_amp, y_override, debug)
        recs.append(rec)
        if debug:
            CHK.check("select: leaky test-MAE <= nested test-MAE (by construction)",
                      rec["test_mae_at_leaky_epoch"] <= rec["test_mae_at_nested_epoch"] + 1e-6
                      or rec["epoch_leaky"] < rec["epoch_nested"],
                      f"fold {fold}: leaky {rec['test_mae_at_leaky_epoch']:.3f} "
                      f"(e{rec['epoch_leaky']}) nested {rec['test_mae_at_nested_epoch']:.3f} "
                      f"(e{rec['epoch_nested']})")
        if verbose:
            print(f"   fold {fold:02d} pid={test_pid:>6} true={rec['y_true']:5.1f} | "
                  f"leaky={rec['leaky_test_es']:5.1f}(e{rec['epoch_leaky']}) "
                  f"nested={rec['nested_refit']:5.1f}(e{rec['epoch_nested']}) "
                  f"inner_ens={rec['nested_inner_ens']:5.1f} "
                  f"final={rec['fixed_final_epoch']:5.1f}  [{time.time()-t0:.0f}s]")
    return recs


def debug_reproduce_original(parts, model_kwargs, hp, seed, device, use_amp, n_folds,
                             inner_splits=5):
    """(1) Verbatim original loop vs our trajectory+patience emulation must give
    the same prediction. (2) Changing ONLY the test label must leave all nested
    protocols unchanged (they never see it) while the leaky one may move."""
    tol = 1e-4 if not use_amp else 0.05
    for fold, test_pid in list(enumerate(parts.pids, 1))[:n_folds]:
        train_pids = [p for p in parts.pids if p != test_pid]
        mu_a, sd_a = parts.age_stats(train_pids)
        tr_b, tr_y, tr_m = parts.bags(train_pids, mu_a, sd_a)
        te_b, te_y, te_m = parts.bags([test_pid], mu_a, sd_a)
        set_seed(seed + fold)
        model = MetaCAMIL(**model_kwargs).to(device)
        orig_pred, orig_ep = original_train_fold(model, tr_b, tr_y, tr_m, te_b, te_y, te_m,
                                                 hp, device, use_amp)
        traj = train_trajectory(tr_b, tr_y, tr_m, te_b, te_m, model_kwargs, hp,
                                seed + fold, device, use_amp)[:, 0]
        e = patience_select(np.abs(traj - te_y[0]), hp["patience"])
        diff = abs(float(orig_pred[0]) - float(traj[e]))
        CHK.check("repro: emulated leaky == verbatim original loop (prediction)", diff <= tol,
                  f"fold {fold}: original {orig_pred[0]:.4f} (e{orig_ep+1}) vs emulated "
                  f"{traj[e]:.4f} (e{e+1}), |diff|={diff:.2e}, tol={tol}")
        CHK.check("repro: emulated leaky epoch == original epoch", e == orig_ep,
                  f"fold {fold}: {e+1} vs {orig_ep+1}", critical=(device != "cuda"))

    # label-invariance on fold 1
    fold, test_pid = 1, parts.pids[0]
    ys = dict(parts.y)
    rec_a = run_fold(parts, fold, test_pid, ys, model_kwargs, hp, seed, inner_splits, device, use_amp,
                     None, debug=False)
    ys_b = dict(ys); ys_b[test_pid] = CARS_MAX if ys[test_pid] < 37 else CARS_MIN
    rec_b = run_fold(parts, fold, test_pid, ys_b, model_kwargs, hp, seed, inner_splits, device, use_amp,
                     ys_b, debug=False)
    for proto in ["nested_refit", "nested_inner_ens", "fixed_final_epoch"]:
        d = abs(rec_a[proto] - rec_b[proto])
        CHK.check(f"invariance: {proto} unchanged when only test label changes", d <= tol,
                  f"{rec_a[proto]:.4f} vs {rec_b[proto]:.4f}", critical=True)
    CHK.check("invariance: nested epoch unchanged when only test label changes",
              rec_a["epoch_nested"] == rec_b["epoch_nested"],
              f"{rec_a['epoch_nested']} vs {rec_b['epoch_nested']}"
              + (" (GPU/AMP run-to-run noise can shift this by a few epochs)" if device == "cuda" else ""),
              critical=(device != "cuda"))
    CHK.info(f"leaky prediction with true label {ys[test_pid]:.1f}: {rec_a['leaky_test_es']:.2f} "
             f"(e{rec_a['epoch_leaky']}); with fake label {ys_b[test_pid]:.1f}: "
             f"{rec_b['leaky_test_es']:.2f} (e{rec_b['epoch_leaky']})  <- leaky follows the test label")


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────
def ccc(y, p):
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = ((y - my) * (p - mp)).mean()
    d = vy + vp + (my - mp) ** 2
    return float(2 * cov / d) if d > 0 else np.nan


def safe_corr(fn, y, p):
    if np.ptp(p) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(fn(y, p)[0])


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan


def metrics(y, p, base=None):
    e = p - y
    out = dict(n=len(y), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
               R2=r2(y, p), Spearman=safe_corr(spearmanr, y, p),
               Pearson=safe_corr(pearsonr, y, p), CCC=ccc(y, p),
               bias=float(e.mean()), LoA_low=float(e.mean() - 1.96 * e.std(ddof=1)),
               LoA_high=float(e.mean() + 1.96 * e.std(ddof=1)),
               pred_SD=float(p.std(ddof=1)), true_SD=float(y.std(ddof=1)))
    if base is not None:
        eb = base - y
        out["baseline_MAE"] = float(np.abs(eb).mean())
        out["baseline_R2"] = r2(y, base)
        out["R2_oos_vs_train_mean"] = float(1 - (e ** 2).sum() / (eb ** 2).sum())
        out["MAE_skill_vs_train_mean"] = float(1 - out["MAE"] / out["baseline_MAE"])
    return out


def bootstrap_ci(y, p, base, n_boot, rng):
    stats = {k: [] for k in ["MAE", "RMSE", "R2", "Spearman", "CCC", "R2_oos_vs_train_mean"]}
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(y[i]) == 0:
            continue
        m = metrics(y[i], p[i], base[i])
        for k in stats:
            stats[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in stats.items()}


def severity_table(y, p):
    rows = []
    for lbl, (lo, hi) in SEVERITY_BINS.items():
        m = (y >= lo) & (y < hi)
        if m.sum() == 0:
            continue
        e = p[m] - y[m]
        rows.append(dict(severity=lbl, n=int(m.sum()), MAE=float(np.abs(e).mean()),
                         RMSE=float(np.sqrt((e ** 2).mean())), bias=float(e.mean())))
    return rows


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=Path, default=DATA_PATH)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / CACHE_FILE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--configs", nargs="+", default=["best_cand"], choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--patience", type=int, default=HP["patience"])
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=0,
                    help="full-pipeline label permutations (single seed each; slow)")
    ap.add_argument("--perm-config", default="best_cand", choices=list(CONFIGS))
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, tiny run")
    ap.add_argument("--check-folds", type=int, default=2,
                    help="folds used to verify the leaky emulation against the verbatim "
                         "original training loop (0 = skip the pre-run checks)")
    ap.add_argument("--strict", action="store_true",
                    help="stop immediately if a critical (leakage) check fails")
    ap.add_argument("--no-debug", action="store_true", help="disable per-fold checks")
    args = ap.parse_args()

    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"
    use_amp = device == "cuda" and not args.no_amp
    hp = dict(HP, epochs=args.epochs, patience=args.patience)

    if args.smoke:
        X, y, g, M = make_synthetic()
        args.out_dir = Path("smoke_out")
    else:
        X, y, g, M = load_data(args.data_path, args.cache)
    parts = Participants(X, y, g, M)
    print(f"[DATA] {len(parts.pids)} participants, {len(X)} images, device={device}, amp={use_amp}")
    CHK.strict = args.strict
    debug = not args.no_debug
    print("\n[DEBUG] pre-run checks")
    debug_metric_selftest()
    debug_data_checks(parts, X, y, g, M)
    if args.check_folds > 0:
        print(f"[DEBUG] verifying leaky emulation against the original loop on "
              f"{args.check_folds} fold(s) + label-invariance test (config={args.configs[0]}, "
              f"seed={args.seeds[0]}) ...")
        t0 = time.time()
        debug_reproduce_original(parts, CONFIGS[args.configs[0]], hp, args.seeds[0], device,
                                 use_amp, args.check_folds, args.inner_splits)
        print(f"[DEBUG] pre-run checks done ({(time.time()-t0)/60:.1f} min)")
    n_crit = sum(1 for r in CHK.rows if r["critical"] and not r["ok"])
    if n_crit:
        print(f"\n[WARNING] {n_crit} critical check(s) failed before the main run. "
              f"Results would not be trustworthy; rerun with --strict to stop here.")
    if len(parts.pids) != 26 and not args.smoke:
        print(f"[WARN] expected 26 participants, found {len(parts.pids)}")

    out = args.out_dir; cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    # cache fingerprint: any change to this script, hyper-parameters, inner
    # folds or input data invalidates previously cached seeds/permutations
    import hashlib
    fp_src = Path(__file__).read_bytes()
    fp_meta = json.dumps(dict(hp=hp, inner=args.inner_splits, configs=CONFIGS,
                              pids=parts.pids, y=[parts.y[p] for p in parts.pids],
                              n_img=[len(parts.X[p]) for p in parts.pids],
                              xsum=float(np.float64(X.sum())), amp=use_amp), sort_keys=True)
    fingerprint = hashlib.sha1(fp_src + fp_meta.encode()).hexdigest()[:10]
    print(f"[CACHE] fingerprint {fingerprint} (cached runs with another fingerprint are ignored)")
    json.dump(dict(vars(args), hp=hp, fingerprint=fingerprint, configs_kwargs={c: CONFIGS[c] for c in args.configs},
                   device=device, amp=use_amp, n_participants=len(parts.pids),
                   n_images=int(len(X)), torch=torch.__version__),
              open(out / "run_config.json", "w"), indent=2, default=str)

    # ---------------- main runs ----------------
    all_recs = []
    t_start = time.time()
    for cfg in args.configs:
        for seed in args.seeds:
            cf = cache / f"{cfg}_seed{seed}_{fingerprint}.json"
            if cf.exists():
                recs = json.load(open(cf))
                print(f"[CACHE] {cfg} seed {seed} loaded")
            else:
                print(f"\n=== config={cfg}  seed={seed} ===")
                t0 = time.time()
                recs = run_seed(parts, CONFIGS[cfg], hp, seed, args.inner_splits, device, use_amp,
                                debug=debug)
                json.dump(recs, open(cf, "w"))
                print(f"   -> {(time.time()-t0)/60:.1f} min")
            for r in recs:
                r.update(config=cfg, seed=seed)
            all_recs.extend(recs)
    pred = pd.DataFrame(all_recs)
    print("\n[DEBUG] post-run checks")
    for cfg, d in pred.groupby("config"):
        cnt = d.groupby("seed").pid.nunique()
        CHK.check("post: every seed has all participants once",
                  (cnt == len(parts.pids)).all() and not d.duplicated(["seed", "pid"]).any(),
                  cnt.to_dict(), critical=True)
        for seed, ds in d.groupby("seed"):
            yy = ds.y_true.values; n = len(yy)
            CHK.check("post: LOPO train-mean baseline R2 == 1-(n/(n-1))^2",
                      abs(r2(yy, ds.train_mean.values) - (1 - (n / (n - 1)) ** 2)) < 1e-6,
                      f"{cfg} seed {seed}: {r2(yy, ds.train_mean.values):.4f}")
            CHK.check("post: y_true matches stored participant labels",
                      all(abs(parts.y[p] - v) < 1e-6 for p, v in zip(ds.pid, ds.y_true)),
                      critical=True)
        CHK.info(f"{cfg}: fraction of folds where nested epoch == leaky epoch: "
                 f"{(d.epoch_nested == d.epoch_leaky).mean():.2f}; "
                 f"median test |err| at leaky epoch {d.test_mae_at_leaky_epoch.median():.2f} vs "
                 f"at nested epoch {d.test_mae_at_nested_epoch.median():.2f}")
        CHK.info(f"{cfg}: leaky_test_es seed-mean MAE = "
                 f"{d.groupby('seed').apply(lambda x: np.abs(x.leaky_test_es - x.y_true).mean()).mean():.3f} "
                 f"-> compare with the metacamil_stability.py log (should be close to the "
                 f"published non-nested value if that script was its source)")
    pred.to_csv(out / "participant_predictions.csv", index=False)

    ep_cols = ["config", "seed", "fold", "pid", "epoch_leaky", "epoch_oracle",
               "epoch_nested", "epoch_inner_mean", "epoch_inner_list", "inner_val_mae_at_nested"]
    pred[ep_cols].to_csv(out / "epoch_selection.csv", index=False)

    rng = np.random.default_rng(20260930)
    per_seed, summary, sev_rows, paired = [], [], [], []
    for cfg, dcfg in pred.groupby("config"):
        # per-seed metrics
        for seed, ds in dcfg.groupby("seed"):
            ds = ds.sort_values("pid")
            yv, bv = ds.y_true.values, ds.train_mean.values
            for proto in PROTOCOLS:
                per_seed.append(dict(config=cfg, seed=seed, protocol=proto,
                                     **metrics(yv, ds[proto].values, bv)))
                for s in severity_table(yv, ds[proto].values):
                    sev_rows.append(dict(config=cfg, seed=seed, protocol=proto, **s))
        ps = pd.DataFrame([r for r in per_seed if r["config"] == cfg])

        # seed-ensemble predictions (mean over seeds per participant)
        ens = dcfg.groupby("pid").agg({**{p: "mean" for p in PROTOCOLS},
                                       "y_true": "first", "train_mean": "first"}).sort_index()
        yv, bv = ens.y_true.values, ens.train_mean.values
        for proto in PROTOCOLS:
            pv = ens[proto].values
            m_ens = metrics(yv, pv, bv)
            ci = bootstrap_ci(yv, pv, bv, args.n_boot, rng)
            row = dict(config=cfg, protocol=proto, n_seeds=dcfg.seed.nunique())
            sp = ps[ps.protocol == proto]
            for k in ["MAE", "RMSE", "R2", "R2_oos_vs_train_mean", "Spearman", "CCC", "bias"]:
                row[f"{k}_seed_mean"] = float(sp[k].mean())
                row[f"{k}_seed_sd"] = float(sp[k].std(ddof=1)) if len(sp) > 1 else np.nan
                row[f"{k}_seed_min"] = float(sp[k].min()); row[f"{k}_seed_max"] = float(sp[k].max())
            for k, v in m_ens.items():
                row[f"ens_{k}"] = v
            for k, (lo, hi) in ci.items():
                row[f"ens_{k}_CI_low"] = lo; row[f"ens_{k}_CI_high"] = hi
            summary.append(row)
            for s in severity_table(yv, pv):
                sev_rows.append(dict(config=cfg, seed="ensemble", protocol=proto, **s))

        # paired protocol differences
        for a, b in PAIRS:
            ea, eb = np.abs(ens[a].values - yv), np.abs(ens[b].values - yv)
            d_mae, d_r2 = [], []
            n = len(yv)
            for _ in range(args.n_boot):
                i = rng.integers(0, n, n)
                if np.ptp(yv[i]) == 0:
                    continue
                d_mae.append(ea[i].mean() - eb[i].mean())
                d_r2.append(r2(yv[i], ens[a].values[i]) - r2(yv[i], ens[b].values[i]))
            try:
                w = wilcoxon(ea, eb)
                w_stat, w_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            sa, sb = ps[ps.protocol == a].set_index("seed"), ps[ps.protocol == b].set_index("seed")
            dseed = (sa.MAE - sb.MAE)
            paired.append(dict(config=cfg, A=a, B=b,
                               dMAE_ens=float(ea.mean() - eb.mean()),
                               dMAE_CI_low=float(np.percentile(d_mae, 2.5)),
                               dMAE_CI_high=float(np.percentile(d_mae, 97.5)),
                               dR2_ens=float(r2(yv, ens[a].values) - r2(yv, ens[b].values)),
                               dR2_CI_low=float(np.percentile(d_r2, 2.5)),
                               dR2_CI_high=float(np.percentile(d_r2, 97.5)),
                               wilcoxon_abs_err_stat=w_stat, wilcoxon_abs_err_p=w_p,
                               n_participants_A_better=int((ea < eb).sum()),
                               dMAE_seed_mean=float(dseed.mean()),
                               dMAE_seed_sd=float(dseed.std(ddof=1)) if len(dseed) > 1 else np.nan))

    pd.DataFrame(per_seed).to_csv(out / "per_seed_metrics.csv", index=False)
    summ = pd.DataFrame(summary); summ.to_csv(out / "summary_metrics.csv", index=False)
    for cfg, d in pred.groupby("config"):
        ens_chk = d.groupby("pid").nested_refit.mean()
        row = summ[(summ.config == cfg) & (summ.protocol == "nested_refit")].iloc[0]
        yy = d.groupby("pid").y_true.first().loc[ens_chk.index].values
        CHK.check("post: seed-ensemble MAE recomputes identically",
                  abs(np.abs(ens_chk.values - yy).mean() - row.ens_MAE) < 1e-9)
        CHK.check("post: bootstrap CI brackets point estimate (MAE, nested_refit)",
                  row.ens_MAE_CI_low <= row.ens_MAE <= row.ens_MAE_CI_high,
                  f"{row.ens_MAE_CI_low:.2f} <= {row.ens_MAE:.2f} <= {row.ens_MAE_CI_high:.2f}")
    pd.DataFrame(sev_rows).to_csv(out / "severity_metrics.csv", index=False)
    pair_df = pd.DataFrame(paired); pair_df.to_csv(out / "paired_differences.csv", index=False)

    # ---------------- permutation null ----------------
    perm_summary = None
    if args.n_perm > 0:
        cfg = args.perm_config
        prng = np.random.default_rng(777)
        perm_rows = []
        for i in range(args.n_perm):
            cf = cache / f"perm_{cfg}_{i:04d}_{fingerprint}.json"
            if cf.exists():
                perm_rows.append(json.load(open(cf))); continue
            t0 = time.time()
            shuffled = prng.permutation([parts.y[p] for p in parts.pids])
            CHK.check("perm: permuted labels are a permutation of the originals",
                      sorted(shuffled.tolist()) == sorted(parts.y.values()), critical=True)
            y_perm = dict(zip(parts.pids, shuffled.tolist()))
            recs = run_seed(parts, CONFIGS[cfg], hp, 90000 + i, args.inner_splits, device,
                            use_amp, y_override=y_perm, verbose=False, debug=debug)
            d = pd.DataFrame(recs).sort_values("pid")
            row = dict(perm=i)
            for proto in PROTOCOLS:
                m = metrics(d.y_true.values, d[proto].values, d.train_mean.values)
                row[f"{proto}_MAE"] = m["MAE"]; row[f"{proto}_R2"] = m["R2"]
                row[f"{proto}_Spearman"] = m["Spearman"]
            json.dump(row, open(cf, "w")); perm_rows.append(row)
            print(f"[PERM {i+1}/{args.n_perm}] leaky MAE={row['leaky_test_es_MAE']:.2f} "
                  f"nested MAE={row['nested_refit_MAE']:.2f} ({time.time()-t0:.0f}s)")
        perm = pd.DataFrame(perm_rows); perm.to_csv(out / "permutation_runs.csv", index=False)
        ps_all = pd.DataFrame(per_seed)
        obs_seed = args.seeds[0]
        rows = []
        for proto in PROTOCOLS:
            o = ps_all[(ps_all.config == cfg) & (ps_all.protocol == proto)]
            o1 = o[o.seed == obs_seed].iloc[0]
            for k, better in [("MAE", "lower"), ("R2", "higher"), ("Spearman", "higher")]:
                null = perm[f"{proto}_{k}"].dropna().values
                for label, obs in [("first_seed", o1[k]), ("seed_mean", o[k].mean())]:
                    cnt = (null <= obs).sum() if better == "lower" else (null >= obs).sum()
                    rows.append(dict(config=cfg, protocol=proto, metric=k, observed_type=label,
                                     observed=float(obs), null_mean=float(null.mean()),
                                     null_sd=float(null.std(ddof=1)),
                                     null_p2_5=float(np.percentile(null, 2.5)),
                                     null_p97_5=float(np.percentile(null, 97.5)),
                                     n_perm=len(null),
                                     p_value=float((1 + cnt) / (1 + len(null)))))
        perm_summary = pd.DataFrame(rows)
        perm_summary.to_csv(out / "permutation_summary.csv", index=False)

    # ---------------- paper numbers ----------------
    L = [f"# MetaCA-MIL nested early-stopping results\n",
         f"n participants = {len(parts.pids)}, seeds = {args.seeds}, epochs = {hp['epochs']}, "
         f"patience = {hp['patience']}, inner folds = {args.inner_splits}, bootstrap B = {args.n_boot}\n"]
    for _, r in summ.iterrows():
        L.append(f"## {r.config} / {r.protocol}")
        L.append(f"- per-seed MAE {r.MAE_seed_mean:.2f} ± {r.MAE_seed_sd:.2f} "
                 f"(range {r.MAE_seed_min:.2f}–{r.MAE_seed_max:.2f}); "
                 f"R² {r.R2_seed_mean:.2f} ± {r.R2_seed_sd:.2f}; "
                 f"Spearman {r.Spearman_seed_mean:.2f} ± {r.Spearman_seed_sd:.2f}; "
                 f"CCC {r.CCC_seed_mean:.2f} ± {r.CCC_seed_sd:.2f}")
        L.append(f"- seed-ensemble MAE {r.ens_MAE:.2f} [{r.ens_MAE_CI_low:.2f}, {r.ens_MAE_CI_high:.2f}], "
                 f"RMSE {r.ens_RMSE:.2f}, R² {r.ens_R2:.2f} [{r.ens_R2_CI_low:.2f}, {r.ens_R2_CI_high:.2f}], "
                 f"Spearman {r.ens_Spearman:.2f} [{r.ens_Spearman_CI_low:.2f}, {r.ens_Spearman_CI_high:.2f}], "
                 f"CCC {r.ens_CCC:.2f}, bias {r.ens_bias:.2f} (LoA {r.ens_LoA_low:.2f} to {r.ens_LoA_high:.2f})")
        L.append(f"- train-mean baseline MAE {r.ens_baseline_MAE:.2f}, R² {r.ens_baseline_R2:.3f}; "
                 f"out-of-sample R² vs train mean {r.ens_R2_oos_vs_train_mean:.2f}\n")
    L.append("## Paired differences (seed-ensemble; A − B)")
    for _, r in pair_df.iterrows():
        L.append(f"- {r.config}: {r.A} − {r.B}: ΔMAE {r.dMAE_ens:.2f} "
                 f"[{r.dMAE_CI_low:.2f}, {r.dMAE_CI_high:.2f}], ΔR² {r.dR2_ens:.2f} "
                 f"[{r.dR2_CI_low:.2f}, {r.dR2_CI_high:.2f}], Wilcoxon p = {r.wilcoxon_abs_err_p:.2g}, "
                 f"A better in {r.n_participants_A_better}/{len(parts.pids)}")
    ep = pred.groupby("config")[["epoch_leaky", "epoch_nested", "epoch_inner_mean"]].agg(["median", "min", "max"])
    L.append("\n## Selected epochs (median [min, max] over seeds × folds)")
    for cfg in ep.index:
        e = ep.loc[cfg]
        L.append(f"- {cfg}: leaky {e[('epoch_leaky','median')]:.0f} [{e[('epoch_leaky','min')]:.0f}, "
                 f"{e[('epoch_leaky','max')]:.0f}]; nested {e[('epoch_nested','median')]:.0f} "
                 f"[{e[('epoch_nested','min')]:.0f}, {e[('epoch_nested','max')]:.0f}]; "
                 f"inner-ES {e[('epoch_inner_mean','median')]:.0f}")
    if perm_summary is not None:
        L.append("\n## Full-pipeline label permutation")
        for _, r in perm_summary[perm_summary.observed_type == "seed_mean"].iterrows():
            L.append(f"- {r.protocol} {r.metric}: observed {r.observed:.2f}, null {r.null_mean:.2f} ± "
                     f"{r.null_sd:.2f}, p = {r.p_value:.3f} (N = {r.n_perm})")
    (out / "paper_numbers.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    dbg = CHK.summary(out / "debug_checks.csv")
    print(f"\n[DONE] total {(time.time()-t_start)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
