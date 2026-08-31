# -*- coding: utf-8 -*-
"""
run_gaze_canonical_from_raw.py
==============================

Self-contained canonical re-analysis of the gaze-dynamics CARS experiment
starting from the raw SMI eye-tracking CSVs.

Expected folder layout (run from the folder shown in Explorer):

    Eye-Tracking Dataset\
        Eye-tracking Output\
            1.csv
            2.csv
            ...
            25.csv
        Metadata_Participants.csv
        run_gaze_canonical_from_raw.py

The script:
  1) reconstructs 18 base + 11 enhanced trial-level gaze features;
  2) aggregates trial features to participant mean + sample SD;
  3) retains ASD participants with non-missing CARS (expected n=27);
  4) evaluates Mean predictor / Ridge / SVR / GBR under outer LOPO;
  5) fits median imputation + standardization ONLY on each training fold;
  6) tunes Ridge alpha by inner LOPO on the outer-training participants only;
  7) evaluates a fixed-SVR enhanced-feature comparison;
  8) computes participant bootstrap CIs and paired Delta-R2;
  9) optionally runs permutation tests and fully nested family selection;
 10) writes auditable CSV/JSON/TXT outputs.

Important provenance note:
This is a NEW canonical reconstruction from the retained raw SMI exports.
It is not claimed to reproduce an undocumented historical preprocessing path.

Quick start
-----------
Run Command Prompt in this folder:

    python run_gaze_canonical_from_raw.py

Recommended final run:

    python run_gaze_canonical_from_raw.py ^
        --bootstrap 10000 ^
        --fixed-permutations 2000 ^
        --nested-family-permutations 200

For a quick smoke test:

    python run_gaze_canonical_from_raw.py ^
        --bootstrap 500 ^
        --fixed-permutations 20 ^
        --nested-family-permutations 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

CARS_MIN, CARS_MAX = 15.0, 60.0

BASE_18 = [
    "fixation_fraction",
    "mean_fixation_duration_ms",
    "fixation_count",
    "saccade_fraction",
    "saccade_count",
    "mean_saccade_amplitude_px",
    "path_length_px",
    "mean_step_size_px",
    "dispersion_px",
    "bbox_width_px",
    "bbox_height_px",
    "mean_pupil_diameter_mm",
    "pupil_diameter_sd_mm",
    "aoi_entropy",
    "social_aoi_fraction",
    "trial_duration_ms",
    "sample_count",
    "tracking_ratio_pct",
]

ENHANCED_11 = [
    "social_dwell_time_ms",
    "non_social_dwell_time_ms",
    "aoi_transition_count",
    "aoi_transition_rate",
    "pupil_velocity_mean",
    "pupil_asymmetry_mean",
    "pupil_range_mm",
    "time_to_first_social_fixation_ms",
    "social_fixation_count",
    "non_social_fixation_count",
    "social_to_nonsocial_fixation_ratio",
]

ALL_29 = BASE_18 + ENHANCED_11

SOCIAL_AOI = {
    "corps", "visage", "visage1", "yeux", "yeux (1)", "bouche", "mains",
    "pointage d", "pointage g", "pointage chat", "pointage chien",
    "pointage chien (1)", "coucou", "joie droite", "joie gauche",
    "triste droite", "triste gauche", "neutre", "neutre fichier b",
    "bouche joie", "bouche neutre", "bouche triste",
    "yeux joie", "yeux neutre", "yeux triste",
}
NON_SOCIAL_AOI = {
    "ballonvisible", "balloninvisible", "chat", "chat (1)",
    "chien", "chien (1)", "white space", "aoi 001",
}
VALID_EVENTS = {"Fixation", "Saccade", "Blink"}
INVALID_PARTICIPANTS = {"Unidentified(Neg)", "Unidentified(Pos)"}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def detect_col(df: pd.DataFrame, candidates, required=True):
    norm = {
        "".join(ch for ch in str(c).lower() if ch.isalnum()): c
        for c in df.columns
    }
    for cand in candidates:
        key = "".join(ch for ch in str(cand).lower() if ch.isalnum())
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"Could not detect one of {candidates}. Columns={list(df.columns)}")
    return None


def aoi_class(x) -> str:
    s = str(x).strip().lower()
    if s in SOCIAL_AOI:
        return "social"
    if s in NON_SOCIAL_AOI:
        return "non_social"
    return "unknown"


def count_runs(mask: np.ndarray) -> int:
    mask = np.asarray(mask, bool)
    if len(mask) == 0:
        return 0
    return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))


def mean_run_duration(times: np.ndarray, mask: np.ndarray) -> float:
    times = np.asarray(times, float)
    mask = np.asarray(mask, bool)
    if len(mask) == 0 or not mask.any():
        return np.nan
    vals = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if np.isfinite(times[i - 1]) and np.isfinite(times[start]):
                vals.append(times[i - 1] - times[start])
            start = None
    if start is not None and np.isfinite(times[-1]) and np.isfinite(times[start]):
        vals.append(times[-1] - times[start])
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Raw feature extraction
# ---------------------------------------------------------------------

def load_raw_file(path: Path, eye_channel="right") -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Cannot read {path.name}: {e}")
        return None

    df.columns = [str(c).strip() for c in df.columns]

    required = {"Participant", "Trial", "Stimulus", "Category Group", "Category Right"}
    if not required.issubset(df.columns):
        print(f"[WARN] {path.name}: missing required columns {sorted(required - set(df.columns))}; skipped")
        return None

    df = df[df["Category Group"].astype(str).str.strip() == "Eye"].copy()
    df = df[~df["Participant"].astype(str).isin(INVALID_PARTICIPANTS)].copy()
    df = df[df["Category Right"].astype(str).isin(VALID_EVENTS)].copy()

    if df.empty:
        return None

    numeric_cols = [
        "RecordingTime [ms]",
        "Pupil Diameter Right [mm]",
        "Pupil Diameter Left [mm]",
        "Point of Regard Right X [px]",
        "Point of Regard Right Y [px]",
        "Point of Regard Left X [px]",
        "Point of Regard Left Y [px]",
        "Tracking Ratio [%]",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if eye_channel == "left":
        px, py = "Point of Regard Left X [px]", "Point of Regard Left Y [px]"
        pupil_col = "Pupil Diameter Left [mm]"
        other_pupil_col = "Pupil Diameter Right [mm]"
        aoi_col = "AOI Name Left"
    else:
        px, py = "Point of Regard Right X [px]", "Point of Regard Right Y [px]"
        pupil_col = "Pupil Diameter Right [mm]"
        other_pupil_col = "Pupil Diameter Left [mm]"
        aoi_col = "AOI Name Right"

    df["gx"] = df[px] if px in df.columns else np.nan
    df["gy"] = df[py] if py in df.columns else np.nan
    df["pupil"] = df[pupil_col] if pupil_col in df.columns else np.nan
    df["pupil_other"] = df[other_pupil_col] if other_pupil_col in df.columns else np.nan
    df["aoi_raw"] = df[aoi_col] if aoi_col in df.columns else "-"
    df["aoi_class"] = df["aoi_raw"].map(aoi_class)
    df["event"] = df["Category Right"].astype(str)
    return df


def compute_trial_features(g: pd.DataFrame) -> dict:
    feat = {}
    n = len(g)

    event = g["event"].to_numpy()
    is_fix = event == "Fixation"
    is_sac = event == "Saccade"

    feat["fixation_fraction"] = float(np.mean(is_fix)) if n else np.nan
    feat["saccade_fraction"] = float(np.mean(is_sac)) if n else np.nan
    feat["fixation_count"] = count_runs(is_fix)
    feat["saccade_count"] = count_runs(is_sac)

    if "RecordingTime [ms]" in g.columns:
        feat["mean_fixation_duration_ms"] = mean_run_duration(
            g["RecordingTime [ms]"].to_numpy(float), is_fix
        )
    else:
        feat["mean_fixation_duration_ms"] = np.nan

    gx = pd.to_numeric(g["gx"], errors="coerce").to_numpy(float)
    gy = pd.to_numeric(g["gy"], errors="coerce").to_numpy(float)
    valid_xy = np.isfinite(gx) & np.isfinite(gy) & (gx != 0) & (gy != 0)
    gxv, gyv = gx[valid_xy], gy[valid_xy]

    if len(gxv) > 1:
        dx, dy = np.diff(gxv), np.diff(gyv)
        step = np.sqrt(dx * dx + dy * dy)
        feat["path_length_px"] = float(np.nansum(step))
        feat["mean_step_size_px"] = float(np.nanmean(step))
        feat["dispersion_px"] = float(np.sqrt(np.nanvar(gxv) + np.nanvar(gyv)))
        feat["bbox_width_px"] = float(np.nanmax(gxv) - np.nanmin(gxv))
        feat["bbox_height_px"] = float(np.nanmax(gyv) - np.nanmin(gyv))

        sac_aligned = is_sac[valid_xy]
        sac_step_mask = sac_aligned[1:] if len(sac_aligned) > 1 else np.array([], bool)
        sac_steps = step[sac_step_mask] if len(sac_step_mask) == len(step) else np.array([])
        feat["mean_saccade_amplitude_px"] = (
            float(np.nanmean(sac_steps)) if len(sac_steps) else np.nan
        )
    else:
        for k in [
            "path_length_px", "mean_step_size_px", "dispersion_px",
            "bbox_width_px", "bbox_height_px", "mean_saccade_amplitude_px"
        ]:
            feat[k] = np.nan

    pupil = pd.to_numeric(g["pupil"], errors="coerce").to_numpy(float)
    pupil_valid = pupil[np.isfinite(pupil) & (pupil > 0)]
    feat["mean_pupil_diameter_mm"] = (
        float(np.mean(pupil_valid)) if len(pupil_valid) else np.nan
    )
    feat["pupil_diameter_sd_mm"] = (
        float(np.std(pupil_valid, ddof=0)) if len(pupil_valid) else np.nan
    )

    known = g[g["aoi_class"].isin(["social", "non_social"])]
    if len(known):
        p = known["aoi_raw"].astype(str).value_counts(normalize=True)
        feat["aoi_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
        feat["social_aoi_fraction"] = float((known["aoi_class"] == "social").mean())
    else:
        feat["aoi_entropy"] = np.nan
        feat["social_aoi_fraction"] = np.nan

    if "RecordingTime [ms]" in g.columns:
        tt = pd.to_numeric(g["RecordingTime [ms]"], errors="coerce").dropna()
        feat["trial_duration_ms"] = float(tt.max() - tt.min()) if len(tt) > 1 else np.nan
    else:
        feat["trial_duration_ms"] = np.nan

    feat["sample_count"] = int(n)
    feat["tracking_ratio_pct"] = (
        float(pd.to_numeric(g["Tracking Ratio [%]"], errors="coerce").dropna().mean())
        if "Tracking Ratio [%]" in g.columns
        and pd.to_numeric(g["Tracking Ratio [%]"], errors="coerce").notna().any()
        else np.nan
    )

    # Enhanced 11
    social_rows = g[g["aoi_class"] == "social"]
    nonsocial_rows = g[g["aoi_class"] == "non_social"]
    feat["social_dwell_time_ms"] = float(len(social_rows))
    feat["non_social_dwell_time_ms"] = float(len(nonsocial_rows))

    seq = g["aoi_class"].to_numpy()
    transitions = int(np.sum(seq[1:] != seq[:-1])) if len(seq) > 1 else 0
    feat["aoi_transition_count"] = transitions
    feat["aoi_transition_rate"] = transitions / n if n else np.nan

    if len(pupil_valid) > 1:
        feat["pupil_velocity_mean"] = float(np.mean(np.abs(np.diff(pupil_valid))))
        feat["pupil_range_mm"] = float(np.max(pupil_valid) - np.min(pupil_valid))
    else:
        feat["pupil_velocity_mean"] = np.nan
        feat["pupil_range_mm"] = np.nan

    pupil_other = pd.to_numeric(g["pupil_other"], errors="coerce").to_numpy(float)
    both = (
        np.isfinite(pupil) & np.isfinite(pupil_other)
        & (pupil > 0) & (pupil_other > 0)
    )
    feat["pupil_asymmetry_mean"] = (
        float(np.mean(np.abs(pupil[both] - pupil_other[both]))) if both.any() else np.nan
    )

    social_fix_mask = is_fix & (g["aoi_class"].to_numpy() == "social")
    nonsocial_fix_mask = is_fix & (g["aoi_class"].to_numpy() == "non_social")
    feat["social_fixation_count"] = int(np.sum(social_fix_mask))
    feat["non_social_fixation_count"] = int(np.sum(nonsocial_fix_mask))
    denom = feat["non_social_fixation_count"]
    feat["social_to_nonsocial_fixation_ratio"] = (
        feat["social_fixation_count"] / denom if denom else np.nan
    )

    feat["time_to_first_social_fixation_ms"] = np.nan
    if social_fix_mask.any() and "RecordingTime [ms]" in g.columns:
        idx = np.flatnonzero(social_fix_mask)[0]
        times = pd.to_numeric(g["RecordingTime [ms]"], errors="coerce").to_numpy(float)
        if len(times) and np.isfinite(times[0]) and np.isfinite(times[idx]):
            feat["time_to_first_social_fixation_ms"] = float(times[idx] - times[0])

    return feat


def reconstruct_features(raw_dir: Path, metadata_path: Path, outdir: Path, eye_channel="right"):
    print("\n[1/6] Reconstructing trial-level gaze features...")
    files = sorted(
        raw_dir.glob("*.csv"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir}")

    rows = []
    for i, f in enumerate(files, 1):
        df = load_raw_file(f, eye_channel=eye_channel)
        if df is None:
            continue
        n_pairs = 0
        for (pid, trial), g in df.groupby(["Participant", "Trial"], sort=False):
            if len(g) < 5:
                continue
            feat = compute_trial_features(g)
            try:
                pid_norm = str(int(float(pid)))
            except Exception:
                pid_norm = str(pid).strip()
            feat.update({
                "participant_id": pid_norm,
                "trial": trial,
                "source_file": f.name,
                "stimulus": g["Stimulus"].iloc[0] if "Stimulus" in g.columns else "",
            })
            rows.append(feat)
            n_pairs += 1
        print(f"  [{i:02d}/{len(files):02d}] {f.name}: {n_pairs} retained participant-trials")

    trial_df = pd.DataFrame(rows)
    if trial_df.empty:
        raise RuntimeError("No trial-level feature rows were produced.")

    meta = pd.read_csv(metadata_path)
    meta.columns = [str(c).strip() for c in meta.columns]
    pid_col = detect_col(meta, ["ParticipantID", "participant_id", "ID", "Participant"])
    meta[pid_col] = pd.to_numeric(meta[pid_col], errors="coerce")
    meta = meta[meta[pid_col].notna()].copy()
    meta["participant_id"] = meta[pid_col].astype(int).astype(str)

    drop_cols = [c for c in [pid_col] if c != "participant_id"]
    meta_merge = meta.drop(columns=drop_cols, errors="ignore")
    trial_df = trial_df.merge(meta_merge, on="participant_id", how="left")

    trial_path = outdir / "gaze_features_trial_level_canonical.csv"
    trial_df.to_csv(trial_path, index=False, encoding="utf-8-sig")

    feature_cols = [c for c in ALL_29 if c in trial_df.columns]
    agg = trial_df.groupby("participant_id")[feature_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg = agg.merge(meta_merge.drop_duplicates("participant_id"), on="participant_id", how="left")
    agg["n_trials"] = trial_df.groupby("participant_id").size().reindex(
        agg["participant_id"]
    ).to_numpy()

    participant_path = outdir / "gaze_features_participant_level_canonical.csv"
    agg.to_csv(participant_path, index=False, encoding="utf-8-sig")

    print(f"  Saved trial-level:       {trial_path}")
    print(f"  Saved participant-level: {participant_path}")
    return trial_df, agg


# ---------------------------------------------------------------------
# Fold-safe regression
# ---------------------------------------------------------------------

def feature_cols(features):
    return [f"{f}_mean" for f in features] + [f"{f}_std" for f in features]


def make_preprocessed_model(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def fit_predict_fixed_family(X_train, y_train, X_test, family, ridge_alpha=None, seed=0):
    if family == "ridge":
        alpha = 1.0 if ridge_alpha is None else float(ridge_alpha)
        model = Ridge(alpha=alpha)
    elif family == "svr":
        model = SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")
    elif family == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=seed
        )
    else:
        raise KeyError(family)

    pipe = make_preprocessed_model(model)
    pipe.fit(X_train, y_train)
    pred = float(pipe.predict(X_test)[0])
    return float(np.clip(pred, CARS_MIN, CARS_MAX))


def tune_ridge_alpha_inner_lopo(X_train, y_train, alphas=(0.1, 1, 10, 100, 1000)):
    loo = LeaveOneOut()
    best = None
    for alpha in alphas:
        preds = np.empty(len(y_train), float)
        for tr, va in loo.split(X_train):
            preds[va[0]] = fit_predict_fixed_family(
                X_train[tr], y_train[tr], X_train[va],
                family="ridge", ridge_alpha=alpha
            )
        mae = mean_absolute_error(y_train, preds)
        cand = (mae, float(alpha))
        if best is None or cand < best:
            best = cand
    return float(best[1])


def outer_lopo_family(X, y, family, seed=0):
    loo = LeaveOneOut()
    preds = np.empty(len(y), float)
    ridge_alphas = []
    for fold, (tr, te) in enumerate(loo.split(X)):
        alpha = None
        if family == "ridge":
            alpha = tune_ridge_alpha_inner_lopo(X[tr], y[tr])
            ridge_alphas.append(alpha)
        preds[te[0]] = fit_predict_fixed_family(
            X[tr], y[tr], X[te], family=family,
            ridge_alpha=alpha, seed=seed + fold
        )
    return preds, ridge_alphas


def mean_predictor_lopo(y):
    preds = np.empty(len(y), float)
    for i in range(len(y)):
        preds[i] = float(np.mean(np.delete(y, i)))
    return preds


def regression_metrics(y, pred):
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "pearson": float(stats.pearsonr(y, pred).statistic),
    }


def ccc(y, pred):
    y = np.asarray(y, float)
    p = np.asarray(pred, float)
    my, mp = np.mean(y), np.mean(p)
    vy, vp = np.var(y), np.var(p)
    cov = np.mean((y - my) * (p - mp))
    den = vy + vp + (my - mp) ** 2
    return float(2 * cov / den) if den > 0 else np.nan


def bootstrap_metric_ci(y, pred, metric_fn, n_boot, seed=20260831):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y[idx], pred[idx]
        if np.std(yt) == 0:
            continue
        try:
            v = float(metric_fn(yt, yp))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return tuple(map(float, np.quantile(vals, [0.025, 0.975])))


def paired_bootstrap_delta_r2(y, base_pred, enh_pred, n_boot, seed=20260831):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y[idx]
        if np.std(yt) == 0:
            continue
        vals.append(r2_score(yt, enh_pred[idx]) - r2_score(yt, base_pred[idx]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def permutation_test_fixed_family(X, y, family, n_perm, seed=20260831):
    obs_pred, _ = outer_lopo_family(X, y, family, seed=0)
    obs = r2_score(y, obs_pred)
    if n_perm <= 0:
        return obs, np.nan, []

    rng = np.random.default_rng(seed)
    null = []
    for b in range(n_perm):
        yp = rng.permutation(y)
        pred, _ = outer_lopo_family(X, yp, family, seed=b)
        null.append(r2_score(yp, pred))
        if (b + 1) % max(1, n_perm // 10) == 0:
            print(f"    {family}: permutation {b+1}/{n_perm}")
    null = np.asarray(null)
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return float(obs), float(p), null.tolist()


def nested_family_selection(X, y, families=("ridge", "svr", "gbr"), seed=0):
    loo = LeaveOneOut()
    preds = np.empty(len(y), float)
    chosen = []
    inner_scores = []

    for fold, (tr, te) in enumerate(loo.split(X)):
        Xtr, ytr = X[tr], y[tr]
        inner_loo = LeaveOneOut()
        fam_scores = {}

        for fam in families:
            ip = np.empty(len(ytr), float)
            for a, b in inner_loo.split(Xtr):
                alpha = None
                if fam == "ridge":
                    alpha = tune_ridge_alpha_inner_lopo(Xtr[a], ytr[a])
                ip[b[0]] = fit_predict_fixed_family(
                    Xtr[a], ytr[a], Xtr[b], fam, ridge_alpha=alpha,
                    seed=seed + fold
                )
            fam_scores[fam] = float(mean_absolute_error(ytr, ip))

        best = min(fam_scores, key=lambda k: (fam_scores[k], k))
        alpha = tune_ridge_alpha_inner_lopo(Xtr, ytr) if best == "ridge" else None
        preds[te[0]] = fit_predict_fixed_family(
            Xtr, ytr, X[te], best, ridge_alpha=alpha, seed=seed + fold
        )
        chosen.append(best)
        inner_scores.append(fam_scores)

    return preds, chosen, inner_scores


def nested_family_permutation(X, y, n_perm, seed=20260831):
    pred, chosen, scores = nested_family_selection(X, y, seed=0)
    obs = r2_score(y, pred)
    if n_perm <= 0:
        return obs, np.nan, pred, chosen, scores, []

    rng = np.random.default_rng(seed)
    null = []
    for b in range(n_perm):
        yp = rng.permutation(y)
        pp, _, _ = nested_family_selection(X, yp, seed=b + 1)
        null.append(r2_score(yp, pp))
        if (b + 1) % max(1, n_perm // 10) == 0:
            print(f"    nested-family: permutation {b+1}/{n_perm}")
    null = np.asarray(null)
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return float(obs), float(p), pred, chosen, scores, null.tolist()


def missingness_audit(df, cols, pid_col):
    rows = []
    for _, r in df.iterrows():
        miss = [c for c in cols if pd.isna(r[c])]
        if miss:
            rows.append({
                "participant_id": str(r[pid_col]),
                "n_missing": len(miss),
                "missing_fields": ";".join(miss),
            })
    return pd.DataFrame(rows)


def severity_band(v):
    if v < 30:
        return "<30"
    if v < 37:
        return "30-36"
    return ">=37"


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="Folder containing 'Eye-tracking Output' and Metadata_Participants.csv")
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--eye-channel", choices=["right", "left"], default="right")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--fixed-permutations", type=int, default=0,
                    help="0 skips fixed-family permutation tests; use 2000 for final audit.")
    ap.add_argument("--nested-family-permutations", type=int, default=0,
                    help="0 skips nested-family permutation; use 200 for final audit.")
    ap.add_argument("--skip-extraction", action="store_true",
                    help="Reuse canonical participant CSV already in outdir.")
    args = ap.parse_args()

    root = args.root.resolve()
    raw_dir = (args.raw_dir or (root / "Eye-tracking Output")).resolve()
    metadata = (args.metadata or (root / "Metadata_Participants.csv")).resolve()
    outdir = (args.outdir or (root / "outputs" / "gaze_canonical_reproduction")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("GAZE-DYNAMICS CANONICAL REPRODUCTION")
    print("=" * 78)
    print(f"Root:      {root}")
    print(f"Raw dir:   {raw_dir}")
    print(f"Metadata:  {metadata}")
    print(f"Output:    {outdir}")

    if not metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw eye-tracking folder not found: {raw_dir}")

    if args.skip_extraction:
        participant_path = outdir / "gaze_features_participant_level_canonical.csv"
        if not participant_path.exists():
            raise FileNotFoundError(
                f"--skip-extraction requested but missing {participant_path}"
            )
        agg = pd.read_csv(participant_path)
    else:
        _, agg = reconstruct_features(
            raw_dir, metadata, outdir, eye_channel=args.eye_channel
        )

    print("\n[2/6] Selecting ASD participants with CARS...")
    class_col = detect_col(agg, ["Class", "Diagnosis", "Group"])
    cars_col = detect_col(agg, ["CARS Score", "CARS_Score", "CARS"])
    pid_col = detect_col(agg, ["participant_id", "ParticipantID", "ID"])

    asd = agg[agg[class_col].astype(str).str.strip().str.upper().eq("ASD")].copy()
    asd[cars_col] = pd.to_numeric(asd[cars_col], errors="coerce")
    asd = asd[asd[cars_col].notna()].copy()
    asd = asd.sort_values(pid_col, key=lambda s: pd.to_numeric(s, errors="coerce"))

    print(f"  ASD + CARS n = {len(asd)} (expected 27)")

    base_cols = feature_cols(BASE_18)
    enh_cols = feature_cols(ALL_29)
    for c in base_cols + enh_cols:
        if c not in asd.columns:
            raise KeyError(f"Missing aggregated column: {c}")

    audit = missingness_audit(asd, enh_cols, pid_col)
    audit.to_csv(outdir / "missingness_audit.csv", index=False)
    print(f"  Participants with >=1 missing aggregated field: {len(audit)}")
    if len(audit):
        print(audit.to_string(index=False))

    Xb = asd[base_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Xe = asd[enh_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = asd[cars_col].to_numpy(float)
    pids = asd[pid_col].astype(str).to_numpy()

    print("\n[3/6] Base 18-feature models under outer LOPO...")
    results = []
    pred_map = {}

    mean_pred = mean_predictor_lopo(y)
    pred_map["mean"] = mean_pred
    m = regression_metrics(y, mean_pred)
    m.update({"model": "Mean predictor", "r2_ci_lo": np.nan, "r2_ci_hi": np.nan})
    results.append(m)

    for fam in ["ridge", "svr", "gbr"]:
        print(f"  Running {fam.upper()}...")
        pred, ridge_alphas = outer_lopo_family(Xb, y, fam)
        pred_map[fam] = pred
        mm = regression_metrics(y, pred)
        lo, hi = bootstrap_metric_ci(y, pred, r2_score, args.bootstrap, seed=20260831)
        mm.update({
            "model": fam.upper(),
            "r2_ci_lo": lo,
            "r2_ci_hi": hi,
            "ccc": ccc(y, pred),
            "ridge_alpha_counts": (
                json.dumps(dict(Counter(ridge_alphas))) if fam == "ridge" else ""
            ),
        })
        results.append(mm)
        print(
            f"    MAE={mm['mae']:.3f} RMSE={mm['rmse']:.3f} "
            f"R2={mm['r2']:.3f} 95%CI=[{lo:.3f},{hi:.3f}] "
            f"rho={mm['spearman']:.3f}"
        )

    table = pd.DataFrame(results)
    table.to_csv(outdir / "table4_base_model_results.csv", index=False)

    print("\n[4/6] Enhanced 29-feature fixed-SVR comparison...")
    base_svr = pred_map["svr"]
    enh_svr, _ = outer_lopo_family(Xe, y, "svr")
    base_metrics = regression_metrics(y, base_svr)
    enh_metrics = regression_metrics(y, enh_svr)

    enh_ci = bootstrap_metric_ci(y, enh_svr, r2_score, args.bootstrap, seed=20260832)
    delta_r2 = enh_metrics["r2"] - base_metrics["r2"]
    delta_ci = paired_bootstrap_delta_r2(
        y, base_svr, enh_svr, args.bootstrap, seed=20260833
    )
    abs_base = np.abs(y - base_svr)
    abs_enh = np.abs(y - enh_svr)
    try:
        wil = stats.wilcoxon(abs_enh, abs_base, zero_method="wilcox", alternative="two-sided")
        wil_p = float(wil.pvalue)
    except Exception:
        wil_p = np.nan

    improved = int(np.sum(abs_enh < abs_base))
    enhanced_summary = {
        "n": len(y),
        "base_svr_r2": base_metrics["r2"],
        "enhanced_svr_r2": enh_metrics["r2"],
        "enhanced_r2_ci_lo": enh_ci[0],
        "enhanced_r2_ci_hi": enh_ci[1],
        "delta_r2_enh_minus_base": delta_r2,
        "delta_r2_ci_lo": delta_ci[0],
        "delta_r2_ci_hi": delta_ci[1],
        "wilcoxon_abs_error_p": wil_p,
        "participants_improved": improved,
        "participants_total": len(y),
        "base_mae": base_metrics["mae"],
        "enhanced_mae": enh_metrics["mae"],
        "base_spearman": base_metrics["spearman"],
        "enhanced_spearman": enh_metrics["spearman"],
        "base_ccc": ccc(y, base_svr),
        "enhanced_ccc": ccc(y, enh_svr),
    }
    pd.DataFrame([enhanced_summary]).to_csv(
        outdir / "enhanced_vs_base_svr_summary.csv", index=False
    )

    print(
        f"  Base SVR:     R2={base_metrics['r2']:.3f}, MAE={base_metrics['mae']:.3f}"
    )
    print(
        f"  Enhanced SVR: R2={enh_metrics['r2']:.3f}, MAE={enh_metrics['mae']:.3f}, "
        f"95% CI=[{enh_ci[0]:.3f},{enh_ci[1]:.3f}]"
    )
    print(
        f"  Delta R2={delta_r2:+.3f}, paired 95% CI=[{delta_ci[0]:.3f},{delta_ci[1]:.3f}], "
        f"Wilcoxon p={wil_p:.4f}"
    )

    pred_df = pd.DataFrame({
        "participant_id": pids,
        "cars_true": y,
        "pred_mean": mean_pred,
        "pred_ridge": pred_map["ridge"],
        "pred_svr_base": base_svr,
        "pred_gbr": pred_map["gbr"],
        "pred_svr_enhanced": enh_svr,
        "abs_err_svr_base": abs_base,
        "abs_err_svr_enhanced": abs_enh,
        "severity_band": [severity_band(v) for v in y],
    })
    pred_df.to_csv(outdir / "participant_predictions.csv", index=False)

    sev_rows = []
    for band in ["<30", "30-36", ">=37"]:
        mask = pred_df["severity_band"].eq(band).to_numpy()
        for model_col in ["pred_ridge", "pred_svr_base", "pred_gbr", "pred_svr_enhanced"]:
            if mask.sum() == 0:
                continue
            yt = y[mask]
            yp = pred_df.loc[mask, model_col].to_numpy(float)
            sev_rows.append({
                "severity_band": band,
                "model": model_col,
                "n": int(mask.sum()),
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            })
    pd.DataFrame(sev_rows).to_csv(
        outdir / "severity_stratified_metrics.csv", index=False
    )

    print("\n[5/6] Fully nested family selection...")
    nested_obs, nested_p, nested_pred, chosen, inner_scores, nested_null = (
        nested_family_permutation(
            Xb, y, args.nested_family_permutations, seed=20260834
        )
    )
    nested_metrics = regression_metrics(y, nested_pred)
    nested_metrics.update({
        "permutation_p": nested_p,
        "n_permutations": args.nested_family_permutations,
        "chosen_family_counts": json.dumps(dict(Counter(chosen))),
    })
    pd.DataFrame([nested_metrics]).to_csv(
        outdir / "nested_family_selection_summary.csv", index=False
    )
    pd.DataFrame({
        "participant_id": pids,
        "cars_true": y,
        "nested_family_prediction": nested_pred,
        "chosen_family": chosen,
    }).to_csv(outdir / "nested_family_predictions.csv", index=False)

    print(
        f"  Nested family selection: R2={nested_obs:.3f}, "
        f"MAE={nested_metrics['mae']:.3f}, p={nested_p}"
    )
    print(f"  Family counts: {dict(Counter(chosen))}")

    fixed_perm_summary = []
    if args.fixed_permutations > 0:
        print("\n[6/6] Fixed-family permutation tests...")
        for fam in ["ridge", "svr", "gbr"]:
            obs, p, null = permutation_test_fixed_family(
                Xb, y, fam, args.fixed_permutations, seed=20260840
            )
            fixed_perm_summary.append({
                "family": fam,
                "observed_r2": obs,
                "p": p,
                "n_permutations": args.fixed_permutations,
                "null_mean": float(np.mean(null)),
                "null_min": float(np.min(null)),
                "null_max": float(np.max(null)),
            })
        pd.DataFrame(fixed_perm_summary).to_csv(
            outdir / "fixed_family_permutation_summary.csv", index=False
        )
    else:
        print("\n[6/6] Fixed-family permutation tests skipped (--fixed-permutations 0).")

    provenance = {
        "root": str(root),
        "raw_dir": str(raw_dir),
        "metadata": str(metadata),
        "eye_channel": args.eye_channel,
        "n_asd_cars": int(len(y)),
        "base_trial_features": BASE_18,
        "enhanced_extra_trial_features": ENHANCED_11,
        "base_aggregated_dimension": len(base_cols),
        "enhanced_aggregated_dimension": len(enh_cols),
        "missing_value_policy": (
            "Within every model-training fold only: median imputation, then "
            "StandardScaler; held-out participant never contributes to either."
        ),
        "ridge_tuning": "alpha in [0.1,1,10,100,1000] selected by inner LOPO MAE",
        "svr": "RBF C=10 epsilon=0.5 gamma=scale",
        "gbr": "100 trees, max_depth=2, learning_rate=0.05",
        "bootstrap_replicates": args.bootstrap,
        "fixed_permutations": args.fixed_permutations,
        "nested_family_permutations": args.nested_family_permutations,
        "historical_reproduction_claim": False,
        "provenance_note": (
            "New canonical reconstruction from raw SMI exports; does not claim "
            "to reproduce undocumented historical preprocessing."
        ),
    }
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = []
    txt.append("GAZE-DYNAMICS CANONICAL REPRODUCTION SUMMARY")
    txt.append("=" * 60)
    txt.append(f"n ASD+CARS = {len(y)}")
    txt.append("")
    txt.append("Base models:")
    for _, r in table.iterrows():
        txt.append(
            f"  {r['model']}: MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}, "
            f"R2={r['r2']:.4f}, Spearman={r['spearman']:.4f}"
        )
    txt.append("")
    txt.append(
        f"Enhanced fixed-SVR: R2={enh_metrics['r2']:.4f}; "
        f"DeltaR2={delta_r2:+.4f} [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]; "
        f"Wilcoxon p={wil_p:.6f}"
    )
    txt.append(
        f"Nested family selection: R2={nested_obs:.4f}; "
        f"p={nested_p}; counts={dict(Counter(chosen))}"
    )
    txt.append("")
    txt.append("IMPORTANT: This is a new canonical reconstruction, not an undocumented")
    txt.append("historical preprocessing reproduction.")
    (outdir / "canonical_summary.txt").write_text("\n".join(txt), encoding="utf-8")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Results written to:\n  {outdir}")
    print("\nMost important files:")
    print("  table4_base_model_results.csv")
    print("  enhanced_vs_base_svr_summary.csv")
    print("  participant_predictions.csv")
    print("  nested_family_selection_summary.csv")
    print("  missingness_audit.csv")
    print("  provenance.json")
    print("  canonical_summary.txt")


if __name__ == "__main__":
    main()
