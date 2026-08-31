#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_table4_gaze_reproduction.py

Reproducible audit/re-generation script for the gaze-dynamics CARS-regression
table using gaze_features_participant_level.csv as the canonical input.

Design:
- ASD-only cohort with available CARS scores (expected n=27).
- Base representation: 18 trial-level features aggregated as mean + std = 36D.
- Enhanced representation: base 18 + 11 additional features = 29 trial-level
  features aggregated as mean + std = 58D.
- Outer Leave-One-Participant-Out (LOPO) evaluation.
- ALL preprocessing is fit on the outer-training fold only:
    SimpleImputer(strategy="median") -> StandardScaler()
- Models:
    * Mean predictor: outer-training mean CARS
    * Ridge: alpha selected by INNER LOPO over {0.1, 1, 10, 100, 1000},
      minimizing inner MAE.
    * RBF-SVR: C=10, epsilon=0.5, gamma="scale".
    * Gradient Boosting: 100 trees, max_depth=2, learning_rate=0.05.
- Predictions clipped to [15, 60].
- Reports MAE, RMSE, R2, Spearman, Pearson, CCC.
- Bootstrap 95% CIs from the fixed outer-LOPO predictions.
- Fixed-prediction permutation p-values for R2.
- Severity-stratified MAE/RMSE.
- Base-vs-enhanced paired comparison.
- Optional fully nested family-selection audit and optional nested-family
  permutation test.

IMPORTANT:
This script intentionally makes missing-value handling explicit. It does NOT
claim to reproduce an undocumented historical preprocessing path. Its purpose
is to establish a clean, fold-safe canonical Table-4 pipeline from the
currently retained participant-level CSV.

Example:
    cd /d <repository-or-data-directory>
    python run_table4_gaze_reproduction.py

Optional full nested family-selection permutation test:
    python run_table4_gaze_reproduction.py --nested-family-permutations 200

Outputs are written to:
    outputs/table4_reproduction/
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr, wilcoxon

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut


# ---------------------------------------------------------------------
# Canonical feature definitions from the manuscript
# ---------------------------------------------------------------------

BASE_FEATURES = [
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

ENHANCED_ADDITIONAL_FEATURES = [
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

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
CLIP_LOW = 15.0
CLIP_HIGH = 60.0


def aggregated_columns(feature_names: List[str]) -> List[str]:
    cols = []
    for name in feature_names:
        cols.extend([f"{name}_mean", f"{name}_std"])
    return cols


BASE_COLS = aggregated_columns(BASE_FEATURES)
ENHANCED_FEATURES = BASE_FEATURES + ENHANCED_ADDITIONAL_FEATURES
ENHANCED_COLS = aggregated_columns(ENHANCED_FEATURES)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mu_y = np.mean(y_true)
    mu_p = np.mean(y_pred)
    var_y = np.var(y_true, ddof=0)
    var_p = np.var(y_pred, ddof=0)
    cov = np.mean((y_true - mu_y) * (y_pred - mu_p))

    denom = var_y + var_p + (mu_y - mu_p) ** 2
    if denom == 0:
        return np.nan
    return float(2.0 * cov / denom)


def safe_spearman(y_true, y_pred) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = spearmanr(y_true, y_pred)
    return float(out.statistic if hasattr(out, "statistic") else out[0])


def safe_pearson(y_true, y_pred) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pearsonr(y_true, y_pred)
    return float(out.statistic if hasattr(out, "statistic") else out[0])


def metric_dict(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": safe_spearman(y_true, y_pred),
        "Pearson": safe_pearson(y_true, y_pred),
        "CCC": concordance_correlation_coefficient(y_true, y_pred),
    }


def bootstrap_metrics(y_true, y_pred, n_boot=10000, seed=20260830):
    """Participant bootstrap over the already-generated held-out predictions."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    names = ["MAE", "RMSE", "R2", "Spearman", "Pearson", "CCC"]
    vals = {k: [] for k in names}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        # R2 is undefined if bootstrap target sample has zero variance.
        if np.var(yt) == 0:
            continue

        m = metric_dict(yt, yp)
        for k in names:
            if np.isfinite(m[k]):
                vals[k].append(m[k])

    rows = []
    for k in names:
        arr = np.asarray(vals[k], dtype=float)
        if len(arr) == 0:
            lo = hi = np.nan
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append({
            "metric": k,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n_valid_boot": int(len(arr)),
        })
    return pd.DataFrame(rows)


def fixed_prediction_permutation_r2(
    y_true, y_pred, n_perm=2000, seed=20260830
) -> Tuple[float, int]:
    """
    Fixed-prediction test:
    shuffle true CARS labels against the fixed outer-LOPO predictions.
    p=(1 + number(null R2 >= observed R2))/(1+B)
    """
    if n_perm <= 0:
        return np.nan, 0

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    obs = r2_score(y_true, y_pred)
    ge = 0
    for _ in range(n_perm):
        yp_true = rng.permutation(y_true)
        null_r2 = r2_score(yp_true, y_pred)
        if null_r2 >= obs:
            ge += 1
    p = (1 + ge) / (1 + n_perm)
    return float(p), int(ge)


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------

def base_preprocessor_and_model(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def build_svr():
    return base_preprocessor_and_model(
        SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")
    )


def build_gbr(seed=20260830):
    return base_preprocessor_and_model(
        GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            random_state=seed,
        )
    )


def build_ridge(alpha):
    return base_preprocessor_and_model(Ridge(alpha=float(alpha)))


def choose_ridge_alpha_inner_lopo(X_train, y_train) -> Tuple[float, pd.DataFrame]:
    """
    Choose ridge alpha using only the current OUTER training partition.
    Criterion: minimum mean inner-LOPO MAE.
    """
    loo = LeaveOneOut()
    rows = []

    for alpha in RIDGE_ALPHAS:
        pred = np.empty(len(y_train), dtype=float)

        for inner_train, inner_val in loo.split(X_train):
            pipe = build_ridge(alpha)
            pipe.fit(X_train[inner_train], y_train[inner_train])
            p = pipe.predict(X_train[inner_val])[0]
            pred[inner_val[0]] = np.clip(p, CLIP_LOW, CLIP_HIGH)

        rows.append({
            "alpha": alpha,
            "inner_MAE": mean_absolute_error(y_train, pred),
            "inner_R2": r2_score(y_train, pred),
        })

    table = pd.DataFrame(rows).sort_values(
        ["inner_MAE", "alpha"], ascending=[True, True]
    )
    best_alpha = float(table.iloc[0]["alpha"])
    return best_alpha, table


# ---------------------------------------------------------------------
# Outer LOPO
# ---------------------------------------------------------------------

def lopo_predictions(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    model_name: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate one held-out prediction for every participant.
    Ridge gets inner-LOPO alpha selection inside every outer fold.
    """
    loo = LeaveOneOut()
    pred = np.empty(len(y), dtype=float)
    chosen_alpha = np.full(len(y), np.nan)
    ridge_tuning_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]

        if model_name == "mean":
            p = float(np.mean(ytr))

        elif model_name == "ridge":
            alpha, tuning = choose_ridge_alpha_inner_lopo(Xtr, ytr)
            chosen_alpha[test_idx[0]] = alpha
            tuning = tuning.copy()
            tuning["outer_fold"] = fold_idx
            tuning["held_out_participant"] = participant_ids[test_idx[0]]
            tuning["selected_alpha"] = alpha
            ridge_tuning_rows.append(tuning)

            pipe = build_ridge(alpha)
            pipe.fit(Xtr, ytr)
            p = float(pipe.predict(Xte)[0])

        elif model_name == "svr":
            pipe = build_svr()
            pipe.fit(Xtr, ytr)
            p = float(pipe.predict(Xte)[0])

        elif model_name == "gbr":
            pipe = build_gbr(seed=seed)
            pipe.fit(Xtr, ytr)
            p = float(pipe.predict(Xte)[0])

        else:
            raise ValueError(f"Unknown model_name={model_name}")

        pred[test_idx[0]] = np.clip(p, CLIP_LOW, CLIP_HIGH)

    out = pd.DataFrame({
        "participant_id": participant_ids,
        "y_true": y,
        "y_pred": pred,
        "abs_error": np.abs(y - pred),
    })
    if model_name == "ridge":
        out["ridge_alpha"] = chosen_alpha

    tuning_df = (
        pd.concat(ridge_tuning_rows, ignore_index=True)
        if ridge_tuning_rows else
        pd.DataFrame()
    )
    return out, tuning_df


def severity_band(y):
    if y < 30:
        return "Below cutoff (<30)"
    if y < 37:
        return "Mild-moderate (30-36.5)"
    return "Severe (>=37)"


def severity_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    tmp = pred_df.copy()
    tmp["band"] = tmp["y_true"].map(severity_band)

    order = [
        "Below cutoff (<30)",
        "Mild-moderate (30-36.5)",
        "Severe (>=37)",
    ]
    rows = []
    for band in order:
        g = tmp[tmp["band"] == band]
        rows.append({
            "band": band,
            "n": len(g),
            "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
            "RMSE": math.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
        })

    rows.append({
        "band": "All",
        "n": len(tmp),
        "MAE": mean_absolute_error(tmp["y_true"], tmp["y_pred"]),
        "RMSE": math.sqrt(mean_squared_error(tmp["y_true"], tmp["y_pred"])),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Fully nested family-selection audit
# ---------------------------------------------------------------------

def inner_family_cv_mae(Xtr, ytr, family: str, seed: int) -> float:
    loo = LeaveOneOut()
    p = np.empty(len(ytr), dtype=float)

    for itr, iva in loo.split(Xtr):
        if family == "ridge":
            # Ridge alpha itself must also be tuned strictly inside this split.
            # With only ~25 participants this is still tractable.
            alpha, _ = choose_ridge_alpha_inner_lopo(Xtr[itr], ytr[itr])
            model = build_ridge(alpha)
        elif family == "svr":
            model = build_svr()
        elif family == "gbr":
            model = build_gbr(seed=seed)
        else:
            raise ValueError(family)

        model.fit(Xtr[itr], ytr[itr])
        q = model.predict(Xtr[iva])[0]
        p[iva[0]] = np.clip(q, CLIP_LOW, CLIP_HIGH)

    return float(mean_absolute_error(ytr, p))


def nested_family_selection_predictions(
    X, y, participant_ids, seed=20260830
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Family selection is performed inside each outer fold using inner-LOPO MAE.
    """
    loo = LeaveOneOut()
    pred = np.empty(len(y), dtype=float)
    selected = []
    audit = []

    families = ["ridge", "svr", "gbr"]

    for fold_idx, (tr, te) in enumerate(loo.split(X), start=1):
        scores = {}
        for family in families:
            scores[family] = inner_family_cv_mae(
                X[tr], y[tr], family=family, seed=seed + fold_idx
            )

        best = min(scores, key=scores.get)
        selected.append(best)

        for fam, mae in scores.items():
            audit.append({
                "outer_fold": fold_idx,
                "held_out_participant": participant_ids[te[0]],
                "family": fam,
                "inner_MAE": mae,
                "selected": fam == best,
            })

        if best == "ridge":
            alpha, _ = choose_ridge_alpha_inner_lopo(X[tr], y[tr])
            model = build_ridge(alpha)
        elif best == "svr":
            model = build_svr()
        else:
            model = build_gbr(seed=seed + fold_idx)

        model.fit(X[tr], y[tr])
        q = model.predict(X[te])[0]
        pred[te[0]] = np.clip(q, CLIP_LOW, CLIP_HIGH)

    result = pd.DataFrame({
        "participant_id": participant_ids,
        "y_true": y,
        "y_pred": pred,
        "selected_family": selected,
        "abs_error": np.abs(y - pred),
    })
    return result, pd.DataFrame(audit)


def nested_family_permutation_test(
    X, y, participant_ids, n_perm, seed=20260830
) -> Dict[str, float]:
    """
    Expensive audit:
    reruns the full nested family-selection pipeline after permuting CARS labels.
    """
    if n_perm <= 0:
        return {
            "observed_R2": np.nan,
            "n_perm": 0,
            "n_null_ge_observed": 0,
            "p_value": np.nan,
        }

    rng = np.random.default_rng(seed)

    obs_pred, _ = nested_family_selection_predictions(
        X, y, participant_ids, seed=seed
    )
    obs_r2 = r2_score(obs_pred["y_true"], obs_pred["y_pred"])

    ge = 0
    for b in range(n_perm):
        yp = rng.permutation(y)
        null_pred, _ = nested_family_selection_predictions(
            X, yp, participant_ids, seed=seed + 10000 + b
        )
        null_r2 = r2_score(null_pred["y_true"], null_pred["y_pred"])
        if null_r2 >= obs_r2:
            ge += 1
        print(
            f"[nested permutation] {b+1:4d}/{n_perm} "
            f"null R2={null_r2: .4f} | ge={ge}",
            flush=True,
        )

    p = (1 + ge) / (1 + n_perm)
    return {
        "observed_R2": float(obs_r2),
        "n_perm": int(n_perm),
        "n_null_ge_observed": int(ge),
        "p_value": float(p),
    }


# ---------------------------------------------------------------------
# Data and diagnostics
# ---------------------------------------------------------------------

def check_required_columns(df: pd.DataFrame):
    required = (
        ["participant_id", "Class", "CARS Score"]
        + BASE_COLS
        + [c for c in ENHANCED_COLS if c not in BASE_COLS]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Input CSV is missing required columns:\n  " + "\n  ".join(missing)
        )


def missingness_report(asd: pd.DataFrame, cols: List[str], representation: str):
    rows = []
    for _, row in asd.iterrows():
        miss = [c for c in cols if pd.isna(row[c])]
        rows.append({
            "participant_id": row["participant_id"],
            "representation": representation,
            "n_columns": len(cols),
            "n_missing": len(miss),
            "missing_columns": ";".join(miss),
        })
    return pd.DataFrame(rows)


def print_metric_line(name, m):
    print(
        f"{name:18s} "
        f"MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
        f"R2={m['R2']:.3f}  rho={m['Spearman']:.3f}  "
        f"r={m['Pearson']:.3f}  CCC={m['CCC']:.3f}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="gaze_features_participant_level.csv",
        help="Participant-level feature CSV.",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path("outputs") / "table4_reproduction"),
        help="Output directory.",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--fixed-permutations", type=int, default=2000)
    parser.add_argument(
        "--nested-family-permutations",
        type=int,
        default=0,
        help=(
            "0 = skip expensive nested-family permutation. "
            "Use 200 to match the manuscript-style audit."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TABLE 4 GAZE-DYNAMICS REPRODUCTION")
    print("=" * 80)
    print(f"Input : {input_path.resolve()}")
    print(f"Output: {outdir.resolve()}")

    df = pd.read_csv(input_path)
    check_required_columns(df)

    asd = df[
        df["Class"].astype(str).str.upper().eq("ASD")
        & pd.to_numeric(df["CARS Score"], errors="coerce").notna()
    ].copy()

    asd["CARS Score"] = pd.to_numeric(asd["CARS Score"], errors="raise")
    asd = asd.sort_values("participant_id").reset_index(drop=True)

    if len(asd) != 27:
        warnings.warn(
            f"Expected 27 ASD participants with CARS, found {len(asd)}."
        )

    participant_ids = asd["participant_id"].to_numpy()
    y = asd["CARS Score"].to_numpy(dtype=float)

    X_base = asd[BASE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X_enh = asd[ENHANCED_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    print(f"ASD cohort n={len(asd)}")
    print(f"Base matrix    : {X_base.shape} | NaN cells={np.isnan(X_base).sum()}")
    print(f"Enhanced matrix: {X_enh.shape} | NaN cells={np.isnan(X_enh).sum()}")

    # Save exact column provenance.
    (outdir / "feature_columns.json").write_text(
        json.dumps(
            {
                "base_features_18": BASE_FEATURES,
                "base_columns_36": BASE_COLS,
                "enhanced_additional_features_11": ENHANCED_ADDITIONAL_FEATURES,
                "enhanced_features_29": ENHANCED_FEATURES,
                "enhanced_columns_58": ENHANCED_COLS,
                "imputation": (
                    "SimpleImputer(strategy='median') fit inside each "
                    "outer-training fold; never fit on the held-out participant."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    miss = pd.concat([
        missingness_report(asd, BASE_COLS, "base_36D"),
        missingness_report(asd, ENHANCED_COLS, "enhanced_58D"),
    ], ignore_index=True)
    miss.to_csv(outdir / "missingness_audit.csv", index=False)

    # -------------------------------------------------------------
    # Base models
    # -------------------------------------------------------------
    base_predictions = {}
    base_metrics_rows = []
    all_boot = []

    for model_name in ["mean", "ridge", "gbr", "svr"]:
        print(f"\nRunning BASE {model_name.upper()}...")
        pred_df, tuning_df = lopo_predictions(
            X_base, y, participant_ids, model_name, args.seed
        )
        pred_df["representation"] = "base_18"
        pred_df["model"] = model_name
        base_predictions[model_name] = pred_df

        pred_df.to_csv(
            outdir / f"predictions_base_{model_name}.csv", index=False
        )
        if not tuning_df.empty:
            tuning_df.to_csv(
                outdir / "ridge_inner_tuning_base.csv", index=False
            )

        m = metric_dict(pred_df["y_true"], pred_df["y_pred"])
        perm_p, ge = fixed_prediction_permutation_r2(
            pred_df["y_true"],
            pred_df["y_pred"],
            n_perm=args.fixed_permutations,
            seed=args.seed + 101,
        )
        row = {
            "representation": "base_18",
            "model": model_name,
            **m,
            "fixed_prediction_perm_B": args.fixed_permutations,
            "fixed_prediction_perm_ge": ge,
            "fixed_prediction_perm_p": perm_p,
        }
        base_metrics_rows.append(row)
        print_metric_line(f"base/{model_name}", m)

        boot = bootstrap_metrics(
            pred_df["y_true"],
            pred_df["y_pred"],
            n_boot=args.bootstrap,
            seed=args.seed + 201,
        )
        boot["representation"] = "base_18"
        boot["model"] = model_name
        all_boot.append(boot)

        severity_table(pred_df).to_csv(
            outdir / f"severity_base_{model_name}.csv", index=False
        )

    # -------------------------------------------------------------
    # Enhanced fixed SVR
    # -------------------------------------------------------------
    print("\nRunning ENHANCED SVR...")
    enh_pred, _ = lopo_predictions(
        X_enh, y, participant_ids, "svr", args.seed
    )
    enh_pred["representation"] = "enhanced_29"
    enh_pred["model"] = "svr"
    enh_pred.to_csv(outdir / "predictions_enhanced_svr.csv", index=False)

    enh_m = metric_dict(enh_pred["y_true"], enh_pred["y_pred"])
    enh_perm_p, enh_ge = fixed_prediction_permutation_r2(
        enh_pred["y_true"],
        enh_pred["y_pred"],
        n_perm=args.fixed_permutations,
        seed=args.seed + 102,
    )
    enh_row = {
        "representation": "enhanced_29",
        "model": "svr",
        **enh_m,
        "fixed_prediction_perm_B": args.fixed_permutations,
        "fixed_prediction_perm_ge": enh_ge,
        "fixed_prediction_perm_p": enh_perm_p,
    }
    print_metric_line("enhanced/svr", enh_m)

    enh_boot = bootstrap_metrics(
        enh_pred["y_true"],
        enh_pred["y_pred"],
        n_boot=args.bootstrap,
        seed=args.seed + 202,
    )
    enh_boot["representation"] = "enhanced_29"
    enh_boot["model"] = "svr"
    all_boot.append(enh_boot)

    severity_table(enh_pred).to_csv(
        outdir / "severity_enhanced_svr.csv", index=False
    )

    # -------------------------------------------------------------
    # Paired base-vs-enhanced comparison
    # -------------------------------------------------------------
    base_svr = base_predictions["svr"].sort_values("participant_id").reset_index(drop=True)
    enh_cmp = enh_pred.sort_values("participant_id").reset_index(drop=True)

    if not np.array_equal(
        base_svr["participant_id"].to_numpy(),
        enh_cmp["participant_id"].to_numpy(),
    ):
        raise RuntimeError("Base and enhanced participant order mismatch.")

    base_abs = np.abs(base_svr["y_true"] - base_svr["y_pred"])
    enh_abs = np.abs(enh_cmp["y_true"] - enh_cmp["y_pred"])

    try:
        w = wilcoxon(base_abs, enh_abs, alternative="two-sided")
        wilcoxon_stat = float(w.statistic)
        wilcoxon_p = float(w.pvalue)
    except Exception:
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan

    # Paired participant bootstrap for Delta R2 = enhanced - base.
    rng = np.random.default_rng(args.seed + 303)
    delta_r2 = []
    y_arr = base_svr["y_true"].to_numpy()
    pb = base_svr["y_pred"].to_numpy()
    pe = enh_cmp["y_pred"].to_numpy()

    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if np.var(y_arr[idx]) == 0:
            continue
        delta_r2.append(
            r2_score(y_arr[idx], pe[idx]) - r2_score(y_arr[idx], pb[idx])
        )

    delta_r2 = np.asarray(delta_r2, dtype=float)
    d_lo, d_hi = np.percentile(delta_r2, [2.5, 97.5])

    paired_df = pd.DataFrame({
        "participant_id": base_svr["participant_id"],
        "y_true": y_arr,
        "base_svr_pred": pb,
        "enhanced_svr_pred": pe,
        "base_abs_error": base_abs,
        "enhanced_abs_error": enh_abs,
        "enhanced_improves": enh_abs < base_abs,
    })
    paired_df.to_csv(
        outdir / "paired_base_vs_enhanced_predictions.csv", index=False
    )

    paired_summary = {
        "base_R2": float(r2_score(y_arr, pb)),
        "enhanced_R2": float(r2_score(y_arr, pe)),
        "delta_R2_enhanced_minus_base": float(
            r2_score(y_arr, pe) - r2_score(y_arr, pb)
        ),
        "paired_bootstrap_delta_R2_ci_low": float(d_lo),
        "paired_bootstrap_delta_R2_ci_high": float(d_hi),
        "n_participants_enhanced_lower_abs_error": int(np.sum(enh_abs < base_abs)),
        "n_participants": int(len(y_arr)),
        "wilcoxon_statistic_abs_errors": wilcoxon_stat,
        "wilcoxon_p_abs_errors": wilcoxon_p,
    }
    pd.DataFrame([paired_summary]).to_csv(
        outdir / "paired_base_vs_enhanced_summary.csv", index=False
    )

    # -------------------------------------------------------------
    # Core result tables
    # -------------------------------------------------------------
    model_results = pd.DataFrame(base_metrics_rows + [enh_row])
    model_results.to_csv(outdir / "table4_model_results.csv", index=False)

    bootstrap_df = pd.concat(all_boot, ignore_index=True)
    bootstrap_df.to_csv(outdir / "table4_bootstrap_cis.csv", index=False)

    # Combined severity table in manuscript-friendly form.
    sev_base = severity_table(base_svr).rename(
        columns={"MAE": "base_MAE", "RMSE": "base_RMSE"}
    )
    sev_enh = severity_table(enh_pred).rename(
        columns={"MAE": "enhanced_MAE", "RMSE": "enhanced_RMSE"}
    )
    severity_combined = sev_base.merge(
        sev_enh[["band", "enhanced_MAE", "enhanced_RMSE"]],
        on="band",
        how="left",
    )
    severity_combined.to_csv(
        outdir / "table4_severity_stratified.csv", index=False
    )

    # -------------------------------------------------------------
    # Fully nested family-selection audit
    # -------------------------------------------------------------
    print("\nRunning fully nested family-selection point-estimate audit...")
    nested_pred, nested_audit = nested_family_selection_predictions(
        X_base, y, participant_ids, seed=args.seed
    )
    nested_pred.to_csv(
        outdir / "nested_family_selection_predictions.csv", index=False
    )
    nested_audit.to_csv(
        outdir / "nested_family_selection_inner_audit.csv", index=False
    )
    nested_metrics = metric_dict(
        nested_pred["y_true"], nested_pred["y_pred"]
    )
    pd.DataFrame([nested_metrics]).to_csv(
        outdir / "nested_family_selection_metrics.csv", index=False
    )
    print_metric_line("nested/family", nested_metrics)

    nested_perm_summary = nested_family_permutation_test(
        X_base,
        y,
        participant_ids,
        n_perm=args.nested_family_permutations,
        seed=args.seed + 404,
    )
    pd.DataFrame([nested_perm_summary]).to_csv(
        outdir / "nested_family_permutation_summary.csv", index=False
    )

    # -------------------------------------------------------------
    # Text summary
    # -------------------------------------------------------------
    lines = []
    lines.append("TABLE 4 GAZE-DYNAMICS REPRODUCTION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Input: {input_path.resolve()}")
    lines.append(f"ASD n={len(asd)}")
    lines.append(f"Base shape={X_base.shape}, NaN cells={np.isnan(X_base).sum()}")
    lines.append(f"Enhanced shape={X_enh.shape}, NaN cells={np.isnan(X_enh).sum()}")
    lines.append("")
    lines.append("Missingness handling:")
    lines.append(
        "  Median imputation fit on each outer-training fold only, followed by "
        "training-fold StandardScaler."
    )
    lines.append("")
    lines.append("Core results:")
    for _, r in model_results.iterrows():
        lines.append(
            f"  {r['representation']:12s} {r['model']:6s} "
            f"MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} "
            f"R2={r['R2']:.3f} Spearman={r['Spearman']:.3f} "
            f"Pearson={r['Pearson']:.3f} CCC={r['CCC']:.3f} "
            f"fixed-perm-p={r['fixed_prediction_perm_p']:.6g}"
        )
    lines.append("")
    lines.append("Base vs enhanced SVR:")
    for k, v in paired_summary.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Fully nested family-selection point estimate:")
    for k, v in nested_metrics.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Nested family-selection permutation:")
    for k, v in nested_perm_summary.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(
        "NOTE: This is a new canonical, fold-safe reproduction from the retained "
        "participant-level feature CSV. It should not be described as an exact "
        "reproduction of an undocumented historical missing-value pipeline."
    )

    (outdir / "table4_reproduction_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Save run config
    config = vars(args).copy()
    config.update({
        "base_features_18": BASE_FEATURES,
        "enhanced_additional_features_11": ENHANCED_ADDITIONAL_FEATURES,
        "ridge_alphas": RIDGE_ALPHAS,
        "svr": {"kernel": "rbf", "C": 10, "epsilon": 0.5, "gamma": "scale"},
        "gbr": {
            "n_estimators": 100,
            "max_depth": 2,
            "learning_rate": 0.05,
        },
        "prediction_clip": [CLIP_LOW, CLIP_HIGH],
        "imputation": "training-fold median",
        "standardization": "training-fold StandardScaler",
        "ridge_inner_selection_metric": "MAE",
    })
    (outdir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(model_results.to_string(index=False))
    print(f"\nOutputs saved under: {outdir.resolve()}")
    print(
        "\nFor the expensive nested-family permutation test, rerun with:\n"
        "  python run_table4_gaze_reproduction.py --nested-family-permutations 200"
    )


if __name__ == "__main__":
    main()
