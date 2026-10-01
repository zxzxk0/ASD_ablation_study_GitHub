# -*- coding: utf-8 -*-
r"""
Participant-leakage stress test for scanpath-image embeddings.

This script reuses the six frozen-backbone caches already used by
``run_shallow_probe_all_backbones.py``.  It does not extract embeddings or
train a vision backbone.

Primary comparison
------------------
For each backbone, the exact same Ridge probe is evaluated under:

1. ``image``: shuffled K-fold splitting of individual scanpath images.  Images
   from the same participant can occur in both train and test folds.
2. ``participant``: shuffled K-fold splitting of participant IDs.  Every image
   from a held-out participant stays in the test fold.

Predictions are always averaged within participant before MAE/RMSE/R2 are
computed.  Thus the metric unit is a participant in both protocols.

Ridge alpha is selected inside each outer training fold.  The inner split
matches the outer protocol: image-level for ``image`` and participant-level
for ``participant``.  This deliberately estimates the full consequence of the
two evaluation protocols while keeping the representation, estimator, alpha
grid, metric unit, and number of folds fixed.

Negative control
----------------
CARS labels are permuted between participants while all images from one
participant retain the same permuted label.  If an image-level split still
achieves positive R2, the embedding/probe can recover arbitrary participant-
specific labels through repeated-participant overlap.  Participant-level CV
should not be able to do this for unseen participants.  By default, every
permutation repeats the full nested alpha-selection procedure with the same
split seeds used for the observed labels.  ``--fixed-permutation-alpha`` is
available only as an explicitly labeled exploratory shortcut.

Expected cache format
---------------------
Each NPZ file contains:
    X     : (n_images, embedding_dim) float array
    paths : (n_images,) image path/name array

The default filenames match ``outputs/cache_v3_unambiguous`` in the existing
project.  By default, participant IDs are parsed from names such as
``TS001_39.png`` using the final numeric token (``39``).

Quick test
----------
    python run_participant_leakage_stress_test.py --self-test-only

Fast real-data smoke test
-------------------------
    python run_participant_leakage_stress_test.py ^
      --cache-dir outputs\cache_v3_unambiguous ^
      --metadata-csv Metadata\Metadata\Metadata_Participants.csv ^
      --n-repeats 2 --n-label-permutations 2

Recommended final run
---------------------
    python run_participant_leakage_stress_test.py ^
      --cache-dir outputs\cache_v3_unambiguous ^
      --metadata-csv Metadata\Metadata\Metadata_Participants.csv ^
      --n-repeats 10 --n-label-permutations 100

Outputs
-------
``protocol_metrics.csv``
    One row per backbone, repeat, and split protocol.
``participant_predictions.csv``
    Participant-level out-of-fold predictions for the real labels.
``label_permutation_metrics.csv``
    One row per backbone, permutation, and protocol.
``summary_by_backbone.csv``
    Mean/SD metrics and image-minus-participant inflation.
``paired_protocol_differences.csv``
    Repeat-wise paired differences and their source metrics.
``inflation_summary.csv``
    One row per backbone summarizing repeated-partition inflation and empirical
    2.5/97.5 percentiles (not confidence intervals).
``permutation_summary.csv``
    Shuffled-label null summaries and protocol-specific empirical p-values.
``permutation_null_statistics.csv``
    One repeat-averaged null statistic per permutation and protocol.
``participant_image_counts.csv``
    Participant IDs, CARS labels, and image counts for every backbone.
``input_exclusions.csv``
    Paths excluded because IDs were unparsable, labels were missing, or values
    were non-finite.
``run_config.json``
    Full command-line configuration and cache filenames.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BACKBONES = {
    "EVA02-B/16": "embeddings_EVA02-B16.npz",
    "EVA02-L/14": "embeddings_EVA02-L14.npz",
    "DINOv2 ViT-B/14": "embeddings_DINOv2-ViT-B14.npz",
    "DINOv2 ViT-S/14": "embeddings_DINOv2-ViT-S14.npz",
    "CLIP ViT-L/14": "embeddings_CLIP-ViT-L14.npz",
    "ResNet-50": "embeddings_ResNet50.npz",
}

CARS_MIN = 15.0
CARS_MAX = 60.0


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def normalize_pid(value: object) -> str:
    """Normalize participant IDs so metadata values like 39.0 match '39'."""
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def parse_pid(path: object, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(path))
    if match is None:
        return None
    return normalize_pid(match.group(1))


def find_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def load_cars_lookup(
    metadata_csv: Path,
    pid_column: str | None,
    cars_column: str | None,
    class_column: str | None,
    asd_label: str,
) -> dict[str, float]:
    meta = pd.read_csv(metadata_csv)
    meta.columns = [str(column).strip() for column in meta.columns]

    pid_column = pid_column or find_column(
        meta.columns, ["ParticipantID", "Participant", "SubjectID", "Subject", "ID"]
    )
    if cars_column is None:
        cars_column = next(
            (column for column in meta.columns if "cars" in column.lower()), None
        )
    class_column = class_column or find_column(
        meta.columns, ["Class", "Diagnosis", "Group", "Dx"]
    )

    if pid_column is None or cars_column is None:
        raise ValueError(
            "Could not detect participant/CARS columns. "
            f"Available columns: {list(meta.columns)}. Use --pid-column and "
            "--cars-column explicitly."
        )

    work = meta.copy()
    work["_pid"] = work[pid_column].map(normalize_pid)
    work["_cars"] = pd.to_numeric(work[cars_column], errors="coerce")

    if class_column is not None:
        wanted = str(asd_label).strip().upper()
        observed = work[class_column].astype(str).str.strip().str.upper()
        filtered = work[observed == wanted]
        if filtered.empty:
            raise ValueError(
                f"No rows matched --asd-label={asd_label!r} in {class_column!r}. "
                f"Observed values: {sorted(observed.unique().tolist())}"
            )
        work = filtered

    work = work.dropna(subset=["_cars"])
    duplicated = work[work["_pid"].duplicated(keep=False)]["_pid"].unique()
    if len(duplicated):
        raise ValueError(
            "Metadata contains repeated participant IDs after filtering: "
            f"{duplicated[:10].tolist()}"
        )

    lookup = dict(zip(work["_pid"], work["_cars"].astype(float)))
    if not lookup:
        raise ValueError("No usable ASD participants with numeric CARS scores were found.")

    print(
        f"[metadata] pid={pid_column!r}, CARS={cars_column!r}, "
        f"class={class_column!r}; loaded {len(lookup)} labels"
    )
    return lookup


def load_image_embeddings(
    npz_path: Path,
    cars_lookup: dict[str, float],
    pid_pattern: re.Pattern[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    data = np.load(npz_path, allow_pickle=True)
    missing_keys = {"X", "paths"}.difference(data.files)
    if missing_keys:
        raise KeyError(f"{npz_path} is missing NPZ keys: {sorted(missing_keys)}")

    X_all = np.asarray(data["X"])
    paths_all = np.asarray(data["paths"], dtype=object)
    if X_all.ndim != 2 or len(X_all) != len(paths_all):
        raise ValueError(
            f"Invalid arrays in {npz_path}: X={X_all.shape}, paths={paths_all.shape}"
        )

    pids: list[str] = []
    keep: list[int] = []
    exclusions: list[dict[str, object]] = []
    for index, path in enumerate(paths_all):
        pid = parse_pid(path, pid_pattern)
        if pid is None:
            exclusions.append(
                {"path": str(path), "participant_id": "", "reason": "unparsable_pid"}
            )
            continue
        if pid not in cars_lookup:
            exclusions.append(
                {"path": str(path), "participant_id": pid, "reason": "missing_cars_label"}
            )
            continue
        keep.append(index)
        pids.append(pid)

    if not keep:
        raise ValueError(
            f"No embeddings in {npz_path} matched metadata. Check --pid-regex and IDs."
        )

    X = np.asarray(X_all[keep], dtype=np.float64)
    paths = paths_all[keep]
    groups = np.asarray(pids, dtype=object)
    y = np.asarray([cars_lookup[pid] for pid in groups], dtype=np.float64)

    finite_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
    dropped_nonfinite = int((~finite_rows).sum())
    for path, pid in zip(paths[~finite_rows], groups[~finite_rows]):
        exclusions.append(
            {"path": str(path), "participant_id": str(pid), "reason": "nonfinite_value"}
        )
    X, y, groups, paths = (
        X[finite_rows],
        y[finite_rows],
        groups[finite_rows],
        paths[finite_rows],
    )

    participant_counts = (
        pd.DataFrame(
            {
                "participant_id": groups.astype(str),
                "cars": y.astype(float),
            }
        )
        .groupby("participant_id", as_index=False)
        .agg(cars=("cars", "first"), n_images=("participant_id", "size"))
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    exclusions_df = pd.DataFrame(
        exclusions, columns=["path", "participant_id", "reason"]
    )

    reason_counts = exclusions_df["reason"].value_counts().to_dict()

    print(
        f"[cache] {npz_path.name}: {len(X)} images, "
        f"{len(np.unique(groups))} participants, d={X.shape[1]}; "
        f"excluded={reason_counts}, nonfinite={dropped_nonfinite}"
    )
    return X, y, groups, paths, participant_counts, exclusions_df


def shuffled_kfold_indices(
    n_samples: int, n_splits: int, seed: int
) -> Iterator[Fold]:
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError(f"n_splits={n_splits} is invalid for n={n_samples}")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    fold_parts = np.array_split(indices, n_splits)
    all_indices = np.arange(n_samples)
    for test_idx in fold_parts:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        yield Fold(all_indices[train_mask], np.asarray(test_idx, dtype=int))


def shuffled_group_kfold_indices(
    groups: np.ndarray, n_splits: int, seed: int
) -> Iterator[Fold]:
    unique_groups = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError(
            f"n_splits={n_splits} is invalid for {len(unique_groups)} participants"
        )
    rng = np.random.RandomState(seed)
    shuffled_groups = rng.permutation(unique_groups)
    group_parts = np.array_split(shuffled_groups, n_splits)
    for test_groups in group_parts:
        test_mask = np.isin(groups, test_groups)
        yield Fold(np.flatnonzero(~test_mask), np.flatnonzero(test_mask))


def make_folds(
    protocol: str, groups: np.ndarray, n_splits: int, seed: int
) -> list[Fold]:
    if protocol == "image":
        return list(shuffled_kfold_indices(len(groups), n_splits, seed))
    if protocol == "participant":
        return list(shuffled_group_kfold_indices(groups, n_splits, seed))
    raise ValueError(f"Unknown protocol: {protocol}")


def aggregate_prediction_arrays(
    y_true_image: np.ndarray,
    y_pred_image: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate image predictions with NumPy; each participant gets equal metric weight."""
    participant_ids, inverse = np.unique(groups.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(int)
    y_true_image = np.asarray(y_true_image, dtype=float)
    y_pred_image = np.asarray(y_pred_image, dtype=float)
    y_true = np.bincount(inverse, weights=y_true_image) / counts
    y_pred = np.bincount(inverse, weights=y_pred_image) / counts

    group_min = np.full(len(participant_ids), np.inf)
    group_max = np.full(len(participant_ids), -np.inf)
    np.minimum.at(group_min, inverse, y_true_image)
    np.maximum.at(group_max, inverse, y_true_image)
    inconsistent = np.flatnonzero(~np.isclose(group_min, group_max))
    if len(inconsistent):
        bad = participant_ids[inconsistent[:10]].tolist()
        raise ValueError(f"Participants have inconsistent labels: {bad}")
    return participant_ids, y_true, y_pred, counts


def aggregate_predictions(
    y_true_image: np.ndarray,
    y_pred_image: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    participant_ids, y_true, y_pred, counts = aggregate_prediction_arrays(
        y_true_image, y_pred_image, groups
    )
    return pd.DataFrame(
        {
            "participant_id": participant_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "n_images": counts,
        }
    )


def participant_metrics_from_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
        spearman = np.nan
    else:
        # Tuple indexing supports SciPy versions before the .statistic attribute.
        pearson = float(pearsonr(y_true, y_pred)[0])
        spearman = float(spearmanr(y_true, y_pred)[0])

    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true, ddof=0))
    var_pred = float(np.var(y_pred, ddof=0))
    covariance = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
    ccc_denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = np.nan if ccc_denominator == 0 else 2.0 * covariance / ccc_denominator

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": pearson,
        "spearman": spearman,
        "ccc": float(ccc),
    }


def participant_metrics(participant_predictions: pd.DataFrame) -> dict[str, float]:
    return participant_metrics_from_arrays(
        participant_predictions["y_true"].to_numpy(dtype=float),
        participant_predictions["y_pred"].to_numpy(dtype=float),
    )


def fit_ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=float(alpha))
    model.fit(X_train_scaled, y_train)
    return np.clip(model.predict(X_test_scaled), CARS_MIN, CARS_MAX)


def ridge_path_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    """Predict the full Ridge path after one scaling and one thin SVD.

    Returns an array with shape ``(n_alphas, n_test)`` and is algebraically
    equivalent to fitting sklearn Ridge(fit_intercept=True) separately for each
    alpha on the standardized training data.
    """
    scaler = StandardScaler()
    A = scaler.fit_transform(X_train)
    B = scaler.transform(X_test)
    y_mean = float(np.mean(y_train))
    centered_y = np.asarray(y_train, dtype=float) - y_mean
    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    uty = U.T @ centered_y
    bvt = B @ Vt.T
    predictions = []
    for alpha in alphas:
        shrinkage = singular_values / (singular_values**2 + float(alpha))
        pred = bvt @ (shrinkage * uty) + y_mean
        predictions.append(np.clip(pred, CARS_MIN, CARS_MAX))
    return np.asarray(predictions, dtype=float)


def select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    protocol: str,
    alpha_grid: Sequence[float],
    inner_splits: int,
    seed: int,
) -> float:
    n_units = len(groups) if protocol == "image" else len(np.unique(groups))
    actual_splits = min(inner_splits, n_units)
    if actual_splits < 2:
        return float(alpha_grid[0])
    folds = make_folds(protocol, groups, actual_splits, seed)

    fold_mae_by_alpha: list[list[float]] = [[] for _ in alpha_grid]
    for fold in folds:
        path_predictions = ridge_path_predict(
            X[fold.train_idx], y[fold.train_idx], X[fold.test_idx], alpha_grid
        )
        for alpha_index, pred in enumerate(path_predictions):
            _, y_true_participant, y_pred_participant, _ = aggregate_prediction_arrays(
                y[fold.test_idx], pred, groups[fold.test_idx]
            )
            fold_mae_by_alpha[alpha_index].append(
                float(mean_absolute_error(y_true_participant, y_pred_participant))
            )
    mean_mae = [float(np.mean(values)) for values in fold_mae_by_alpha]

    # np.argmin deterministically chooses the first/smaller listed alpha on ties.
    return float(alpha_grid[int(np.argmin(mean_mae))])


def cross_validated_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    protocol: str,
    outer_splits: int,
    inner_splits: int,
    alpha_grid: Sequence[float],
    seed: int,
    fixed_alpha: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, float]]:
    folds = make_folds(protocol, groups, outer_splits, seed)
    image_predictions = np.full(len(y), np.nan, dtype=float)
    baseline_predictions = np.full(len(y), np.nan, dtype=float)
    selected_alphas: list[float] = []
    overlap_rates: list[float] = []
    alpha_max_over_s2_max: list[float] = []
    alpha_min_over_s2_min: list[float] = []

    for outer_index, fold in enumerate(folds):
        train_groups = groups[fold.train_idx]
        test_groups = groups[fold.test_idx]
        overlap = np.intersect1d(np.unique(train_groups), np.unique(test_groups))
        overlap_rates.append(len(overlap) / max(1, len(np.unique(test_groups))))

        if fixed_alpha is None:
            alpha = select_alpha(
                X[fold.train_idx],
                y[fold.train_idx],
                train_groups,
                protocol,
                alpha_grid,
                inner_splits,
                seed + 10_000 + outer_index,
            )
        else:
            alpha = float(fixed_alpha)
        selected_alphas.append(alpha)
        image_predictions[fold.test_idx] = fit_ridge_predict(
            X[fold.train_idx], y[fold.train_idx], X[fold.test_idx], alpha
        )
        # Intercept-only reference on the identical folds.  Grouped CV R2 is
        # negatively biased even for a label-free predictor, so inflation
        # should be read against this per-protocol baseline.
        # Participant-weighted training mean, matching the participant-level
        # metric (each participant counts once regardless of image count).
        _, train_y_participant, _, _ = aggregate_prediction_arrays(
            y[fold.train_idx], y[fold.train_idx], groups[fold.train_idx]
        )
        baseline_predictions[fold.test_idx] = float(np.mean(train_y_participant))

        # Spectrum of the standardized outer-training matrix.  If
        # max(alpha_grid) >> s2_max, the upper grid edge is the intercept-only
        # limit; if min(alpha_grid) << smallest nonzero s2, the lower edge is
        # the minimum-norm (interpolation) limit.  Boundary selections are then
        # near-asymptotic regimes.  Report these together with the boundary
        # selection rates; the flags alone do not say which alpha was chosen.
        s2 = np.linalg.svd(
            StandardScaler().fit_transform(X[fold.train_idx]), compute_uv=False
        ) ** 2
        s2_nonzero = s2[s2 > 1e-10 * s2.max()]
        alpha_max_over_s2_max.append(float(np.max(alpha_grid)) / float(s2.max()))
        alpha_min_over_s2_min.append(float(np.min(alpha_grid)) / float(s2_nonzero.min()))

    if np.isnan(image_predictions).any():
        raise RuntimeError("Some samples did not receive an out-of-fold prediction.")

    participant_pred = aggregate_predictions(y, image_predictions, groups)
    metrics = participant_metrics(participant_pred)
    _, base_true, base_pred, _ = aggregate_prediction_arrays(
        y, baseline_predictions, groups
    )
    baseline_metrics = participant_metrics_from_arrays(base_true, base_pred)
    metrics["baseline_mae"] = baseline_metrics["mae"]
    metrics["baseline_r2"] = baseline_metrics["r2"]
    metrics["r2_gain_over_intercept_baseline"] = metrics["r2"] - baseline_metrics["r2"]
    sse_model = float(np.sum((participant_pred["y_true"] - participant_pred["y_pred"]) ** 2))
    sse_base = float(np.sum((base_true - base_pred) ** 2))
    # Skill score relative to the fold-specific intercept-only baseline.
    metrics["skill_vs_intercept_baseline"] = (
        np.nan if sse_base == 0 else 1.0 - sse_model / sse_base
    )
    selected_array = np.asarray(selected_alphas, dtype=float)
    if fixed_alpha is None:
        alpha_min = float(np.min(alpha_grid))
        alpha_max = float(np.max(alpha_grid))
        lower_boundary_rate = float(np.mean(np.isclose(selected_array, alpha_min)))
        upper_boundary_rate = float(np.mean(np.isclose(selected_array, alpha_max)))
        boundary_rate = float(
            np.mean(
                np.isclose(selected_array, alpha_min)
                | np.isclose(selected_array, alpha_max)
            )
        )
    else:
        lower_boundary_rate = np.nan
        upper_boundary_rate = np.nan
        boundary_rate = np.nan
    diagnostics = {
        "mean_selected_alpha": float(np.mean(selected_alphas)),
        "median_selected_alpha": float(np.median(selected_alphas)),
        "lower_alpha_boundary_rate": lower_boundary_rate,
        "upper_alpha_boundary_rate": upper_boundary_rate,
        "any_alpha_boundary_rate": boundary_rate,
        "mean_test_participant_overlap": float(np.mean(overlap_rates)),
        "min_alpha_max_over_s2_max": float(np.min(alpha_max_over_s2_max)),
        "max_alpha_min_over_s2_min": float(np.max(alpha_min_over_s2_min)),
    }
    return metrics, participant_pred, diagnostics


def participant_label_permutation(
    y: np.ndarray, groups: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    unique_groups = np.unique(groups)
    group_labels = np.asarray([y[groups == group][0] for group in unique_groups])
    shuffled_labels = rng.permutation(group_labels)
    mapping = dict(zip(unique_groups, shuffled_labels))
    return np.asarray([mapping[group] for group in groups], dtype=float)


def summarize_real_metrics(
    metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics_df.groupby(["backbone", "protocol"], as_index=False)
        .agg(
            n_participants=("n_participants", "first"),
            n_images=("n_images", "first"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_sd=("pearson", "std"),
            pearson_valid_n=("pearson", "count"),
            pearson_nan_n=("pearson", lambda s: int(s.isna().sum())),
            spearman_mean=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
            spearman_valid_n=("spearman", "count"),
            spearman_nan_n=("spearman", lambda s: int(s.isna().sum())),
            ccc_mean=("ccc", "mean"),
            ccc_sd=("ccc", "std"),
            ccc_valid_n=("ccc", "count"),
            ccc_nan_n=("ccc", lambda s: int(s.isna().sum())),
            lower_alpha_boundary_rate_mean=("lower_alpha_boundary_rate", "mean"),
            upper_alpha_boundary_rate_mean=("upper_alpha_boundary_rate", "mean"),
            any_alpha_boundary_rate_mean=("any_alpha_boundary_rate", "mean"),
            overlap_mean=("mean_test_participant_overlap", "mean"),
            baseline_r2_mean=("baseline_r2", "mean"),
            baseline_mae_mean=("baseline_mae", "mean"),
            skill_vs_intercept_baseline_mean=("skill_vs_intercept_baseline", "mean"),
            skill_vs_intercept_baseline_sd=("skill_vs_intercept_baseline", "std"),
            r2_gain_over_intercept_baseline_mean=("r2_gain_over_intercept_baseline", "mean"),
            r2_gain_over_intercept_baseline_sd=("r2_gain_over_intercept_baseline", "std"),
            min_alpha_max_over_s2_max=("min_alpha_max_over_s2_max", "min"),
            max_alpha_min_over_s2_min=("max_alpha_min_over_s2_min", "max"),
        )
        .sort_values(["backbone", "protocol"])
    )
    # Grid-coverage flags (factor 100 on each side).  Interpret only together
    # with lower/upper_alpha_boundary_rate_mean.
    summary["upper_edge_near_intercept_only_limit"] = summary["min_alpha_max_over_s2_max"] >= 100
    summary["lower_edge_near_unregularized_limit"] = (
        summary["max_alpha_min_over_s2_min"] <= 0.01
    )

    wide = metrics_df.pivot_table(
        index=["backbone", "repeat"], columns="protocol", values=["mae", "r2", "r2_gain_over_intercept_baseline"]
    )
    wide.columns = [f"{metric}_{protocol}" for metric, protocol in wide.columns]
    wide = wide.reset_index()
    inflation_summary = pd.DataFrame()
    required = {"r2_image", "r2_participant", "mae_image", "mae_participant"}
    if required.issubset(wide.columns):
        wide["r2_inflation_image_minus_participant"] = (
            wide["r2_image"] - wide["r2_participant"]
        )
        wide["mae_optimism_participant_minus_image"] = (
            wide["mae_participant"] - wide["mae_image"]
        )
        wide["baseline_relative_protocol_difference"] = (
            wide["r2_gain_over_intercept_baseline_image"] - wide["r2_gain_over_intercept_baseline_participant"]
        )
        inflation_summary = (
            wide.groupby("backbone", as_index=False)
            .agg(
                r2_inflation_mean=("r2_inflation_image_minus_participant", "mean"),
                r2_inflation_sd=("r2_inflation_image_minus_participant", "std"),
                r2_inflation_q025=(
                    "r2_inflation_image_minus_participant",
                    lambda s: float(np.quantile(s, 0.025)),
                ),
                r2_inflation_q975=(
                    "r2_inflation_image_minus_participant",
                    lambda s: float(np.quantile(s, 0.975)),
                ),
                baseline_relative_protocol_difference_mean=(
                    "baseline_relative_protocol_difference", "mean"
                ),
                baseline_relative_protocol_difference_q025=(
                    "baseline_relative_protocol_difference",
                    lambda s: float(np.quantile(s, 0.025)),
                ),
                baseline_relative_protocol_difference_q975=(
                    "baseline_relative_protocol_difference",
                    lambda s: float(np.quantile(s, 0.975)),
                ),
                mae_optimism_mean=("mae_optimism_participant_minus_image", "mean"),
                mae_optimism_sd=("mae_optimism_participant_minus_image", "std"),
                mae_optimism_q025=(
                    "mae_optimism_participant_minus_image",
                    lambda s: float(np.quantile(s, 0.025)),
                ),
                mae_optimism_q975=(
                    "mae_optimism_participant_minus_image",
                    lambda s: float(np.quantile(s, 0.975)),
                ),
            )
        )
    return summary, wide, inflation_summary


def summarize_permutations(
    permutation_df: pd.DataFrame,
    observed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if permutation_df.empty:
        return permutation_df.copy(), permutation_df.copy()

    # Each null statistic is the mean over the exact same repeated split seeds
    # used for the observed statistic.  Individual repeat rows remain available
    # in label_permutation_metrics.csv for auditing.
    null_statistics = (
        permutation_df.groupby(
            ["backbone", "protocol", "permutation"], as_index=False
        )
        .agg(
            n_repeats=("repeat", "nunique"),
            r2=("r2", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            pearson=("pearson", "mean"),
            pearson_valid_repeats=("pearson", "count"),
            pearson_nan_repeats=("pearson", lambda s: int(s.isna().sum())),
            spearman=("spearman", "mean"),
            spearman_valid_repeats=("spearman", "count"),
            spearman_nan_repeats=("spearman", lambda s: int(s.isna().sum())),
            ccc=("ccc", "mean"),
            ccc_valid_repeats=("ccc", "count"),
            ccc_nan_repeats=("ccc", lambda s: int(s.isna().sum())),
        )
    )
    observed_statistics = (
        observed_df.groupby(["backbone", "protocol"], as_index=False)
        .agg(observed_r2=("r2", "mean"), observed_mae=("mae", "mean"))
    )

    summary = (
        null_statistics.groupby(["backbone", "protocol"], as_index=False)
        .agg(
            n_permutations=("permutation", "nunique"),
            shuffled_r2_mean=("r2", "mean"),
            shuffled_r2_median=("r2", "median"),
            shuffled_r2_q025=("r2", lambda s: float(np.quantile(s, 0.025))),
            shuffled_r2_q975=("r2", lambda s: float(np.quantile(s, 0.975))),
            proportion_r2_above_zero=("r2", lambda s: float(np.mean(s > 0))),
            shuffled_mae_mean=("mae", "mean"),
        )
        .sort_values(["backbone", "protocol"])
    )
    summary = summary.merge(
        observed_statistics, on=["backbone", "protocol"], how="left"
    )
    nan_diagnostics = (
        permutation_df.groupby(["backbone", "protocol"], as_index=False)
        .agg(
            pearson_valid_n=("pearson", "count"),
            pearson_nan_n=("pearson", lambda s: int(s.isna().sum())),
            spearman_valid_n=("spearman", "count"),
            spearman_nan_n=("spearman", lambda s: int(s.isna().sum())),
            ccc_valid_n=("ccc", "count"),
            ccc_nan_n=("ccc", lambda s: int(s.isna().sum())),
        )
    )
    summary = summary.merge(
        nan_diagnostics, on=["backbone", "protocol"], how="left"
    )

    p_values: list[dict[str, object]] = []
    for (backbone, protocol), group in null_statistics.groupby(
        ["backbone", "protocol"]
    ):
        observed_match = observed_statistics[
            (observed_statistics["backbone"] == backbone)
            & (observed_statistics["protocol"] == protocol)
        ]
        observed_r2 = float(observed_match["observed_r2"].iloc[0])
        null_r2 = group["r2"].to_numpy(dtype=float)
        empirical_p = (1 + int(np.sum(null_r2 >= observed_r2))) / (len(null_r2) + 1)
        p_values.append(
            {
                "backbone": backbone,
                "protocol": protocol,
                "empirical_p_r2": empirical_p,
            }
        )
    summary = summary.merge(
        pd.DataFrame(p_values), on=["backbone", "protocol"], how="left"
    )
    return summary, null_statistics


def run_synthetic_self_test() -> None:
    print("=" * 72)
    print("SYNTHETIC SELF-TEST")
    print("=" * 72)
    rng = np.random.RandomState(7)
    n_participants = 20
    images_per_participant = 18
    embedding_dim = 48

    participant_signature = rng.normal(size=(n_participants, embedding_dim))
    participant_labels = rng.uniform(18, 48, size=n_participants)
    X = np.vstack(
        [
            participant_signature[pid]
            + rng.normal(scale=0.08, size=(images_per_participant, embedding_dim))
            for pid in range(n_participants)
        ]
    )
    groups = np.repeat(np.arange(n_participants).astype(str), images_per_participant)
    y = np.repeat(participant_labels, images_per_participant)
    alpha_grid = np.logspace(-3, 6, 10).tolist()

    # Verify the accelerated path against independent sklearn fits.
    train_idx = np.arange(0, len(X), 2)
    test_idx = np.arange(1, len(X), 2)
    path_pred = ridge_path_predict(X[train_idx], y[train_idx], X[test_idx], alpha_grid)
    sklearn_pred = np.vstack(
        [fit_ridge_predict(X[train_idx], y[train_idx], X[test_idx], alpha) for alpha in alpha_grid]
    )
    max_abs_difference = float(np.max(np.abs(path_pred - sklearn_pred)))
    print(f"SVD path vs sklearn max |difference|: {max_abs_difference:.3e}")
    if max_abs_difference >= 1e-10:
        raise AssertionError(
            "Accelerated Ridge path does not match sklearn Ridge closely enough."
        )

    image_metrics, _, image_diag = cross_validated_probe(
        X, y, groups, "image", 5, 3, alpha_grid, seed=0
    )
    participant_metrics_result, _, participant_diag = cross_validated_probe(
        X, y, groups, "participant", 5, 3, alpha_grid, seed=0
    )
    print(
        f"image split:       R2={image_metrics['r2']:.3f}, "
        f"MAE={image_metrics['mae']:.3f}, "
        f"overlap={image_diag['mean_test_participant_overlap']:.3f}"
    )
    print(
        f"participant split: R2={participant_metrics_result['r2']:.3f}, "
        f"MAE={participant_metrics_result['mae']:.3f}, "
        f"overlap={participant_diag['mean_test_participant_overlap']:.3f}"
    )
    if image_diag["mean_test_participant_overlap"] <= 0.99:
        raise AssertionError("Image-level self-test did not create participant overlap.")
    if participant_diag["mean_test_participant_overlap"] != 0.0:
        raise AssertionError("Participant-level self-test leaked participant IDs.")
    if image_metrics["r2"] <= participant_metrics_result["r2"]:
        raise AssertionError(
            "Synthetic leakage check failed: image-level R2 should exceed grouped R2."
        )
    print("SELF-TEST PASSED\n")


def run_integration_self_test() -> None:
    """Create a fake NPZ/metadata pair and exercise the complete CLI pipeline."""
    print("=" * 72)
    print("FAKE-CACHE INTEGRATION SELF-TEST")
    print("=" * 72)
    with tempfile.TemporaryDirectory(prefix="cars-leakage-test-") as temporary:
        root = Path(temporary)
        cache_dir = root / "cache"
        output_dir = root / "outputs"
        cache_dir.mkdir()

        rng = np.random.RandomState(21)
        n_participants, images_per_participant, dimension = 12, 8, 16
        signatures = rng.normal(size=(n_participants, dimension))
        X = np.vstack(
            [
                signatures[pid]
                + rng.normal(scale=0.12, size=(images_per_participant, dimension))
                for pid in range(n_participants)
            ]
        )
        paths = np.asarray(
            [
                f"TS{image + 1:03d}_{pid + 1}.png"
                for pid in range(n_participants)
                for image in range(images_per_participant)
            ],
            dtype=object,
        )
        np.savez(cache_dir / DEFAULT_BACKBONES["EVA02-B/16"], X=X, paths=paths)
        metadata_path = root / "Metadata_Participants.csv"
        pd.DataFrame(
            {
                "Participant": np.arange(1, n_participants + 1),
                "CARS": rng.uniform(18, 48, n_participants),
                "Class": ["TS"] * n_participants,
            }
        ).to_csv(metadata_path, index=False)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--skip-self-test",
            "--cache-dir",
            str(cache_dir),
            "--metadata-csv",
            str(metadata_path),
            "--output-dir",
            str(output_dir),
            "--backbones",
            "EVA02-B/16",
            "--expected-participants",
            str(n_participants),
            "--n-repeats",
            "2",
            "--n-label-permutations",
            "2",
            "--inner-splits",
            "3",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(
                "Integration subprocess failed:\n"
                + completed.stdout
                + "\nSTDERR:\n"
                + completed.stderr
            )

        expected_files = {
            "protocol_metrics.csv",
            "participant_predictions.csv",
            "label_permutation_metrics.csv",
            "summary_by_backbone.csv",
            "paired_protocol_differences.csv",
            "inflation_summary.csv",
            "permutation_summary.csv",
            "permutation_null_statistics.csv",
            "participant_image_counts.csv",
            "input_exclusions.csv",
            "run_config.json",
        }
        observed_files = {path.name for path in output_dir.iterdir() if path.is_file()}
        missing = expected_files.difference(observed_files)
        if missing:
            raise AssertionError(f"Integration test outputs missing: {sorted(missing)}")

        protocol = pd.read_csv(output_dir / "protocol_metrics.csv")
        permutation = pd.read_csv(output_dir / "label_permutation_metrics.csv")
        permutation_summary = pd.read_csv(output_dir / "permutation_summary.csv")
        counts = pd.read_csv(output_dir / "participant_image_counts.csv")
        required_metric_columns = {
            "pearson",
            "spearman",
            "ccc",
            "lower_alpha_boundary_rate",
            "upper_alpha_boundary_rate",
            "any_alpha_boundary_rate",
            "baseline_r2",
            "r2_gain_over_intercept_baseline",
            "skill_vs_intercept_baseline",
            "min_alpha_max_over_s2_max",
            "max_alpha_min_over_s2_min",
        }
        if not required_metric_columns.issubset(protocol.columns):
            raise AssertionError("Integration metrics/diagnostics are incomplete.")
        if set(protocol["seed"]) != set(permutation["seed"]):
            raise AssertionError("Observed and permutation split seeds do not match.")
        if not {"empirical_p_r2", "observed_r2"}.issubset(
            permutation_summary.columns
        ):
            raise AssertionError("Permutation p-value output is incomplete.")
        if len(counts) != n_participants or not counts["n_images"].eq(
            images_per_participant
        ).all():
            raise AssertionError("Participant image-count audit is incorrect.")
    print("INTEGRATION SELF-TEST PASSED\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/cache_v3_unambiguous")
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("Metadata/Metadata/Metadata_Participants.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/leakage_stress_test"))
    parser.add_argument("--pid-column", type=str, default=None)
    parser.add_argument("--cars-column", type=str, default=None)
    parser.add_argument("--class-column", type=str, default=None)
    parser.add_argument(
        "--asd-label",
        type=str,
        default="TS",
        help="Value in the metadata class column identifying ASD participants",
    )
    parser.add_argument(
        "--pid-regex",
        type=str,
        default=r"_(\d+)\.[^.]+$",
        help="Regex with participant ID in capture group 1",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument(
        "--expected-participants",
        type=int,
        default=26,
        help="Fail if a backbone has a different participant count; use 0 to disable",
    )
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-label-permutations", type=int, default=20)
    parser.add_argument(
        "--permutation-alpha",
        type=float,
        default=1.0,
        help="Exploratory fixed alpha used only with --fixed-permutation-alpha",
    )
    parser.add_argument(
        "--fixed-permutation-alpha",
        action="store_true",
        help="Skip nested tuning in permutations (exploratory shortcut; not recommended for the paper)",
    )
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=np.logspace(-3, 6, 10).tolist(),
    )
    parser.add_argument(
        "--backbones",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset using display names, e.g. 'EVA02-B/16' 'ResNet-50'",
    )
    parser.add_argument("--base-seed", type=int, default=20260924)
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--integration-self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.integration_self_test_only:
        run_synthetic_self_test()
        run_integration_self_test()
        return
    if not args.skip_self_test or args.self_test_only:
        run_synthetic_self_test()
    if args.self_test_only:
        return

    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1")
    if args.n_label_permutations < 0:
        raise ValueError("--n-label-permutations cannot be negative")
    if args.expected_participants < 0:
        raise ValueError("--expected-participants cannot be negative")
    if any(alpha <= 0 for alpha in args.alpha_grid):
        raise ValueError("All Ridge alphas must be positive")
    if args.permutation_alpha <= 0:
        raise ValueError("--permutation-alpha must be positive")

    selected_backbones = DEFAULT_BACKBONES
    if args.backbones:
        unknown = sorted(set(args.backbones).difference(DEFAULT_BACKBONES))
        if unknown:
            raise ValueError(
                f"Unknown backbones: {unknown}. Available: {list(DEFAULT_BACKBONES)}"
            )
        selected_backbones = {
            name: DEFAULT_BACKBONES[name] for name in args.backbones
        }

    pid_pattern = re.compile(args.pid_regex)
    cars_lookup = load_cars_lookup(
        args.metadata_csv,
        args.pid_column,
        args.cars_column,
        args.class_column,
        args.asd_label,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    permutation_rows: list[dict[str, object]] = []
    participant_count_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    completed_caches: dict[str, str] = {}
    start_time = time.time()

    for backbone_index, (backbone, filename) in enumerate(selected_backbones.items()):
        cache_path = args.cache_dir / filename
        if not cache_path.exists():
            print(f"[skip] {backbone}: missing {cache_path}")
            continue

        print("\n" + "=" * 72)
        print(f"BACKBONE: {backbone}")
        print("=" * 72)
        X, y, groups, _, participant_counts, exclusions = load_image_embeddings(
            cache_path, cars_lookup, pid_pattern
        )
        n_participants = len(np.unique(groups))
        participant_counts.insert(0, "backbone", backbone)
        participant_count_frames.append(participant_counts)
        if not exclusions.empty:
            exclusions.insert(0, "backbone", backbone)
            exclusion_frames.append(exclusions)
        if (
            args.expected_participants > 0
            and n_participants != args.expected_participants
        ):
            pd.concat(participant_count_frames, ignore_index=True).to_csv(
                args.output_dir / "participant_image_counts.csv", index=False
            )
            if exclusion_frames:
                pd.concat(exclusion_frames, ignore_index=True).to_csv(
                    args.output_dir / "input_exclusions.csv", index=False
                )
            raise ValueError(
                f"{backbone} contains {n_participants} participants; "
                f"expected {args.expected_participants}. Check --pid-regex, metadata, "
                "and participant_image_counts.csv. Use --expected-participants 0 "
                "only after manually verifying the mapping."
            )
        if args.outer_splits > n_participants:
            raise ValueError(
                f"outer_splits={args.outer_splits} exceeds n={n_participants} for {backbone}"
            )
        completed_caches[backbone] = str(cache_path)

        repeat_seeds = [
            args.base_seed + backbone_index * 100_000 + repeat
            for repeat in range(args.n_repeats)
        ]
        for repeat, seed in enumerate(repeat_seeds):
            for protocol in ("image", "participant"):
                metrics, participant_pred, diagnostics = cross_validated_probe(
                    X,
                    y,
                    groups,
                    protocol,
                    args.outer_splits,
                    args.inner_splits,
                    args.alpha_grid,
                    seed,
                )
                protocol_rows.append(
                    {
                        "backbone": backbone,
                        "repeat": repeat,
                        "seed": seed,
                        "protocol": protocol,
                        "n_images": len(X),
                        "n_participants": n_participants,
                        **metrics,
                        **diagnostics,
                    }
                )
                participant_pred.insert(0, "protocol", protocol)
                participant_pred.insert(0, "seed", seed)
                participant_pred.insert(0, "repeat", repeat)
                participant_pred.insert(0, "backbone", backbone)
                prediction_frames.append(participant_pred)
                print(
                    f"[real] repeat={repeat:02d} {protocol:11s} "
                    f"MAE={metrics['mae']:.3f} R2={metrics['r2']:.3f} "
                    f"overlap={diagnostics['mean_test_participant_overlap']:.3f}"
                )

        rng = np.random.RandomState(args.base_seed + backbone_index * 100_000 + 50_000)
        for permutation in range(args.n_label_permutations):
            y_permuted = participant_label_permutation(y, groups, rng)
            for repeat, seed in enumerate(repeat_seeds):
                for protocol in ("image", "participant"):
                    metrics, _, diagnostics = cross_validated_probe(
                        X,
                        y_permuted,
                        groups,
                        protocol,
                        args.outer_splits,
                        args.inner_splits,
                        args.alpha_grid,
                        seed,
                        fixed_alpha=(
                            args.permutation_alpha
                            if args.fixed_permutation_alpha
                            else None
                        ),
                    )
                    permutation_rows.append(
                        {
                            "backbone": backbone,
                            "permutation": permutation,
                            "repeat": repeat,
                            "seed": seed,
                            "protocol": protocol,
                            "n_images": len(X),
                            "n_participants": n_participants,
                            **metrics,
                            **diagnostics,
                        }
                    )
            if (permutation + 1) % max(1, min(10, args.n_label_permutations)) == 0:
                print(
                    f"[permutation] {permutation + 1}/{args.n_label_permutations} complete"
                )

    if not protocol_rows:
        raise FileNotFoundError(
            "No backbone caches were processed. Check --cache-dir and filenames."
        )

    protocol_df = pd.DataFrame(protocol_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    permutation_df = pd.DataFrame(permutation_rows)
    participant_counts_df = pd.concat(participant_count_frames, ignore_index=True)
    exclusions_df = (
        pd.concat(exclusion_frames, ignore_index=True)
        if exclusion_frames
        else pd.DataFrame(columns=["backbone", "path", "participant_id", "reason"])
    )
    summary_df, paired_differences_df, inflation_summary_df = summarize_real_metrics(
        protocol_df
    )
    permutation_summary_df, permutation_null_df = summarize_permutations(
        permutation_df, protocol_df
    )

    protocol_df.to_csv(args.output_dir / "protocol_metrics.csv", index=False)
    predictions_df.to_csv(args.output_dir / "participant_predictions.csv", index=False)
    permutation_df.to_csv(args.output_dir / "label_permutation_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_by_backbone.csv", index=False)
    paired_differences_df.to_csv(
        args.output_dir / "paired_protocol_differences.csv", index=False
    )
    inflation_summary_df.to_csv(
        args.output_dir / "inflation_summary.csv", index=False
    )
    permutation_summary_df.to_csv(
        args.output_dir / "permutation_summary.csv", index=False
    )
    permutation_null_df.to_csv(
        args.output_dir / "permutation_null_statistics.csv", index=False
    )
    participant_counts_df.to_csv(
        args.output_dir / "participant_image_counts.csv", index=False
    )
    exclusions_df.to_csv(args.output_dir / "input_exclusions.csv", index=False)

    config = {
        **vars(args),
        "cache_dir": str(args.cache_dir),
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
        "completed_caches": completed_caches,
        "elapsed_seconds": time.time() - start_time,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    if not permutation_summary_df.empty:
        print("\nSHUFFLED-LABEL SUMMARY")
        print(permutation_summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")
    print(f"Elapsed: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
