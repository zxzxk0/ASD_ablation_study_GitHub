# -*- coding: utf-8 -*-
r"""
Identity-decodability and metadata-leakage follow-up to the Ridge stress test.

Motivation
----------
The Ridge stress test (``run_participant_leakage_stress_test.py``) found that an
image-level split did not inflate CARS performance much for frozen scanpath
embeddings.  This script tests the two remaining explanations for why a
participant-overlapping evaluation could still look good:

A. Identity decodability (descriptive, no model fitting)
   Are images of the same child closer to each other than to other children's
   images in embedding space?  If not, an image-level split has little identity
   signal to exploit, which explains the Ridge result.

B. Memorizing predictors under both split protocols
   * k-NN regression on image embeddings (k = 1, 5).  k-NN is the canonical
     "memorizing" head: if identity is decodable, an image-level split lets it
     copy the CARS label of the same child's other images.
   * Metadata only (age + sex), replicated to every image of a participant.
     Evaluated with k-NN (k = 1, 5) and with nested Ridge.  Age and sex are
     replicated at the image level to match their use in the image-based
     pipeline, so k counts nearest replicated *image records*, not nearest
     participants (under a participant split several of the k neighbours can
     be records of one training participant).  A 2-D linear model
     cannot look up identity, whereas 1-NN can whenever (age, sex) is nearly
     unique per child.  This isolates leakage that needs no image at all.

Everything reuses the stress-test module (same fold generator, seeds, participant
aggregation, metrics, intercept baseline and label-permutation scheme), so the
numbers are directly comparable with ``summary_by_backbone.csv``.

This script must sit in the same folder as
``run_participant_leakage_stress_test.py``.

Quick test
----------
    python run_identity_and_metadata_leakage.py --self-test-only

Final run
---------
    python run_identity_and_metadata_leakage.py ^
      --cache-dir outputs\cache_v3_unambiguous ^
      --metadata-csv Metadata\Metadata\Metadata_Participants.csv ^
      --n-repeats 10 --n-label-permutations 100

If age/sex columns are not auto-detected, pass --age-column / --sex-column.

Outputs
-------
``identity_decodability.csv``
    Per backbone and similarity space: 1-NN same-participant rate, exact chance
    rate, top-5 same-participant fraction, within/between similarity, pairwise
    AUROC (within-pair more similar than between-pair) and permutation p.
``metadata_audit.csv``
    Per-participant age, sex, CARS and image count used for the metadata arm.
``metadata_uniqueness.csv``
    How many participants share an identical (age, sex) profile, and the
    Spearman correlation between age and CARS.
``memorizer_protocol_metrics.csv``
    One row per feature set, model, repeat and protocol (all metrics + baseline).
``memorizer_participant_predictions.csv``
    Participant-level out-of-fold predictions for real labels.
``memorizer_summary.csv``
    Mean/SD per feature set, model and protocol.
``memorizer_inflation_summary.csv``
    Raw and baseline-relative image-minus-participant differences with empirical
    2.5/97.5 percentiles across repeated partitions (not confidence intervals).
``memorizer_permutation_metrics.csv`` / ``memorizer_permutation_summary.csv``
    Shuffled-label null (same split seeds, repeat-averaged) and empirical p.
``run_config.json``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import warnings

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_participant_leakage_stress_test as st  # noqa: E402

# Memorizing predictors (and metadata-only models) often emit near-constant
# participant predictions in single folds; Pearson/Spearman are then already
# recorded as NaN or unstable and counted in the *_nan_n columns.  Silence only
# this specific SciPy warning so real warnings stay visible.
for _name in ("NearConstantInputWarning", "ConstantInputWarning"):
    _cls = getattr(_scipy_stats, _name, None)
    if _cls is not None:
        warnings.filterwarnings("ignore", category=_cls)


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------

SEX_SYNONYMS = {
    "M": "M", "MALE": "M", "H": "M", "HOMME": "M", "MASCULIN": "M", "BOY": "M", "GARCON": "M", "GARÇON": "M",
    "F": "F", "FEMALE": "F", "FEMME": "F", "FEMININ": "F", "FÉMININ": "F", "GIRL": "F", "FILLE": "F",
}
MISSING_TOKENS = {"", "NAN", "NA", "N/A", "NONE", "NULL", "?", "-"}


def normalize_sex(value: object) -> float | str:
    """Map sex/gender tokens to 'M'/'F'; missing -> NaN; unknown token -> error."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    token = str(value).strip().upper()
    if token in MISSING_TOKENS:
        return np.nan
    if token not in SEX_SYNONYMS:
        raise ValueError(
            f"Unrecognized sex/gender value {value!r}; extend SEX_SYNONYMS explicitly."
        )
    return SEX_SYNONYMS[token]


def load_participant_table(
    metadata_csv: Path,
    pid_column: str | None,
    cars_column: str | None,
    class_column: str | None,
    age_column: str | None,
    sex_column: str | None,
    asd_label: str,
) -> pd.DataFrame:
    """Return one row per ASD participant: pid, cars, age, sex (raw), sex_code."""
    meta = pd.read_csv(metadata_csv)
    meta.columns = [str(c).strip() for c in meta.columns]

    pid_column = pid_column or st.find_column(
        meta.columns, ["ParticipantID", "Participant", "SubjectID", "Subject", "ID"]
    )
    if cars_column is None:
        cars_column = next((c for c in meta.columns if "cars" in c.lower()), None)
    class_column = class_column or st.find_column(
        meta.columns, ["Class", "Diagnosis", "Group", "Dx"]
    )
    age_column = age_column or st.find_column(
        meta.columns, ["Age", "AgeYears", "Age (years)", "AgeInYears", "age_years"]
    )
    sex_column = sex_column or st.find_column(
        meta.columns, ["Gender", "Sex", "Gender (M/F)", "Sexe"]
    )
    missing = {
        name: col
        for name, col in {
            "pid": pid_column, "cars": cars_column,
            "age": age_column, "sex": sex_column,
        }.items()
        if col is None
    }
    if missing:
        raise ValueError(
            f"Could not detect columns {sorted(missing)}. Available: "
            f"{list(meta.columns)}. Pass them explicitly."
        )

    work = meta.copy()
    if class_column is not None:
        observed = work[class_column].astype(str).str.strip().str.upper()
        work = work[observed == str(asd_label).strip().upper()]
    table = pd.DataFrame(
        {
            "participant_id": work[pid_column].map(st.normalize_pid),
            "cars": pd.to_numeric(work[cars_column], errors="coerce"),
            "age": pd.to_numeric(work[age_column], errors="coerce"),
            "sex": work[sex_column].map(normalize_sex),
        }
    ).dropna(subset=["cars"])
    if table["participant_id"].duplicated().any():
        raise ValueError("Repeated participant IDs in metadata after filtering.")
    # Binary sex code; missing values stay NaN (checked later for the
    # participants actually used) and unknown tokens raise in normalize_sex.
    levels = sorted(table["sex"].dropna().unique().tolist())
    if len(levels) > 2:
        raise ValueError(f"Expected at most two sex levels, found {levels}.")
    table["sex_code"] = table["sex"].map({lvl: i for i, lvl in enumerate(levels)}).astype(float)
    print(
        f"[metadata] pid={pid_column!r} CARS={cars_column!r} age={age_column!r} "
        f"sex={sex_column!r} levels={levels}; {len(table)} ASD rows"
    )
    return table.set_index("participant_id")


# ----------------------------------------------------------------------------
# A. Identity decodability
# ----------------------------------------------------------------------------

def chance_same_participant_rate(groups: np.ndarray) -> float:
    """P(a random *other* image belongs to the same participant), image-averaged."""
    _, counts = np.unique(groups, return_counts=True)
    n = counts.sum()
    return float(np.sum(counts * (counts - 1)) / (n * (n - 1)))


def similarity_matrix(X: np.ndarray, space: str) -> np.ndarray:
    if space == "raw_cosine":
        Z = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
        return Z @ Z.T
    if space == "standardized_neg_euclidean":
        Z = StandardScaler().fit_transform(X)
        sq = np.sum(Z**2, axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
        return -np.sqrt(d2)
    raise ValueError(space)


def identity_decodability(
    X: np.ndarray, groups: np.ndarray, space: str, n_perm: int, seed: int
) -> dict[str, float]:
    S = similarity_matrix(X, space)
    n = len(groups)
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    nn1 = order[:, 0]
    top5 = order[:, :5]
    codes = np.unique(groups, return_inverse=True)[1]

    nn1_rate = float(np.mean(codes[nn1] == codes))
    top5_frac = float(np.mean(codes[top5] == codes[:, None]))

    # Null: randomly relabel images while preserving participant image counts.
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(codes)
        null[b] = np.mean(perm[nn1] == perm)
    p_nn1 = (1 + int(np.sum(null >= nn1_rate))) / (n_perm + 1)

    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    same = codes[iu[0]] == codes[iu[1]]
    within, between = sims[same], sims[~same]
    # AUROC = P(within-pair similarity > between-pair similarity).
    u = mannwhitneyu(within, between, alternative="two-sided").statistic
    auroc = float(u / (len(within) * len(between)))
    return {
        "space": space,
        "n_images": n,
        "n_participants": int(len(np.unique(codes))),
        "nn1_same_participant_rate": nn1_rate,
        "chance_rate": chance_same_participant_rate(groups),
        "nn1_rate_over_chance": nn1_rate / chance_same_participant_rate(groups),
        "top5_same_participant_fraction": top5_frac,
        "nn1_null_mean": float(null.mean()),
        "nn1_null_q975": float(np.quantile(null, 0.975)),
        "nn1_permutation_p": p_nn1,
        "within_similarity_mean": float(within.mean()),
        "between_similarity_mean": float(between.mean()),
        "within_minus_between": float(within.mean() - between.mean()),
        "within_vs_between_auroc": auroc,
        "n_within_pairs": int(len(within)),
        "n_between_pairs": int(len(between)),
    }


def adjust_pvalues(p: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni and Benjamini-Hochberg adjusted p-values for one test family."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    bonf = np.minimum(p * m, 1.0)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    bh_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    bh = np.empty(m)
    bh[order] = np.minimum(bh_sorted, 1.0)
    return bonf, bh


# ----------------------------------------------------------------------------
# B. Memorizing predictors under both protocols
# ----------------------------------------------------------------------------

def knn_cross_validated(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    protocol: str,
    outer_splits: int,
    seed: int,
    k: int,
    metric: str,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, float]]:
    """k-NN regression with the stress test's folds, aggregation and baseline.

    k is prespecified (no tuning), so there is no inner loop to leak through.
    ``metric='cosine'`` uses raw embeddings; ``metric='euclidean'`` standardizes
    features on the outer-training images only.
    """
    folds = st.make_folds(protocol, groups, outer_splits, seed)
    pred = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    overlap_rates = []
    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx
        overlap = np.intersect1d(np.unique(groups[tr]), np.unique(groups[te]))
        overlap_rates.append(len(overlap) / max(1, len(np.unique(groups[te]))))
        Xtr, Xte = X[tr], X[te]
        if metric == "euclidean":
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        model = KNeighborsRegressor(
            n_neighbors=min(k, len(tr)), metric=metric, algorithm="brute"
        )
        model.fit(Xtr, y[tr])
        pred[te] = np.clip(model.predict(Xte), st.CARS_MIN, st.CARS_MAX)
        _, ytr_participant, _, _ = st.aggregate_prediction_arrays(
            y[tr], y[tr], groups[tr]
        )
        base[te] = float(np.mean(ytr_participant))
    if np.isnan(pred).any():
        raise RuntimeError("Missing out-of-fold predictions.")
    participant_pred = st.aggregate_predictions(y, pred, groups)
    metrics = st.participant_metrics(participant_pred)
    _, bt, bp, _ = st.aggregate_prediction_arrays(y, base, groups)
    bm = st.participant_metrics_from_arrays(bt, bp)
    sse_m = float(np.sum((participant_pred["y_true"] - participant_pred["y_pred"]) ** 2))
    sse_b = float(np.sum((bt - bp) ** 2))
    metrics.update(
        baseline_mae=bm["mae"],
        baseline_r2=bm["r2"],
        r2_gain_over_intercept_baseline=metrics["r2"] - bm["r2"],
        skill_vs_intercept_baseline=np.nan if sse_b == 0 else 1 - sse_m / sse_b,
    )
    return metrics, participant_pred, {
        "mean_test_participant_overlap": float(np.mean(overlap_rates))
    }


def ridge_cross_validated(
    X, y, groups, protocol, outer_splits, inner_splits, alpha_grid, seed
):
    """Nested Ridge from the stress-test module (same baseline and diagnostics)."""
    return st.cross_validated_probe(
        X, y, groups, protocol, outer_splits, inner_splits, alpha_grid, seed
    )


def run_config(cfg: dict, X, y, groups, protocol, seed, args):
    if cfg["model"].startswith("ridge"):
        return ridge_cross_validated(
            X, y, groups, protocol, args.outer_splits, args.inner_splits,
            args.alpha_grid, seed,
        )
    return knn_cross_validated(
        X, y, groups, protocol, args.outer_splits, seed, cfg["k"], cfg["metric"]
    )


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------

KEYS = ["feature_set", "model"]


def q(p):
    return lambda s: float(np.quantile(s, p))


def summarize(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics_df.groupby(KEYS + ["protocol"], as_index=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
            r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
            pearson_mean=("pearson", "mean"), pearson_nan_n=("pearson", lambda s: int(s.isna().sum())),
            spearman_mean=("spearman", "mean"), spearman_nan_n=("spearman", lambda s: int(s.isna().sum())),
            ccc_mean=("ccc", "mean"), ccc_sd=("ccc", "std"),
            baseline_r2_mean=("baseline_r2", "mean"),
            baseline_mae_mean=("baseline_mae", "mean"),
            r2_gain_over_intercept_baseline_mean=("r2_gain_over_intercept_baseline", "mean"),
            skill_vs_intercept_baseline_mean=("skill_vs_intercept_baseline", "mean"),
            overlap_mean=("mean_test_participant_overlap", "mean"),
        )
    )
    wide = metrics_df.pivot_table(
        index=KEYS + ["repeat"], columns="protocol",
        values=["r2", "mae", "r2_gain_over_intercept_baseline"],
    )
    wide.columns = [f"{m}_{p}" for m, p in wide.columns]
    wide = wide.reset_index()
    wide["r2_difference_image_minus_participant"] = wide["r2_image"] - wide["r2_participant"]
    wide["baseline_relative_protocol_difference"] = (
        wide["r2_gain_over_intercept_baseline_image"]
        - wide["r2_gain_over_intercept_baseline_participant"]
    )
    wide["mae_optimism_participant_minus_image"] = wide["mae_participant"] - wide["mae_image"]
    inflation = wide.groupby(KEYS, as_index=False).agg(
        r2_image_mean=("r2_image", "mean"),
        r2_participant_mean=("r2_participant", "mean"),
        r2_difference_mean=("r2_difference_image_minus_participant", "mean"),
        r2_difference_q025=("r2_difference_image_minus_participant", q(0.025)),
        r2_difference_q975=("r2_difference_image_minus_participant", q(0.975)),
        baseline_relative_difference_mean=("baseline_relative_protocol_difference", "mean"),
        baseline_relative_difference_q025=("baseline_relative_protocol_difference", q(0.025)),
        baseline_relative_difference_q975=("baseline_relative_protocol_difference", q(0.975)),
        mae_optimism_mean=("mae_optimism_participant_minus_image", "mean"),
        mae_optimism_q025=("mae_optimism_participant_minus_image", q(0.025)),
        mae_optimism_q975=("mae_optimism_participant_minus_image", q(0.975)),
    )
    return summary, wide, inflation


def summarize_permutations(perm_df: pd.DataFrame, observed_df: pd.DataFrame) -> pd.DataFrame:
    if perm_df.empty:
        return perm_df
    null = perm_df.groupby(KEYS + ["protocol", "permutation"], as_index=False).agg(
        r2=("r2", "mean"), mae=("mae", "mean")
    )
    obs = observed_df.groupby(KEYS + ["protocol"], as_index=False).agg(
        observed_r2=("r2", "mean"), observed_mae=("mae", "mean")
    )
    rows = []
    for key, g in null.groupby(KEYS + ["protocol"]):
        o = obs.set_index(KEYS + ["protocol"]).loc[key]
        r2 = g["r2"].to_numpy()
        rows.append(
            dict(
                zip(KEYS + ["protocol"], key),
                n_permutations=len(r2),
                shuffled_r2_mean=float(r2.mean()),
                shuffled_r2_q025=float(np.quantile(r2, 0.025)),
                shuffled_r2_q975=float(np.quantile(r2, 0.975)),
                proportion_shuffled_r2_above_zero=float(np.mean(r2 > 0)),
                observed_r2=float(o["observed_r2"]),
                observed_mae=float(o["observed_mae"]),
                empirical_p_r2=(1 + int(np.sum(r2 >= o["observed_r2"]))) / (len(r2) + 1),
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def run_self_test() -> None:
    print("=" * 72 + "\nSELF-TEST\n" + "=" * 72)
    rng = np.random.RandomState(3)
    P, k, d = 20, 8, 32
    groups = np.repeat(np.arange(P).astype(str), k)
    y = np.repeat(rng.uniform(18, 45, P), k)

    # Strong identity signature -> high 1-NN rate, image-split k-NN inflation.
    X_id = np.repeat(rng.normal(size=(P, d)), k, axis=0) + 0.1 * rng.normal(size=(P * k, d))
    r = identity_decodability(X_id, groups, "raw_cosine", 200, 0)
    print(f"signature: nn1={r['nn1_same_participant_rate']:.2f} chance={r['chance_rate']:.3f} AUROC={r['within_vs_between_auroc']:.2f}")
    assert r["nn1_same_participant_rate"] > 0.9 and r["within_vs_between_auroc"] > 0.9

    # No identity signature -> near chance.
    X_no = rng.normal(size=(P * k, d))
    r0 = identity_decodability(X_no, groups, "raw_cosine", 200, 0)
    print(f"no signature: nn1={r0['nn1_same_participant_rate']:.2f} chance={r0['chance_rate']:.3f} AUROC={r0['within_vs_between_auroc']:.2f}")
    assert r0["nn1_same_participant_rate"] < 0.2 and abs(r0["within_vs_between_auroc"] - 0.5) < 0.05

    m_img, _, _ = knn_cross_validated(X_id, y, groups, "image", 5, 0, 1, "cosine")
    m_par, _, d_par = knn_cross_validated(X_id, y, groups, "participant", 5, 0, 1, "cosine")
    print(f"k-NN on signature: image R2={m_img['r2']:.2f} participant R2={m_par['r2']:.2f}")
    assert m_img["r2"] > 0.9 and m_par["r2"] < 0.3 and d_par["mean_test_participant_overlap"] == 0

    # Unique metadata -> 1-NN recovers labels only under image split.
    meta = np.column_stack([np.repeat(rng.uniform(3, 12, P), k), np.repeat(rng.randint(0, 2, P), k)])
    m_img, _, _ = knn_cross_validated(meta, y, groups, "image", 5, 0, 1, "euclidean")
    m_par, _, _ = knn_cross_validated(meta, y, groups, "participant", 5, 0, 1, "euclidean")
    print(f"metadata 1-NN: image R2={m_img['r2']:.2f} participant R2={m_par['r2']:.2f}")
    assert m_img["r2"] > 0.95 and m_par["r2"] < 0.3
    # Helpers added for the final run.
    assert normalize_sex("Male") == "M" and normalize_sex(" f ") == "F"
    assert np.isnan(normalize_sex(np.nan)) and np.isnan(normalize_sex("nan"))
    try:
        normalize_sex("X")
        raise AssertionError("unknown sex token was accepted")
    except ValueError:
        pass
    bonf, bh = adjust_pvalues(pd.Series([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(bonf, [0.04, 0.16, 0.12, 1.0])
    assert np.allclose(bh, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)
    print("SELF-TEST PASSED\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/cache_v3_unambiguous"))
    ap.add_argument("--metadata-csv", type=Path, default=Path("Metadata/Metadata/Metadata_Participants.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/identity_metadata_leakage"))
    ap.add_argument("--pid-column"); ap.add_argument("--cars-column")
    ap.add_argument("--class-column"); ap.add_argument("--age-column"); ap.add_argument("--sex-column")
    ap.add_argument("--asd-label", default="TS")
    ap.add_argument("--pid-regex", default=r"_(\d+)\.[^.]+$")
    ap.add_argument("--expected-participants", type=int, default=26)
    ap.add_argument(
        "--metadata-reference-backbone", default="EVA02-B/16",
        help="Cache whose images/participants define the metadata arm (identical image set)",
    )
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--alpha-grid", type=float, nargs="+", default=np.logspace(-3, 8, 12).tolist())
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--n-label-permutations", type=int, default=100)
    ap.add_argument("--n-identity-permutations", type=int, default=10000)
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--base-seed", type=int, default=20260924)
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_self_test or args.self_test_only:
        run_self_test()
    if args.self_test_only:
        return

    backbones = st.DEFAULT_BACKBONES
    if args.backbones:
        unknown = set(args.backbones) - set(backbones)
        if unknown:
            raise ValueError(f"Unknown backbones {sorted(unknown)}")
        backbones = {b: backbones[b] for b in args.backbones}

    table = load_participant_table(
        args.metadata_csv, args.pid_column, args.cars_column, args.class_column,
        args.age_column, args.sex_column, args.asd_label,
    )
    cars_lookup = table["cars"].to_dict()
    pid_pattern = re.compile(args.pid_regex)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load caches ------------------------------------------------------
    data = {}
    for name, fname in backbones.items():
        path = args.cache_dir / fname
        if not path.exists():
            print(f"[skip] {name}: missing {path}")
            continue
        X, y, groups, paths, counts, _ = st.load_image_embeddings(path, cars_lookup, pid_pattern)
        n_p = len(np.unique(groups))
        if args.expected_participants and n_p != args.expected_participants:
            raise ValueError(f"{name}: {n_p} participants, expected {args.expected_participants}")
        data[name] = (X, y, groups, paths)
    if not data:
        raise FileNotFoundError("No caches found.")

    # ---- verify every backbone uses the identical image set ---------------
    # Reference = first loaded cache.  Each other cache must contain exactly the
    # same image paths (hence same per-participant counts); rows are re-ordered
    # to the reference order so image-level folds index the same images.
    ref_name = next(iter(data))
    ref_paths = np.asarray([str(p) for p in data[ref_name][3]], dtype=object)
    if len(set(ref_paths)) != len(ref_paths):
        raise ValueError(f"{ref_name}: duplicated image paths in cache.")
    ref_pos = {p: i for i, p in enumerate(ref_paths)}
    image_set_rows = []
    for name in list(data):
        X, y, groups, paths = data[name]
        paths = np.asarray([str(p) for p in paths], dtype=object)
        same_set = set(paths) == set(ref_paths) and len(paths) == len(ref_paths)
        same_order = same_set and bool(np.all(paths == ref_paths))
        image_set_rows.append({
            "backbone": name, "reference": ref_name, "n_images": len(paths),
            "identical_path_set": same_set, "identical_order_before_alignment": same_order,
        })
        if not same_set:
            only_here = sorted(set(paths) - set(ref_paths))[:10]
            only_ref = sorted(set(ref_paths) - set(paths))[:10]
            pd.DataFrame(image_set_rows).to_csv(args.output_dir / "image_set_check.csv", index=False)
            raise ValueError(
                f"{name} image set differs from {ref_name}: only in {name}={only_here}, "
                f"only in {ref_name}={only_ref}"
            )
        if not same_order:
            order = np.argsort([ref_pos[p] for p in paths])
            X, y, groups, paths = X[order], y[order], groups[order], paths[order]
        assert np.all(paths == ref_paths)
        assert np.all(groups == data[ref_name][2]) and np.allclose(y, data[ref_name][1])
        data[name] = (X, y, groups, paths)
    pd.DataFrame(image_set_rows).to_csv(args.output_dir / "image_set_check.csv", index=False)
    print(f"[image-set] all {len(data)} caches share the same {len(ref_paths)} images "
          f"(reference {ref_name}); see image_set_check.csv")

    # ---- A. identity decodability ----------------------------------------
    id_rows = []
    for bi, (name, (X, y, groups, _)) in enumerate(data.items()):
        for space in ("raw_cosine", "standardized_neg_euclidean"):
            r = identity_decodability(
                X, groups, space, args.n_identity_permutations, args.base_seed + bi
            )
            id_rows.append({"backbone": name, **r})
            print(
                f"[identity] {name:16s} {space:27s} nn1={r['nn1_same_participant_rate']:.3f} "
                f"(chance {r['chance_rate']:.3f}, p={r['nn1_permutation_p']:.4f}) "
                f"AUROC={r['within_vs_between_auroc']:.3f}"
            )
    id_df = pd.DataFrame(id_rows)
    id_df["nn1_p_bonferroni"], id_df["nn1_p_bh"] = adjust_pvalues(id_df["nn1_permutation_p"])
    id_df["n_tests_in_family"] = len(id_df)
    id_df.to_csv(args.output_dir / "identity_decodability.csv", index=False)

    # ---- metadata arm on the reference image set -------------------------
    ref = args.metadata_reference_backbone
    if ref not in data:
        ref = next(iter(data))
        print(f"[metadata] reference backbone not loaded; using {ref}")
    _, y_ref, g_ref, _ = data[ref]
    age = table.loc[g_ref, "age"].to_numpy(float)
    sex = table.loc[g_ref, "sex_code"].to_numpy(float)
    if np.isnan(age).any() or np.isnan(sex).any():
        bad = sorted(set(g_ref[np.isnan(age) | np.isnan(sex)]))
        raise ValueError(f"Missing age/sex for participants {bad}")
    X_meta = np.column_stack([age, sex])

    audit = (
        table.loc[np.unique(g_ref), ["age", "sex", "sex_code", "cars"]]
        .assign(n_images=pd.Series(g_ref).value_counts())
        .reset_index()
    )
    audit.to_csv(args.output_dir / "metadata_audit.csv", index=False)
    profile = audit.groupby(["age", "sex_code"]).size()
    rho, p_rho = spearmanr(audit["age"], audit["cars"])
    pd.DataFrame([{
        "n_participants": len(audit),
        "n_unique_age_sex_profiles": int(len(profile)),
        "n_participants_sharing_a_profile": int(profile[profile > 1].sum()),
        "max_participants_per_profile": int(profile.max()),
        "n_unique_ages": int(audit["age"].nunique()),
        "age_cars_spearman_rho": float(rho),
        "age_cars_spearman_p": float(p_rho),
        "reference_backbone": ref,
    }]).to_csv(args.output_dir / "metadata_uniqueness.csv", index=False)
    print(f"[metadata] {len(profile)} unique (age, sex) profiles among {len(audit)} participants; "
          f"age-CARS rho={rho:.3f} (p={p_rho:.3f})")

    # ---- B. memorizer configs --------------------------------------------
    configs = []
    for name, (X, y, groups, _) in data.items():
        for k in args.k_values:
            configs.append(dict(feature_set=name, model=f"knn{k}_cosine", k=k,
                                metric="cosine", X=X, y=y, groups=groups))
    for k in args.k_values:
        configs.append(dict(feature_set="metadata_age_sex", model=f"knn{k}_euclidean", k=k,
                            metric="euclidean", X=X_meta, y=y_ref, groups=g_ref))
    # NOTE: metadata rows are image-level replicates (see module docstring).
    configs.append(dict(feature_set="metadata_age_sex", model="ridge_nested", k=None,
                        metric=None, X=X_meta, y=y_ref, groups=g_ref))

    rows, preds, perm_rows = [], [], []
    start = time.time()
    for ci, cfg in enumerate(configs):
        X, y, groups = cfg["X"], cfg["y"], cfg["groups"]
        # Shared across every config: identical partitions (and, below, identical
        # shuffled labels), so models differ only in features/predictor.  These
        # equal the stress test's seeds for its first backbone (EVA02-B/16).
        seeds = [args.base_seed + r for r in range(args.n_repeats)]
        tag = {"feature_set": cfg["feature_set"], "model": cfg["model"]}
        for r, seed in enumerate(seeds):
            for protocol in ("image", "participant"):
                m, pp, diag = run_config(cfg, X, y, groups, protocol, seed, args)
                rows.append({**tag, "repeat": r, "seed": seed, "protocol": protocol,
                             "n_images": len(X), "n_participants": len(np.unique(groups)),
                             **m, **diag})
                pp.insert(0, "protocol", protocol); pp.insert(0, "repeat", r)
                pp.insert(0, "model", cfg["model"]); pp.insert(0, "feature_set", cfg["feature_set"])
                preds.append(pp)
        rng = np.random.RandomState(args.base_seed + 50_000)
        for b in range(args.n_label_permutations):
            y_perm = st.participant_label_permutation(y, groups, rng)
            for r, seed in enumerate(seeds):
                for protocol in ("image", "participant"):
                    m, _, _ = run_config(cfg, X, y_perm, groups, protocol, seed, args)
                    perm_rows.append({**tag, "permutation": b, "repeat": r, "seed": seed,
                                      "protocol": protocol, **m})
        last = [x for x in rows if x["feature_set"] == cfg["feature_set"] and x["model"] == cfg["model"]]
        r2i = np.mean([x["r2"] for x in last if x["protocol"] == "image"])
        r2p = np.mean([x["r2"] for x in last if x["protocol"] == "participant"])
        print(f"[{ci + 1}/{len(configs)}] {cfg['feature_set']:16s} {cfg['model']:18s} "
              f"R2 image={r2i:.3f} participant={r2p:.3f}  ({(time.time() - start) / 60:.1f} min)")

    metrics_df = pd.DataFrame(rows)
    perm_df = pd.DataFrame(perm_rows)
    summary, wide, inflation = summarize(metrics_df)
    perm_summary = summarize_permutations(perm_df, metrics_df)
    if not perm_summary.empty:
        perm_summary["p_bonferroni_within_protocol"] = np.nan
        perm_summary["p_bh_within_protocol"] = np.nan
        for protocol, idx in perm_summary.groupby("protocol").groups.items():
            b, h = adjust_pvalues(perm_summary.loc[idx, "empirical_p_r2"])
            perm_summary.loc[idx, "p_bonferroni_within_protocol"] = b
            perm_summary.loc[idx, "p_bh_within_protocol"] = h

    metrics_df.to_csv(args.output_dir / "memorizer_protocol_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(
        args.output_dir / "memorizer_participant_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "memorizer_summary.csv", index=False)
    wide.to_csv(args.output_dir / "memorizer_paired_differences.csv", index=False)
    inflation.to_csv(args.output_dir / "memorizer_inflation_summary.csv", index=False)
    perm_df.to_csv(args.output_dir / "memorizer_permutation_metrics.csv", index=False)
    perm_summary.to_csv(args.output_dir / "memorizer_permutation_summary.csv", index=False)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump({**vars(args), "loaded_backbones": list(data),
                   "metadata_reference_backbone_used": ref,
                   "elapsed_seconds": time.time() - start}, fh, indent=2, default=str)

    pd.set_option("display.width", 250)
    print("\nIDENTITY DECODABILITY")
    print(pd.DataFrame(id_rows)[["backbone", "space", "nn1_same_participant_rate", "chance_rate",
                                 "nn1_permutation_p", "within_vs_between_auroc"]].to_string(index=False))
    print("\nPROTOCOL DIFFERENCES")
    print(inflation.to_string(index=False))
    if not perm_summary.empty:
        print("\nSHUFFLED-LABEL NULL")
        print(perm_summary.to_string(index=False))
    print(f"\nSaved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
