# -*- coding: utf-8 -*-
"""
run_canonical_followup_audits.py
================================

Follow-up audits for the canonical gaze-dynamics analysis.

A) Clinical follow-up from nested_family_predictions.csv:
   - severity-band MAE/RMSE
   - Bland-Altman bias and 95% limits of agreement
   - tolerance rates within +/-2 / +/-3 / +/-5 CARS points
   - leave-one-out conformal coverage at nominal 90%

B) Section 5 fixed complete-case quality-confound provenance audit:
   - full 18-feature fixed-SVR regression
   - quality-removed 15-feature fixed-SVR regression
   - quality-only 3-feature fixed-SVR regression
   - full / no-quality / quality-only logistic classification

Expected dataset-root layout:
    Eye-Tracking Dataset/
        outputs/gaze_canonical_reproduction/
            gaze_features_participant_level_canonical.csv
            nested_family_predictions.csv
        run_canonical_followup_audits.py

Run:
    python run_canonical_followup_audits.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BASE_18 = [
    "fixation_fraction", "mean_fixation_duration_ms", "fixation_count",
    "saccade_fraction", "saccade_count", "mean_saccade_amplitude_px",
    "path_length_px", "mean_step_size_px", "dispersion_px",
    "bbox_width_px", "bbox_height_px",
    "mean_pupil_diameter_mm", "pupil_diameter_sd_mm",
    "aoi_entropy", "social_aoi_fraction",
    "trial_duration_ms", "sample_count", "tracking_ratio_pct",
]
QUALITY_3 = ["trial_duration_ms", "sample_count", "tracking_ratio_pct"]
NO_QUALITY_15 = [f for f in BASE_18 if f not in QUALITY_3]
CARS_MIN, CARS_MAX = 15.0, 60.0


def agg_cols(features):
    return [f"{f}_mean" for f in features] + [f"{f}_std" for f in features]


def detect_col(df, candidates):
    norm = {"".join(ch for ch in str(c).lower() if ch.isalnum()): c for c in df.columns}
    for cand in candidates:
        key = "".join(ch for ch in str(cand).lower() if ch.isalnum())
        if key in norm:
            return norm[key]
    raise KeyError(f"Could not detect one of {candidates}; columns={list(df.columns)}")


def severity_band(y):
    if y < 30:
        return "Below cutoff (<30)"
    if y < 37:
        return "Mild-to-moderate (30-36.5)"
    return "Severe (>=37)"


def regression_metrics(y, pred):
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "pearson": float(stats.pearsonr(y, pred).statistic),
    }


def fixed_svr_lopo_complete_case(X, y):
    if not np.all(np.isfinite(X)):
        raise ValueError("Complete-case SVR received non-finite X")
    loo = LeaveOneOut()
    pred = np.empty(len(y), float)
    for tr, te in loo.split(X):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")),
        ])
        model.fit(X[tr], y[tr])
        pred[te[0]] = np.clip(model.predict(X[te])[0], CARS_MIN, CARS_MAX)
    return pred


def nested_logistic_lopo_complete_case(X, y, seed=0):
    if not np.all(np.isfinite(X)):
        raise ValueError("Complete-case logistic received non-finite X")
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    loo = LeaveOneOut()
    proba = np.empty(len(y), float)
    for fold, (tr, te) in enumerate(loo.split(X)):
        Xtr, ytr = X[tr], y[tr]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(X[te])
        counts = np.bincount(ytr.astype(int))
        min_class = int(counts[counts > 0].min())
        n_splits = min(5, min_class)
        best_C = 1.0
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + fold)
            best_score = -np.inf
            for C in Cs:
                scores = []
                for itr, iva in cv.split(Xtr_s, ytr):
                    clf = LogisticRegression(C=C, penalty="l2", max_iter=5000)
                    clf.fit(Xtr_s[itr], ytr[itr])
                    pv = clf.predict_proba(Xtr_s[iva])[:, 1]
                    try:
                        scores.append(roc_auc_score(ytr[iva], pv))
                    except ValueError:
                        pass
                score = np.mean(scores) if scores else -np.inf
                if score > best_score:
                    best_score, best_C = score, C
        clf = LogisticRegression(C=best_C, penalty="l2", max_iter=5000)
        clf.fit(Xtr_s, ytr)
        proba[te[0]] = clf.predict_proba(Xte_s)[:, 1][0]
    return proba


def fixed_complete_case(df, cols, target_col=None):
    work = df.copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    required = list(cols)
    if target_col is not None:
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        required.append(target_col)
    keep = work.dropna(subset=required).copy()
    drop = work.loc[~work.index.isin(keep.index)].copy()
    return keep, drop


def clinical_followup(nested_path, outdir):
    print("\n[A] Clinical follow-up from nested-family predictions")
    df = pd.read_csv(nested_path)
    req = {"participant_id", "cars_true", "nested_family_prediction"}
    if not req.issubset(df.columns):
        raise KeyError(f"Missing columns in {nested_path}: {sorted(req - set(df.columns))}")
    y = pd.to_numeric(df["cars_true"], errors="raise").to_numpy(float)
    pred = pd.to_numeric(df["nested_family_prediction"], errors="raise").to_numpy(float)

    rows = []
    bands = np.array([severity_band(v) for v in y])
    for band in ["Below cutoff (<30)", "Mild-to-moderate (30-36.5)", "Severe (>=37)"]:
        m = bands == band
        rows.append({
            "severity_band": band,
            "n": int(m.sum()),
            "mae": float(mean_absolute_error(y[m], pred[m])),
            "rmse": float(np.sqrt(mean_squared_error(y[m], pred[m]))),
        })
    rows.append({
        "severity_band": "All",
        "n": len(y),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    })
    pd.DataFrame(rows).to_csv(outdir / "nested_severity_stratified_metrics.csv", index=False)

    diff = pred - y
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    abs_err = np.abs(diff)

    tol = []
    for k in [2, 3, 5]:
        count = int(np.sum(abs_err <= k))
        tol.append({"tolerance_cars_points": k, "count": count, "n": len(y), "fraction": count / len(y)})
    pd.DataFrame(tol).to_csv(outdir / "nested_clinical_tolerance.csv", index=False)

    covered, radii, lower, upper = [], [], [], []
    for i in range(len(y)):
        other = np.delete(abs_err, i)
        radius = float(np.quantile(other, 0.90, method="higher"))
        lo, hi = pred[i] - radius, pred[i] + radius
        radii.append(radius); lower.append(lo); upper.append(hi); covered.append(lo <= y[i] <= hi)
    conf = df[["participant_id", "cars_true", "nested_family_prediction"]].copy()
    conf["radius_90"] = radii; conf["lower_90"] = lower; conf["upper_90"] = upper; conf["covered_90"] = covered
    conf.to_csv(outdir / "nested_conformal_intervals.csv", index=False)

    overall = regression_metrics(y, pred)
    summary = {
        **overall,
        "bland_altman_bias_pred_minus_true": bias,
        "bland_altman_loa_lo": loa_lo,
        "bland_altman_loa_hi": loa_hi,
        "conformal_nominal": 0.90,
        "conformal_coverage": float(np.mean(covered)),
        "conformal_covered_count": int(np.sum(covered)),
        "within_2_count": int(np.sum(abs_err <= 2)),
        "within_3_count": int(np.sum(abs_err <= 3)),
        "within_5_count": int(np.sum(abs_err <= 5)),
        "within_2_fraction": float(np.mean(abs_err <= 2)),
        "within_3_fraction": float(np.mean(abs_err <= 3)),
        "within_5_fraction": float(np.mean(abs_err <= 5)),
    }
    pd.DataFrame([summary]).to_csv(outdir / "nested_clinical_summary.csv", index=False)
    print(f"  n={len(y)} MAE={overall['mae']:.3f} RMSE={overall['rmse']:.3f} R2={overall['r2']:.3f}")
    print(f"  Bland-Altman bias={bias:+.3f}, LoA=[{loa_lo:.3f}, {loa_hi:.3f}]")
    print(f"  conformal coverage={np.mean(covered):.3f}")
    return summary


def quality_audit(participant_path, outdir):
    print("\n[B] Section 5 complete-case quality-confound audit")
    df = pd.read_csv(participant_path)
    df.columns = [str(c).strip() for c in df.columns]
    class_col = detect_col(df, ["Class", "Diagnosis", "Group"])
    cars_col = detect_col(df, ["CARS Score", "CARS_Score", "CARS"])
    pid_col = detect_col(df, ["participant_id", "ParticipantID", "ID"])

    full_cols = agg_cols(BASE_18)
    noqual_cols = agg_cols(NO_QUALITY_15)
    quality_mean_cols = [f"{f}_mean" for f in QUALITY_3]

    asd = df[df[class_col].astype(str).str.strip().str.upper().eq("ASD")].copy()
    asd[cars_col] = pd.to_numeric(asd[cars_col], errors="coerce")
    asd = asd[asd[cars_col].notna()].copy()
    reg_keep, reg_drop = fixed_complete_case(asd, full_cols, cars_col)
    reg_keep = reg_keep.sort_values(pid_col, key=lambda s: pd.to_numeric(s, errors="coerce"))

    y = reg_keep[cars_col].to_numpy(float)
    X18, X15, Xq = reg_keep[full_cols].to_numpy(float), reg_keep[noqual_cols].to_numpy(float), reg_keep[quality_mean_cols].to_numpy(float)
    p18, p15, pq = fixed_svr_lopo_complete_case(X18, y), fixed_svr_lopo_complete_case(X15, y), fixed_svr_lopo_complete_case(Xq, y)
    m18, m15, mq = regression_metrics(y, p18), regression_metrics(y, p15), regression_metrics(y, pq)
    reg = pd.DataFrame([
        {"model": "Full 18-feature SVR", **m18},
        {"model": "No-quality 15-feature SVR", **m15},
        {"model": "Quality-only 3-feature SVR", **mq},
    ])
    reg["delta_r2_vs_full18"] = reg["r2"] - m18["r2"]
    reg.to_csv(outdir / "quality_audit_regression_summary.csv", index=False)
    pd.DataFrame({
        "participant_id": reg_keep[pid_col].astype(str), "cars_true": y,
        "pred_full18": p18, "pred_noquality15": p15, "pred_quality3": pq,
    }).to_csv(outdir / "quality_audit_regression_predictions.csv", index=False)

    cls = df[df[class_col].astype(str).str.strip().str.upper().isin(["ASD", "TD"])].copy()
    cls_keep, cls_drop = fixed_complete_case(cls, full_cols)
    cls_keep = cls_keep.sort_values(pid_col, key=lambda s: pd.to_numeric(s, errors="coerce"))
    yc = cls_keep[class_col].astype(str).str.strip().str.upper().eq("ASD").astype(int).to_numpy()
    Xc18, Xc15, Xcq = cls_keep[full_cols].to_numpy(float), cls_keep[noqual_cols].to_numpy(float), cls_keep[quality_mean_cols].to_numpy(float)
    c18, c15, cq = nested_logistic_lopo_complete_case(Xc18, yc), nested_logistic_lopo_complete_case(Xc15, yc), nested_logistic_lopo_complete_case(Xcq, yc)
    a18, a15, aq = float(roc_auc_score(yc, c18)), float(roc_auc_score(yc, c15)), float(roc_auc_score(yc, cq))
    cls_sum = pd.DataFrame([
        {"model": "Full 18-feature logistic", "n": len(yc), "auroc": a18},
        {"model": "No-quality 15-feature logistic", "n": len(yc), "auroc": a15},
        {"model": "Quality-only 3-feature logistic", "n": len(yc), "auroc": aq},
    ])
    cls_sum["delta_auroc_vs_full18"] = cls_sum["auroc"] - a18
    cls_sum.to_csv(outdir / "quality_audit_classification_summary.csv", index=False)
    pd.DataFrame({
        "participant_id": cls_keep[pid_col].astype(str), "class_true": yc,
        "prob_full18": c18, "prob_noquality15": c15, "prob_quality3": cq,
    }).to_csv(outdir / "quality_audit_classification_predictions.csv", index=False)

    comp = pd.DataFrame([
        {"quantity": "reg_full18_r2", "old": 0.351, "canonical": m18["r2"]},
        {"quantity": "reg_noquality15_r2", "old": 0.332, "canonical": m15["r2"]},
        {"quantity": "reg_quality3_r2", "old": 0.419, "canonical": mq["r2"]},
        {"quantity": "cls_full18_auroc", "old": 0.884, "canonical": a18},
        {"quantity": "cls_noquality15_auroc", "old": 0.877, "canonical": a15},
        {"quantity": "cls_quality3_auroc", "old": 0.901, "canonical": aq},
    ])
    comp["difference"] = comp["canonical"] - comp["old"]
    comp.to_csv(outdir / "quality_audit_old_vs_canonical.csv", index=False)

    pd.DataFrame({"participant_id": reg_drop[pid_col].astype(str)}).to_csv(outdir / "quality_audit_regression_dropped.csv", index=False)
    pd.DataFrame({"participant_id": cls_drop[pid_col].astype(str)}).to_csv(outdir / "quality_audit_classification_dropped.csv", index=False)

    print(f"  regression n={len(y)} | full18 R2={m18['r2']:.4f} | noqual R2={m15['r2']:.4f} | quality3 R2={mq['r2']:.4f}")
    print(f"  classification n={len(yc)} | full18 AUROC={a18:.4f} | noqual={a15:.4f} | quality3={aq:.4f}")
    print(f"  dropped regression IDs: {reg_drop[pid_col].astype(str).tolist()}")
    print(f"  dropped classification IDs: {cls_drop[pid_col].astype(str).tolist()}")
    return reg, cls_sum, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--canonical-dir", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    canonical_dir = (args.canonical_dir or (root / "outputs" / "gaze_canonical_reproduction")).resolve()
    outdir = (args.outdir or (root / "outputs" / "gaze_canonical_followup")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    participant_path = canonical_dir / "gaze_features_participant_level_canonical.csv"
    nested_path = canonical_dir / "nested_family_predictions.csv"
    if not participant_path.exists():
        raise FileNotFoundError(f"Missing {participant_path}")
    if not nested_path.exists():
        raise FileNotFoundError(f"Missing {nested_path}")

    print("=" * 78)
    print("CANONICAL GAZE FOLLOW-UP AUDITS")
    print("=" * 78)
    print(f"Canonical dir: {canonical_dir}")
    print(f"Output dir:    {outdir}")

    clinical = clinical_followup(nested_path, outdir)
    reg, cls, comp = quality_audit(participant_path, outdir)

    lines = [
        "CANONICAL GAZE FOLLOW-UP AUDIT SUMMARY", "=" * 60, "",
        "[A] Fully nested family-selection clinical analysis",
        f"n={clinical['n']}", f"MAE={clinical['mae']:.4f}", f"RMSE={clinical['rmse']:.4f}", f"R2={clinical['r2']:.4f}",
        f"Spearman={clinical['spearman']:.4f}", f"Pearson={clinical['pearson']:.4f}",
        f"Bland-Altman bias={clinical['bland_altman_bias_pred_minus_true']:+.4f}",
        f"95% LoA=[{clinical['bland_altman_loa_lo']:.4f}, {clinical['bland_altman_loa_hi']:.4f}]",
        f"Within +/-2={clinical['within_2_count']}/{clinical['n']} ({clinical['within_2_fraction']:.3f})",
        f"Within +/-3={clinical['within_3_count']}/{clinical['n']} ({clinical['within_3_fraction']:.3f})",
        f"Within +/-5={clinical['within_5_count']}/{clinical['n']} ({clinical['within_5_fraction']:.3f})",
        f"LOO conformal 90% coverage={clinical['conformal_covered_count']}/{clinical['n']} ({clinical['conformal_coverage']:.3f})",
        "", "[B] Complete-case quality-confound regression audit", reg.to_string(index=False),
        "", "[C] Complete-case quality-confound classification audit", cls.to_string(index=False),
        "", "[D] Old manuscript values vs canonical complete-case audit", comp.to_string(index=False),
    ]
    (outdir / "followup_audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nDONE")
    print(f"Outputs: {outdir}")
    print("Important files:")
    for name in [
        "followup_audit_summary.txt",
        "nested_severity_stratified_metrics.csv",
        "nested_clinical_summary.csv",
        "nested_clinical_tolerance.csv",
        "nested_conformal_intervals.csv",
        "quality_audit_regression_summary.csv",
        "quality_audit_classification_summary.csv",
        "quality_audit_old_vs_canonical.csv",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
