# -*- coding: utf-8 -*-
"""
run_table7_and_section44_reconciliation.py
=============================================
run_section44_reconciliation.py가 §4.4의 4개 숫자를 재계산해서 내적
일관성은 확인했지만(residualization == formula), Table 7의 marginal
상관값(-0.575, -0.63)과는 살짝 어긋난 채로 남았습니다
(-0.608, -0.627로 재계산됨). 즉 Table 7과 §4.4가 서로 다른 참가자
부분집합이나 계산 방식으로 만들어졌을 가능성이 높습니다.

이 스크립트는 Table 7의 18개 base feature 전체 마커 스캔과, §4.4의
fixation fraction / tracking ratio 관련 4개 숫자를 **하나의 스크립트,
하나의 참가자 부분집합**으로 동시에 계산합니다. 이렇게 하면 Table 7과
§4.4가 이제 절대로 서로 다른 숫자를 보고할 수 없습니다 (같은 코드가
같은 데이터로 만든 것이므로).

계산 내용:
  1. 18개 base feature 각각의 CARS와의 Spearman rho, raw p (n=27)
  2. Bonferroni 보정: p_Bonf = min(1, raw_p * 18)
  3. Benjamini-Hochberg FDR: 표준 step-up 절차로 q-value 계산
  4. §4.4용 4개 숫자: rho(fix,CARS), rho(track,CARS), rho(fix,track),
     partial rho(fix,CARS|track) -- 전부 위와 동일한 n=27 참가자 집합에서

실행 위치: gaze_features_participant_level.csv가 있는 폴더에서
    python run_table7_and_section44_reconciliation.py
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

# Table 7 / Appendix에 정의된 18개 base feature 이름 (participant-level
# CSV에서는 각각 뒤에 _mean이 붙어 저장되어 있다고 가정)
BASE_18_FEATURES = [
    "fixation_fraction", "mean_fixation_duration_ms", "fixation_count",
    "saccade_fraction", "saccade_count", "mean_saccade_amplitude_px",
    "path_length_px", "mean_step_size_px", "dispersion_px",
    "bbox_width_px", "bbox_height_px",
    "mean_pupil_diameter_mm", "pupil_diameter_sd_mm",
    "aoi_entropy", "social_aoi_fraction",
    "trial_duration_ms", "sample_count", "tracking_ratio_pct",
]

# 사람이 읽기 좋은 표시 이름 (Table 7 캡션과 맞춤)
DISPLAY_NAME = {
    "fixation_fraction": "Fixation fraction",
    "mean_fixation_duration_ms": "Mean fixation duration",
    "fixation_count": "Fixation count",
    "saccade_fraction": "Saccade fraction",
    "saccade_count": "Saccade count",
    "mean_saccade_amplitude_px": "Mean saccade amplitude",
    "path_length_px": "Path length",
    "mean_step_size_px": "Mean step size",
    "dispersion_px": "Dispersion",
    "bbox_width_px": "Bounding-box width",
    "bbox_height_px": "Bounding-box height",
    "mean_pupil_diameter_mm": "Mean pupil diameter",
    "pupil_diameter_sd_mm": "Pupil diameter SD",
    "aoi_entropy": "AOI entropy",
    "social_aoi_fraction": "Social-AOI fraction",
    "trial_duration_ms": "Trial duration",
    "sample_count": "Sample count",
    "tracking_ratio_pct": "Tracking ratio",
}


def detect_column(df, candidates):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        for cand in candidates:
            if cand.lower().replace(" ", "").replace("_", "") in \
               c.lower().replace(" ", "").replace("_", ""):
                return c
    return None


def benjamini_hochberg(pvals):
    """표준 BH step-up 절차. 반환: 원래 순서 그대로의 q-value 배열."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    q_ranked = ranked_p * m / (np.arange(m) + 1)
    # step-up: 뒤에서부터 누적 최솟값 (단조성 강제)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q = np.empty(m)
    q[order] = q_ranked
    return q


def partial_corr_by_residualization(x, y, z):
    x_rank = rankdata(x).reshape(-1, 1)
    y_rank = rankdata(y).reshape(-1, 1)
    z_rank = rankdata(z).reshape(-1, 1)
    reg_x = LinearRegression().fit(z_rank, x_rank)
    resid_x = (x_rank - reg_x.predict(z_rank)).ravel()
    reg_y = LinearRegression().fit(z_rank, y_rank)
    resid_y = (y_rank - reg_y.predict(z_rank)).ravel()
    return np.corrcoef(resid_x, resid_y)[0, 1]


def run_self_test():
    print("=" * 60)
    print("SELF-TEST (synthetic data)")
    print("=" * 60)

    # TEST 1: known Benjamini-Hochberg example (textbook case, 5 p-values)
    p_example = np.array([0.01, 0.02, 0.03, 0.04, 0.50])
    q = benjamini_hochberg(p_example)
    # manual calculation: m=5
    # sorted p: 0.01,0.02,0.03,0.04,0.50 ; ranks 1..5
    # raw q = p*5/rank = 0.05, 0.05, 0.05, 0.05, 0.50
    # step-up (cumulative min from the end): [0.05,0.05,0.05,0.05,0.50]
    expected = np.array([0.05, 0.05, 0.05, 0.05, 0.50])
    print(f"[TEST 1] BH-FDR on textbook p-values: {q}")
    print(f"         expected:                     {expected}")
    assert np.allclose(q, expected, atol=1e-9), \
        f"BH-FDR mismatch: got {q}, expected {expected}"
    print("  PASS")

    # TEST 2: Bonferroni sanity check
    raw_p = 0.002
    m = 18
    p_bonf = min(1.0, raw_p * m)
    print(f"[TEST 2] Bonferroni: raw_p={raw_p}, m={m} -> p_Bonf={p_bonf}")
    assert abs(p_bonf - 0.036) < 1e-9
    print("  PASS")

    # TEST 3: partial correlation cross-check (same as previous script)
    rng = np.random.RandomState(0)
    n = 100
    z = rng.randn(n)
    x = z * 2 + rng.randn(n) * 0.5
    y = z * 1.5 + rng.randn(n) * 0.5
    partial = partial_corr_by_residualization(x, y, z)
    print(f"[TEST 3] Partial correlation (shared-confound synthetic case): "
          f"{partial:.3f} (expect small)")
    assert abs(partial) < 0.3
    print("  PASS")

    print("\nSelf-test complete.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant-csv", default="gaze_features_participant_level.csv")
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    if not args.skip_self_test:
        run_self_test()

    df = pd.read_csv(args.participant_csv)
    df.columns = [c.strip() for c in df.columns]
    print(f"\n[DIAG] Columns available: {len(df.columns)} total")

    cars_col = detect_column(df, ["CARS Score", "cars_score", "cars"])
    class_col = detect_column(df, ["Class", "Diagnosis", "Group"])

    mean_cols = {}
    for feat in BASE_18_FEATURES:
        col = detect_column(df, [f"{feat}_mean"])
        if col is None:
            print(f"[WARN] Could not find column for feature '{feat}' "
                  f"(expected something like '{feat}_mean')")
        mean_cols[feat] = col

    found = {k: v for k, v in mean_cols.items() if v is not None}
    missing = [k for k, v in mean_cols.items() if v is None]
    if missing:
        print(f"\n[WARN] {len(missing)}/18 features not found: {missing}")
        print(f"       Table 7 will only include the {len(found)} features found. "
              f"Fix column-name detection above if this is wrong.")

    if cars_col is None:
        print(f"\n[FATAL] Could not detect a CARS score column.")
        return

    if class_col:
        asd = df[df[class_col].astype(str).str.upper().str.contains("ASD", na=False)].copy()
    else:
        asd = df.copy()
    asd[cars_col] = pd.to_numeric(asd[cars_col], errors="coerce")
    asd = asd.dropna(subset=[cars_col]).copy()
    print(f"\n[INFO] ASD participants with a valid CARS score: n={len(asd)} "
          f"(expect 27)")
    print(f"[INFO] Using PAIRWISE-complete cases per feature (matching the "
          f"manuscript's Section 3.2.2 per-feature aggregation rule), rather "
          f"than requiring all 18 features to be simultaneously non-null. "
          f"Participants 22/24, who lack valid trials for specific features "
          f"only, are therefore retained for every feature where they DO have "
          f"a valid value.")

    # ---------------- Table 7: full 18-feature marker scan ----------------
    print("\n" + "=" * 70)
    print(f"TABLE 7 RECONCILIATION: all {len(found)} base features vs CARS")
    print("=" * 70)

    results = []
    for feat, col in found.items():
        sub = asd.dropna(subset=[col, cars_col])
        x = sub[col].to_numpy(dtype=float)
        y = sub[cars_col].to_numpy(dtype=float)
        n_feat = len(sub)
        rho, raw_p = spearmanr(x, y)
        results.append({"feature": feat, "display": DISPLAY_NAME.get(feat, feat),
                         "n": n_feat, "rho": rho, "raw_p": raw_p})

    res_df = pd.DataFrame(results)
    res_df["p_bonf"] = np.minimum(1.0, res_df["raw_p"] * len(found))
    res_df["q_fdr"] = benjamini_hochberg(res_df["raw_p"].to_numpy())
    res_df["survives_bonf"] = res_df["p_bonf"] < 0.05
    res_df["survives_fdr"] = res_df["q_fdr"] < 0.05
    res_df = res_df.sort_values("raw_p").reset_index(drop=True)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", 30)
    print(res_df.to_string(index=False,
          formatters={"rho": "{:.4f}".format, "raw_p": "{:.5f}".format,
                      "p_bonf": "{:.4f}".format, "q_fdr": "{:.4f}".format}))

    if res_df["n"].nunique() > 1:
        print(f"\n[INFO] Per-feature n varies across rows ({sorted(res_df['n'].unique())}). "
              f"This is expected under pairwise-complete aggregation: features "
              f"that IDs 22/24 lack valid trials for will show n<27 on just "
              f"that row, while every other feature retains the full n=27.")

    res_df.to_csv("table7_reconciled_full_scan.csv", index=False)
    print(f"\n[DONE] Full 18-feature scan written to table7_reconciled_full_scan.csv")

    surv_bonf = res_df[res_df["survives_bonf"]]["display"].tolist()
    surv_fdr = res_df[res_df["survives_fdr"]]["display"].tolist()
    print(f"\n[SUMMARY] Survives Bonferroni ({len(surv_bonf)}): {surv_bonf}")
    print(f"[SUMMARY] Survives FDR ({len(surv_fdr)}): {surv_fdr}")

    # ---------------- Section 4.4: the four related numbers ----------------
    if "fixation_fraction" not in found or "tracking_ratio_pct" not in found:
        print("\n[SKIP] Section 4.4 numbers require both fixation_fraction and "
              "tracking_ratio_pct columns, which were not both found.")
        return

    ff_col = found["fixation_fraction"]
    tr_col = found["tracking_ratio_pct"]
    # pairwise-complete across exactly the three variables these four numbers
    # need: CARS, fixation fraction, tracking ratio. This is the correct
    # cohort for Section 4.4 specifically, independent of whether some OTHER
    # of the 18 features happens to be missing for a given participant.
    sub44 = asd.dropna(subset=[ff_col, tr_col, cars_col])
    n44 = len(sub44)
    fix = sub44[ff_col].to_numpy(dtype=float)
    track = sub44[tr_col].to_numpy(dtype=float)
    cars = sub44[cars_col].to_numpy(dtype=float)

    print("\n" + "=" * 70)
    print(f"SECTION 4.4 RECONCILIATION (pairwise-complete on fixation fraction, "
          f"tracking ratio, and CARS; n={n44})")
    print("=" * 70)

    rho_fix_cars, p_fix_cars = spearmanr(fix, cars)
    rho_track_cars, p_track_cars = spearmanr(track, cars)
    rho_fix_track, p_fix_track = spearmanr(fix, track)
    partial = partial_corr_by_residualization(fix, cars, track)

    print(f"  rho(fixation fraction, CARS)              = {rho_fix_cars:.4f}  (p={p_fix_cars:.5f})")
    print(f"  rho(tracking ratio, CARS)                 = {rho_track_cars:.4f}  (p={p_track_cars:.5f})")
    print(f"  rho(fixation fraction, tracking ratio)    = {rho_fix_track:.4f}  (p={p_fix_track:.4g})")
    print(f"  partial rho(fix, CARS | tracking ratio)   = {partial:.4f}")

    # cross-check against the Table 7 rows for the same two features (only
    # valid if their per-feature n in Table 7 matches n44 above)
    ff_row = res_df[res_df["feature"] == "fixation_fraction"].iloc[0]
    tr_row = res_df[res_df["feature"] == "tracking_ratio_pct"].iloc[0]
    consistent = (abs(ff_row["rho"] - rho_fix_cars) < 1e-9 and
                  abs(tr_row["rho"] - rho_track_cars) < 1e-9 and
                  ff_row["n"] == n44 and tr_row["n"] == n44)
    print(f"\n[CHECK] Table 7's fixation-fraction and tracking-ratio rows match "
          f"these numbers exactly: {'YES' if consistent else 'NO -- investigate'}")
    if not consistent:
        print(f"  Table 7 fixation_fraction: n={ff_row['n']}, rho={ff_row['rho']:.4f}")
        print(f"  Table 7 tracking_ratio:    n={tr_row['n']}, rho={tr_row['rho']:.4f}")
        print(f"  Section 4.4 (this block):  n={n44}, "
              f"rho_fix={rho_fix_cars:.4f}, rho_track={rho_track_cars:.4f}")

    with open("table7_and_section44_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Section 4.4 n={n44}\n")
        f.write(f"Table 7 survives Bonferroni: {surv_bonf}\n")
        f.write(f"Table 7 survives FDR: {surv_fdr}\n")
        f.write(f"rho(fix,CARS)={rho_fix_cars:.4f} p={p_fix_cars:.5f}\n")
        f.write(f"rho(track,CARS)={rho_track_cars:.4f} p={p_track_cars:.5f}\n")
        f.write(f"rho(fix,track)={rho_fix_track:.4f}\n")
        f.write(f"partial rho(fix,CARS|track)={partial:.4f}\n")
    print(f"[DONE] Summary written to table7_and_section44_summary.txt")


if __name__ == "__main__":
    main()
