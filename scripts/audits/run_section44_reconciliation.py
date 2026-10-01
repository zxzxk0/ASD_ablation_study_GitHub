# -*- coding: utf-8 -*-
"""
run_section44_reconciliation.py
==================================
§4.4의 4개 숫자가 서로 수학적으로 모순됨이 발견됐습니다:
  rho(fixation fraction, CARS)     = -0.575
  rho(tracking ratio, CARS)        = -0.63
  rho(fixation fraction, tracking) =  0.984
  partial rho(fix, CARS | track)   =  0.063

partial correlation 공식으로 역산하면 위 세 개는 partial rho ≈ 0.325를
암시하는데, 실제 보고된 값은 0.063입니다. 넷 다 동시에 참일 수 없습니다.

원인 후보: 각 숫자가 서로 다른 스크립트/세션에서, 어쩌면 서로 다른
참가자 부분집합으로 계산됐을 가능성이 높습니다.

이 스크립트는 네 숫자를 전부 하나의 실행에서, 동일한 n=27 참가자
벡터로 동시에 계산합니다. 그래서 이번에 나오는 값들은 서로 내적
일관성이 보장됩니다 (partial correlation 공식이 자동으로 성립).

partial correlation은 두 가지 방식으로 계산해서 서로 대조합니다:
  (a) rank-residualization 방식 (원래 §4.4가 썼다고 명시한 방법:
      순위로 변환 후 linear regression으로 잔차를 구하고, 그 잔차끼리
      Pearson correlation)
  (b) 표준 partial-correlation 공식 (Spearman rho 세 개로부터 대수적으로
      역산)
두 방식이 크게 다르면 그 자체가 흥미로운 진단 정보입니다.

실행 위치: gaze_features_participant_level.csv가 있는 폴더에서
    python run_section44_reconciliation.py
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


def partial_corr_by_residualization(x, y, z):
    """원래 §4.4가 썼다고 명시한 방법: 세 변수 다 rank로 변환한 뒤,
    x와 y 각각을 z(rank)에 대해 linear regression으로 적합시키고,
    그 잔차끼리 Pearson correlation."""
    x_rank = rankdata(x).reshape(-1, 1)
    y_rank = rankdata(y).reshape(-1, 1)
    z_rank = rankdata(z).reshape(-1, 1)

    reg_x = LinearRegression().fit(z_rank, x_rank)
    resid_x = (x_rank - reg_x.predict(z_rank)).ravel()

    reg_y = LinearRegression().fit(z_rank, y_rank)
    resid_y = (y_rank - reg_y.predict(z_rank)).ravel()

    r = np.corrcoef(resid_x, resid_y)[0, 1]
    return r


def partial_corr_formula(rho_xy, rho_xz, rho_yz):
    """표준 1차 partial correlation 공식 (Spearman rho 세 개로부터
    대수적으로 역산)."""
    num = rho_xy - rho_xz * rho_yz
    den = np.sqrt((1 - rho_xz ** 2) * (1 - rho_yz ** 2))
    return num / den


def run_self_test():
    print("=" * 60)
    print("SELF-TEST (synthetic data)")
    print("=" * 60)
    rng = np.random.RandomState(0)
    n = 100  # larger n for a cleaner self-test signal

    # Construct z, then x and y each as a function of z plus independent
    # noise, so that the TRUE partial correlation between x and y given z
    # should be close to zero (both residualization and formula methods
    # should agree and both report near-zero).
    z = rng.randn(n)
    x = z * 2 + rng.randn(n) * 0.5
    y = z * 1.5 + rng.randn(n) * 0.5

    rho_xy, _ = spearmanr(x, y)
    rho_xz, _ = spearmanr(x, z)
    rho_yz, _ = spearmanr(y, z)

    partial_resid = partial_corr_by_residualization(x, y, z)
    partial_formula = partial_corr_formula(rho_xy, rho_xz, rho_yz)

    print(f"[TEST 1] Synthetic data where x,y share only z-driven variance:")
    print(f"  rho(x,y)={rho_xy:.3f}  rho(x,z)={rho_xz:.3f}  rho(y,z)={rho_yz:.3f}")
    print(f"  partial (residualization) = {partial_resid:.3f}")
    print(f"  partial (formula)         = {partial_formula:.3f}")
    assert abs(partial_resid) < 0.3, \
        f"Expected near-zero partial correlation, got {partial_resid:.3f}"
    assert abs(partial_resid - partial_formula) < 0.15, \
        (f"The two methods disagree by more than expected: "
         f"{partial_resid:.3f} vs {partial_formula:.3f}")
    print("  PASS (both methods agree and both are near zero, as expected)")

    # Second synthetic case: x and y have a genuine independent
    # relationship not mediated by z, so partial correlation should stay
    # sizeable even after controlling for z.
    z2 = rng.randn(n)
    x2 = z2 + rng.randn(n) * 0.3
    y2 = x2 * 0.8 + rng.randn(n) * 0.3  # y depends on x directly, not just via z
    rho_xy2, _ = spearmanr(x2, y2)
    rho_xz2, _ = spearmanr(x2, z2)
    rho_yz2, _ = spearmanr(y2, z2)
    partial_resid2 = partial_corr_by_residualization(x2, y2, z2)
    partial_formula2 = partial_corr_formula(rho_xy2, rho_xz2, rho_yz2)
    print(f"\n[TEST 2] Synthetic data where x,y share a direct relationship "
          f"beyond z:")
    print(f"  partial (residualization) = {partial_resid2:.3f}")
    print(f"  partial (formula)         = {partial_formula2:.3f}")
    assert partial_resid2 > 0.3, \
        f"Expected a sizeable positive partial correlation, got {partial_resid2:.3f}"
    assert abs(partial_resid2 - partial_formula2) < 0.15, \
        (f"The two methods disagree by more than expected: "
         f"{partial_resid2:.3f} vs {partial_formula2:.3f}")
    print("  PASS (both methods agree and both detect the direct relationship)")

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
    print(f"\n[DIAG] Columns: {list(df.columns)}")

    pid_col = detect_column(df, ["participant_id", "ParticipantID", "id"])
    cars_col = detect_column(df, ["CARS Score", "cars_score", "cars"])
    class_col = detect_column(df, ["Class", "Diagnosis", "Group"])
    fixfrac_col = detect_column(df, ["fixation_fraction_mean"])
    trackratio_col = detect_column(df, ["tracking_ratio_pct_mean"])

    missing = [n for n, c in [("cars", cars_col), ("fixfrac", fixfrac_col),
                                ("trackratio", trackratio_col)]
               if c is None]
    if missing:
        print(f"\n[FATAL] Could not detect required columns: {missing}. "
              f"Check DIAG output above.")
        return

    if class_col:
        asd = df[df[class_col].astype(str).str.upper().str.contains("ASD", na=False)].copy()
    else:
        asd = df.copy()
    asd[cars_col] = pd.to_numeric(asd[cars_col], errors="coerce")

    # CRITICAL: use exactly one complete-case filter, applied once, and use
    # this SAME n for every one of the four numbers below. This is the
    # actual fix -- previously these four numbers likely came from separate
    # computations that silently used different participant subsets.
    asd_complete = asd.dropna(subset=[cars_col, fixfrac_col, trackratio_col]).copy()
    n = len(asd_complete)
    print(f"\n[INFO] n={n} ASD participants with complete CARS, fixation_fraction, "
          f"and tracking_ratio (expect 27)")
    if n != 27:
        print(f"[WARN] n != 27. If the manuscript's headline marker analysis is "
              f"defined on exactly n=27, investigate which participant(s) are "
              f"missing here and why, before using these numbers in the paper.")

    fix = asd_complete[fixfrac_col].to_numpy(dtype=float)
    track = asd_complete[trackratio_col].to_numpy(dtype=float)
    cars = asd_complete[cars_col].to_numpy(dtype=float)

    print("\n" + "=" * 70)
    print("ALL FOUR NUMBERS, COMPUTED TOGETHER ON THE SAME n={} VECTORS".format(n))
    print("=" * 70)

    rho_fix_cars, p_fix_cars = spearmanr(fix, cars)
    print(f"\n1. rho(fixation fraction, CARS) = {rho_fix_cars:.4f}  "
          f"(p={p_fix_cars:.4g})")

    rho_track_cars, p_track_cars = spearmanr(track, cars)
    print(f"2. rho(tracking ratio, CARS)    = {rho_track_cars:.4f}  "
          f"(p={p_track_cars:.4g})")

    rho_fix_track, p_fix_track = spearmanr(fix, track)
    print(f"3. rho(fixation fraction, tracking ratio) = {rho_fix_track:.4f}  "
          f"(p={p_fix_track:.4g})")

    partial_resid = partial_corr_by_residualization(fix, cars, track)
    partial_formula = partial_corr_formula(rho_fix_cars, rho_fix_track, rho_track_cars)
    print(f"4a. partial rho(fix, CARS | track), rank-residualization method "
          f"= {partial_resid:.4f}")
    print(f"4b. partial rho(fix, CARS | track), algebraic formula "
          f"= {partial_formula:.4f}")

    disagreement = abs(partial_resid - partial_formula)
    print(f"\n[CHECK] Agreement between the two partial-correlation methods: "
          f"|difference| = {disagreement:.4f}")
    if disagreement > 0.1:
        print(f"[WARN] The two methods disagree by more than 0.1. This is larger "
              f"than expected for n={n} and warrants a closer look at the "
              f"residualization code before trusting either number blindly.")
    else:
        print(f"[OK] The two methods agree closely, as they should when computed "
              f"on the identical n={n} sample.")

    print("\n" + "=" * 70)
    print("COMPARE AGAINST WHAT'S CURRENTLY IN THE MANUSCRIPT")
    print("=" * 70)
    print(f"  Manuscript rho(fix, CARS):     -0.575   vs. this run: {rho_fix_cars:.4f}")
    print(f"  Manuscript rho(track, CARS):   -0.63    vs. this run: {rho_track_cars:.4f}")
    print(f"  Manuscript rho(fix, track):     0.984   vs. this run: {rho_fix_track:.4f}")
    print(f"  Manuscript partial rho:         0.063   vs. this run: "
          f"{partial_resid:.4f} (residualization) / {partial_formula:.4f} (formula)")
    print(f"\n  If any of the four numbers above differs meaningfully from the "
          f"manuscript's current value, ALL FOUR should be replaced with this "
          f"run's numbers everywhere they appear (Section 4.4 text, Table 8, "
          f"Figure 6 caption, abstract, intro bullet 4), since this run is the "
          f"one that guarantees internal consistency.")

    out = pd.DataFrame([{
        "quantity": "rho(fixation_fraction, CARS)", "value": rho_fix_cars, "p": p_fix_cars, "n": n},
        {"quantity": "rho(tracking_ratio, CARS)", "value": rho_track_cars, "p": p_track_cars, "n": n},
        {"quantity": "rho(fixation_fraction, tracking_ratio)", "value": rho_fix_track, "p": p_fix_track, "n": n},
        {"quantity": "partial_rho_residualization(fix,CARS|track)", "value": partial_resid, "p": np.nan, "n": n},
        {"quantity": "partial_rho_formula(fix,CARS|track)", "value": partial_formula, "p": np.nan, "n": n},
    ])
    out.to_csv("section44_reconciled_numbers.csv", index=False)
    print(f"\n[DONE] Written to section44_reconciled_numbers.csv")


if __name__ == "__main__":
    main()
