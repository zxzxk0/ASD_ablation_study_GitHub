# -*- coding: utf-8 -*-
"""
run_marker_confound_correlations.py
======================================
두 가지 단순 Spearman 상관을 계산합니다. 둘 다 nested LOPO나 permutation
같은 무거운 절차가 필요 없는 가벼운 분석입니다.

B-8: rho(fixation fraction, tracking ratio)
  §4.4의 partial correlation 결과(controlling for tracking ratio 시
  fixation fraction의 CARS 연관성이 rho=-0.575 -> 0.063으로 붕괴)를 설명하는
  가장 직접적인 숫자입니다. 두 변수 자체가 얼마나 강하게 얽혀 있는지
  직접 보고합니다.

B-10: rho(trial-exclusion fraction, CARS)
  <5-valid-samples 규칙으로 트라이얼이 제외된 비율이 참가자마다 다른데,
  이 제외율 자체가 CARS와 상관관계가 있는지 확인합니다. 만약 상관관계가
  있다면, "제외율이 그 자체로 recording-quality confound의 또 다른
  경로"라는 걸 시사하며 quality confound 스토리를 강화합니다.

=== 필요한 입력 파일 ===
1. --participant-csv: gaze_features_participant_level.csv
   (fixation_fraction_mean, tracking_ratio_pct_mean, CARS Score, Class,
   participant_id 컬럼 필요. n_trials 컬럼이 있으면 필터 후 트라이얼 수로
   자동 인식)
2. --raw-trials-csv: N4 감사에서 만든 n4_trials_per_participant_raw.csv
   (participant_id, n_trials 컬럼 필요 -- 이건 <5-sample 필터 적용 전
   raw 트라이얼 수)

두 파일 다 이전 세션에서 이미 만드셨던 파일입니다 -- 재실행이 아니라
기존 산출물을 다시 읽어서 계산만 하는 스크립트입니다.

실행:
    python run_marker_confound_correlations.py \
        --participant-csv gaze_features_participant_level.csv \
        --raw-trials-csv n4_trials_per_participant_raw.csv
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def run_self_test():
    print("=" * 60)
    print("SELF-TEST (synthetic data)")
    print("=" * 60)
    rng = np.random.RandomState(0)
    n = 27

    # TEST 1: strongly correlated pair -> rho should be close to the
    # true correlation used to generate the data
    x = rng.randn(n)
    y = x * 0.95 + rng.randn(n) * 0.1  # near-deterministic relationship
    rho, p = spearmanr(x, y)
    print(f"[TEST 1] Strongly correlated synthetic pair: rho={rho:.3f} (expect ~0.9+)")
    assert rho > 0.85, f"Expected rho > 0.85 for near-deterministic pair, got {rho:.3f}"
    print("  PASS")

    # TEST 2: independent pair -> rho should be small
    x2 = rng.randn(n)
    y2 = rng.randn(n)
    rho2, p2 = spearmanr(x2, y2)
    print(f"[TEST 2] Independent synthetic pair: rho={rho2:.3f} (expect near 0, "
          f"not asserting a strict bound since this is a single random draw)")
    assert -1.0 <= rho2 <= 1.0
    print("  PASS")

    # TEST 3: exclusion-fraction-style bounded [0,1] variable vs CARS
    exclusion_frac = rng.uniform(0, 0.5, n)
    cars = exclusion_frac * 20 + 25 + rng.randn(n) * 2  # designed positive relationship
    rho3, p3 = spearmanr(exclusion_frac, cars)
    print(f"[TEST 3] Designed positive relationship (exclusion frac vs CARS-like): "
          f"rho={rho3:.3f} (expect positive)")
    assert rho3 > 0, f"Expected positive correlation by construction, got {rho3:.3f}"
    print("  PASS")

    print("\nSelf-test complete.\n")


def detect_column(df, candidates):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fuzzy fallback: substring match
    for c in df.columns:
        for cand in candidates:
            if cand.lower().replace(" ", "").replace("_", "") in \
               c.lower().replace(" ", "").replace("_", ""):
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant-csv", default="gaze_features_participant_level.csv")
    ap.add_argument("--raw-trials-csv", default="n4_trials_per_participant_raw.csv",
                     help="Optional; if not found, B-10 analysis is skipped with a "
                          "clear message rather than failing.")
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    if not args.skip_self_test:
        run_self_test()

    # ---------- Load participant-level features ----------
    df = pd.read_csv(args.participant_csv)
    df.columns = [c.strip() for c in df.columns]
    print(f"\n[DIAG] Participant CSV columns: {list(df.columns)}")

    pid_col = detect_column(df, ["participant_id", "ParticipantID", "id"])
    cars_col = detect_column(df, ["CARS Score", "cars_score", "cars"])
    class_col = detect_column(df, ["Class", "Diagnosis", "Group"])
    fixfrac_col = detect_column(df, ["fixation_fraction_mean"])
    trackratio_col = detect_column(df, ["tracking_ratio_pct_mean"])
    ntrials_col = detect_column(df, ["n_trials", "num_trials", "trial_count"])

    print(f"[DIAG] Detected: pid={pid_col}, cars={cars_col}, class={class_col}, "
          f"fixfrac={fixfrac_col}, trackratio={trackratio_col}, n_trials={ntrials_col}")

    missing = [n for n, c in [("pid", pid_col), ("cars", cars_col),
                                ("fixfrac", fixfrac_col), ("trackratio", trackratio_col)]
               if c is None]
    if missing:
        print(f"\n[FATAL] Could not detect required columns: {missing}. "
              f"Check the DIAG output above and adjust the candidate lists "
              f"in detect_column() calls if the real column names differ.")
        return

    if class_col:
        asd = df[df[class_col].astype(str).str.upper().str.contains("ASD", na=False)].copy()
    else:
        asd = df.copy()
    asd[cars_col] = pd.to_numeric(asd[cars_col], errors="coerce")
    asd = asd.dropna(subset=[cars_col, fixfrac_col, trackratio_col])
    print(f"\n[INFO] n={len(asd)} ASD participants with complete fixation_fraction, "
          f"tracking_ratio, and CARS (expect 27, or fewer if some lack these features)")

    # ================================================================
    # B-8: rho(fixation fraction, tracking ratio)
    # ================================================================
    print("\n" + "=" * 70)
    print("B-8: Spearman rho(fixation fraction, tracking ratio)")
    print("=" * 70)
    rho_b8, p_b8 = spearmanr(asd[fixfrac_col], asd[trackratio_col])
    print(f"[RESULT] rho={rho_b8:.4f}  p={p_b8:.4g}  n={len(asd)}")
    print(f"[INFO] This is the direct pairwise correlation between the two raw")
    print(f"       features. It is expected to be sizeable in magnitude given")
    print(f"       that controlling for tracking ratio collapsed fixation")
    print(f"       fraction's CARS association from rho=-0.575 to partial rho=0.063")
    print(f"       (Table 8), but the exact value should be taken from this")
    print(f"       computation rather than assumed.")

    # ================================================================
    # B-10: rho(trial-exclusion fraction, CARS)
    # ================================================================
    print("\n" + "=" * 70)
    print("B-10: Spearman rho(trial-exclusion fraction, CARS)")
    print("=" * 70)

    try:
        raw = pd.read_csv(args.raw_trials_csv)
        raw.columns = [c.strip() for c in raw.columns]
        raw_pid_col = detect_column(raw, ["participant_id", "ParticipantID", "id"])
        raw_ntrials_col = detect_column(raw, ["n_trials", "num_trials", "trial_count"])

        if raw_pid_col is None or raw_ntrials_col is None or ntrials_col is None:
            print(f"[SKIP] Could not detect required columns in one of the two "
                  f"files. raw_trials file columns: {list(raw.columns)}. "
                  f"Need a participant-id column and an n_trials column in both "
                  f"files (post-filter n_trials must exist in the participant-level "
                  f"CSV: '{ntrials_col}').")
        else:
            raw["_pid"] = raw[raw_pid_col].astype(str).str.strip()
            asd["_pid"] = asd[pid_col].astype(str).str.strip()

            merged = asd.merge(
                raw[["_pid", raw_ntrials_col]].rename(
                    columns={raw_ntrials_col: "_raw_n_trials"}),
                on="_pid", how="left")

            merged = merged.dropna(subset=["_raw_n_trials", ntrials_col])
            merged["_raw_n_trials"] = pd.to_numeric(merged["_raw_n_trials"], errors="coerce")
            merged[ntrials_col] = pd.to_numeric(merged[ntrials_col], errors="coerce")
            merged = merged[merged["_raw_n_trials"] > 0]

            merged["_exclusion_fraction"] = (
                1 - merged[ntrials_col] / merged["_raw_n_trials"]
            )

            print(f"[INFO] n={len(merged)} participants merged across both files "
                  f"(expect 27 if all raw-trial-count participants match)")
            print(f"[INFO] Exclusion fraction summary: "
                  f"mean={merged['_exclusion_fraction'].mean():.3f}, "
                  f"min={merged['_exclusion_fraction'].min():.3f}, "
                  f"max={merged['_exclusion_fraction'].max():.3f}")
            print(f"[INFO] Compare mean exclusion fraction against the manuscript's "
                  f"reported cohort-level 266/1,416 = 18.8\\% figure as a sanity check.")

            if len(merged) < 5:
                print(f"[SKIP] Too few merged participants ({len(merged)}) to "
                      f"compute a meaningful correlation.")
            else:
                rho_b10, p_b10 = spearmanr(merged["_exclusion_fraction"], merged[cars_col])
                print(f"\n[RESULT] rho={rho_b10:.4f}  p={p_b10:.4g}  n={len(merged)}")

                out = merged[["_pid", "_raw_n_trials", ntrials_col,
                               "_exclusion_fraction", cars_col]].copy()
                out.columns = ["participant_id", "raw_n_trials", "used_n_trials",
                                "exclusion_fraction", "cars_score"]
                out.to_csv("trial_exclusion_fraction_per_participant.csv", index=False)
                print(f"[DONE] Per-participant detail written to "
                      f"trial_exclusion_fraction_per_participant.csv")

    except FileNotFoundError:
        print(f"[SKIP] Raw trial-count file not found at '{args.raw_trials_csv}'. "
              f"B-10 requires this file (from the N4 audit script's "
              f"n4_trials_per_participant_raw.csv output) to compute per-participant "
              f"raw trial counts before the <5-sample filter. Locate that file and "
              f"re-run with --raw-trials-csv <path>.")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"B-8  rho(fixation_fraction, tracking_ratio) = {rho_b8:.4f}  "
          f"(p={p_b8:.4g}, n={len(asd)})")
    print("B-10 see result above (or SKIP message if the raw-trials file was "
          "not found)")

    with open("marker_confound_correlations_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"B-8: rho(fixation_fraction, tracking_ratio) = {rho_b8:.4f}, "
                f"p={p_b8:.4g}, n={len(asd)}\n")
    print(f"\n[DONE] Summary written to marker_confound_correlations_summary.txt")


if __name__ == "__main__":
    main()
