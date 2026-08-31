# -*- coding: utf-8 -*-
"""
check_missing_features_22_24.py
==================================
v4 체크리스트 item 6 후반부: §3.2.2의 "insufficient valid trials to compute
one or more of the 18 features"라는 뭉뚱그린 표현을, 실제로 22번/24번
참가자가 어느 feature에서 결측인지 확인해서 구체적인 이름으로 교체하기
위한 조회 스크립트입니다.

재실행이 아니라 단순 조회입니다 -- 이미 가지고 계신
gaze_features_participant_level.csv를 읽어서, participant_id가 22, 24인
행에서 어느 {feature}_mean 컬럼이 NaN인지만 출력합니다.

실행 위치: gaze_features_participant_level.csv가 있는 폴더에서
    python check_missing_features_22_24.py
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import pandas as pd

BASE_18_FEATURES = [
    "fixation_fraction", "mean_fixation_duration_ms", "fixation_count",
    "saccade_fraction", "saccade_count", "mean_saccade_amplitude_px",
    "path_length_px", "mean_step_size_px", "dispersion_px",
    "bbox_width_px", "bbox_height_px",
    "mean_pupil_diameter_mm", "pupil_diameter_sd_mm",
    "aoi_entropy", "social_aoi_fraction",
    "trial_duration_ms", "sample_count", "tracking_ratio_pct",
]

DISPLAY_NAME = {
    "fixation_fraction": "fixation fraction",
    "mean_fixation_duration_ms": "mean fixation duration",
    "fixation_count": "fixation count",
    "saccade_fraction": "saccade fraction",
    "saccade_count": "saccade count",
    "mean_saccade_amplitude_px": "mean saccade amplitude",
    "path_length_px": "path length",
    "mean_step_size_px": "mean step size",
    "dispersion_px": "dispersion",
    "bbox_width_px": "bounding-box width",
    "bbox_height_px": "bounding-box height",
    "mean_pupil_diameter_mm": "mean pupil diameter",
    "pupil_diameter_sd_mm": "pupil diameter SD",
    "aoi_entropy": "AOI entropy",
    "social_aoi_fraction": "social-AOI fraction",
    "trial_duration_ms": "trial duration",
    "sample_count": "sample count",
    "tracking_ratio_pct": "tracking ratio",
}


def detect_column(df, candidates):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant-csv", default="gaze_features_participant_level.csv")
    ap.add_argument("--target-ids", default="22,24",
                     help="Comma-separated participant IDs to check (default: 22,24)")
    args = ap.parse_args()

    df = pd.read_csv(args.participant_csv)
    df.columns = [c.strip() for c in df.columns]
    print(f"[DIAG] Columns available: {len(df.columns)} total")

    pid_col = detect_column(df, ["participant_id", "ParticipantID", "id"])
    if pid_col is None:
        print(f"[FATAL] Could not detect a participant ID column. "
              f"Available columns: {list(df.columns)}")
        return

    df["_pid_str"] = df[pid_col].astype(str).str.strip()
    target_ids = [t.strip() for t in args.target_ids.split(",")]

    print(f"\n[INFO] Checking missingness for participant IDs: {target_ids}")
    print(f"[INFO] Against the 18 base features (mean columns)\n")

    results = {}
    for pid in target_ids:
        row = df[df["_pid_str"] == pid]
        if len(row) == 0:
            print(f"[WARN] Participant ID '{pid}' not found in CSV.")
            continue
        if len(row) > 1:
            print(f"[WARN] Participant ID '{pid}' matched {len(row)} rows; "
                  f"using the first.")
        row = row.iloc[0]

        missing_features = []
        for feat in BASE_18_FEATURES:
            col = detect_column(df, [f"{feat}_mean"])
            if col is None:
                print(f"  [WARN] Column for '{feat}_mean' not found in CSV at all.")
                continue
            if pd.isna(row[col]):
                missing_features.append(DISPLAY_NAME.get(feat, feat))

        results[pid] = missing_features
        if missing_features:
            print(f"[RESULT] Participant {pid}: missing {DISPLAY_NAME.get('', '')}"
                  f"{len(missing_features)} feature(s): {missing_features}")
        else:
            print(f"[RESULT] Participant {pid}: no missing base features found "
                  f"(all 18 _mean columns present). If the manuscript's claim "
                  f"that this participant has incomplete coverage is still "
                  f"correct, the missingness may instead be in per-trial data "
                  f"not visible at the participant-level aggregate -- check the "
                  f"trial-level CSV instead.")

    print("\n" + "=" * 70)
    print("SUGGESTED TEXT FOR SECTION 3.2.2")
    print("=" * 70)
    all_missing = sorted(set(f for feats in results.values() for f in feats))
    if all_missing:
        if len(all_missing) == 1:
            feat_text = all_missing[0]
        elif len(all_missing) == 2:
            feat_text = f"{all_missing[0]} and {all_missing[1]}"
        else:
            feat_text = ", ".join(all_missing[:-1]) + f", and {all_missing[-1]}"
        print(f'Replace "insufficient valid trials to compute one or more of '
              f'the 18 features" with:')
        print(f'  "insufficient valid trials to compute {feat_text}"')
        print(f"\nPer-participant detail:")
        for pid, feats in results.items():
            print(f"  ID {pid}: {feats if feats else '(none found -- check trial-level data)'}")
    else:
        print("[WARN] No missing base features were found for either participant "
              "at the participant level. Before editing the manuscript text, "
              "verify against the trial-level CSV (gaze_features_trial_level.csv) "
              "whether the incompleteness is a per-trial phenomenon that gets "
              "washed out at the participant-level aggregate.")


if __name__ == "__main__":
    main()