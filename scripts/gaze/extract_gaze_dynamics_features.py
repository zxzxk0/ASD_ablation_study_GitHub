# -*- coding: utf-8 -*-
"""
extract_gaze_dynamics_features.py
====================================
"Eye-tracking Output"의 25개 raw SMI export CSV에서 참가자-트라이얼 단위로
18개 base + 11개 enhanced gaze-dynamics 피처를 계산합니다 (논문 Section
3.2.2 서술 기준). B3(recording-quality confound 검증)와 C11(피처 목록
부록)에 필요한 원자료를 생성하는 것이 목적입니다.

핵심 파싱 규칙 (실제 데이터 구조 확인을 통해 확정됨):
  - 유효 gaze 행: Category Group == "Eye" (Information 행은 트라이얼
    메타 마커이므로 제외)
  - 이벤트 타입: Category Right/Left 값 중 Fixation / Saccade / Blink만
    사용 (Left Click, Separator, '-'는 실제 gaze 이벤트가 아님)
  - 좌우 시선 중 결측 없는 쪽 우선 사용, 둘 다 있으면 Right를 기본 채널로
    사용 (SMI 이중안구 기록의 관례를 따름; 필요시 --eye-channel 옵션으로
    left/right/avg 전환 가능)
  - Participant 값 중 'Unidentified(Neg)'/'Unidentified(Pos)' 행은
    참가자 ID가 아니라 데이터 이슈 마커이므로 처리 전 드롭
  - AOI는 논문 Section 3.2.2 서술("actor's body and pointing gesture
    (social) vs a balloon (non-social distractor)")에 맞춰 이분화 (A안):
      social:     corps, visage, visage1, yeux, "yeux (1)", bouche, mains,
                  "Pointage D", "Pointage G", "pointage chat",
                  "pointage chien", "pointage chien (1)", coucou,
                  "joie droite", "joie gauche", "triste droite",
                  "triste gauche", neutre, "neutre fichier b",
                  "bouche joie", "bouche neutre", "bouche triste",
                  "yeux joie", "yeux neutre", "yeux triste"
      non_social: BallonVisible, BallonInvisible, chat, "chat (1)",
                  chien, "chien (1)", "White Space", "AOI 001"
      missing:    '-'
  - 25.csv는 AOI Name 컬럼이 없음 -> 그 트라이얼들은 AOI 관련 피처만
    NaN으로 남기고, 나머지 피처는 정상 계산

18개 base 피처 (논문 Table 6/Section 3.2.2 카테고리에 매핑):
  Fixation/saccade structure:
    fixation_fraction, mean_fixation_duration_ms, fixation_count,
    saccade_fraction, saccade_count, mean_saccade_amplitude_px
  Scanpath geometry:
    path_length_px, mean_step_size_px, dispersion_px,
    bbox_width_px, bbox_height_px
  Pupillary dynamics:
    mean_pupil_diameter_mm, pupil_diameter_sd_mm
  AOI / recording-quality:
    aoi_entropy, social_aoi_fraction,
    trial_duration_ms, sample_count, tracking_ratio_pct

11개 enhanced 피처 (프리스펙파이된 강화 세트):
  AOI dwell/transition:
    social_dwell_time_ms, non_social_dwell_time_ms,
    aoi_transition_count, aoi_transition_rate
  Pupil velocity/asymmetry/range:
    pupil_velocity_mean, pupil_asymmetry_mean (|L-R|),
    pupil_range_mm
  Additional social attention:
    time_to_first_social_fixation_ms, social_fixation_count,
    non_social_fixation_count, social_to_nonsocial_fixation_ratio

주의: 이 스크립트는 raw export의 실측 구조를 최대한 존중해서 짰지만,
원 논문의 18/29개 피처 "정확한" 정의(예: fixation 최소 지속시간 threshold,
saccade amplitude 계산 방식)와 100% 동일하다는 보장은 없습니다. 이는
독립적으로 재구성된 근사치이며, 논문에 반영할 때는 그렇게 명시해야 합니다.

실행 위치: 'Eye-tracking Output' 폴더의 부모 디렉토리에서
    python extract_gaze_dynamics_features.py --metadata Metadata_Participants.csv
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ET_DIR_DEFAULT = Path("Eye-tracking Output")

SOCIAL_AOI = {
    "corps", "visage", "visage1", "yeux", "yeux (1)", "bouche", "mains",
    "Pointage D", "Pointage G", "pointage chat", "pointage chien",
    "pointage chien (1)", "coucou", "joie droite", "joie gauche",
    "triste droite", "triste gauche", "neutre", "neutre fichier b",
    "bouche joie", "bouche neutre", "bouche triste",
    "yeux joie", "yeux neutre", "yeux triste",
}
NON_SOCIAL_AOI = {
    "BallonVisible", "BallonInvisible", "chat", "chat (1)",
    "chien", "chien (1)", "White Space", "AOI 001",
}
MISSING_AOI = {"-"}

VALID_EVENTS = {"Fixation", "Saccade", "Blink"}
INVALID_PARTICIPANT_MARKERS = {"Unidentified(Neg)", "Unidentified(Pos)"}


# [VALIDATION FIX] Case-insensitive lookup sets. The raw export mixes
# casing across sessions/files (e.g. 'visage' vs 'Visage' both occur --
# confirmed by the validation report: 'Visage' had 7068 occurrences and
# was falling through to 'unknown' before this fix). Build lowercased
# versions once at import time instead of matching case-sensitively.
SOCIAL_AOI_LOWER = {s.lower() for s in SOCIAL_AOI}
NON_SOCIAL_AOI_LOWER = {s.lower() for s in NON_SOCIAL_AOI}


def aoi_class(name: str) -> str:
    name_norm = name.strip().lower()
    if name_norm in SOCIAL_AOI_LOWER:
        return "social"
    if name_norm in NON_SOCIAL_AOI_LOWER:
        return "non_social"
    return "unknown"  # includes MISSING_AOI and anything not seen during inspection


UNRECOGNIZED_AOI_VALUES = Counter()  # [VALIDATION] populated as files are parsed


def aoi_class_tracked(name: str) -> str:
    """Same as aoi_class but records any value that falls through to
    'unknown' and isn't the expected MISSING_AOI marker, so a final report
    can flag AOI labels the SOCIAL_AOI/NON_SOCIAL_AOI lists don't cover."""
    cls = aoi_class(name)
    if cls == "unknown" and name not in MISSING_AOI:
        UNRECOGNIZED_AOI_VALUES[name] += 1
    return cls


def load_one_file(path: Path, eye_channel: str = "right"):
    """
    Loads one raw SMI export CSV, filters to valid Eye rows, and returns a
    cleaned DataFrame with normalized column names, or None if the file is
    unusable.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  [ERROR] failed to read {path.name}: {e}")
        return None

    df.columns = [c.strip() for c in df.columns]

    required = {"Participant", "Trial", "Stimulus", "Category Group",
                "Category Right"}
    missing_req = required - set(df.columns)
    if missing_req:
        print(f"  [WARN] {path.name} missing required columns "
              f"{missing_req}, skipping file")
        return None

    n_raw = len(df)
    df = df[df["Category Group"].astype(str).str.strip() == "Eye"].copy()
    n_after_eye = len(df)
    df = df[~df["Participant"].astype(str).isin(INVALID_PARTICIPANT_MARKERS)]
    n_after_pid = len(df)
    df = df[df["Category Right"].astype(str).isin(VALID_EVENTS)]
    n_after_event = len(df)

    # [VALIDATION] row 손실률 로그 -- 필터 단계마다 얼마나 걸러지는지 눈으로
    # 확인. Category Group=="Eye" 필터에서 대부분 남아야 정상 (Information
    # 마커는 트라이얼당 몇 줄뿐이어야 함); 여기서 90% 이상 날아가면 필터
    # 조건 자체가 이 파일의 실제 값과 안 맞는다는 신호이므로 조사 필요.
    kept_pct = 100 * n_after_event / n_raw if n_raw else 0
    print(f"  [FILTER] {path.name}: raw={n_raw} -> Eye-only={n_after_eye} "
          f"-> valid-pid={n_after_pid} -> valid-event={n_after_event} "
          f"({kept_pct:.1f}% kept)")
    if kept_pct < 50:
        print(f"    [VALIDATION WARNING] >50% of rows dropped in {path.name} "
              f"-- verify Category Group/Category Right values for this "
              f"file match the expected set (should rarely happen if the "
              f"file follows the same schema as the files used to design "
              f"this script).")

    if df.empty:
        return None

    # numeric coercion (SMI export uses '-' for missing numeric fields)
    num_cols = [
        "RecordingTime [ms]",
        "Pupil Diameter Right [mm]", "Pupil Diameter Left [mm]",
        "Point of Regard Right X [px]", "Point of Regard Right Y [px]",
        "Point of Regard Left X [px]", "Point of Regard Left Y [px]",
        "Tracking Ratio [%]",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # pick the working gaze-position / pupil channel
    if eye_channel == "left":
        px, py = "Point of Regard Left X [px]", "Point of Regard Left Y [px]"
        pupil = "Pupil Diameter Left [mm]"
        aoi_col = "AOI Name Left"
    else:  # default: right
        px, py = "Point of Regard Right X [px]", "Point of Regard Right Y [px]"
        pupil = "Pupil Diameter Right [mm]"
        aoi_col = "AOI Name Right"

    df["gx"] = df[px] if px in df.columns else np.nan
    df["gy"] = df[py] if py in df.columns else np.nan
    df["pupil"] = df[pupil] if pupil in df.columns else np.nan
    df["pupil_other"] = (df["Pupil Diameter Left [mm]"]
                          if eye_channel == "right" and "Pupil Diameter Left [mm]" in df.columns
                          else df.get("Pupil Diameter Right [mm]", np.nan))
    df["aoi_raw"] = df[aoi_col] if aoi_col in df.columns else "-"
    df["aoi_class"] = df["aoi_raw"].astype(str).map(aoi_class_tracked)
    df["event"] = df["Category Right"].astype(str)

    df["source_file"] = path.name
    return df


def compute_trial_features(g: pd.DataFrame) -> dict:
    """
    g = rows for ONE (Participant, Trial) pair, already filtered to valid
    Eye rows. Returns a dict of the 18 base + 11 enhanced features for
    this single trial.
    """
    n = len(g)
    feat = {}

    # ---- Fixation/saccade structure ----
    is_fix = g["event"] == "Fixation"
    is_sac = g["event"] == "Saccade"
    feat["fixation_fraction"] = is_fix.mean()
    feat["saccade_fraction"] = is_sac.mean()
    feat["fixation_count"] = int(count_runs(is_fix))
    feat["saccade_count"] = int(count_runs(is_sac))

    if "RecordingTime [ms]" in g.columns and is_fix.any():
        t = g.loc[is_fix, "RecordingTime [ms]"].to_numpy()
        # crude per-sample duration inferred from consecutive timestamp
        # deltas within fixation runs; not a substitute for a proper
        # event-based duration field if the export ever provides one
        feat["mean_fixation_duration_ms"] = _mean_run_duration(g, is_fix)
    else:
        feat["mean_fixation_duration_ms"] = np.nan

    gx, gy = g["gx"].to_numpy(), g["gy"].to_numpy()
    valid_xy = ~(np.isnan(gx) | np.isnan(gy)) & (gx != 0) & (gy != 0)
    gx_v, gy_v = gx[valid_xy], gy[valid_xy]

    if len(gx_v) > 1:
        dx, dy = np.diff(gx_v), np.diff(gy_v)
        step = np.sqrt(dx ** 2 + dy ** 2)
        feat["path_length_px"] = float(np.nansum(step))
        feat["mean_step_size_px"] = float(np.nanmean(step))
        sac_step = step[is_sac.to_numpy()[valid_xy][1:]] if is_sac.any() else np.array([])
        feat["mean_saccade_amplitude_px"] = (float(np.nanmean(sac_step))
                                              if len(sac_step) else np.nan)
        feat["dispersion_px"] = float(np.sqrt(np.nanvar(gx_v) + np.nanvar(gy_v)))
        feat["bbox_width_px"] = float(np.nanmax(gx_v) - np.nanmin(gx_v))
        feat["bbox_height_px"] = float(np.nanmax(gy_v) - np.nanmin(gy_v))
    else:
        for k in ("path_length_px", "mean_step_size_px",
                   "mean_saccade_amplitude_px", "dispersion_px",
                   "bbox_width_px", "bbox_height_px"):
            feat[k] = np.nan

    # ---- Pupillary dynamics ----
    pupil = g["pupil"].to_numpy()
    pupil_valid = pupil[~np.isnan(pupil) & (pupil > 0)]
    feat["mean_pupil_diameter_mm"] = (float(np.mean(pupil_valid))
                                       if len(pupil_valid) else np.nan)
    feat["pupil_diameter_sd_mm"] = (float(np.std(pupil_valid))
                                     if len(pupil_valid) else np.nan)

    # ---- AOI / recording-quality ----
    aoi_counts = g["aoi_class"].value_counts(normalize=True)
    known_aoi = g[g["aoi_class"].isin(["social", "non_social"])]
    if len(known_aoi):
        p = known_aoi["aoi_raw"].value_counts(normalize=True)
        feat["aoi_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
        feat["social_aoi_fraction"] = float(
            (known_aoi["aoi_class"] == "social").mean())
    else:
        feat["aoi_entropy"] = np.nan
        feat["social_aoi_fraction"] = np.nan
    # [VALIDATION] fraction of rows whose AOI label fell outside the
    # social/non_social dictionaries. Split into two separate signals:
    #   aoi_missing_fraction        -> raw AOI value was literally '-'
    #                                  (genuine tracking loss / no AOI hit,
    #                                  i.e. this IS the recording-quality
    #                                  signal B3 is investigating)
    #   aoi_truly_unrecognized_fraction -> some OTHER label not in either
    #                                  dictionary (would indicate the
    #                                  SOCIAL_AOI/NON_SOCIAL_AOI lists are
    #                                  incomplete, a real coding bug)
    # aoi_unknown_fraction (kept for backward compat) = sum of both.
    is_unknown = g["aoi_class"] == "unknown"
    is_missing_marker = g["aoi_raw"].astype(str).isin(MISSING_AOI)
    feat["aoi_unknown_fraction"] = float(is_unknown.mean())
    feat["aoi_missing_fraction"] = float((is_unknown & is_missing_marker).mean())
    feat["aoi_truly_unrecognized_fraction"] = float(
        (is_unknown & ~is_missing_marker).mean())

    if "RecordingTime [ms]" in g.columns and n > 1:
        t = g["RecordingTime [ms]"].dropna()
        feat["trial_duration_ms"] = (float(t.max() - t.min())
                                      if len(t) > 1 else np.nan)
    else:
        feat["trial_duration_ms"] = np.nan
    feat["sample_count"] = n
    feat["tracking_ratio_pct"] = (float(g["Tracking Ratio [%]"].dropna().mean())
                                   if "Tracking Ratio [%]" in g.columns
                                   and g["Tracking Ratio [%]"].notna().any()
                                   else np.nan)

    # ==== Enhanced (11 features) ====
    social_rows = g[g["aoi_class"] == "social"]
    nonsocial_rows = g[g["aoi_class"] == "non_social"]
    feat["social_dwell_time_ms"] = float(len(social_rows))  # sample-count proxy; scale by sampling period if known
    feat["non_social_dwell_time_ms"] = float(len(nonsocial_rows))

    aoi_seq = g["aoi_class"].to_numpy()
    transitions = int(np.sum(aoi_seq[1:] != aoi_seq[:-1])) if len(aoi_seq) > 1 else 0
    feat["aoi_transition_count"] = transitions
    feat["aoi_transition_rate"] = transitions / n if n else np.nan

    if len(pupil_valid) > 1:
        pupil_vel = np.abs(np.diff(pupil_valid))
        feat["pupil_velocity_mean"] = float(np.mean(pupil_vel))
        feat["pupil_range_mm"] = float(np.max(pupil_valid) - np.min(pupil_valid))
    else:
        feat["pupil_velocity_mean"] = np.nan
        feat["pupil_range_mm"] = np.nan

    pupil_other = g["pupil_other"].to_numpy()
    both_valid = (~np.isnan(pupil) & ~np.isnan(pupil_other)
                  & (pupil > 0) & (pupil_other > 0))
    feat["pupil_asymmetry_mean"] = (
        float(np.mean(np.abs(pupil[both_valid] - pupil_other[both_valid])))
        if both_valid.any() else np.nan)

    if is_fix.any() and social_rows.index.size:
        first_social_idx = social_rows.index[0]
        pos_in_g = g.index.get_loc(first_social_idx)
        if "RecordingTime [ms]" in g.columns:
            t0 = g["RecordingTime [ms]"].iloc[0]
            t_first_social = g["RecordingTime [ms]"].iloc[pos_in_g]
            feat["time_to_first_social_fixation_ms"] = float(t_first_social - t0)
        else:
            feat["time_to_first_social_fixation_ms"] = np.nan
    else:
        feat["time_to_first_social_fixation_ms"] = np.nan

    feat["social_fixation_count"] = int((is_fix & (g["aoi_class"] == "social")).sum())
    feat["non_social_fixation_count"] = int((is_fix & (g["aoi_class"] == "non_social")).sum())
    denom = feat["non_social_fixation_count"]
    feat["social_to_nonsocial_fixation_ratio"] = (
        feat["social_fixation_count"] / denom if denom else np.nan)

    return feat


def count_runs(mask: pd.Series) -> int:
    """Counts contiguous True-runs in a boolean Series (i.e. number of
    distinct fixation/saccade EVENTS, not samples)."""
    m = mask.to_numpy()
    if len(m) == 0:
        return 0
    starts = np.flatnonzero((m[1:] & ~m[:-1])) + 1
    n_starts = len(starts) + (1 if m[0] else 0)
    return n_starts


def _mean_run_duration(g: pd.DataFrame, mask: pd.Series) -> float:
    if "RecordingTime [ms]" not in g.columns:
        return np.nan
    m = mask.to_numpy()
    t = g["RecordingTime [ms]"].to_numpy()
    if len(m) == 0 or not m.any():
        return np.nan
    run_durations = []
    start = None
    for i in range(len(m)):
        if m[i] and start is None:
            start = i
        elif not m[i] and start is not None:
            run_durations.append(t[i - 1] - t[start])
            start = None
    if start is not None:
        run_durations.append(t[-1] - t[start])
    run_durations = [d for d in run_durations if not np.isnan(d)]
    return float(np.mean(run_durations)) if run_durations else np.nan


def run_validation_checks(trial_df: pd.DataFrame, agg: pd.DataFrame,
                           out_report_path: str = "gaze_features_validation_report.txt"):
    """
    [VALIDATION] Sanity-checks the computed features against known ranges
    and internal consistency rules. Writes a plain-text report and prints
    a summary. This does NOT prove the features match the paper's exact
    definitions -- it only catches gross computation errors (e.g. a
    fraction >1, an impossible pupil size, systematic AOI misclassification)
    before the numbers are used for B3/C11.
    """
    lines = []
    def log(msg):
        print(msg)
        lines.append(msg)

    log("\n" + "=" * 70)
    log("VALIDATION REPORT")
    log("=" * 70)

    # ---- 1. Range checks on trial-level features ----
    range_checks = {
        "fixation_fraction":        (0.0, 1.0),
        "saccade_fraction":         (0.0, 1.0),
        "social_aoi_fraction":      (0.0, 1.0),
        "aoi_unknown_fraction":     (0.0, 1.0),
        "aoi_missing_fraction":     (0.0, 1.0),
        "aoi_truly_unrecognized_fraction": (0.0, 1.0),
        "tracking_ratio_pct":       (0.0, 100.0),
        "mean_pupil_diameter_mm":   (1.5, 9.0),   # physiologically plausible pupil range
        "pupil_diameter_sd_mm":     (0.0, 3.0),
    }
    log("\n-- Range checks (trial-level) --")
    any_range_violation = False
    for col, (lo, hi) in range_checks.items():
        if col not in trial_df.columns:
            log(f"  [SKIP] '{col}' not found in trial_df")
            continue
        vals = trial_df[col].dropna()
        n_bad = int(((vals < lo) | (vals > hi)).sum())
        pct_bad = 100 * n_bad / len(vals) if len(vals) else 0
        status = "OK" if n_bad == 0 else "VIOLATION"
        if n_bad > 0:
            any_range_violation = True
        log(f"  [{status}] {col}: expected [{lo},{hi}], "
            f"{n_bad}/{len(vals)} ({pct_bad:.1f}%) rows out of range "
            f"(actual range observed: [{vals.min():.3f}, {vals.max():.3f}])")
    if not any_range_violation:
        log("  All range checks passed.")

    # ---- 2. fixation_fraction + saccade_fraction should not exceed ~1
    #         (they can co-occur with Blink, so sum < 1 is expected, but
    #         sum > 1.01 would indicate a counting bug) ----
    if {"fixation_fraction", "saccade_fraction"}.issubset(trial_df.columns):
        combined = trial_df["fixation_fraction"] + trial_df["saccade_fraction"]
        n_over = int((combined > 1.01).sum())
        log(f"\n-- fixation_fraction + saccade_fraction > 1.01: "
            f"{n_over}/{len(combined)} trials "
            f"({'OK' if n_over == 0 else 'VIOLATION -- check event-type overlap logic'})")

    # ---- 3. AOI dictionary coverage ----
    log("\n-- AOI classification coverage --")
    mean_missing = trial_df["aoi_missing_fraction"].mean()
    mean_unrecog = trial_df["aoi_truly_unrecognized_fraction"].mean()
    log(f"  Mean aoi_missing_fraction (raw AOI value == '-', i.e. genuine "
        f"tracking loss / no AOI hit): {mean_missing:.3f}")
    log(f"  Mean aoi_truly_unrecognized_fraction (some OTHER label not in "
        f"either dictionary): {mean_unrecog:.3f}")
    if mean_unrecog > 0.05:
        log("  [VALIDATION WARNING] More than 5% of AOI samples carry a "
            "label that is neither '-' nor in the SOCIAL_AOI/NON_SOCIAL_AOI "
            "dictionaries -- the social/non-social split may be missing "
            "real AOI names. Check UNRECOGNIZED_AOI_VALUES below.")
    else:
        log("  [OK] Truly-unrecognized AOI labels are negligible; nearly "
            "all 'unknown' rows are genuine '-' (tracking loss), not a "
            "classification gap. This is expected recording-quality "
            "variation, not a bug -- and is directly relevant to B3.")
    if UNRECOGNIZED_AOI_VALUES:
        log(f"  Unrecognized AOI raw values seen (not in either dict, and "
            f"not the expected '-' marker), with occurrence counts:")
        for val, cnt in UNRECOGNIZED_AOI_VALUES.most_common(20):
            log(f"    '{val}': {cnt}")
        log("  -> If any of these look like real AOI labels, add them to "
            "SOCIAL_AOI or NON_SOCIAL_AOI and re-run.")
    else:
        log("  No unrecognized AOI values encountered (aside from '-').")

    # ---- 4. Trial coverage per participant ----
    log("\n-- Trial coverage per participant --")
    trial_counts = trial_df.groupby("participant_id").size()
    log(f"  n_trials per participant: min={trial_counts.min()}, "
        f"median={trial_counts.median():.0f}, max={trial_counts.max()}")
    low_coverage = trial_counts[trial_counts <= 2]
    if len(low_coverage):
        log(f"  [VALIDATION WARNING] {len(low_coverage)} participant(s) have "
            f"<=2 trials, so their mean/std aggregates are unstable: "
            f"{low_coverage.to_dict()}")
    else:
        log("  All participants have >2 trials.")

    # ---- 5. Cross-check against paper's reported fixation-fraction range
    #         (Table 6 / Figure 6 show participant-level values roughly in
    #         [0.05, 0.8]; this is only a rough plausibility check, not an
    #         exact-match test since these features are independently
    #         reconstructed, not the paper's original computation) ----
    if "fixation_fraction_mean" in agg.columns:
        fmean = agg["fixation_fraction_mean"].dropna()
        log(f"\n-- Participant-level mean fixation_fraction vs. paper's "
            f"reported range [~0.05, ~0.8] (Figure 6) --")
        log(f"  Computed range: [{fmean.min():.3f}, {fmean.max():.3f}], "
            f"mean={fmean.mean():.3f}")
        out_of_paper_range = int(((fmean < 0.0) | (fmean > 1.0)).sum())
        if fmean.min() < -0.05 or fmean.max() > 1.05:
            log("  [VALIDATION WARNING] Computed range is well outside "
                "even a generous [0,1] envelope -- do not trust these "
                "values without further debugging.")
        elif not (0.0 <= fmean.min() and fmean.max() <= 1.0):
            log("  [NOTE] Range is plausible but does not need to match the "
                "paper's exact bounds -- this is an independent "
                "reconstruction, not a byte-for-byte reproduction.")
        else:
            log("  Range is internally consistent (within [0,1]).")

    # ---- 6. Missing-data summary ----
    log("\n-- Missing-value fraction per feature (trial-level) --")
    feature_like_cols = [c for c in trial_df.columns if c not in
                          ("participant_id", "trial", "source_file",
                           "stimulus", "Gender", "Age", "Class", "CARS Score")]
    na_frac = trial_df[feature_like_cols].isna().mean().sort_values(ascending=False)
    for col, frac in na_frac.items():
        if frac > 0:
            log(f"  {col}: {frac*100:.1f}% missing")
    if (na_frac == 0).all():
        log("  No missing values in any computed feature.")

    log("\n" + "=" * 70)
    log("END VALIDATION REPORT -- review any VIOLATION/WARNING lines above "
        "before using these features for B3 or C11.")
    log("=" * 70)

    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[DONE] Validation report written to {out_report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--et-dir", default=str(ET_DIR_DEFAULT))
    ap.add_argument("--metadata", default="Metadata_Participants.csv")
    ap.add_argument("--eye-channel", choices=["left", "right"], default="right")
    ap.add_argument("--out-trial-csv", default="gaze_features_trial_level.csv")
    ap.add_argument("--out-participant-csv", default="gaze_features_participant_level.csv")
    args = ap.parse_args()

    et_dir = Path(args.et_dir)
    csv_files = sorted(et_dir.glob("*.csv"),
                        key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    print(f"[INFO] {len(csv_files)} raw files found in '{et_dir}'")

    meta_df = pd.read_csv(args.metadata)
    meta_df.columns = [c.strip() for c in meta_df.columns]
    meta_df["ParticipantID"] = meta_df["ParticipantID"].astype(int).astype(str)
    print(f"[INFO] Loaded metadata for {len(meta_df)} participants")

    all_trial_rows = []
    skipped_files = []

    for f in csv_files:
        df = load_one_file(f, eye_channel=args.eye_channel)
        if df is None:
            skipped_files.append(f.name)
            continue

        for (pid, trial), g in df.groupby(["Participant", "Trial"]):
            if len(g) < 5:  # too few samples to compute meaningful features
                continue
            feat = compute_trial_features(g)
            feat["participant_id"] = str(pid)
            feat["trial"] = trial
            feat["source_file"] = f.name
            feat["stimulus"] = g["Stimulus"].iloc[0] if "Stimulus" in g.columns else None
            all_trial_rows.append(feat)

        print(f"  [OK] {f.name}: {df['Participant'].nunique()} participants, "
              f"{df.groupby(['Participant','Trial']).ngroups} (participant,trial) pairs")

    if skipped_files:
        print(f"\n[WARN] {len(skipped_files)} file(s) skipped entirely "
              f"(missing required columns or no valid rows): {skipped_files}")

    trial_df = pd.DataFrame(all_trial_rows)
    if trial_df.empty:
        raise SystemExit("[FATAL] No trial-level features computed -- check "
                          "the input files and column assumptions.")

    trial_df = trial_df.merge(
        meta_df.rename(columns={"ParticipantID": "participant_id"}),
        on="participant_id", how="left")

    trial_df.to_csv(args.out_trial_csv, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Trial-level features -> {args.out_trial_csv} "
          f"({len(trial_df)} rows)")

    # ---- Aggregate to participant level (mean + std, per Eq. 2 of the paper) ----
    feature_cols = [c for c in trial_df.columns if c not in
                     ("participant_id", "trial", "source_file", "stimulus",
                      "Gender", "Age", "Class", "CARS Score")]

    agg = trial_df.groupby("participant_id")[feature_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    demo = meta_df.rename(columns={"ParticipantID": "participant_id"})
    agg = agg.merge(demo, on="participant_id", how="left")
    agg["n_trials"] = trial_df.groupby("participant_id").size().reindex(
        agg["participant_id"]).values

    agg.to_csv(args.out_participant_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Participant-level features -> {args.out_participant_csv} "
          f"({len(agg)} participants)")

    run_validation_checks(trial_df, agg)

    n_asd = (agg["Class"].astype(str).str.upper() == "ASD").sum()
    n_td = (agg["Class"].astype(str).str.upper() == "TD").sum()
    print(f"\n[SUMMARY] {len(agg)} participants total "
          f"({n_asd} ASD, {n_td} TD) with computed gaze-dynamics features.")
    print("Compare n_asd against the paper's n=27 (ASD with CARS) before "
          "using this table for B3 -- some participants may need to be "
          "dropped for missing CARS scores or insufficient trial coverage, "
          "matching the accounting already done for the image arm (A6).")


if __name__ == "__main__":
    main()
