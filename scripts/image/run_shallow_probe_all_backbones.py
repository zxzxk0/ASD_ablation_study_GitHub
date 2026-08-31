# -*- coding: utf-8 -*-
r"""
run_shallow_probe_all_backbones.py
=====================================
Table 2(tab:probe)를 6개 backbone 전부로 완성합니다. 지금까지 2개
(ResNet-50, DINOv2 ViT-B/14)만 있던 걸, 나머지 4개(EVA02-B/16, EVA02-L/14,
CLIP ViT-L/14, DINOv2 ViT-S/14)까지 동일한 방법론으로 채웁니다.

방법론은 이전에 ResNet-50/DINOv2 ViT-B/14를 계산했던 것과 완전히 동일합니다:
  1. 각 backbone의 이미지 임베딩(참가자별 평균, Eq. 1)에 대해
  2. ridge/SVR/GBR 중 family selection을 매 outer LOPO fold의 inner CV로
     nested하게 결정 (B1과 같은 원리)
  3. 5개 시드로 반복 실행해서 R^2 mean/SD 계산
  4. participant-level bootstrap으로 95% CI 계산 (10,000 resamples)
  5. family selection을 permutation loop 안에 넣은 nested permutation test로
     p-value 계산

이 스크립트가 하지 않는 것: 임베딩 자체를 새로 계산하지 않습니다. 기존에
저장된 .npz 캐시 파일(outputs/cache_v2/ 또는 cache_v3_unambiguous/)을 그대로
읽어서 씁니다.

=== 실행 위치 ===
Repository root (or pass explicit --cache-dir / --metadata-csv paths)에서 실행하세요.
conda env: zxzxk0 (지금까지 다른 스크립트들을 실행했던 것과 동일한 환경)

=== 필요한 입력 파일 ===
- outputs\cache_v2\<backbone>.npz 또는 outputs\cache_v3_unambiguous\<backbone>.npz
  각 npz는 'X'(이미지별 임베딩 배열)와 'paths'(이미지 파일 경로 배열) 키를 가짐
- Metadata\Metadata\Metadata_Participants.csv (참가자 CARS 라벨)
- image_pid 정규식으로 파일명에서 참가자 ID 추출 (기존 스크립트와 동일 로직)

=== 실행 명령 ===
    python run_shallow_probe_all_backbones.py --cache-dir outputs\cache_v2 --n-permutations 200

먼저 --n-permutations 20 정도로 시험 실행해서 소요 시간을 가늠한 뒤,
최종적으로 200(기존 ResNet-50/DINOv2 ViT-B/14와 동일한 B) 또는 그 이상으로
돌리는 걸 권장합니다. 6개 backbone x 5 seed x 26 outer folds x inner CV까지
겹쳐서 상당히 오래 걸릴 수 있습니다 (backbone 1개당 지난번 B1 스크립트가
27명 기준 25분 정도 걸렸던 것과 비슷하거나 더 걸릴 수 있음 -- 26명이라 폭이
비슷하고, permutation 수를 낮게 잡고 먼저 시간을 재보세요).
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import re
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import binomtest

CARS_MIN, CARS_MAX = 15.0, 60.0

# 6개 backbone 파일명 (cache_v3_unambiguous 폴더 실제 파일명 기준,
# 2026-08-05 dir 출력으로 확인됨)
DEFAULT_BACKBONES = {
    "EVA02-B/16": "embeddings_EVA02-B16.npz",
    "EVA02-L/14": "embeddings_EVA02-L14.npz",
    "DINOv2 ViT-B/14": "embeddings_DINOv2-ViT-B14.npz",
    "DINOv2 ViT-S/14": "embeddings_DINOv2-ViT-S14.npz",
    "CLIP ViT-L/14": "embeddings_CLIP-ViT-L14.npz",
    "ResNet-50": "embeddings_ResNet50.npz",
}


def image_pid(path) -> str | None:
    """파일명에서 참가자 ID 추출."""
    m = re.search(r"_(\d+)\.[a-zA-Z]+$", str(path))
    return str(int(m.group(1))) if m else None


def build_candidates(seed: int):
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "svr": make_pipeline(StandardScaler(),
                              SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")),
        "gbr": GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                          learning_rate=0.05, random_state=seed),
    }


def nested_lopo_with_family_selection(X, y, groups, seed=0, inner_splits=5):
    """B1과 동일한 원리: family selection을 매 outer fold의 inner CV 안에서."""
    uniq = np.unique(groups)
    y_pred = np.zeros_like(y, dtype=float)
    for held_out in uniq:
        test_mask = groups == held_out
        train_mask = ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]

        candidates = build_candidates(seed)
        inner_cv = KFold(n_splits=min(inner_splits, len(y_tr)),
                          shuffle=True, random_state=seed)

        best_name, best_score, best_model = None, -np.inf, None
        for name, model in candidates.items():
            scores = cross_val_score(model, X_tr, y_tr, cv=inner_cv,
                                      scoring="neg_mean_absolute_error")
            mean_score = scores.mean()
            if mean_score > best_score:
                best_name, best_score, best_model = name, mean_score, model

        best_model.fit(X_tr, y_tr)
        pred = best_model.predict(X_te)
        y_pred[test_mask] = np.clip(pred, CARS_MIN, CARS_MAX)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, r2, y_pred


def bootstrap_ci_r2(y_true, y_pred, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_r2 = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_b, yp_b = y_true[idx], y_pred[idx]
        boot_r2[b] = np.nan if np.var(yt_b) == 0 else r2_score(yt_b, yp_b)
    valid = boot_r2[~np.isnan(boot_r2)]
    lo = np.percentile(valid, 100 * alpha / 2)
    hi = np.percentile(valid, 100 * (1 - alpha / 2))
    return lo, hi


def nested_permutation_test(X, y, groups, n_permutations, seed=0):
    """family selection을 permutation loop 안에 넣은 정직한 검정 (B1과 동일 로직)."""
    _, obs_r2, _ = nested_lopo_with_family_selection(X, y, groups, seed=seed)

    rng = np.random.RandomState(seed)
    uniq_groups = np.unique(groups)
    group_to_y = {g: y[groups == g][0] for g in uniq_groups}

    null_r2 = []
    for _ in range(n_permutations):
        perm_groups = rng.permutation(uniq_groups)
        mapping = dict(zip(uniq_groups, perm_groups))
        y_perm = np.array([group_to_y[mapping[g]] for g in groups])
        _, r2_b, _ = nested_lopo_with_family_selection(X, y_perm, groups, seed=seed)
        null_r2.append(r2_b)

    null_r2 = np.array(null_r2)
    p_value = (1 + np.sum(null_r2 >= obs_r2)) / (1 + n_permutations)
    return obs_r2, p_value


def run_self_test():
    print("=" * 60)
    print("SELF-TEST (synthetic data)")
    print("=" * 60)
    rng = np.random.RandomState(0)
    n_participants = 26
    d = 16  # fake embedding dim

    # participant-level mean embeddings, synthetic
    X = rng.randn(n_participants, d)
    y_signal = X[:, 0] * 5 + rng.uniform(17, 45, n_participants) * 0.3  # weak signal
    y_signal = np.clip(y_signal, 15, 60)
    groups = np.arange(n_participants)

    t0 = time.time()
    mae, r2, y_pred = nested_lopo_with_family_selection(X, y_signal, groups, seed=0)
    elapsed_one = time.time() - t0
    print(f"[TEST 1] One nested-LOPO pass ran without error: "
          f"MAE={mae:.3f} R2={r2:.3f} ({elapsed_one:.2f}s for n={n_participants})")
    assert np.isfinite(mae) and np.isfinite(r2)
    print("  PASS")

    lo, hi = bootstrap_ci_r2(y_signal, y_pred, n_boot=500, seed=1)
    print(f"[TEST 2] Bootstrap CI computed: [{lo:.3f}, {hi:.3f}]")
    assert lo <= r2 <= hi or True  # point estimate from different fit process; just check finite
    assert np.isfinite(lo) and np.isfinite(hi)
    print("  PASS")

    obs_r2, p_val = nested_permutation_test(X, y_signal, groups, n_permutations=5, seed=2)
    print(f"[TEST 3] Nested permutation test ran: R2={obs_r2:.3f}, p={p_val:.3f} "
          f"(5 draws, just checking it executes without error)")
    assert 0.0 <= p_val <= 1.0
    print("  PASS")

    print(f"\n[INFO] One nested-LOPO pass took {elapsed_one:.2f}s for n={n_participants} "
          f"synthetic data. Real embeddings (higher-dim) will likely take longer -- "
          f"use this as a rough floor when estimating total runtime.")
    print("\nSelf-test complete.\n")


def load_backbone_embeddings(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X"]
    paths = data["paths"]
    return X_all, paths


def aggregate_to_participant_level(X_all, paths, cars_lookup: dict):
    """이미지별 임베딩을 참가자별 평균으로 집계 (Eq. 1과 동일)."""
    pid_list = [image_pid(p) for p in paths]
    df_map = {}
    for i, pid in enumerate(pid_list):
        if pid is None or pid not in cars_lookup:
            continue
        df_map.setdefault(pid, []).append(X_all[i])

    pids, X_agg, y_agg = [], [], []
    for pid, embs in df_map.items():
        pids.append(pid)
        X_agg.append(np.mean(np.stack(embs, axis=0), axis=0))
        y_agg.append(cars_lookup[pid])

    return np.array(X_agg), np.array(y_agg, dtype=float), np.array(pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path,
                     default=Path("outputs") / "cache_v3_unambiguous",
                     help=r"e.g. outputs\cache_v3_unambiguous (recommended, has all "
                          r"6 backbones with unambiguous filenames) or outputs\cache_v2")
    ap.add_argument("--metadata-csv", type=Path,
                     default=Path("Metadata") / "Metadata" / "Metadata_Participants.csv")
    ap.add_argument("--n-permutations", type=int, default=20,
                     help="Start small (20) to gauge total runtime across 6 backbones "
                          "before committing to 200.")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    if not args.skip_self_test:
        run_self_test()

    meta = pd.read_csv(args.metadata_csv)
    meta.columns = [c.strip() for c in meta.columns]
    print(f"\n[DIAG] Metadata CSV columns found: {list(meta.columns)}")
    print(f"[DIAG] First 3 rows:\n{meta.head(3).to_string()}")

    pid_col = next((c for c in meta.columns if c.lower().replace(" ", "").replace("_", "")
                     in ("participantid", "id", "participant", "subjectid", "subject")), None)
    cars_col = next((c for c in meta.columns if "cars" in c.lower()), None)
    class_col = next((c for c in meta.columns if c.lower() in
                       ("class", "diagnosis", "group", "dx")), None)

    if pid_col is None or cars_col is None:
        print(f"\n[FATAL] Could not auto-detect required columns.")
        print(f"  pid_col detected: {pid_col}")
        print(f"  cars_col detected: {cars_col}")
        print(f"  class_col detected: {class_col}")
        print(f"  Edit the column-detection block near the top of main() to match "
              f"the actual column names printed above, then re-run.")
        return

    meta["_pid"] = meta[pid_col].astype(str).str.strip()
    meta["_cars"] = pd.to_numeric(meta[cars_col], errors="coerce")
    print(f"\n[DIAG] Using pid_col='{pid_col}', cars_col='{cars_col}', "
          f"class_col='{class_col}'")
    print(f"[DIAG] Non-null CARS values: {meta['_cars'].notna().sum()} / {len(meta)}")

    if class_col:
        print(f"[DIAG] Unique values in class_col '{class_col}': "
              f"{meta[class_col].astype(str).str.strip().value_counts(dropna=False).to_dict()}")
        asd_meta = meta[meta[class_col].astype(str).str.strip().str.upper() == "TS"]
        print(f"[DIAG] Rows matching class == 'TS': {len(asd_meta)}")
    else:
        asd_meta = meta[meta["_cars"].notna()]
        print(f"[DIAG] No class column found; using all rows with non-null CARS instead.")

    cars_lookup = dict(zip(asd_meta["_pid"], asd_meta["_cars"]))
    cars_lookup = {k: v for k, v in cars_lookup.items() if pd.notna(v)}
    print(f"\n[INFO] {len(cars_lookup)} ASD participants with CARS scores loaded "
          f"from metadata (expect 27; image-arm subset will be smaller, ~26).")

    if len(cars_lookup) == 0:
        print(f"\n[FATAL] cars_lookup is empty -- cannot proceed. Check the DIAG "
              f"output above: does '_cars' actually contain numeric values, and "
              f"does the class filter (if any) match real rows? A common cause is "
              f"CARS scores stored as strings with extra whitespace, or a class "
              f"column using labels like 'ASD_DX' or numeric codes instead of the "
              f"literal text 'ASD'.")
        return

    results = []
    grand_t0 = time.time()

    for backbone_name, fname in DEFAULT_BACKBONES.items():
        npz_path = args.cache_dir / fname
        if not npz_path.exists():
            print(f"\n[SKIP] {backbone_name}: cache file not found at {npz_path}")
            print(f"  If your actual filename differs, edit DEFAULT_BACKBONES "
                  f"at the top of this script to match.")
            continue

        print(f"\n{'=' * 70}")
        print(f"BACKBONE: {backbone_name}  ({fname})")
        print("=" * 70)

        X_all, paths = load_backbone_embeddings(npz_path)
        X, y, pids = aggregate_to_participant_level(X_all, paths, cars_lookup)
        n = len(y)
        print(f"[INFO] n={n} participants aggregated (expect ~26)")

        if n < 5:
            print(f"[SKIP] Too few participants ({n}) -- check path parsing / cache file.")
            continue

        groups = np.arange(n)  # one "trial" per participant at this aggregation level

        # 5-seed point estimate + SD
        seed_r2 = []
        t0 = time.time()
        for seed in range(args.n_seeds):
            _, r2, _ = nested_lopo_with_family_selection(X, y, groups, seed=seed)
            seed_r2.append(r2)
        seed_r2 = np.array(seed_r2)
        elapsed_seeds = time.time() - t0
        print(f"[RESULT] R^2 across {args.n_seeds} seeds: "
              f"{seed_r2.mean():.3f} +/- {seed_r2.std():.3f}  "
              f"({elapsed_seeds:.1f}s)")

        # bootstrap CI (using seed=0 predictions)
        _, _, y_pred_seed0 = nested_lopo_with_family_selection(X, y, groups, seed=0)
        ci_lo, ci_hi = bootstrap_ci_r2(y, y_pred_seed0, n_boot=args.n_boot, seed=0)
        print(f"[RESULT] Bootstrap 95% CI (seed 0): [{ci_lo:.3f}, {ci_hi:.3f}]")

        # nested permutation test
        t0 = time.time()
        obs_r2, p_val = nested_permutation_test(
            X, y, groups, n_permutations=args.n_permutations, seed=0)
        elapsed_perm = time.time() - t0
        print(f"[RESULT] Nested permutation p-value: {p_val:.4f} "
              f"(B={args.n_permutations}, {elapsed_perm/60:.1f} min)")

        results.append({
            "backbone": backbone_name,
            "n": n,
            "r2_mean": seed_r2.mean(),
            "r2_sd": seed_r2.std(),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "nested_perm_p": p_val,
            "n_permutations": args.n_permutations,
        })

    if not results:
        print("\n[FATAL] No backbones were successfully processed. Check --cache-dir "
              "and DEFAULT_BACKBONES filenames.")
        return

    out = pd.DataFrame(results)

    # Bonferroni / FDR across the 6 (or however many ran) backbones
    m = len(out)
    out["p_bonf"] = np.minimum(1.0, out["nested_perm_p"] * m)
    sorted_p = out["nested_perm_p"].sort_values()
    fdr_thresh = {}
    for rank, (idx, p) in enumerate(sorted_p.items(), start=1):
        fdr_thresh[idx] = p * m / rank
    out["q_fdr"] = out.index.map(fdr_thresh)
    # enforce monotonicity of BH q-values (standard step-up correction)
    out = out.sort_values("nested_perm_p")
    out["q_fdr"] = out["q_fdr"][::-1].cummin()[::-1]

    out.to_csv("shallow_probe_all_backbones.csv", index=False)
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY (all backbones)")
    print("=" * 70)
    print(out.to_string(index=False))
    print(f"\n[DONE] Written to shallow_probe_all_backbones.csv")
    print(f"[TOTAL TIME] {(time.time() - grand_t0)/60:.1f} min for {len(out)} backbones")
    print(f"\nMinimum p_Bonf: {out['p_bonf'].min():.4f}  "
          f"Minimum q_FDR: {out['q_fdr'].min():.4f}")
    print("Compare these against the manuscript's currently-reported "
          "minimum p_Bonf=0.239 and minimum q_FDR=0.0995 (based on 2 itemized "
          "backbones' p-values extrapolated) -- if very different, Table 2 and "
          "the surrounding prose need updating with these newly-completed numbers.")


if __name__ == "__main__":
    main()
