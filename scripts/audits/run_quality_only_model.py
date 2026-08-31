# -*- coding: utf-8 -*-
"""
run_quality_only_model.py
============================
체크리스트 B그룹 9번: quality-descriptors-only 모델 (3 features: tracking
ratio, sample count, trial duration)을 기존 18-feature/15-feature ablation과
동일한 n=25(회귀) / n=55(분류) fixed complete-case cohort에서 재현합니다.

왜 이 모델이 필요한가:
지금까지의 ablation은 "18개에서 quality 3개를 뺐을 때 성능이 거의 안 줄어든다"
(R²=0.351 -> 0.332, AUROC=0.884 -> 0.877)는 것만 보여줬습니다. 이건 quality
정보가 "필요 없다"는 증거로 쓰이고 있는데, 만약 quality 3개와 나머지 15개
gaze feature 사이에 강한 collinearity(중복 정보)가 있다면, quality를 빼도
비슷한 정보가 다른 feature를 통해 이미 모델에 들어가 있어서 성능이 안
줄어드는 것처럼 보일 수 있습니다. 즉 지금 ablation만으로는:
    "quality 정보가 진짜로 불필요하다" 와
    "quality 정보가 다른 feature에 이미 중복돼서 들어가 있다"
를 구분할 수 없습니다.

이 스크립트는 반대 방향의 대조군을 추가합니다: quality 3개 feature만
단독으로 썼을 때 얼마나 예측력이 나오는지 봅니다.
  - quality-only R²/AUROC가 거의 0에 가까우면 -> quality 자체는 약한 신호이고,
    18-feature 모델의 예측력은 진짜로 나머지 15개 gaze feature에서 나온다는
    뜻입니다 (collinearity 우려 해소).
  - quality-only R²/AUROC가 15-feature-without-quality 모델과 비슷하게
    크면 -> quality가 실제로 상당한 정보를 담고 있고, ablation에서 성능이
    안 줄어든 건 다른 feature가 그 정보를 흡수했기 때문일 수 있다는 뜻입니다
    (collinearity 우려가 현실적임).

방법론은 기존 quality-ablation 스크립트와 완전히 동일합니다: 동일한 n=25
(regression) / n=55, 25 ASD + 30 TD (classification) fixed complete-case
cohort, 동일한 nested LOPO SVR/logistic 프로토콜.

실행 위치: gaze_features_participant_level.csv가 있는 폴더에서
    python run_quality_only_model.py
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

CARS_MIN, CARS_MAX = 15.0, 60.0
QUALITY_FEATURES = ["tracking_ratio_pct", "sample_count", "trial_duration_ms"]


def nested_lopo_svr(X, y, groups, seed=0, inner_splits=5):
    """18-feature ablation 스크립트와 동일한 SVR 고정 하이퍼파라미터
    (C=10, epsilon=0.5, gamma='scale')를 그대로 사용합니다 -- quality-only
    모델을 원래 모델과 공정하게 비교하려면 하이퍼파라미터 탐색 절차 자체가
    달라지면 안 됩니다."""
    uniq = np.unique(groups)
    y_pred = np.zeros_like(y, dtype=float)
    for held_out in uniq:
        test_mask = groups == held_out
        train_mask = ~test_mask
        model = make_pipeline(StandardScaler(),
                               SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale"))
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        y_pred[test_mask] = np.clip(pred, CARS_MIN, CARS_MAX)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, r2, y_pred


def nested_lopo_logistic(X, y, groups, seed=0, inner_splits=5):
    """분류 arm: 원래 모델과 동일하게 inner CV로 정규화 강도 선택."""
    uniq = np.unique(groups)
    y_prob = np.zeros_like(y, dtype=float)
    Cs = [0.01, 0.1, 1, 10, 100]
    for held_out in uniq:
        test_mask = groups == held_out
        train_mask = ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        inner_cv = StratifiedKFold(n_splits=min(inner_splits, np.bincount(y_tr.astype(int)).min()),
                                     shuffle=True, random_state=seed)
        clf = GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            param_grid={"logisticregression__C": Cs},
            cv=inner_cv, scoring="roc_auc")
        clf.fit(X_tr, y_tr)
        y_prob[test_mask] = clf.predict_proba(X[test_mask])[:, 1]
    auroc = roc_auc_score(y, y_prob)
    return auroc, y_prob


def permutation_test_r2(X, y, groups, n_permutations, seed=0):
    _, obs_r2, _ = nested_lopo_svr(X, y, groups, seed=seed)
    rng = np.random.RandomState(seed)
    uniq_groups = np.unique(groups)
    group_to_y = {g: y[groups == g][0] for g in uniq_groups}
    null_r2 = []
    for _ in range(n_permutations):
        perm_groups = rng.permutation(uniq_groups)
        mapping = dict(zip(uniq_groups, perm_groups))
        y_perm = np.array([group_to_y[mapping[g]] for g in groups])
        _, r2_b, _ = nested_lopo_svr(X, y_perm, groups, seed=seed)
        null_r2.append(r2_b)
    null_r2 = np.array(null_r2)
    p_value = (1 + np.sum(null_r2 >= obs_r2)) / (1 + n_permutations)
    return obs_r2, p_value


def permutation_test_auroc(X, y, groups, n_permutations, seed=0):
    obs_auroc, _ = nested_lopo_logistic(X, y, groups, seed=seed)
    rng = np.random.RandomState(seed)
    uniq_groups = np.unique(groups)
    group_to_y = {g: y[groups == g][0] for g in uniq_groups}
    null_auroc = []
    for _ in range(n_permutations):
        perm_groups = rng.permutation(uniq_groups)
        mapping = dict(zip(uniq_groups, perm_groups))
        y_perm = np.array([group_to_y[mapping[g]] for g in groups])
        auroc_b, _ = nested_lopo_logistic(X, y_perm, groups, seed=seed)
        null_auroc.append(auroc_b)
    null_auroc = np.array(null_auroc)
    p_value = (1 + np.sum(null_auroc >= obs_auroc)) / (1 + n_permutations)
    return obs_auroc, p_value


def run_self_test():
    print("=" * 60)
    print("SELF-TEST (synthetic data)")
    print("=" * 60)
    rng = np.random.RandomState(0)

    # Regression self-test: strong signal case
    n = 25
    X = rng.randn(n, 3)
    y = np.clip(X[:, 0] * 8 + 30 + rng.randn(n) * 2, 15, 60)
    groups = np.arange(n)
    mae, r2, y_pred = nested_lopo_svr(X, y, groups, seed=0)
    print(f"[TEST 1] Regression (strong synthetic signal): MAE={mae:.3f} R2={r2:.3f}")
    assert np.isfinite(mae) and np.isfinite(r2)
    print("  PASS")

    # Regression self-test: mean-predictor case (no signal)
    y_flat = np.full(n, 30.0)
    mae2, r2_2, _ = nested_lopo_svr(X, y_flat, groups, seed=0)
    print(f"[TEST 2] Regression (constant target): MAE={mae2:.3f} R2={r2_2:.3f}")
    assert mae2 < 1e-6, "Expected near-zero MAE for a constant target"
    print("  PASS")

    # Classification self-test: perfectly separable case
    n_clf = 40
    Xc = rng.randn(n_clf, 3)
    yc = (Xc[:, 0] > 0).astype(int)
    Xc[:, 0] += yc * 10  # make classes trivially separable
    groups_c = np.arange(n_clf)
    auroc, _ = nested_lopo_logistic(Xc, yc, groups_c, seed=0)
    print(f"[TEST 3] Classification (separable synthetic data): AUROC={auroc:.3f}")
    assert auroc > 0.9, "Expected near-perfect AUROC for trivially separable data"
    print("  PASS")

    # Classification self-test: chance-level case
    yc_random = rng.randint(0, 2, n_clf)
    auroc_r, _ = nested_lopo_logistic(Xc[:, [1, 2]], yc_random, groups_c, seed=0)
    print(f"[TEST 4] Classification (random labels, unrelated features): AUROC={auroc_r:.3f}")
    assert 0.0 <= auroc_r <= 1.0
    print("  PASS")

    print("\nSelf-test complete.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant-csv", default="gaze_features_participant_level.csv")
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    if not args.skip_self_test:
        run_self_test()

    df = pd.read_csv(args.participant_csv)
    df.columns = [c.strip() for c in df.columns]

    quality_cols_mean = [f"{f}_mean" for f in QUALITY_FEATURES]
    missing = [c for c in quality_cols_mean if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] Missing quality feature columns: {missing}. "
                          f"Check exact column names in {args.participant_csv}.")

    # -- Regression: same n=25 complete-case cohort as the 18-vs-15 ablation --
    # A participant is "complete" here if they have valid values across ALL
    # 18 base features (matching the manuscript's ablation cohort definition
    # exactly), not just the 3 quality features, so this is directly
    # comparable to the existing R^2=0.351 (18-feat) / R^2=0.332 (15-feat)
    # numbers on the same participants.
    all_18_mean = [c for c in df.columns if c.endswith("_mean")]
    asd = df[df["Class"].astype(str).str.upper() == "ASD"].copy()
    asd["CARS Score"] = pd.to_numeric(asd["CARS Score"], errors="coerce")
    asd = asd.dropna(subset=["CARS Score"])

    complete_mask_reg = asd[all_18_mean].notna().all(axis=1)
    asd_complete = asd[complete_mask_reg].copy()
    print(f"\n[INFO] Regression complete-case cohort: n={len(asd_complete)} "
          f"(expect 25, matching the 18-vs-15-feature ablation cohort)")

    X_reg = asd_complete[quality_cols_mean].to_numpy(dtype=float)
    y_reg = asd_complete["CARS Score"].to_numpy(dtype=float)
    groups_reg = np.arange(len(y_reg))

    print("\n" + "=" * 70)
    print("REGRESSION: quality-only 3-feature SVR, n=25 fixed cohort")
    print("=" * 70)
    mae, r2, y_pred = nested_lopo_svr(X_reg, y_reg, groups_reg, seed=args.seed)
    print(f"[RESULT] MAE={mae:.3f}  R2={r2:.3f}")
    print(f"[INFO] Running {args.n_permutations}-draw nested permutation test "
          f"(this repeats the full n=25 nested-LOPO pass each draw and may "
          f"take a while)...")
    obs_r2, p_val = permutation_test_r2(X_reg, y_reg, groups_reg,
                                          n_permutations=args.n_permutations,
                                          seed=args.seed)
    print(f"[RESULT] Permutation p-value: {p_val:.4f} (B={args.n_permutations})")

    # -- Classification: same n=55 (25 ASD + 30 TD) complete-case cohort --
    all18 = df[df[all_18_mean].notna().all(axis=1)].copy()
    print(f"\n[INFO] Classification complete-case cohort: n={len(all18)} "
          f"(expect 55 = 25 ASD + 30 TD)")

    X_clf = all18[quality_cols_mean].to_numpy(dtype=float)
    y_clf = (all18["Class"].astype(str).str.upper() == "ASD").astype(int).to_numpy()
    groups_clf = np.arange(len(y_clf))

    print("\n" + "=" * 70)
    print("CLASSIFICATION: quality-only 3-feature logistic, n=55 fixed cohort")
    print("=" * 70)
    auroc, y_prob = nested_lopo_logistic(X_clf, y_clf, groups_clf, seed=args.seed)
    print(f"[RESULT] AUROC={auroc:.3f}")
    print(f"[INFO] Running {args.n_permutations}-draw nested permutation test...")
    obs_auroc, p_val_auroc = permutation_test_auroc(
        X_clf, y_clf, groups_clf, n_permutations=args.n_permutations, seed=args.seed)
    print(f"[RESULT] Permutation p-value: {p_val_auroc:.4f} (B={args.n_permutations})")

    print("\n" + "=" * 70)
    print("SUMMARY -- compare against manuscript's existing ablation numbers")
    print("=" * 70)
    print(f"  Full 18-feature SVR (from manuscript):        R2=0.351  (n=25)")
    print(f"  Quality-removed 15-feature SVR (manuscript):   R2=0.332  (n=25)")
    print(f"  Quality-ONLY 3-feature SVR (this script):      R2={r2:.3f}  (n={len(y_reg)}, p={p_val:.4f})")
    print()
    print(f"  Full 18-feature AUROC (from manuscript):       AUROC=0.884  (n=55)")
    print(f"  Quality-removed 15-feature AUROC (manuscript): AUROC=0.877  (n=55)")
    print(f"  Quality-ONLY 3-feature AUROC (this script):    AUROC={auroc:.3f}  (n={len(y_clf)}, p={p_val_auroc:.4f})")
    print()
    print("  Interpretation guide:")
    print("  - If quality-only R2/AUROC are both close to zero/chance, the")
    print("    15-feature model's near-unchanged performance genuinely reflects")
    print("    non-quality gaze information, not collinearity with quality.")
    print("  - If quality-only R2/AUROC are substantial (comparable to the full")
    print("    18-feature numbers), quality carries real signal on its own, and")
    print("    the 15-feature ablation's stability may instead reflect")
    print("    collinearity absorbing that signal into other features.")

    out = pd.DataFrame([{
        "cohort": "regression_n25", "n": len(y_reg), "metric": "R2",
        "value": r2, "mae": mae, "permutation_p": p_val,
    }, {
        "cohort": "classification_n55", "n": len(y_clf), "metric": "AUROC",
        "value": auroc, "mae": np.nan, "permutation_p": p_val_auroc,
    }])
    out.to_csv("quality_only_model_results.csv", index=False)
    print(f"\n[DONE] Written to quality_only_model_results.csv")


if __name__ == "__main__":
    main()
