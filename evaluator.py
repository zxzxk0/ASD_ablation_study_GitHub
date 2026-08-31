"""
evaluator.py — 회귀 평가 지표 계산 및 결과 저장
"""

import json
import os
import numpy as np
from typing import List, Optional, Dict, Any
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import RESULTS_DIR


def compute_metrics(
    y_true: List[float],
    y_pred: List[Optional[float]],
    model_name: str,
) -> Dict[str, Any]:
    """
    MAE, RMSE, R², Spearman 상관계수 계산.
    예측 실패(None)인 샘플은 평균값으로 대체하여 보수적으로 평가.
    """
    y_true = np.array(y_true)
    y_pred_raw = list(y_pred)

    # None 처리: 예측 실패한 샘플 수 기록 후 평균값 대체
    mean_true = float(np.mean(y_true))
    y_pred_filled = np.array([
        p if p is not None else mean_true
        for p in y_pred_raw
    ])

    n_failed  = sum(1 for p in y_pred_raw if p is None)
    n_total   = len(y_true)

    mae      = float(mean_absolute_error(y_true, y_pred_filled))
    rmse     = float(np.sqrt(mean_squared_error(y_true, y_pred_filled)))
    r2       = float(r2_score(y_true, y_pred_filled))
    spearman = float(spearmanr(y_true, y_pred_filled).statistic)

    metrics = {
        "model":          model_name,
        "n_total":        n_total,
        "n_failed":       n_failed,
        "mae":            round(mae,      4),
        "rmse":           round(rmse,     4),
        "r2":             round(r2,       4),
        "spearman":       round(spearman, 4),
        "y_true":         y_true.tolist(),
        "y_pred_raw":     [p if p is not None else None for p in y_pred_raw],
        "y_pred_filled":  y_pred_filled.tolist(),
    }
    return metrics


def print_metrics_table(all_results: List[Dict[str, Any]]) -> None:
    """모든 모델의 평가 지표를 비교 테이블로 출력."""
    print("\n" + "=" * 72)
    print("  ABLATION STUDY — LLM Baseline Comparison")
    print("=" * 72)
    header = f"{'Model':<22} {'MAE↓':>8} {'RMSE↓':>8} {'R²↑':>8} {'Spearman↑':>10} {'Failed':>8}"
    print(header)
    print("-" * 72)
    for r in all_results:
        row = (
            f"{r['model']:<22}"
            f"{r['mae']:>8.3f}"
            f"{r['rmse']:>8.3f}"
            f"{r['r2']:>8.3f}"
            f"{r['spearman']:>10.3f}"
            f"{r['n_failed']:>6}/{r['n_total']:<3}"
        )
        print(row)
    print("=" * 72)


def save_results(results: Dict[str, Any], model_key: str) -> str:
    """개별 모델 결과를 JSON으로 저장. 저장 경로 반환."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{model_key}_results.json")
    # y_true / y_pred는 numpy float → python float 변환
    to_save = {k: v for k, v in results.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)
    print(f"  Saved results → {path}")
    return path

def save_summary(all_results: List[Dict[str, Any]]) -> str:
    """전체 ablation 요약을 JSON으로 저장."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "ablation_summary.json")
    summary = [
        {k: v for k, v in r.items() if k not in ("y_true", "y_pred_raw", "y_pred_filled")}
        for r in all_results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved → {path}")
    return path


def save_csv(all_results: List[Dict[str, Any]]) -> str:
    """전체 ablation 결과를 CSV로 저장."""
    import csv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "ablation_results.csv")

    fieldnames = ["model", "n_shots", "mae", "rmse", "r2", "spearman", "n_failed", "n_total"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "model":    r["model"],
                "n_shots":  r.get("n_shots", ""),
                "mae":      r["mae"],
                "rmse":     r["rmse"],
                "r2":       r["r2"],
                "spearman": r["spearman"],
                "n_failed": r["n_failed"],
                "n_total":  r["n_total"],
            })
    print(f"  CSV saved → {path}")
    return path