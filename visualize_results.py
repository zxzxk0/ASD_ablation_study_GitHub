"""
visualize_results.py — Ablation 결과 시각화
사용법: python visualize_results.py [--results_dir ./results]
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any

from config import RESULTS_DIR


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """results 디렉토리에서 각 모델 결과 JSON 로드."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_results.json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    return results


def plot_metric_bars(all_results: List[Dict], ax: plt.Axes, metric: str,
                     label: str, higher_better: bool = False) -> None:
    """단일 지표 바 차트."""
    models = [r["model"] for r in all_results]
    values = [r[metric] for r in all_results]

    colors = ["#4C9BE8", "#E8844C", "#4CE884"][:len(models)]
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=1.2)

    best_idx = np.argmax(values) if higher_better else np.argmin(values)
    bars[best_idx].set_edgecolor("#FFD700")
    bars[best_idx].set_linewidth(2.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(metric.upper(), fontsize=9)
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=15)


def plot_scatter_comparison(all_results: List[Dict], axs: List[plt.Axes]) -> None:
    """모델별 Predicted vs. True 산점도."""
    colors = ["#4C9BE8", "#E8844C", "#4CE884"]

    for ax, result, color in zip(axs, all_results, colors):
        y_true   = result["y_true"]
        y_pred   = result["y_pred_filled"]
        model    = result["model"]

        ax.scatter(y_true, y_pred, alpha=0.6, s=40, color=color, edgecolors="white", linewidth=0.5)

        # Perfect prediction line
        lo, hi = min(y_true + y_pred), max(y_true + y_pred)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, alpha=0.6, label="Perfect")

        mae = result["mae"]
        r2  = result["r2"]
        ax.set_title(f"{model}\nMAE={mae:.3f}  R²={r2:.3f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("True CARS", fontsize=8)
        ax.set_ylabel("Predicted CARS", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_error_distribution(all_results: List[Dict], ax: plt.Axes) -> None:
    """모델별 예측 오차 분포 (KDE + Histogram)."""
    colors = ["#4C9BE8", "#E8844C", "#4CE884"]
    for result, color in zip(all_results, colors):
        y_true = np.array(result["y_true"])
        y_pred = np.array(result["y_pred_filled"])
        errors = y_true - y_pred
        ax.hist(errors, bins=12, alpha=0.45, color=color, edgecolor="white",
                label=result["model"], density=True)

    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Zero error")
    ax.set_title("Prediction Error Distribution\n(True − Predicted)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Error", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_summary_figure(all_results: List[Dict], save_path: str) -> None:
    """전체 ablation 요약 Figure 생성."""
    n_models = len(all_results)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "LLM Ablation Study — CARS Regression from Eye Scanpath Images",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # Row 1: Metric bars (MAE, RMSE, R², Spearman)
    ax_mae      = fig.add_subplot(gs[0, 0])
    ax_rmse     = fig.add_subplot(gs[0, 1])
    ax_r2       = fig.add_subplot(gs[0, 2])
    ax_spearman = fig.add_subplot(gs[0, 3])

    plot_metric_bars(all_results, ax_mae,      "mae",      "MAE ↓",       higher_better=False)
    plot_metric_bars(all_results, ax_rmse,     "rmse",     "RMSE ↓",      higher_better=False)
    plot_metric_bars(all_results, ax_r2,       "r2",       "R² ↑",        higher_better=True)
    plot_metric_bars(all_results, ax_spearman, "spearman", "Spearman ↑",  higher_better=True)

    # Row 2: Predicted vs True scatter per model (up to 3)
    scatter_axs = [fig.add_subplot(gs[1, i]) for i in range(min(n_models, 3))]
    plot_scatter_comparison(all_results[:3], scatter_axs)

    # Row 3: Error distribution + summary table
    ax_err = fig.add_subplot(gs[2, :2])
    plot_error_distribution(all_results, ax_err)

    # Summary text table
    ax_tbl = fig.add_subplot(gs[2, 2:])
    ax_tbl.axis("off")
    col_labels = ["Model", "MAE↓", "RMSE↓", "R²↑", "Spearman↑", "Failed"]
    table_data = [
        [r["model"], f"{r['mae']:.3f}", f"{r['rmse']:.3f}",
         f"{r['r2']:.3f}", f"{r['spearman']:.3f}",
         f"{r['n_failed']}/{r['n_total']}"]
        for r in all_results
    ]
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax_tbl.set_title("Summary Table", fontsize=11, fontweight="bold", pad=10)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Figure saved → {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize ablation results")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--output",      default=None,
                        help="Output PNG path. Default: <results_dir>/ablation_figure.png")
    args = parser.parse_args()

    all_results = load_results(args.results_dir)
    if not all_results:
        print(f"No result JSON files found in '{args.results_dir}'")
        return

    output_path = args.output or os.path.join(args.results_dir, "ablation_figure.png")
    make_summary_figure(all_results, output_path)


if __name__ == "__main__":
    main()
