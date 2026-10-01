"""
generate_plots.py
=================
train_mlp_v2_fixed.py 실행 후 저장된 체크포인트에서
예측값을 뽑아 4-panel 진단 플롯을 생성합니다.

사용법:
    python generate_plots.py
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 경로 설정 (환경에 맞게 수정) ──────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import torch
import numpy as np

CKPT_DIR  = Path(r"D:\download\7073087\outputs\checkpoints_v2_cv")
IMG_ROOT  = Path(r"D:\DOWNLOAD\7073087\IMAGES\IMAGES\TSImages")
DATA_PATH = Path(r"D:\download\7073087\all_scanpath_absolute.jsonl")
OUT_DIR   = Path(r"D:\download\7073087\outputs\checkpoints_v2_cv")
CACHE_DIR = Path(r"D:\download\7073087\outputs\cache_v2")

BACKBONE   = "EVA02-B-16@merged2b_s8b_b131k"
N_SPLITS   = 5
CARS_MIN, CARS_MAX = 15.0, 60.0

SEVERITY_BINS = [
    ("Mild",     15.0, 30.0),
    ("Moderate", 30.0, 37.0),
    ("Severe",   37.0, 61.0),
]

# ── 모델 정의 (train_mlp_v2_fixed.py와 동일) ─────────────────────
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=1024, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.ReLU(True),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── 예측값 수집 ──────────────────────────────────────────────────
def collect_predictions():
    """
    각 fold의 best checkpoint를 로드해 val 예측값을 수집.
    train_mlp_v2_fixed.py와 동일한 데이터 로딩/전처리 사용.
    """
    # 데이터 로드 (train_mlp_v2_fixed.py의 함수 재활용)
    from train_mlp_v2_fixed import (
        load_table_any, embed_path_list_from_df,
        get_target_array, prepare_meta_features,
        cars_to_severity_label,
    )
    from sklearn.model_selection import StratifiedKFold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_all = load_table_any(DATA_PATH)

    backbones = [b.strip() for b in BACKBONE.split(",")]
    X_img_all, df_filtered = embed_path_list_from_df(
        df_all, IMG_ROOT, backbones, device, tta=False, cache_prefix="all_data"
    )
    y_all = get_target_array(df_filtered)
    M_all_raw, num_cont = prepare_meta_features(df_filtered)
    y_strat = cars_to_severity_label(y_all)

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    all_y_true, all_y_pred = [], []
    participant_ids = (
        df_filtered["meta"].apply(lambda m: m.get("participant_id", "?")).tolist()
        if "meta" in df_filtered.columns else [str(i) for i in range(len(df_filtered))]
    )

    safe_bb = BACKBONE.replace("/", "_").replace("@", "_").replace(",", "_")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_img_all, y_strat), 1):
        ckpt_path = CKPT_DIR / f"mlp_best_{safe_bb}_fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"  [WARN] Checkpoint not found: {ckpt_path}")
            continue

        X_img_tr  = X_img_all[tr_idx]
        X_img_val = X_img_all[val_idx]
        y_tr      = y_all[tr_idx]
        y_val     = y_all[val_idx]

        if M_all_raw is not None:
            M_tr  = M_all_raw[tr_idx].copy()
            M_val = M_all_raw[val_idx].copy()
            if num_cont > 0:
                mu_m = M_tr[:, :num_cont].mean(0); sd_m = M_tr[:, :num_cont].std(0)
                sd_m[sd_m == 0] = 1.0
                M_tr[:, :num_cont]  = (M_tr[:, :num_cont]  - mu_m) / sd_m
                M_val[:, :num_cont] = (M_val[:, :num_cont] - mu_m) / sd_m
            X_val = np.concatenate([X_img_val, M_val], axis=1)
            X_tr  = np.concatenate([X_img_tr,  M_tr],  axis=1)
        else:
            X_val = X_img_val
            X_tr  = X_img_tr

        y_mu = float(y_tr.mean()); y_sd = float(y_tr.std())
        if y_sd == 0: y_sd = 1.0

        model = MLPRegressor(in_dim=X_tr.shape[1]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        with torch.no_grad():
            X_t   = torch.from_numpy(X_val).float().to(device)
            preds = model(X_t).cpu().numpy() * y_sd + y_mu
        preds = np.clip(preds, CARS_MIN, CARS_MAX)

        for i, vi in enumerate(val_idx):
            all_y_true.append(float(y_val[i]))
            all_y_pred.append(float(preds[i]))

    return (
        np.array(all_y_true),
        np.array(all_y_pred),
        [participant_ids[i] for fold, (_, val_idx)
         in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=42)
                      .split(X_img_all, y_strat), 1)
         for i in val_idx],
    )


# ── 4-panel 플롯 ─────────────────────────────────────────────────
def make_four_panel(y_true, y_pred, participant_ids, out_path):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    sp   = spearmanr(y_true, y_pred).statistic
    errs = y_true - y_pred
    abs_errs = np.abs(errs)

    # severity 색상
    def sev_color(score):
        if score < 30:   return "#4CAF50"   # Mild  - green
        elif score < 37: return "#FF9800"   # Moderate - orange
        else:            return "#F44336"   # Severe - red

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Diagnostic Plots: EVA02-B-16 CARS Regression\n(5-Fold Stratified CV)",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── (a) Predicted vs True ────────────────────────────────────
    ax = axes[0, 0]
    colors = [sev_color(s) for s in y_true]
    ax.scatter(y_true, y_pred, c=colors, alpha=0.7, edgecolors="white",
               linewidth=0.5, s=60, zorder=3)
    lo, hi = min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction", zorder=2)
    stats_txt = (f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n"
                 f"R²: {r2:.2f}\nSpearman: {sp:.2f}")
    ax.text(0.04, 0.96, stats_txt, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))
    patches = [
        mpatches.Patch(color="#4CAF50", label="Mild (15–29)"),
        mpatches.Patch(color="#FF9800", label="Moderate (30–36)"),
        mpatches.Patch(color="#F44336", label="Severe (37–60)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Perfect prediction"),
    ]
    ax.legend(handles=patches, fontsize=7.5, loc="lower right")
    ax.set_xlabel("True CARS Score", fontsize=11)
    ax.set_ylabel("Predicted CARS Score", fontsize=11)
    ax.set_title("Predicted vs True CARS Score", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── (b) Error Distribution ───────────────────────────────────
    ax = axes[0, 1]
    ax.hist(errs, bins=20, color="steelblue", edgecolor="white",
            alpha=0.85, zorder=3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax.axvline(errs.mean(), color="orange", linestyle="-",
               linewidth=1.2, label=f"Mean={errs.mean():.2f}")
    ax.set_xlabel("Prediction Error (True − Predicted)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Error Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (c) Absolute Error vs True Score ─────────────────────────
    ax = axes[1, 0]
    colors_c = [sev_color(s) for s in y_true]
    ax.scatter(y_true, abs_errs, c=colors_c, alpha=0.7,
               edgecolors="white", linewidth=0.5, s=60, zorder=3)

    # 상위 3개 오차 포인트 레이블
    top3_idx = np.argsort(abs_errs)[-3:][::-1]
    for idx in top3_idx:
        pid = participant_ids[idx] if participant_ids else str(idx)
        ax.annotate(f"P{pid}", (y_true[idx], abs_errs[idx]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color="dimgray")

    # severity별 mean MAE 수평선
    for label, lo_s, hi_s in SEVERITY_BINS:
        mask = (y_true >= lo_s) & (y_true < hi_s)
        if mask.sum() == 0: continue
        mean_ae = abs_errs[mask].mean()
        x_lo = y_true[mask].min() - 0.5
        x_hi = y_true[mask].max() + 0.5
        ax.hlines(mean_ae, x_lo, x_hi,
                  colors=sev_color((lo_s + hi_s) / 2),
                  linestyles="--", linewidth=1.2, alpha=0.8,
                  label=f"{label} mean={mean_ae:.2f}")

    ax.set_xlabel("True CARS Score", fontsize=11)
    ax.set_ylabel("Absolute Error", fontsize=11)
    ax.set_title("Absolute Error vs True CARS Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ── (d) MAE per Participant ──────────────────────────────────
    ax = axes[1, 1]
    pid_arr  = np.array(participant_ids)
    pids_u   = np.unique(pid_arr)
    pid_mae  = [(pid, abs_errs[pid_arr == pid].mean()) for pid in pids_u]
    pid_mae.sort(key=lambda x: -x[1])

    pids_sorted  = [p for p, _ in pid_mae]
    maes_sorted  = [m for _, m in pid_mae]
    true_sorted  = [y_true[pid_arr == p].mean() for p in pids_sorted]
    bar_colors   = [sev_color(s) for s in true_sorted]

    ax.bar(range(len(pids_sorted)), maes_sorted,
           color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(pids_sorted)))
    ax.set_xticklabels([f"P{p}" for p in pids_sorted],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Participant", fontsize=11)
    ax.set_ylabel("Mean Absolute Error", fontsize=11)
    ax.set_title("Mean Absolute Error per Participant", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    patches_d = [
        mpatches.Patch(color="#4CAF50", label="Mild"),
        mpatches.Patch(color="#FF9800", label="Moderate"),
        mpatches.Patch(color="#F44336", label="Severe"),
    ]
    ax.legend(handles=patches_d, fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("Collecting fold predictions from checkpoints...")
    y_true, y_pred, pids = collect_predictions()

    print(f"\nTotal predictions: {len(y_true)}")
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    sp   = spearmanr(y_true, y_pred).statistic
    print(f"Overall: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, Spearman={sp:.3f}")

    out_path = OUT_DIR / "plots_stratified_cv.jpeg"
    make_four_panel(y_true, y_pred, pids, out_path)


if __name__ == "__main__":
    main()
