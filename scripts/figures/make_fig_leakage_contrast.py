# -*- coding: utf-8 -*-
"""Regenerates image/fig_leakage_contrast.png (Supplementary Figure, S-leak) from
the nested early-stopping run. Predictions are averaged over the five seeds.

    python make_fig_leakage_contrast.py ^
      --pred C:\\Users\\zxzxk\\Downloads\\7073087\\outputs\\metacamil_nested_es\\participant_predictions.csv ^
      --out image\\fig_leakage_contrast.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--out", default="fig_leakage_contrast.png")
ap.add_argument("--config", default="best_cand")
a = ap.parse_args()

d = pd.read_csv(a.pred)
d = d[d.config == a.config]
assert d.seed.nunique() == 5, f"expected 5 seeds, found {d.seed.nunique()}"
e = d.groupby("pid").agg(y=("y_true", "first"), leaky=("leaky_test_es", "mean"),
                         nested=("nested_refit", "mean"))


def metrics(y, p):
    return float(np.abs(p - y).mean()), float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
lo, hi = 15, 50
for ax, col, title in [(axes[0], "leaky", "Stopping epoch chosen on held-out participant"),
                       (axes[1], "nested", "Stopping epoch chosen by inner cross-validation")]:
    mae, r2 = metrics(e.y.values, e[col].values)
    ax.scatter(e.y, e[col], s=28, color="#2f5f8a", edgecolor="white", linewidth=0.5, zorder=3)
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", zorder=1)
    ax.set_title(title, fontsize=10)
    ax.text(0.04, 0.95, f"MAE = {mae:.2f}\n$R^2$ = {r2:.2f}", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlabel("Clinician CARS")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    print(f"{col}: MAE={mae:.3f} R2={r2:.3f}  (caption should match)")
axes[0].set_ylabel("Predicted CARS (mean of 5 seeds)")
fig.tight_layout()
fig.savefig(a.out, dpi=300)
print("saved", a.out)
