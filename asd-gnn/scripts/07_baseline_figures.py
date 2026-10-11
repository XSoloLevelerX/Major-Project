#!/usr/bin/env python
"""Presentation figures for the classical baselines (M8).

Reads results/runs/*.json and the cached time series; writes PNGs to
results/figures/presentation/. Run after `make baselines`.

    python scripts/07_baseline_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from asdgnn.data.connectivity import ConnectivityTransformer
from asdgnn.data.loading import load_cohort

ROOT = Path.cwd()
OUT = ROOT / "results/figures/presentation"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, GRID, SURF = "#14213D", "#4A5568", "#8A8F98", "#E4E2DC", "#FBFBF8"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODEL_COLOR = {"logreg": BLUE, "svm": ORANGE, "rf": AQUA, "ridge": YELLOW}
NAMES = {"logreg": "Logistic regression", "svm": "SVM (linear)", "rf": "Random forest",
         "ridge": "Ridge"}
ORDER = ["logreg", "svm", "rf", "ridge"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
})

runs = {}
for p in (ROOT / "results/runs").glob("*.json"):
    b = json.loads(p.read_text())
    runs[(b["model"], b["protocol"])] = b


def pooled(b):
    y = np.concatenate([f["y_true"] for f in b["folds"]])
    s = np.concatenate([f["y_prob"] for f in b["folds"]])
    return y, s


# 1. Balanced accuracy, k-fold vs LOSO, grouped bars with error bars + chance line
fig, ax = plt.subplots(figsize=(12, 6.2))
x = np.arange(len(ORDER))
w = 0.36
for k, (proto, col, lab) in enumerate([("kfold", BLUE, "10-fold CV (5 repeats)"),
                                       ("loso", ORANGE, "Leave-one-site-out")]):
    m = [runs[(o, proto)]["aggregate"]["bal_acc_mean"] * 100 for o in ORDER]
    s = [runs[(o, proto)]["aggregate"]["bal_acc_std"] * 100 for o in ORDER]
    xs = x + (k - 0.5) * (w + 0.02)
    ax.bar(xs, m, w, color=col, label=lab, zorder=2)
    ax.errorbar(xs, m, yerr=s, fmt="none", ecolor=INK2, elinewidth=1.5, capsize=4, zorder=3)
    for xi, mi in zip(xs, m):
        ax.text(xi, 41.5, f"{mi:.1f}", ha="center", va="bottom", color="#FBFBF8",
                fontsize=14, fontweight="bold", zorder=4)
ax.axhline(50, color=INK, lw=1.5, zorder=4)
ax.text(-0.5, 50.6, "chance (50%)", ha="left", color=INK, fontsize=13)
ax.set_xticks(x, [NAMES[o] for o in ORDER])
ax.set_ylim(40, 80)
ax.set_ylabel("Balanced accuracy (%)")
ax.set_yticks([40, 50, 60, 70, 80])
ax.yaxis.grid(True, color=GRID, zorder=0)
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "bal_acc_bars.png", dpi=200)
plt.close(fig)

# 2. Pooled ROC under LOSO (the harder test)
for proto in ("kfold", "loso"):
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.5)
    for o in ORDER:
        y, s = pooled(runs[(o, proto)])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=MODEL_COLOR[o], lw=2.5,
                label=f"{NAMES[o]}  (AUC {roc_auc_score(y, s):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(True, color=GRID)
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"roc_{proto}.png", dpi=200)
    plt.close(fig)

# 3. Per-site balanced accuracy, logistic regression under LOSO
b = runs[("logreg", "loso")]
rows = sorted(((f["fold"], f["metrics"]["n_test"], f["metrics"]["bal_acc"] * 100)
               for f in b["folds"]), key=lambda r: -r[1])
fig, ax = plt.subplots(figsize=(14, 6.4))
xs = np.arange(len(rows))
ax.bar(xs, [r[2] for r in rows], 0.72, color=BLUE, zorder=2)
ax.axhline(50, color=INK, lw=1.5, zorder=3)
ax.axhline(b["aggregate"]["bal_acc_mean"] * 100, color=ORANGE, lw=2, zorder=3)
ax.set_xlim(-2.6, len(rows) - 0.5)
ax.text(-2.5, b["aggregate"]["bal_acc_mean"] * 100 + 1,
        f"mean {b['aggregate']['bal_acc_mean']*100:.1f}%", ha="left", color=INK, fontsize=13)
ax.text(-2.5, 51, "chance", ha="left", color=INK, fontsize=13)
ax.set_xticks(xs, [f"{r[0].replace('loso_', '')} (n={r[1]})" for r in rows], fontsize=12,
              rotation=40, ha="right", rotation_mode="anchor")
ax.set_ylim(30, 90)
ax.set_yticks([30, 40, 50, 60, 70, 80, 90])
ax.set_ylabel("Balanced accuracy (%)")
ax.yaxis.grid(True, color=GRID, zorder=0)
fig.tight_layout()
fig.savefig(OUT / "per_site_logreg.png", dpi=200)
plt.close(fig)

# 4. One subject's connectivity matrix and the ASD - TC group difference
ts, pheno = load_cohort("cc200")
ct = ConnectivityTransformer(kind="correlation", vectorize=False).fit(ts[:2])
M = ct.transform(ts)  # stateless for correlation, so fitting on 2 is irrelevant
y = pheno["y"].to_numpy(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
im = axes[0].imshow(M[0], cmap="RdBu_r", vmin=-1.2, vmax=1.2)
axes[0].set_title("One subject: 179 × 179 Fisher-z correlations", fontsize=15, color=INK)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
d = M[y == 1].mean(0) - M[y == 0].mean(0)
lim = np.abs(d).max()
im2 = axes[1].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[1].set_title("Group mean difference: ASD − Control", fontsize=15, color=INK)
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.03)
for a in axes:
    a.set_xlabel("Brain region")
    a.set_ylabel("Brain region")
    a.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "connectivity.png", dpi=200)
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
