from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def roi_importance_bar(scores, stds, labels, path="results/figures/top_rois.png",
                       k: int = 20):
    """Top-k ROIs with across-fold error bars — the error bar is the point of the
    figure, not decoration."""
    scores, stds = np.asarray(scores), np.asarray(stds)
    idx = np.argsort(-scores)[:k]
    names = [str(labels[i]) for i in idx]
    fig, ax = plt.subplots(figsize=(5, 0.28 * k + 1))
    ax.barh(range(len(idx))[::-1], scores[idx], xerr=stds[idx], capsize=2)
    ax.set_yticks(range(len(idx))[::-1])
    ax.set_yticklabels(names, fontsize=7)
    ax.set(xlabel="importance (mean ± std across folds)", title=f"Top {k} ROIs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def stability_scatter(mean_scores, selection_freq,
                      path="results/figures/roi_stability.png"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(mean_scores, selection_freq, s=12, alpha=0.7)
    ax.set(xlabel="mean importance", ylabel="fraction of folds in top-k",
           title="Saliency stability")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
