from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def roc_panel(runs: dict[str, tuple], path="results/figures/roc.png"):
    """runs: name -> (y_true, y_prob), pooled across folds."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (yt, yp) in runs.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={roc_auc_score(yt, yp):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (pooled)")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def pr_panel(runs: dict[str, tuple], path="results/figures/pr.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (yt, yp) in runs.items():
        prec, rec, _ = precision_recall_curve(yt, yp)
        ax.plot(rec, prec, lw=1.6, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (pooled)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def kfold_vs_loso_scatter(df, path="results/figures/kfold_vs_loso.png",
                          metric: str = "bal_acc_mean"):
    """The headline generalisation figure: each configuration's k-fold score against its
    LOSO score. Points below the diagonal are the honest story about site transfer.

    One point per CONFIGURATION, not per model. Collapsing on the model name alone makes
    pivot_table average across connectivity measures and harmonisation settings, so a
    ComBat run and a raw run land as a single fictitious point.
    """
    from asdgnn.viz.tables import CONFIG_KEYS

    keys = [k for k in CONFIG_KEYS if k in df.columns]
    piv = df.pivot_table(index=keys, columns="protocol", values=metric)
    piv = piv.dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(piv.get("kfold"), piv.get("loso"), s=40)
    for key, row in piv.iterrows():
        cfg = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        label = cfg.get("model", "?")
        # Only qualify the label where a run departs from the default configuration,
        # otherwise every point carries the same unhelpful suffix.
        if cfg.get("conn_kind") not in ("correlation", None):
            label += f"/{str(cfg['conn_kind']).split()[0]}"
        if cfg.get("harmonize") not in ("none", None):
            label += f"/{cfg['harmonize']}"
            if cfg.get("unseen_site") not in ("n/a", None):
                label += f"-{str(cfg['unseen_site'])[:4]}"
        ax.annotate(label, (row["kfold"], row["loso"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    lo = float(min(piv.min().min(), 0.45))
    ax.plot([lo, 0.9], [lo, 0.9], "k--", lw=0.8)
    ax.set(xlabel=f"10-fold CV {metric}", ylabel=f"LOSO {metric}",
           title="Within-site vs cross-site generalisation")
    return _save(fig, path)


def per_site_bars(fold_records, path="results/figures/per_site.png",
                  metric: str = "bal_acc"):
    names = [f["fold"] for f in fold_records]
    vals = [f["metrics"][metric] for f in fold_records]
    ns = [f["metrics"]["n_test"] for f in fold_records]
    order = np.argsort(-np.asarray(ns))
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(names)), 3.5))
    ax.bar(range(len(order)), [vals[i] for i in order])
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{names[i]}\n(n={ns[i]})" for i in order], rotation=90, fontsize=6)
    ax.set(ylabel=metric, title="Per-site performance (LOSO), sites sorted by n")
    return _save(fig, path)


def learning_curve(history, path="results/figures/learning_curve.png"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_loss"] for h in history], label="train loss")
    if "auc" in history[0]:
        ax2 = ax.twinx()
        ax2.plot(ep, [h.get("auc") or np.nan for h in history], color="C1",
                 label="val AUC")
        ax2.set_ylabel("val AUC")
    ax.set(xlabel="epoch", ylabel="train loss")
    return _save(fig, path)


def null_distribution(null, observed, path="results/figures/permutation_null.png"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(null, bins=40, alpha=0.8)
    ax.axvline(observed, color="C3", lw=2, label=f"observed {observed:.3f}")
    ax.set(xlabel="balanced accuracy", ylabel="count",
           title="Label-permutation null distribution")
    ax.legend()
    return _save(fig, path)
