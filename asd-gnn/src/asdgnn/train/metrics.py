from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# Primary metric is balanced accuracy: the cohort is mildly imbalanced and LOSO folds
# have wild per-site class ratios, which makes raw accuracy misleading.
PRIMARY = "bal_acc"


def compute_all(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        # Undefined, not 0.5, when a LOSO site has one class only.
        "auc": None if single_class else float(roc_auc_score(y_true, y_prob)),
        "auc_pr": None if single_class else float(average_precision_score(y_true, y_prob)),
        "sens": float(tp / (tp + fn)) if (tp + fn) else None,
        "spec": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if not single_class else None,
        "n_test": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
    }


def aggregate(fold_metrics: list[dict], weight_by_n: bool = False) -> dict:
    """Mean/std across folds, skipping folds where a metric is undefined and recording
    how many were excluded."""
    out: dict = {}
    keys = [k for k in fold_metrics[0] if not k.startswith("n_")]
    for key in keys:
        vals, ws = [], []
        for m in fold_metrics:
            v = m.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            vals.append(float(v))
            ws.append(m.get("n_test", 1))
        n_excl = len(fold_metrics) - len(vals)
        if not vals:
            out[f"{key}_mean"] = None
            out[f"{key}_std"] = None
        else:
            w = np.asarray(ws, float) if weight_by_n else np.ones(len(vals))
            out[f"{key}_mean"] = float(np.average(vals, weights=w))
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f"{key}_n_excluded"] = n_excl
    out["n_folds"] = len(fold_metrics)
    out["n_test_total"] = int(sum(m.get("n_test", 0) for m in fold_metrics))
    return out


def pooled_metrics(fold_records: list[dict], threshold: float = 0.5) -> dict:
    """Metrics over predictions pooled across all folds — the right way to report a
    single AUC when individual folds are tiny."""
    y_true = np.concatenate([f["y_true"] for f in fold_records])
    y_prob = np.concatenate([f["y_prob"] for f in fold_records])
    return compute_all(y_true, y_prob, threshold)
