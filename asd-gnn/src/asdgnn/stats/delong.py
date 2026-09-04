from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _structural_components(preds: np.ndarray, m: int, n: int):
    """preds: (k, m+n) with the m positives first. Returns V10 (k, m), V01 (k, n)."""
    k = preds.shape[0]
    v10 = np.empty((k, m))
    v01 = np.empty((k, n))
    for r in range(k):
        pos, neg = preds[r, :m], preds[r, m:]
        tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(preds[r])
        v10[r] = (tz[:m] - tx) / n
        v01[r] = 1.0 - (tz[m:] - ty) / m
    return v10, v01


def delong_roc_variance(y_true, y_score):
    """AUC and its DeLong variance for a single model."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(y_true))
    y_score = np.asarray(y_score, float)[order]
    m = int((y_true == 1).sum())
    n = len(y_true) - m
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present")
    v10, v01 = _structural_components(y_score[None, :], m, n)
    var = np.var(v10[0], ddof=1) / m + np.var(v01[0], ddof=1) / n
    return float(v10[0].mean()), float(var)


def delong_test(y_true, score_a, score_b) -> dict:
    """Paired DeLong test for two correlated ROC curves on the same subjects.

    This is why per-fold predicted probabilities are saved: without them there is
    nothing to pair.
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    y = y_true[order]
    preds = np.vstack([np.asarray(score_a, float)[order],
                       np.asarray(score_b, float)[order]])
    m = int((y == 1).sum())
    n = len(y) - m
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present")

    v10, v01 = _structural_components(preds, m, n)
    auc = v10.mean(axis=1)
    S = np.cov(v10) / m + np.cov(v01) / n
    L = np.array([1.0, -1.0])
    var = float(L @ S @ L)
    if var <= 0:
        return {"auc_a": float(auc[0]), "auc_b": float(auc[1]), "delta": 0.0,
                "z": 0.0, "p": 1.0}
    z = float((auc[0] - auc[1]) / np.sqrt(var))
    return {
        "auc_a": float(auc[0]),
        "auc_b": float(auc[1]),
        "delta": float(auc[0] - auc[1]),
        "z": z,
        "p": float(2 * stats.norm.sf(abs(z))),
    }
