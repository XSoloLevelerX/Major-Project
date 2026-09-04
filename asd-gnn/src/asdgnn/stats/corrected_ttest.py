from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Nadeau-Bengio corrected paired t-test for repeated k-fold CV.

    A plain paired t-test over CV folds is anti-conservative because the folds share
    training data; the correction inflates the variance by (1/k + n_test/n_train).
    Use this, not scipy.ttest_rel, for every model-vs-model claim from k-fold.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    if a.shape != b.shape:
        raise ValueError("paired test needs the same folds for both models")
    d = a - b
    k = len(d)
    if k < 2:
        raise ValueError("need at least 2 folds")
    var = d.var(ddof=1)
    corrected_var = var * (1.0 / k + n_test / n_train)
    if corrected_var <= 0:
        return {"mean_diff": float(d.mean()), "t": 0.0, "p": 1.0, "df": k - 1}
    t = d.mean() / np.sqrt(corrected_var)
    return {
        "mean_diff": float(d.mean()),
        "t": float(t),
        "p": float(2 * stats.t.sf(abs(t), df=k - 1)),
        "df": k - 1,
    }


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict:
    """Step-down correction across a family of comparisons. Report corrected p-values
    whenever a table has more than a couple of tests in it."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[name] = {"p": p, "p_adj": adj, "significant": adj < alpha}
    return out
