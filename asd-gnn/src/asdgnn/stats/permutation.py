from __future__ import annotations

import numpy as np


def permutation_p(observed: float, null_scores, alternative: str = "greater") -> float:
    """The (r + 1) / (n + 1) estimator — never report p = 0 from a permutation test."""
    null = np.asarray(null_scores, float)
    if alternative == "greater":
        r = int((null >= observed).sum())
    elif alternative == "less":
        r = int((null <= observed).sum())
    else:
        r = int((np.abs(null - null.mean()) >= abs(observed - null.mean())).sum())
    return (r + 1) / (len(null) + 1)


def label_permutation_test(score_fn, y, n_perm: int = 1000, seed: int = 0) -> dict:
    """Permute labels n_perm times and rebuild the null distribution of the score.

    `score_fn(y)` must run the FULL pipeline including every fitted transform — a
    permutation test that reuses features fitted on the true labels tests nothing.
    """
    rng = np.random.RandomState(seed)
    observed = float(score_fn(np.asarray(y)))
    null = [float(score_fn(rng.permutation(np.asarray(y)))) for _ in range(n_perm)]
    return {
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "p": permutation_p(observed, null),
        "n_perm": n_perm,
        "null": null,
    }
