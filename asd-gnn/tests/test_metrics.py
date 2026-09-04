from __future__ import annotations

import numpy as np
import pytest

from asdgnn.stats.corrected_ttest import corrected_resampled_ttest, holm_bonferroni
from asdgnn.stats.delong import delong_test
from asdgnn.stats.permutation import permutation_p
from asdgnn.train.metrics import aggregate, compute_all, pooled_metrics


def test_perfect_and_inverted_predictions():
    y = np.array([0, 0, 1, 1])
    perfect = compute_all(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["acc"] == 1.0
    assert perfect["bal_acc"] == 1.0
    assert perfect["auc"] == 1.0
    assert perfect["sens"] == 1.0 and perfect["spec"] == 1.0

    inverted = compute_all(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert inverted["auc"] == 0.0
    assert inverted["mcc"] == -1.0


def test_known_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1])
    m = compute_all(y, p)
    assert m["sens"] == pytest.approx(2 / 3)
    assert m["spec"] == pytest.approx(4 / 5)
    assert m["bal_acc"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["n_pos"] == 3 and m["n_neg"] == 5


def test_single_class_fold_returns_none_not_half():
    """A LOSO site with one class has an UNDEFINED AUC. Returning 0.5 would quietly
    drag the mean toward chance."""
    m = compute_all(np.ones(6, int), np.linspace(0.1, 0.9, 6))
    assert m["auc"] is None
    assert m["auc_pr"] is None
    assert m["mcc"] is None
    assert m["bal_acc"] is not None


def test_aggregate_counts_exclusions():
    folds = [
        compute_all(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])),
        compute_all(np.array([0, 1, 0, 1]), np.array([0.9, 0.1, 0.8, 0.2])),
        compute_all(np.ones(4, int), np.array([0.1, 0.9, 0.2, 0.8])),
    ]
    agg = aggregate(folds)
    assert agg["n_folds"] == 3
    assert agg["auc_n_excluded"] == 1
    assert agg["auc_mean"] == pytest.approx(0.5)
    assert agg["bal_acc_n_excluded"] == 0


def test_pooled_beats_per_fold_when_folds_are_tiny():
    folds = [
        {"y_true": [0, 1], "y_prob": [0.2, 0.8]},
        {"y_true": [0, 1], "y_prob": [0.3, 0.7]},
    ]
    assert pooled_metrics(folds)["auc"] == 1.0


def test_delong_identical_scores_give_p_one():
    rng = np.random.RandomState(0)
    y = rng.binomial(1, 0.5, 200)
    s = rng.rand(200) + y * 0.4
    out = delong_test(y, s, s.copy())
    assert out["p"] == pytest.approx(1.0)
    assert out["delta"] == pytest.approx(0.0, abs=1e-9)


def test_delong_detects_a_real_difference():
    rng = np.random.RandomState(1)
    y = rng.binomial(1, 0.5, 400)
    good = rng.rand(400) + y * 1.5
    bad = rng.rand(400) + y * 0.05
    out = delong_test(y, good, bad)
    assert out["auc_a"] > out["auc_b"]
    assert out["p"] < 0.01


def test_corrected_ttest_is_more_conservative_than_plain():
    from scipy import stats

    rng = np.random.RandomState(0)
    a = rng.normal(0.65, 0.03, 10)
    b = a - 0.02
    plain = stats.ttest_rel(a, b).pvalue
    corrected = corrected_resampled_ttest(a, b, n_train=780, n_test=87)["p"]
    assert corrected > plain


def test_holm_is_monotone_and_bounded():
    out = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.4})
    assert out["a"]["p_adj"] <= out["b"]["p_adj"] <= out["c"]["p_adj"]
    assert all(0 <= v["p_adj"] <= 1 for v in out.values())


def test_permutation_p_never_returns_zero():
    assert permutation_p(1.0, np.zeros(100)) == pytest.approx(1 / 101)
