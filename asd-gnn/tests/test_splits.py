from __future__ import annotations

import numpy as np

from asdgnn.train.splits import (
    build_splits,
    cache_splits,
    inner_val_split,
    kfold_splits,
    load_splits,
    loso_splits,
    site_proportion_drift,
    underpowered_folds,
)


def test_kfold_counts(cohort):
    _, pheno = cohort
    y, sites = pheno["y"].to_numpy(), pheno["SITE_ID"].to_numpy()
    splits = list(kfold_splits(y, sites, n_splits=5, n_seeds=3))
    assert len(splits) == 15
    assert len({name for name, _, _ in splits}) == 15


def test_kfold_preserves_site_proportions(cohort):
    """Stratifying on y must not skew site composition beyond sampling noise.

    The tolerance is 3 binomial standard deviations for the largest site at this fold
    size, not a fixed number: with 12 test subjects a per-site share can only move in
    steps of 1/12, so a constant threshold would either be vacuous on 871 subjects or
    flaky on a 60-subject fixture.
    """
    _, pheno = cohort
    y, sites = pheno["y"].to_numpy(), pheno["SITE_ID"].to_numpy()
    n_splits = 5
    n_test = len(y) // n_splits
    p_max = max((sites == s).mean() for s in set(sites))
    tol = 3 * np.sqrt(p_max * (1 - p_max) / n_test)
    drifts = [site_proportion_drift(sites, tr, te)
              for _, tr, te in kfold_splits(y, sites, n_splits=n_splits, n_seeds=1)]
    assert max(drifts) < tol


def test_loso_one_site_per_fold(cohort):
    _, pheno = cohort
    sites = pheno["SITE_ID"].to_numpy()
    folds = list(loso_splits(sites))
    assert len(folds) == len(set(sites))
    assert {name for name, _, _ in folds} == set(sites)


def test_underpowered_flagging(cohort):
    _, pheno = cohort
    sites = pheno["SITE_ID"].to_numpy()
    flagged = underpowered_folds(sites, min_site_n=10_000)
    assert set(flagged) == set(sites)
    assert underpowered_folds(sites, min_site_n=1) == []


def test_inner_val_split_comes_out_of_train(cohort):
    _, pheno = cohort
    y = pheno["y"].to_numpy()
    train_idx = np.arange(0, 40)
    tr, va = inner_val_split(train_idx, y, frac=0.2, seed=0)
    assert set(tr).isdisjoint(set(va))
    assert set(tr) | set(va) == set(train_idx)
    assert 0.15 <= len(va) / len(train_idx) <= 0.25


def test_splits_roundtrip_through_cache(cohort, tmp_path):
    _, pheno = cohort
    y, sites = pheno["y"].to_numpy(), pheno["SITE_ID"].to_numpy()
    splits = build_splits(y, sites, "kfold", n_splits=5, n_seeds=1)
    path = tmp_path / "splits.json"
    cache_splits(splits, pheno.index, "kfold", path)
    loaded = load_splits("kfold", pheno.index, path)
    assert len(loaded) == len(splits)
    for (n1, a1, b1), (n2, a2, b2) in zip(splits, loaded):
        assert n1 == n2
        assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_cached_splits_reject_a_different_cohort(cohort, tmp_path):
    import pytest

    _, pheno = cohort
    y, sites = pheno["y"].to_numpy(), pheno["SITE_ID"].to_numpy()
    path = tmp_path / "splits.json"
    cache_splits(build_splits(y, sites, "kfold", 5, 1), pheno.index, "kfold", path)
    with pytest.raises(ValueError):
        load_splits("kfold", pheno.index[::-1], path)
