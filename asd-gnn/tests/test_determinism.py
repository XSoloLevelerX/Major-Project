from __future__ import annotations

import numpy as np
import pytest

from asdgnn.data.synthetic import make_cohort
from asdgnn.models.classical import build_classical, decision_scores
from asdgnn.train.splits import build_splits
from asdgnn.utils.hashing import config_hash
from asdgnn.utils.seed import set_seed


def test_same_seed_same_splits(cohort):
    _, pheno = cohort
    y, sites = pheno["y"].to_numpy(), pheno["SITE_ID"].to_numpy()
    a = build_splits(y, sites, "kfold", 5, 2)
    b = build_splits(y, sites, "kfold", 5, 2)
    for (n1, t1, s1), (n2, t2, s2) in zip(a, b):
        assert n1 == n2
        assert np.array_equal(t1, t2) and np.array_equal(s1, s2)


def test_classical_pipeline_is_reproducible(cohort):
    ts_list, pheno = cohort
    y = pheno["y"].to_numpy()
    tr, te = np.arange(45), np.arange(45, len(y))

    def run():
        set_seed(0)
        pipe = build_classical({"model": "svm", "select_k": 20, "seed": 0})
        pipe.fit([ts_list[i] for i in tr], y[tr])
        return decision_scores(pipe, [ts_list[i] for i in te])

    assert np.allclose(run(), run())


def test_synthetic_cohort_is_seeded():
    a, pa = make_cohort(n_subjects=10, n_rois=6, seed=3)
    b, pb = make_cohort(n_subjects=10, n_rois=6, seed=3)
    assert list(a) == list(b)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert pa.equals(pb)


def test_config_hash_is_order_independent_and_change_sensitive():
    a = {"model": "gat", "conn_kind": "tangent", "seed": 0}
    b = {"seed": 0, "conn_kind": "tangent", "model": "gat"}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({**a, "conn_kind": "correlation"})


@pytest.mark.parametrize("seed", [0, 7])
def test_torch_seeding(seed, torch_available):
    if not torch_available:
        pytest.skip("torch not installed")
    import torch

    set_seed(seed)
    a = torch.randn(8)
    set_seed(seed)
    assert torch.allclose(a, torch.randn(8))
