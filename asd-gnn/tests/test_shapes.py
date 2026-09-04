from __future__ import annotations

import numpy as np
import pytest

from asdgnn.data.connectivity import (
    ConnectivityTransformer,
    matrix_to_vec,
    n_features,
    vec_index_to_roi_pair,
    vec_to_matrix,
)
from asdgnn.data.timeseries import qc_filter, stack_aligned


def test_timeseries_contract(cohort):
    ts_list, pheno = cohort
    assert len(ts_list) == len(pheno)
    assert all(a.ndim == 2 for a in ts_list)
    assert len({a.shape[1] for a in ts_list}) == 1, "R must be identical across subjects"
    assert len({a.shape[0] for a in ts_list}) > 1, "T varies across sites — by design"
    assert all(np.isfinite(a).all() for a in ts_list)
    assert all(a.dtype == np.float32 for a in ts_list)


def test_qc_drops_short_and_constant(cohort):
    ts_list, pheno = cohort
    ts = {int(s): a for s, a in zip(pheno.index, ts_list)}
    short = int(pheno.index[0])
    ts[short] = ts[short][:20]
    dead = int(pheno.index[1])
    ts[dead] = ts[dead].copy()
    ts[dead][:, 3] = 0.0

    kept, pheno_f, report = qc_filter(ts, pheno, min_timepoints=100, report_path=None)
    assert short not in kept
    assert len(pheno_f) == len(kept)
    assert 3 in report["dropped_rois"]
    assert all(a.std(axis=0).min() > 0 for a in kept.values())


def test_stack_aligned_preserves_pheno_order(cohort):
    ts_list, pheno = cohort
    ts = {int(s): a for s, a in zip(pheno.index, ts_list)}
    stacked = stack_aligned(ts, pheno)
    assert [a.shape for a in stacked] == [a.shape for a in ts_list]


@pytest.mark.parametrize("n_rois", [5, 12, 33])
def test_vec_matrix_roundtrip_is_exact(n_rois):
    rng = np.random.RandomState(0)
    v = rng.randn(n_features(n_rois))
    M = vec_to_matrix(v, n_rois)
    assert M.shape == (n_rois, n_rois)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0)
    assert np.array_equal(matrix_to_vec(M), v)


def test_vec_matrix_roundtrip_batched():
    rng = np.random.RandomState(1)
    n_rois = 9
    V = rng.randn(7, n_features(n_rois))
    assert np.array_equal(matrix_to_vec(vec_to_matrix(V, n_rois)), V)


def test_vec_index_maps_to_the_right_cell():
    n_rois = 6
    for idx in range(n_features(n_rois)):
        v = np.zeros(n_features(n_rois))
        v[idx] = 1.0
        i, j = vec_index_to_roi_pair(idx, n_rois)
        M = vec_to_matrix(v, n_rois)
        assert M[i, j] == 1.0 and M[j, i] == 1.0
        assert M.sum() == 2.0


@pytest.mark.parametrize("kind", ["correlation", "tangent", "partial correlation"])
def test_connectivity_output_shapes(dummy_data, kind):
    R = dummy_data[0].shape[1]
    ct = ConnectivityTransformer(kind=kind, vectorize=True).fit(dummy_data[:30])
    X = ct.transform(dummy_data[:10])
    assert X.shape == (10, n_features(R))
    assert np.isfinite(X).all()

    ct_m = ConnectivityTransformer(kind=kind, vectorize=False).fit(dummy_data[:30])
    M = ct_m.transform(dummy_data[:10])
    assert M.shape == (10, R, R)
    assert np.allclose(M, np.transpose(M, (0, 2, 1)), atol=1e-5)
    assert np.allclose(M[:, np.arange(R), np.arange(R)], 0)


def test_vectorised_matches_matrix_form(dummy_data):
    R = dummy_data[0].shape[1]
    a = ConnectivityTransformer(kind="correlation", vectorize=True).fit(dummy_data[:30])
    b = ConnectivityTransformer(kind="correlation", vectorize=False).fit(dummy_data[:30])
    X, M = a.transform(dummy_data[:5]), b.transform(dummy_data[:5])
    assert np.allclose(X, matrix_to_vec(M), atol=1e-5)
    assert np.allclose(vec_to_matrix(X, R), M, atol=1e-5)
