from __future__ import annotations

import numpy as np
import pytest

from asdgnn.data.graphs_brain import build_adjacency_mask, build_brain_graphs, mean_degree

pyg = pytest.importorskip("torch_geometric")


def test_topk_hits_the_requested_density(conn_and_y):
    conn, _ = conn_and_y
    R = conn.shape[1]
    for density in (0.05, 0.10, 0.25):
        mask = build_adjacency_mask(conn[0], "topk", density)
        actual = mask.sum() / (R * R - R)
        assert abs(actual - density) < 0.2 * density + 1e-6


def test_knn_is_symmetric_and_has_no_isolated_nodes(conn_and_y):
    conn, _ = conn_and_y
    mask = build_adjacency_mask(conn[0], "knn", 4)
    assert np.array_equal(mask, mask.T)
    assert (mask.sum(1) > 0).all()
    assert not np.diag(mask).any()


def test_thresholding_uses_absolute_weight(conn_and_y):
    """A -0.9 correlation is a strong edge; it must survive an |w| threshold."""
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 0] = -0.9
    A[2, 3] = A[3, 2] = 0.1
    mask = build_adjacency_mask(A, "threshold", 0.5)
    assert mask[0, 1] and not mask[2, 3]


def test_signed_weight_survives_into_edge_attr(conn_and_y):
    conn, y = conn_and_y
    graphs = build_brain_graphs(conn, y, edge_strategy="topk", edge_param=0.2)
    attrs = graphs[0].edge_attr.numpy().ravel()
    assert (attrs < 0).any(), "all edge weights positive — the sign was discarded"


def test_graph_contract(conn_and_y):
    conn, y = conn_and_y
    R = conn.shape[1]
    graphs = build_brain_graphs(conn, y, edge_strategy="knn", edge_param=3,
                                sub_ids=np.arange(len(y)) + 100,
                                site_codes=np.zeros(len(y), int))
    assert len(graphs) == len(y)
    for g in graphs:
        assert int(g.edge_index.max()) < R
        assert g.edge_index.shape[0] == 2
        assert g.edge_attr.shape[0] == g.edge_index.shape[1]
        assert g.x.shape == (R, R)
        assert np.isfinite(g.x.numpy()).all()
        # degree > 0 for every node
        deg = np.bincount(g.edge_index[0].numpy(), minlength=R)
        assert (deg > 0).all()
    assert int(graphs[0].sub_id.item()) == 100


def test_extra_attributes_are_tensors_so_pyg_can_collate(conn_and_y):
    from torch_geometric.loader import DataLoader

    conn, y = conn_and_y
    graphs = build_brain_graphs(conn, y, edge_strategy="topk", edge_param=0.2,
                                sub_ids=np.arange(len(y)), site_codes=np.arange(len(y)) % 3)
    batch = next(iter(DataLoader(graphs, batch_size=8)))
    assert batch.num_graphs == 8
    assert batch.site.shape[0] == 8
    assert batch.y.shape[0] == 8


def test_dense_graph_is_detectable(conn_and_y):
    """A dense graph makes message passing a global average. mean_degree is the check
    to run before debugging anything else."""
    conn, y = conn_and_y
    R = conn.shape[1]
    dense = build_brain_graphs(conn, y, edge_strategy="dense")
    sparse = build_brain_graphs(conn, y, edge_strategy="topk", edge_param=0.05)
    assert mean_degree(dense) == pytest.approx(R - 1)
    assert mean_degree(sparse) < mean_degree(dense) / 4


def test_node_feature_variants(conn_and_y):
    conn, y = conn_and_y
    R = conn.shape[1]
    coords = np.random.RandomState(0).randn(R, 3)
    dims = {
        "profile": R,
        "identity": R,
        "profile+identity": 2 * R,
        "profile+coords": R + 3,
        "degree": 2,
    }
    for kind, dim in dims.items():
        g = build_brain_graphs(conn[:3], y[:3], node_feature=kind, coords=coords,
                               edge_strategy="knn", edge_param=3)
        assert g[0].x.shape == (R, dim)


def test_empty_graph_is_an_error_not_a_silent_nan(conn_and_y):
    conn, y = conn_and_y
    with pytest.raises(ValueError):
        build_brain_graphs(conn[:2], y[:2], edge_strategy="threshold", edge_param=1e9)
