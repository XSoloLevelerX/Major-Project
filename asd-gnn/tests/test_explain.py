from __future__ import annotations

import numpy as np
import pytest

from asdgnn.explain.saliency import (
    aggregate_across_folds,
    edge_importance_to_matrix,
    rank_stability,
)

pyg = pytest.importorskip("torch_geometric")


@pytest.fixture
def trained_gat(conn_and_y):
    from asdgnn.data.graphs_brain import build_brain_graphs
    from asdgnn.models.registry import build_model
    from asdgnn.utils.seed import set_seed

    set_seed(0)
    conn, y = conn_and_y
    graphs = build_brain_graphs(conn, y, edge_strategy="knn", edge_param=4)
    model = build_model("gat", {"in_dim": graphs[0].x.shape[1], "hidden": 16,
                                "n_layers": 2, "dropout": 0.0})
    return model, graphs


def test_attention_is_not_constant_across_rois(trained_gat):
    """Regression guard for a bug that produced a perfectly flat 'importance' map.

    GAT normalises attention over each node's INCOMING edges, so summing by destination
    gives exactly 1.0 per layer for every ROI. That constant looks like a result and is
    not one, so aggregation is by source node and a flat map is now a hard error.
    """
    from torch_geometric.loader import DataLoader

    from asdgnn.explain.saliency import gat_attention_scores

    model, graphs = trained_gat
    scores = gat_attention_scores(model, DataLoader(graphs, batch_size=8))
    assert scores.shape == (graphs[0].num_nodes,)
    assert np.isfinite(scores).all()
    assert np.ptp(scores) > 1e-6, "attention map is flat — the normalisation bug is back"


def test_integrated_gradients_shape_and_finiteness(trained_gat):
    from torch_geometric.loader import DataLoader

    from asdgnn.explain.saliency import integrated_gradients

    model, graphs = trained_gat
    scores = integrated_gradients(model, DataLoader(graphs, batch_size=8), steps=8)
    assert scores.shape == (graphs[0].num_nodes,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_aggregate_reports_spread_not_just_mean():
    folds = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])]
    mean, std = aggregate_across_folds(folds)
    assert np.allclose(mean, [2.0, 2.0, 2.0])
    assert std[0] > 0 and std[1] == 0, "std must expose the disagreement the mean hides"


def test_rank_stability_separates_signal_from_noise():
    """An ROI ranked top in every fold is stable; one that swings is not."""
    rng = np.random.RandomState(0)
    n_rois = 30
    folds = []
    for _ in range(5):
        v = rng.rand(n_rois) * 0.1
        v[0] = 10.0          # always the strongest
        v[1] = rng.rand() * 10  # wildly variable
        folds.append(v)
    stab = rank_stability(folds, top_k=3)
    assert stab["selection_frequency"][0] == 1.0
    assert 0 in stab["stable_rois"]
    assert stab["mean_pairwise_spearman"] is not None


def test_edge_importance_matrix_is_symmetric():
    ei = np.array([[0, 1, 2], [1, 2, 0]])
    M = edge_importance_to_matrix(np.array([0.5, 0.2, 0.9]), ei, 3)
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T)
    assert M[0, 1] == 0.5
