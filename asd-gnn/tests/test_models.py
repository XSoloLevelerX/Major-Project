from __future__ import annotations

import numpy as np
import pytest

from asdgnn.models.registry import available, build_model, model_family

pyg = pytest.importorskip("torch_geometric")


def test_registry_families():
    assert model_family("svm") == "classical"
    assert model_family("gat") == "graph"
    assert model_family("pop_cheb") == "population"
    assert "hybrid" in available()
    with pytest.raises(ValueError):
        model_family("not_a_model")


@pytest.mark.parametrize("conv", ["gcn", "gat", "gin", "sage", "cheb", "transformer"])
def test_forward_shapes(conn_and_y, conv):
    import torch
    from torch_geometric.loader import DataLoader

    from asdgnn.data.graphs_brain import build_brain_graphs

    conn, y = conn_and_y
    graphs = build_brain_graphs(conn, y, edge_strategy="knn", edge_param=3)
    model = build_model(conv, {"in_dim": graphs[0].x.shape[1], "hidden": 16, "n_layers": 2})
    batch = next(iter(DataLoader(graphs, batch_size=8)))
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (8, 2)
    assert torch.isfinite(logits).all()


def test_gat_returns_attention(conn_and_y):
    import torch
    from torch_geometric.loader import DataLoader

    from asdgnn.data.graphs_brain import build_brain_graphs

    conn, y = conn_and_y
    graphs = build_brain_graphs(conn, y, edge_strategy="knn", edge_param=3)
    model = build_model("gat", {"in_dim": graphs[0].x.shape[1], "hidden": 16, "n_layers": 2})
    batch = next(iter(DataLoader(graphs, batch_size=4)))
    model.eval()
    with torch.no_grad():
        logits, att = model(batch, return_attention=True)
    assert logits.shape == (4, 2)
    assert len(att) == 2
    ei, alpha = att[0]
    assert alpha.shape[0] == ei.shape[1]


def test_single_batch_overfit(conn_and_y):
    """Sanity check §8: 24 samples, no regularisation — train accuracy must reach 1.0.
    If it cannot, the architecture, lr or loss is broken, not the data."""
    import torch
    from torch_geometric.loader import DataLoader

    from asdgnn.data.graphs_brain import build_brain_graphs
    from asdgnn.utils.seed import set_seed

    set_seed(0)
    conn, y = conn_and_y
    graphs = build_brain_graphs(conn, y, edge_strategy="knn", edge_param=4)
    model = build_model("gcn", {"in_dim": graphs[0].x.shape[1], "hidden": 32,
                                "n_layers": 2, "dropout": 0.0})
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(300):
        opt.zero_grad()
        crit(model(batch), batch.y.view(-1)).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(batch).argmax(1) == batch.y.view(-1)).float().mean().item()
    assert acc == 1.0, f"single-batch overfit reached only {acc:.2f}"


def test_braincnn_forward():
    import torch

    from asdgnn.models.braincnn import BrainNetCNN

    R = 12
    model = BrainNetCNN(n_rois=R, e2e=4, e2n=8, n2g=16)
    out = model(torch.randn(3, R, R))
    assert out.shape == (3, 2)


def test_population_gnn_forward(cohort):
    import torch

    from asdgnn.data.graphs_pop import build_population_graph

    _, pheno = cohort
    y = pheno["y"].to_numpy()
    X = np.random.RandomState(0).randn(len(y), 16).astype(np.float32)
    n = len(y)
    data = build_population_graph(X, pheno, y, np.arange(0, 40), np.arange(40, 48),
                                  np.arange(48, n))
    model = build_model("pop_cheb", {"in_dim": 16, "hidden": 16, "n_layers": 2})
    model.eval()
    with torch.no_grad():
        logits = model(data)
    assert logits.shape == (n, 2)


def test_gradient_reversal_flips_the_sign():
    import torch

    from asdgnn.models.gnn_pop import grad_reverse

    x = torch.ones(4, requires_grad=True)
    grad_reverse(x, 2.0).sum().backward()
    assert torch.allclose(x.grad, torch.full((4,), -2.0))


def test_adv_lambda_ramps_from_zero():
    from asdgnn.models.hybrid import adv_lambda_schedule

    assert adv_lambda_schedule(0, 100, 0.5) == 0.0
    assert adv_lambda_schedule(15, 100, 0.5) == pytest.approx(0.25)
    assert adv_lambda_schedule(99, 100, 0.5) == 0.5
    assert adv_lambda_schedule(50, 100, 0.0) == 0.0


def test_empty_graph_ablation_matches_an_mlp(conn_and_y):
    """The edge-free control. Its gap to the GNN is what topology actually buys."""
    import torch
    from torch_geometric.loader import DataLoader

    from asdgnn.data.graphs_brain import build_brain_graphs

    conn, y = conn_and_y
    graphs = build_brain_graphs(conn, y, edge_strategy="knn", edge_param=3)
    model = build_model("mlp_graph", {"in_dim": graphs[0].x.shape[1], "hidden": 16})
    batch = next(iter(DataLoader(graphs, batch_size=6)))
    model.eval()
    with torch.no_grad():
        assert model(batch).shape == (6, 2)
