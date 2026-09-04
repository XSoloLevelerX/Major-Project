from __future__ import annotations

import numpy as np


def gat_attention_scores(model, loader, device=None) -> np.ndarray:
    """Per-ROI attention influence, averaged over heads, layers and subjects. (R,)

    Aggregation is by SOURCE node, not destination. GAT's softmax normalises over each
    node's incoming edges, so attention summed per destination is exactly 1.0 per layer
    for every node — a constant that looks like a result and is not one. What varies,
    and what is interpretable, is how much attention a region *sends*: how strongly its
    neighbours weight it when updating themselves.
    """
    import torch

    from asdgnn.train.loop import _device

    device = device or _device()
    model = model.to(device).eval()
    totals, n = None, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, attentions = model(batch, return_attention=True)
            if not attentions:
                raise ValueError("model returned no attention — is conv='gat'?")
            acc = torch.zeros(batch.x.size(0), device=device)
            for ei, alpha in attentions:
                acc.index_add_(0, ei[0], alpha.mean(dim=-1))
            per_graph = acc.view(batch.num_graphs, -1).cpu().numpy()
            totals = per_graph.sum(0) if totals is None else totals + per_graph.sum(0)
            n += batch.num_graphs
    scores = totals / max(1, n)
    if np.ptp(scores) < 1e-9:
        raise ValueError(
            "attention scores are constant across ROIs — this is the destination-node "
            "normalisation bug, not a finding"
        )
    return scores


def integrated_gradients(model, loader, steps: int = 50, target: int = 1,
                         device=None) -> np.ndarray:
    """IG on node features w.r.t. the ASD logit, against a zero baseline.
    Returns (R,) node importance."""
    import torch

    from asdgnn.train.loop import _device

    device = device or _device()
    model = model.to(device).eval()
    totals, n = None, 0
    for batch in loader:
        batch = batch.to(device)
        x0 = torch.zeros_like(batch.x)
        grads = torch.zeros_like(batch.x)
        for a in np.linspace(1.0 / steps, 1.0, steps):
            b = batch.clone()
            b.x = (x0 + a * (batch.x - x0)).requires_grad_(True)
            logit = model(b)[:, target].sum()
            grads += torch.autograd.grad(logit, b.x)[0]
        attr = ((batch.x - x0) * grads / steps).abs().sum(dim=1)
        per_graph = attr.detach().view(batch.num_graphs, -1).cpu().numpy()
        totals = per_graph.sum(0) if totals is None else totals + per_graph.sum(0)
        n += batch.num_graphs
    return totals / max(1, n)


def gnn_explainer_edges(model, data, epochs: int = 200, device=None) -> np.ndarray:
    """Per-edge importance from GNNExplainer for a single subject graph."""
    import torch
    from torch_geometric.explain import Explainer, GNNExplainer

    from asdgnn.train.loop import _device

    device = device or _device()
    explainer = Explainer(
        model=model.to(device).eval(),
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        edge_mask_type="object",
        node_mask_type="attributes",
        model_config=dict(mode="multiclass_classification", task_level="graph",
                          return_type="raw"),
    )
    data = data.to(device)
    with torch.enable_grad():
        expl = explainer(data.x, data.edge_index, edge_attr=getattr(data, "edge_attr", None))
    return expl.edge_mask.detach().cpu().numpy()


def aggregate_across_folds(per_fold: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """mean, std across folds. Report the std: an ROI that ranks #1 in three folds and
    #150 in seven is noise, and reviewers ask about exactly this."""
    A = np.vstack([np.asarray(v, float) for v in per_fold])
    return A.mean(axis=0), A.std(axis=0, ddof=1 if len(A) > 1 else 0)


def rank_stability(per_fold: list[np.ndarray], top_k: int = 20) -> dict:
    """How often each ROI enters the top-k, plus the mean pairwise rank correlation
    between folds — the honest summary of whether the explanation is reproducible."""
    from scipy.stats import spearmanr

    A = np.vstack([np.asarray(v, float) for v in per_fold])
    ranks = np.argsort(np.argsort(-A, axis=1), axis=1)
    in_top = (ranks < top_k).mean(axis=0)
    rhos = [
        float(spearmanr(A[i], A[j]).statistic)
        for i in range(len(A)) for j in range(i + 1, len(A))
    ]
    return {
        "top_k": top_k,
        "selection_frequency": in_top.tolist(),
        "stable_rois": np.flatnonzero(in_top >= 0.8).tolist(),
        "mean_pairwise_spearman": float(np.mean(rhos)) if rhos else None,
        "mean_rank": ranks.mean(axis=0).tolist(),
    }


def edge_importance_to_matrix(edge_scores: np.ndarray, edge_index, n_rois: int) -> np.ndarray:
    ei = np.asarray(edge_index)
    M = np.zeros((n_rois, n_rois), float)
    M[ei[0], ei[1]] = np.asarray(edge_scores, float)
    return np.maximum(M, M.T)
