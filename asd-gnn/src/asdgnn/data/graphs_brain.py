from __future__ import annotations

import numpy as np

from asdgnn.utils.logging import get_logger

log = get_logger(__name__)

EDGE_STRATEGIES = ("topk", "knn", "threshold", "dense")
NODE_FEATURES = ("profile", "profile+coords", "identity", "profile+identity", "degree")


def build_adjacency_mask(A: np.ndarray, edge_strategy: str = "topk",
                         edge_param: float = 0.10, use_abs: bool = True) -> np.ndarray:
    """Boolean (R, R) edge mask. Thresholding is on |weight| — a -0.8 correlation is a
    strong edge — while the SIGNED weight is what ends up in edge_attr."""
    A = np.asarray(A, dtype=np.float64).copy()
    np.fill_diagonal(A, 0.0)
    W = np.abs(A) if use_abs else A
    R = W.shape[0]

    if edge_strategy == "dense":
        mask = ~np.eye(R, dtype=bool)
    elif edge_strategy == "topk":
        off = ~np.eye(R, dtype=bool)
        k = max(1, int(round(edge_param * off.sum())))
        thr = np.partition(W[off].ravel(), -k)[-k]
        mask = (W >= thr) & off
    elif edge_strategy == "knn":
        k = int(edge_param)
        Wk = W.copy()
        np.fill_diagonal(Wk, -np.inf)
        idx = np.argsort(-Wk, axis=1)[:, :k]
        mask = np.zeros_like(W, dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        mask = mask | mask.T  # symmetrise
        np.fill_diagonal(mask, False)
    elif edge_strategy == "threshold":
        mask = (W >= edge_param) & (~np.eye(R, dtype=bool))
    else:
        raise ValueError(f"unknown edge_strategy {edge_strategy!r}")
    return mask


def node_features(A: np.ndarray, kind: str, coords: np.ndarray | None,
                  mask: np.ndarray) -> np.ndarray:
    R = A.shape[0]
    if kind == "profile":
        return A
    if kind == "identity":
        return np.eye(R, dtype=A.dtype)
    if kind == "profile+identity":
        return np.concatenate([A, np.eye(R, dtype=A.dtype)], axis=1)
    if kind == "profile+coords":
        if coords is None:
            raise ValueError("node_feature='profile+coords' requires coords")
        return np.concatenate([A, np.asarray(coords, dtype=A.dtype)], axis=1)
    if kind == "degree":
        deg = mask.sum(1, keepdims=True).astype(A.dtype)
        strength = np.abs(A).sum(1, keepdims=True).astype(A.dtype)
        return np.concatenate([deg, strength], axis=1)
    raise ValueError(f"unknown node_feature {kind!r}")


def build_brain_graphs(
    conn: np.ndarray,
    y: np.ndarray,
    node_feature: str = "profile",
    edge_strategy: str = "topk",
    edge_param: float = 0.10,
    use_abs: bool = True,
    self_loops: bool = False,
    coords: np.ndarray | None = None,
    sub_ids: np.ndarray | None = None,
    site_codes: np.ndarray | None = None,
):
    """One PyG Data per subject, for graph-level classification.

    `conn` must already be fold-safe (built by a transformer fitted on train only).
    `sub_id` and `site` are carried as integer tensors so PyG can collate them — never
    store a site string on a Data object.
    """
    import torch
    from torch_geometric.data import Data

    conn = np.asarray(conn)
    if conn.ndim != 3 or conn.shape[1] != conn.shape[2]:
        raise ValueError(f"expected (N, R, R), got {conn.shape}")
    N, R, _ = conn.shape
    graphs = []
    degrees = []

    for i in range(N):
        A = conn[i].astype(np.float32).copy()
        np.fill_diagonal(A, 0.0)
        mask = build_adjacency_mask(A, edge_strategy, edge_param, use_abs)
        if self_loops:
            np.fill_diagonal(mask, True)
        if not mask.any():
            raise ValueError(f"subject {i}: empty graph at edge_param={edge_param}")
        degrees.append(mask.sum(1).mean())

        ei = np.array(np.nonzero(mask))
        x = node_features(A, node_feature, coords, mask)
        data = Data(
            x=torch.from_numpy(np.ascontiguousarray(x)).float(),
            edge_index=torch.from_numpy(ei).long(),
            edge_attr=torch.from_numpy(A[mask]).float().unsqueeze(-1),  # SIGNED weight
            y=torch.tensor([int(y[i])], dtype=torch.long),
            sub_id=torch.tensor([int(sub_ids[i])] if sub_ids is not None else [i]),
            site=torch.tensor([int(site_codes[i])] if site_codes is not None else [0]),
        )
        data.num_nodes = R
        graphs.append(data)

    log.info(
        "built %d brain graphs: R=%d, node_dim=%d, mean degree %.1f",
        N, R, graphs[0].x.shape[1], float(np.mean(degrees)),
    )
    return graphs


def mean_degree(graphs) -> float:
    """A GNN stuck near 53% is a dense graph 40% of the time — print this first."""
    return float(np.mean([g.edge_index.shape[1] / g.num_nodes for g in graphs]))
