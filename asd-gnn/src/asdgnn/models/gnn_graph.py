from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    ChebConv,
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

CONVS = ("gcn", "gat", "gin", "sage", "cheb", "transformer")
POOLS = ("mean", "max", "sum", "mean+max")


def _make_conv(conv: str, in_dim: int, out_dim: int, heads: int, edge_dim: int):
    if conv == "gcn":
        return GCNConv(in_dim, out_dim), False
    if conv == "gat":
        return GATConv(in_dim, out_dim // heads, heads=heads, edge_dim=edge_dim,
                       dropout=0.0), True
    if conv == "gin":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        return GINConv(mlp), False
    if conv == "sage":
        return SAGEConv(in_dim, out_dim), False
    if conv == "cheb":
        return ChebConv(in_dim, out_dim, K=3), False
    if conv == "transformer":
        return TransformerConv(in_dim, out_dim // heads, heads=heads,
                               edge_dim=edge_dim), True
    raise ValueError(f"unknown conv {conv!r}")


class BrainGNN(nn.Module):
    """Graph-level classifier over one graph per subject.

    Node features are the full connectivity profile (R=200 for CC200), which is a very
    large node feature, so the first thing that happens is a linear projection down to
    `hidden`. `return_attention` is plumbed from day one because the interpretability
    chapter needs it and retrofitting it into a trained model is painful.
    """

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3, conv: str = "gat",
                 heads: int = 4, pool: str = "mean", dropout: float = 0.5,
                 n_classes: int = 2, edge_dim: int = 1, use_edge_weight: bool = True):
        super().__init__()
        if conv not in CONVS:
            raise ValueError(f"unknown conv {conv!r}")
        if pool not in POOLS:
            raise ValueError(f"unknown pool {pool!r}")
        self.conv_name, self.pool_name, self.dropout = conv, pool, dropout
        self.use_edge_weight = use_edge_weight

        self.proj = nn.Linear(in_dim, hidden)
        self.convs, self.norms, self.takes_edge_attr = nn.ModuleList(), nn.ModuleList(), []
        for _ in range(n_layers):
            c, uses_ea = _make_conv(conv, hidden, hidden, heads, edge_dim)
            self.convs.append(c)
            self.norms.append(nn.BatchNorm1d(hidden))
            self.takes_edge_attr.append(uses_ea)

        emb_dim = hidden * (2 if pool == "mean+max" else 1)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def _pool(self, h, batch):
        if self.pool_name == "mean":
            return global_mean_pool(h, batch)
        if self.pool_name == "max":
            return global_max_pool(h, batch)
        if self.pool_name == "sum":
            return global_add_pool(h, batch)
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)

    def embed(self, data, return_attention: bool = False):
        x, ei = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        ea = getattr(data, "edge_attr", None)
        ew = ea.squeeze(-1) if (ea is not None and self.use_edge_weight) else None

        h = F.relu(self.proj(x))
        attentions = []
        for conv, norm, wants_ea in zip(self.convs, self.norms, self.takes_edge_attr):
            if wants_ea and return_attention and self.conv_name == "gat":
                h, (att_ei, alpha) = conv(h, ei, edge_attr=ea, return_attention_weights=True)
                attentions.append((att_ei.detach(), alpha.detach()))
            elif wants_ea:
                h = conv(h, ei, edge_attr=ea)
            elif self.conv_name in ("gcn", "cheb") and ew is not None:
                h = conv(h, ei, edge_weight=ew.abs())
            else:
                h = conv(h, ei)
            h = F.relu(norm(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = self._pool(h, batch)
        return (g, attentions) if return_attention else (g, None)

    def forward(self, data, return_attention: bool = False):
        g, att = self.embed(data, return_attention)
        logits = self.head(g)
        return (logits, att) if return_attention else logits


class MLPBaseline(nn.Module):
    """Edge-free control: the same features with no message passing. The gap between
    this and the GNN is what the topology actually contributes."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5,
                 n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, data, return_attention: bool = False):
        from torch_geometric.nn import global_mean_pool

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = data.x.new_zeros(data.x.size(0), dtype=torch.long)
        logits = self.net(global_mean_pool(data.x, batch))
        return (logits, None) if return_attention else logits
