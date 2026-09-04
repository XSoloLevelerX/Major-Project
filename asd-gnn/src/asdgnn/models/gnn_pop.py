from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv

POP_CONVS = ("cheb", "gcn", "gat", "sage")


class PopulationGNN(nn.Module):
    """Node-level transductive classifier over the cohort graph.

    Message passing deliberately flows through test nodes — that is the semi-supervised
    formulation. The loss is masked, the graph is not. `y[test_mask]` must never reach
    the loss; `train/loop.py` is the only place that indexes it.
    """

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 2, conv: str = "cheb",
                 K: int = 3, heads: int = 4, dropout: float = 0.5, n_classes: int = 2):
        super().__init__()
        if conv not in POP_CONVS:
            raise ValueError(f"unknown conv {conv!r}")
        self.conv_name, self.dropout = conv, dropout
        dims = [in_dim] + [hidden] * (n_layers - 1) + [hidden]
        self.convs = nn.ModuleList()
        for a, b in zip(dims[:-1], dims[1:]):
            if conv == "cheb":
                self.convs.append(ChebConv(a, b, K=K))
            elif conv == "gcn":
                self.convs.append(GCNConv(a, b))
            elif conv == "gat":
                self.convs.append(GATConv(a, b // heads, heads=heads))
            else:
                self.convs.append(SAGEConv(a, b))
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, data, return_attention: bool = False):
        x, ei = data.x, data.edge_index
        ew = data.edge_attr.squeeze(-1) if getattr(data, "edge_attr", None) is not None else None
        h = x
        for conv in self.convs:
            h = conv(h, ei, ew) if self.conv_name in ("cheb", "gcn") else conv(h, ei)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.out(h)
        return (logits, None) if return_attention else logits


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd: float = 1.0):
    return GradientReversal.apply(x, lambd)


class SiteDiscriminator(nn.Module):
    def __init__(self, in_dim: int, n_sites: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_sites))

    def forward(self, z, lambd: float = 1.0):
        return self.net(grad_reverse(z, lambd))
