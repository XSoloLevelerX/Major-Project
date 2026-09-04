from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class E2EBlock(nn.Module):
    """BrainNetCNN edge-to-edge filter: a cross-shaped kernel over the (R, R) matrix,
    so each edge is updated from its own row and column."""

    def __init__(self, in_ch: int, out_ch: int, n_rois: int, bias: bool = True):
        super().__init__()
        self.d = n_rois
        self.row = nn.Conv2d(in_ch, out_ch, (1, n_rois), bias=bias)
        self.col = nn.Conv2d(in_ch, out_ch, (n_rois, 1), bias=bias)

    def forward(self, x):
        a = self.row(x)  # (B, C, R, 1)
        b = self.col(x)  # (B, C, 1, R)
        return a.expand(-1, -1, -1, self.d) + b.expand(-1, -1, self.d, -1)


class BrainNetCNN(nn.Module):
    """Grid-CNN baseline operating on the raw connectivity matrix. Included because it
    is the standard non-GNN deep baseline on ABIDE, and it is the fair comparison for
    'does the graph formulation help at all'."""

    def __init__(self, n_rois: int, e2e: int = 32, e2n: int = 64, n2g: int = 128,
                 dropout: float = 0.5, n_classes: int = 2):
        super().__init__()
        self.e2e1 = E2EBlock(1, e2e, n_rois)
        self.e2e2 = E2EBlock(e2e, e2e, n_rois)
        self.e2n = nn.Conv2d(e2e, e2n, (1, n_rois))
        self.n2g = nn.Conv2d(e2n, n2g, (n_rois, 1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(n2g, 64),
            nn.LeakyReLU(0.33), nn.Dropout(dropout), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, R, R)
        h = F.leaky_relu(self.e2e1(x), 0.33)
        h = F.leaky_relu(self.e2e2(h), 0.33)
        h = F.leaky_relu(self.e2n(h), 0.33)
        h = F.leaky_relu(self.n2g(h), 0.33)
        return self.head(h)
