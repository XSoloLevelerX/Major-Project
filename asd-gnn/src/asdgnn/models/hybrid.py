from __future__ import annotations

import torch
import torch.nn as nn

from asdgnn.models.gnn_graph import BrainGNN
from asdgnn.models.gnn_pop import PopulationGNN, SiteDiscriminator


class HybridModel(nn.Module):
    """brain-graph encoder -> 64-d subject embedding -> population GNN -> node labels,
    with an optional gradient-reversal site discriminator for site invariance.

    Build and validate this STAGE-WISE first: (1) train the encoder as a plain graph
    classifier, (2) freeze it and dump embeddings, (3) train the population GNN on those
    embeddings. Only when both halves work independently is end-to-end worth attempting.
    """

    def __init__(self, encoder_cfg: dict, pop_cfg: dict, lambda_adv: float = 0.0,
                 n_sites: int = 17, emb_dim: int = 64, n_classes: int = 2):
        super().__init__()
        self.encoder = BrainGNN(**{**encoder_cfg, "n_classes": n_classes})
        enc_out = encoder_cfg.get("hidden", 64) * (
            2 if encoder_cfg.get("pool") == "mean+max" else 1
        )
        self.bottleneck = nn.Linear(enc_out, emb_dim)
        self.pop = PopulationGNN(**{**pop_cfg, "in_dim": emb_dim, "n_classes": n_classes})
        self.site_head = SiteDiscriminator(emb_dim, n_sites)
        self.lambda_adv = lambda_adv

    def encode(self, brain_graphs_batch) -> torch.Tensor:
        g, _ = self.encoder.embed(brain_graphs_batch)
        return self.bottleneck(g)

    def forward(self, brain_graphs_batch, pop_graph, lambda_adv: float | None = None):
        z = self.encode(brain_graphs_batch)
        pop_graph = pop_graph.clone()
        pop_graph.x = z
        logits = self.pop(pop_graph)
        lam = self.lambda_adv if lambda_adv is None else lambda_adv
        site_logits = self.site_head(z, lam) if lam > 0 else None
        return logits, site_logits


def adv_lambda_schedule(epoch: int, n_epochs: int, target: float,
                        ramp_frac: float = 0.3) -> float:
    """Ramp lambda from 0 to target over the first ~30% of epochs. A constant large
    lambda from epoch 0 collapses the encoder to a constant."""
    if target <= 0:
        return 0.0
    ramp = max(1, int(ramp_frac * n_epochs))
    return float(target * min(1.0, epoch / ramp))
