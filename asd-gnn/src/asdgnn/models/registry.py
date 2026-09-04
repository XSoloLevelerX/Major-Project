from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Maps a config string to a constructor so scripts/02_run_experiment.py never grows an
# `if model == ...` chain.

CLASSICAL = {"svm", "svm_rbf", "ridge", "logreg", "rf", "mlp"}
GRAPH_LEVEL = {"gcn", "gat", "gin", "sage", "cheb", "transformer", "mlp_graph"}
POP_LEVEL = {"pop_cheb", "pop_gcn", "pop_gat", "pop_sage"}
OTHER = {"braincnn", "hybrid"}

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def model_family(name: str) -> str:
    if name in CLASSICAL:
        return "classical"
    if name in GRAPH_LEVEL:
        return "graph"
    if name in POP_LEVEL:
        return "population"
    if name in OTHER:
        return name
    raise ValueError(f"unknown model {name!r}")


def build_model(name: str, cfg: dict):
    """cfg carries whatever the family needs: in_dim / n_rois / n_sites etc."""
    if name in _REGISTRY:
        return _REGISTRY[name](cfg)
    family = model_family(name)

    if family == "classical":
        from asdgnn.models.classical import build_classical

        return build_classical({**cfg, "model": name})

    if family == "graph":
        from asdgnn.models.gnn_graph import MLPBaseline

        if name == "mlp_graph":
            return MLPBaseline(in_dim=cfg["in_dim"], hidden=cfg.get("hidden", 64),
                               dropout=cfg.get("dropout", 0.5))
        from asdgnn.models.gnn_graph import BrainGNN

        return BrainGNN(
            in_dim=cfg["in_dim"], hidden=cfg.get("hidden", 64),
            n_layers=cfg.get("n_layers", 3), conv=name, heads=cfg.get("heads", 4),
            pool=cfg.get("pool", "mean"), dropout=cfg.get("dropout", 0.5),
            edge_dim=cfg.get("edge_dim", 1),
        )

    if family == "population":
        from asdgnn.models.gnn_pop import PopulationGNN

        return PopulationGNN(
            in_dim=cfg["in_dim"], hidden=cfg.get("hidden", 64),
            n_layers=cfg.get("n_layers", 2), conv=name.removeprefix("pop_"),
            K=cfg.get("K", 3), heads=cfg.get("heads", 4),
            dropout=cfg.get("dropout", 0.5),
        )

    if family == "braincnn":
        from asdgnn.models.braincnn import BrainNetCNN

        return BrainNetCNN(n_rois=cfg["n_rois"], dropout=cfg.get("dropout", 0.5))

    from asdgnn.models.hybrid import HybridModel

    return HybridModel(
        encoder_cfg=cfg["encoder_cfg"], pop_cfg=cfg["pop_cfg"],
        lambda_adv=cfg.get("lambda_adv", 0.0), n_sites=cfg.get("n_sites", 17),
    )


def available() -> list[str]:
    return sorted(CLASSICAL | GRAPH_LEVEL | POP_LEVEL | OTHER | set(_REGISTRY))
