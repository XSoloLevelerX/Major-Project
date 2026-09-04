#!/usr/bin/env python
"""Run one experiment (one resolved config) and write results/runs/<run_id>.json.

    python scripts/02_run_experiment.py --model svm --protocol kfold
    python scripts/02_run_experiment.py --model gat --protocol loso --conn tangent
    python scripts/02_run_experiment.py --sanity          # §8 pre-flight checks
"""
from __future__ import annotations

import argparse

import numpy as np

from asdgnn.data.loading import load_cohort
from asdgnn.train.cv import run_cv
from asdgnn.utils.logging import get_logger, write_json

log = get_logger("02_run_experiment")


# The CLI shorthand differs from nilearn's ConnectivityMeasure kind string, which is
# what configs/conn/partial.yaml already carries.
CONN_ALIASES = {"partial": "partial correlation"}


def build_cfg(args) -> dict:
    cfg = {
        "model": args.model,
        "atlas": args.atlas,
        "conn_kind": CONN_ALIASES.get(args.conn, args.conn),
        "protocol": args.protocol,
        "harmonize": args.harmonize,
        "unseen_site": args.unseen_site,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "n_splits": args.n_splits,
        "device": args.device,
        "edge_strategy": args.edge_strategy,
        "edge_param": args.edge_param,
        "node_feature": args.node_feature,
        "select_k": args.select_k,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    if args.pheno_terms is not None:
        cfg["pheno_terms"] = tuple(t for t in args.pheno_terms.split(",") if t)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="svm")
    ap.add_argument("--atlas", default="cc200")
    ap.add_argument("--conn", default="correlation",
                    help="correlation | partial | tangent (nilearn's 'partial "
                         "correlation' also accepted verbatim)")
    ap.add_argument("--protocol", default="kfold", choices=["kfold", "loso"])
    ap.add_argument("--harmonize", default="none", choices=["none", "combat"])
    ap.add_argument("--unseen-site", default="passthrough",
                    choices=["passthrough", "transductive", "reference"],
                    help="LOSO has no fitted parameters for the held-out site; this is "
                         "the declared policy for it (see harmonize/combat.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--edge-strategy", default="topk")
    ap.add_argument("--edge-param", type=float, default=0.10)
    ap.add_argument("--node-feature", default="profile")
    ap.add_argument("--select-k", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--min-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pheno-terms", default=None,
                    help="comma-separated, e.g. 'sex,age' to drop the site term")
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sanity", action="store_true",
                    help="run the §8 pre-flight checks instead of an experiment")
    ap.add_argument("--gnn-sanity", action="store_true",
                    help="the §8 checks that need a trained GNN (slower)")
    args = ap.parse_args()

    ts_list, pheno = load_cohort(args.atlas, synthetic=args.synthetic)
    cfg = build_cfg(args)
    cfg["_pheno"] = pheno  # population models need the phenotype for edge affinities

    if args.sanity or args.gnn_sanity:
        return run_sanity(cfg, ts_list, pheno, gnn=args.gnn_sanity)
    path = run_cv(cfg, ts_list, pheno, results_dir=args.results_dir)
    print(path)


def run_sanity(cfg, ts_list, pheno, gnn: bool = False) -> None:
    """Nothing is believed until these pass. See §8 of the coding spec."""
    from asdgnn.data.connectivity import ConnectivityTransformer
    from asdgnn.models.classical import build_classical, decision_scores
    from asdgnn.train.metrics import compute_all
    from asdgnn.train.splits import build_splits

    y = pheno["y"].to_numpy(int)
    sites = pheno["SITE_ID"].to_numpy()
    splits = build_splits(y, sites, "kfold", n_splits=5, n_seeds=1)
    checks = {}

    def quick_bal_acc(labels):
        scores = []
        for _, tr, te in splits:
            pipe = build_classical({**cfg, "from_timeseries": True, "model": "svm"})
            pipe.fit([ts_list[i] for i in tr], labels[tr])
            p = decision_scores(pipe, [ts_list[i] for i in te])
            scores.append(compute_all(labels[te], p)["bal_acc"])
        return float(np.mean(scores))

    real = quick_bal_acc(y)
    shuffled = quick_bal_acc(np.random.RandomState(0).permutation(y))
    checks["label_shuffle"] = {
        "bal_acc": shuffled, "expected": "~0.50", "n_splits": len(splits),
        "pass": 0.40 <= shuffled <= 0.60,
    }
    # Informational, NOT a gate. Real failing to beat shuffled means no detectable
    # signal, which is a different diagnosis from leakage and must not be reported
    # with a leakage message. On a weak cohort these two estimates are close enough
    # that their ordering is noise.
    checks["real_labels"] = {
        "bal_acc": real, "margin_over_shuffle": real - shuffled,
        "note": ("real labels do not beat shuffled — no detectable signal at this "
                 "configuration; this is NOT leakage"
                 if real <= shuffled else "real labels beat shuffled, as expected"),
    }

    # Sex and site control tasks: cheap, and both produce paper content.
    checks["sex_control"] = {
        "bal_acc": quick_bal_acc(pheno["sex"].to_numpy(int)),
        "note": "features should carry SOME real signal; chance here means a broken pipeline",
    }
    site_code = pheno["site_code"].to_numpy(int)
    checks["site_control"] = {
        "bal_acc": quick_bal_acc((site_code == site_code[0]).astype(int)),
        "note": "quantifies the site confound — cite this number in the Discussion",
    }

    ct = ConnectivityTransformer(kind=cfg.get("conn_kind", "correlation"))
    _, tr, _ = splits[0]
    ct.fit([ts_list[i] for i in tr])
    checks["connectivity_fit_size"] = {
        "n_fit_samples": ct.n_fit_samples_, "n_total": len(ts_list),
        "pass": ct.n_fit_samples_ < len(ts_list),
    }

    if gnn:
        checks.update(gnn_sanity(cfg, ts_list, pheno, splits))

    path = write_json(checks, "results/qc/sanity.json")
    for name, c in checks.items():
        status = {True: "PASS", False: "FAIL"}.get(c.get("pass"), "info")
        log.info("%-22s %-5s %s", name, status,
                 {k: v for k, v in c.items() if k != "pass"})
    failed = [k for k, c in checks.items() if c.get("pass") is False]
    print(path)
    if failed:
        leaky = [k for k in failed if k in ("label_shuffle", "connectivity_fit_size")]
        msg = f"sanity checks FAILED: {failed}"
        raise SystemExit(
            f"{msg} — LEAKAGE suspected, stop everything" if leaky
            else f"{msg} — configuration or architecture problem, not leakage"
        )


def gnn_sanity(cfg, ts_list, pheno, splits) -> dict:
    """The §8 checks that need a trained graph model.

    The ablations matter more than the headline score: if random node features do just
    as well, the model is reading nothing; if the edge-free control matches the GNN,
    the topology is contributing nothing and the paper should say so.
    """
    import numpy as np

    from asdgnn.data.connectivity import ConnectivityTransformer, vec_to_matrix
    from asdgnn.data.graphs_brain import build_brain_graphs, mean_degree
    from asdgnn.models.registry import build_model
    from asdgnn.train.loop import train_graph_level
    from asdgnn.train.splits import inner_val_split

    y = pheno["y"].to_numpy(int)
    sub_ids = pheno.index.to_numpy()
    site_codes = pheno["site_code"].to_numpy(int)
    n_rois = ts_list[0].shape[1]
    _, tr, te = splits[0]

    ct = ConnectivityTransformer(kind=cfg.get("conn_kind", "correlation"))
    ct.fit([ts_list[i] for i in tr])
    conn_tr = vec_to_matrix(ct.transform([ts_list[i] for i in tr]), n_rois)
    conn_te = vec_to_matrix(ct.transform([ts_list[i] for i in te]), n_rois)
    gkw = dict(edge_strategy=cfg.get("edge_strategy", "topk"),
               edge_param=cfg.get("edge_param", 0.10))
    small = {**cfg, "hidden": 32, "n_layers": 2,
             "epochs": cfg.get("epochs", 60), "min_epochs": 10, "patience": 10}

    def score(model_name, node_feature="profile", randomise=False):
        g_tr = build_brain_graphs(conn_tr, y[tr], node_feature=node_feature,
                                  sub_ids=sub_ids[tr], site_codes=site_codes[tr], **gkw)
        g_te = build_brain_graphs(conn_te, y[te], node_feature=node_feature,
                                  sub_ids=sub_ids[te], site_codes=site_codes[te], **gkw)
        if randomise:
            import torch

            rng = torch.Generator().manual_seed(0)
            for g in list(g_tr) + list(g_te):
                g.x = torch.randn(g.x.shape, generator=rng)
        i_tr, i_va = inner_val_split(np.arange(len(tr)), y[tr], frac=0.15, seed=0)
        model = build_model(model_name, {**small, "in_dim": g_tr[0].x.shape[1]})
        rec = train_graph_level(model, [g_tr[i] for i in i_tr], [g_tr[i] for i in i_va],
                                g_te, small, seed=0)
        return rec["metrics"]["bal_acc"], g_tr

    gnn, g_tr = score("gcn")
    rand, _ = score("gcn", randomise=True)
    edge_free, _ = score("mlp_graph")
    deg = mean_degree(g_tr)

    return {
        "mean_node_degree": {
            "value": deg, "expected": "10-40 for CC200",
            "pass": deg > 2,
            "note": "a dense graph makes message passing a global average",
        },
        "random_node_features": {
            "bal_acc": rand, "gnn_bal_acc": gnn,
            "note": "a large drop is expected; no drop means the model reads structure "
                    "only, or nothing",
        },
        "empty_graph_ablation": {
            "bal_acc": edge_free, "gnn_bal_acc": gnn,
            "note": "edge-free control — the gap quantifies what topology contributes, "
                    "and is a paper table in its own right",
        },
    }


if __name__ == "__main__":
    main()
