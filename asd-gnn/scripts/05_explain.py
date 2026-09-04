#!/usr/bin/env python
"""Train a graph-level model per fold, extract ROI importance, and report its STABILITY.

The headline output is not the top-20 list — it is how often each ROI enters the top-20
across folds. An ROI that ranks #1 in three folds and #150 in the other seven is noise,
and that is the first thing a reviewer will ask about.

    python scripts/05_explain.py --model gat --method attention
    python scripts/05_explain.py --synthetic --method ig --epochs 40 --min-epochs 10
"""
from __future__ import annotations

import argparse

import numpy as np

from asdgnn.data.connectivity import ConnectivityTransformer, vec_to_matrix
from asdgnn.data.graphs_brain import build_brain_graphs, mean_degree
from asdgnn.data.loading import load_cohort
from asdgnn.explain.roi_mapping import roi_to_labels, top_roi_table
from asdgnn.explain.saliency import (
    aggregate_across_folds,
    gat_attention_scores,
    integrated_gradients,
    rank_stability,
)
from asdgnn.models.registry import build_model
from asdgnn.train.loop import train_graph_level
from asdgnn.train.splits import build_splits, inner_val_split
from asdgnn.utils.logging import get_logger, write_json
from asdgnn.utils.seed import set_seed
from asdgnn.viz.brains import roi_importance_bar, stability_scatter

log = get_logger("05_explain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gat")
    ap.add_argument("--method", default="attention", choices=["attention", "ig"])
    ap.add_argument("--atlas", default="cc200")
    ap.add_argument("--conn", default="correlation")
    ap.add_argument("--protocol", default="kfold", choices=["kfold", "loso"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--min-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--edge-strategy", default="topk")
    ap.add_argument("--edge-param", type=float, default=0.10)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results/explain/roi_importance.json")
    args = ap.parse_args()

    if args.method == "attention" and args.model != "gat":
        raise SystemExit("--method attention needs --model gat; use --method ig otherwise")

    from torch_geometric.loader import DataLoader

    ts_list, pheno = load_cohort(args.atlas, synthetic=args.synthetic)
    y = pheno["y"].to_numpy(int)
    sites = pheno["SITE_ID"].to_numpy()
    sub_ids = pheno.index.to_numpy()
    site_codes = pheno["site_code"].to_numpy(int)
    n_rois = ts_list[0].shape[1]

    cfg = dict(hidden=args.hidden, n_layers=args.n_layers, epochs=args.epochs,
               min_epochs=args.min_epochs, patience=args.patience, dropout=0.5)
    gkw = dict(edge_strategy=args.edge_strategy, edge_param=args.edge_param)

    per_fold, fold_names, fold_scores = [], [], []
    for fold_name, tr, te in build_splits(y, sites, args.protocol, args.n_splits, 1):
        set_seed(args.seed)
        # Connectivity fitted on this fold's training subjects only — the explanation
        # inherits the same fold discipline as the performance numbers.
        ct = ConnectivityTransformer(kind=args.conn).fit([ts_list[i] for i in tr])
        conn_tr = vec_to_matrix(ct.transform([ts_list[i] for i in tr]), n_rois)
        conn_te = vec_to_matrix(ct.transform([ts_list[i] for i in te]), n_rois)

        g_tr = build_brain_graphs(conn_tr, y[tr], sub_ids=sub_ids[tr],
                                  site_codes=site_codes[tr], **gkw)
        g_te = build_brain_graphs(conn_te, y[te], sub_ids=sub_ids[te],
                                  site_codes=site_codes[te], **gkw)
        i_tr, i_va = inner_val_split(np.arange(len(tr)), y[tr], frac=0.15, seed=args.seed)

        model = build_model(args.model, {**cfg, "in_dim": g_tr[0].x.shape[1]})
        rec = train_graph_level(model, [g_tr[i] for i in i_tr], [g_tr[i] for i in i_va],
                                g_te, cfg, seed=args.seed)

        loader = DataLoader(g_te, batch_size=32)
        scores = (gat_attention_scores(model, loader) if args.method == "attention"
                  else integrated_gradients(model, loader))
        per_fold.append(scores)
        fold_names.append(fold_name)
        fold_scores.append(rec["metrics"]["bal_acc"])
        log.info("%s bal_acc=%.3f | importance range %.3g–%.3g", fold_name,
                 rec["metrics"]["bal_acc"], float(scores.min()), float(scores.max()))

    mean, std = aggregate_across_folds(per_fold)
    stab = rank_stability(per_fold, top_k=args.top_k)
    labels = roi_to_labels(args.atlas) if not args.synthetic else None
    names = (labels["name"].to_numpy()[:n_rois] if labels is not None
             else np.array([f"ROI_{i:03d}" for i in range(n_rois)]))

    table = top_roi_table(mean, std, __import__("pandas").DataFrame({"name": names}),
                          k=args.top_k)
    table["in_top_k_fraction"] = np.asarray(stab["selection_frequency"])[table["roi_idx"]]

    blob = {
        "model": args.model,
        "method": args.method,
        "protocol": args.protocol,
        "atlas": args.atlas,
        "n_folds": len(per_fold),
        "folds": fold_names,
        "fold_bal_acc": fold_scores,
        "mean_pairwise_spearman": stab["mean_pairwise_spearman"],
        "stable_rois": stab["stable_rois"],
        "top_rois": table.to_dict(orient="records"),
        "mean_importance": mean.tolist(),
        "std_importance": std.tolist(),
        "selection_frequency": stab["selection_frequency"],
        "mean_degree": mean_degree(g_te),
    }
    write_json(blob, args.out)
    roi_importance_bar(mean, std, names, k=args.top_k)
    stability_scatter(mean, stab["selection_frequency"])

    rho = stab["mean_pairwise_spearman"]
    log.info("mean pairwise Spearman across folds: %s", f"{rho:.3f}" if rho else "n/a")
    log.info("%d ROIs appear in the top-%d in >=80%% of folds",
             len(stab["stable_rois"]), args.top_k)
    if rho is not None and rho < 0.3:
        log.warning("saliency is NOT reproducible across folds — report it as unstable "
                    "rather than naming individual ROIs in the text")
    print(args.out)


if __name__ == "__main__":
    main()
