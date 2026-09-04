#!/usr/bin/env python
"""Stage-wise hybrid: brain-graph encoder -> subject embeddings -> population GNN.

Built in the order the spec insists on, because debugging a broken end-to-end hybrid
with an adversarial head is a week nobody has:

  stage 1  train the encoder as a plain graph classifier
  stage 2  freeze it, dump 64-d embeddings for every subject
  stage 3  train the population GNN on those embeddings
  stage 4  (optional) end-to-end fine-tune with the gradient-reversal site head

Each stage reports its own score, so a failure is attributable to one half.

    python scripts/06_hybrid.py --synthetic --protocol loso --epochs 40 --min-epochs 10
    python scripts/06_hybrid.py --lambda-adv 0.3        # site-invariance head
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np
import torch

from asdgnn.data.connectivity import ConnectivityTransformer, vec_to_matrix
from asdgnn.data.graphs_brain import build_brain_graphs
from asdgnn.data.graphs_pop import build_population_graph
from asdgnn.data.loading import load_cohort
from asdgnn.models.gnn_pop import PopulationGNN, SiteDiscriminator
from asdgnn.models.hybrid import adv_lambda_schedule
from asdgnn.models.registry import build_model
from asdgnn.train.loop import _device, _probs, train_graph_level, train_node_level
from asdgnn.train.metrics import aggregate, compute_all
from asdgnn.train.splits import build_splits, inner_val_split
from asdgnn.utils.hashing import config_hash, git_commit
from asdgnn.utils.logging import get_logger, write_json
from asdgnn.utils.seed import set_seed

log = get_logger("06_hybrid")


def embed_all(encoder, graphs, device, batch_size: int = 64) -> torch.Tensor:
    from torch_geometric.loader import DataLoader

    encoder.eval()
    out = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size):
            g, _ = encoder.embed(batch.to(device))
            out.append(g.cpu())
    return torch.cat(out)


def finetune_end_to_end(encoder, pop_model, site_head, graphs, pop_data, y, tr, va, te,
                        cfg, device, seed: int):
    """Stage 4. The lambda ramp matters: a constant large lambda from epoch 0 collapses
    the encoder to a constant."""
    from torch_geometric.loader import DataLoader

    set_seed(seed)
    params = list(encoder.parameters()) + list(pop_model.parameters())
    if cfg["lambda_adv"] > 0:
        params += list(site_head.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"] * 0.1, weight_decay=cfg["weight_decay"])
    crit = torch.nn.CrossEntropyLoss()
    site_crit = torch.nn.CrossEntropyLoss()

    all_loader = DataLoader(graphs, batch_size=len(graphs))
    big = next(iter(all_loader)).to(device)
    pop_data = pop_data.to(device)
    sites = pop_data.site
    best = {"score": -np.inf, "state": None}

    for epoch in range(cfg["ft_epochs"]):
        encoder.train()
        pop_model.train()
        opt.zero_grad()
        z, _ = encoder.embed(big)
        data = pop_data.clone()
        data.x = z
        logits = pop_model(data)
        loss = crit(logits[data.train_mask], data.y[data.train_mask])

        lam = adv_lambda_schedule(epoch, cfg["ft_epochs"], cfg["lambda_adv"])
        if lam > 0:
            loss = loss + site_crit(site_head(z, lam), sites)
        loss.backward()
        opt.step()

        encoder.eval()
        pop_model.eval()
        with torch.no_grad():
            z = encoder.embed(big)[0]
            data = pop_data.clone()
            data.x = z
            prob = _probs(pop_model(data))
        vm = compute_all(y[va], prob[va])
        score = vm.get("auc") or vm["bal_acc"]
        if score > best["score"]:
            best = {"score": score,
                    "state": ({k: v.clone() for k, v in encoder.state_dict().items()},
                              {k: v.clone() for k, v in pop_model.state_dict().items()})}

    if best["state"]:
        encoder.load_state_dict(best["state"][0])
        pop_model.load_state_dict(best["state"][1])
    encoder.eval()
    pop_model.eval()
    with torch.no_grad():
        data = pop_data.clone()
        data.x = encoder.embed(big)[0]
        prob = _probs(pop_model(data))
    return compute_all(y[te], prob[te]), prob[te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="gat")
    ap.add_argument("--pop-model", default="pop_cheb")
    ap.add_argument("--atlas", default="cc200")
    ap.add_argument("--conn", default="correlation")
    ap.add_argument("--protocol", default="kfold", choices=["kfold", "loso"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--min-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--ft-epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--lambda-adv", type=float, default=0.0)
    ap.add_argument("--edge-param", type=float, default=0.10)
    ap.add_argument("--pheno-terms", default="sex,site,age")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--end-to-end", action="store_true",
                    help="run stage 4; only after stages 1-3 both look sane")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results-dir", default="results/runs")
    args = ap.parse_args()

    # The gradient-reversal head only exists inside finetune_end_to_end; sweeping lambda
    # stage-wise silently produces identical numbers for every value.
    if args.lambda_adv > 0 and not args.end_to_end:
        log.warning("--lambda-adv %.3g has no effect without --end-to-end; the "
                    "site-adversarial head is only trained in stage 4",
                    args.lambda_adv)

    ts_list, pheno = load_cohort(args.atlas, synthetic=args.synthetic)
    y = pheno["y"].to_numpy(int)
    sites = pheno["SITE_ID"].to_numpy()
    site_codes = pheno["site_code"].to_numpy(int)
    sub_ids = pheno.index.to_numpy()
    n_rois = ts_list[0].shape[1]
    n_sites = int(pheno["site_code"].nunique())
    device = _device()
    pheno_terms = tuple(t for t in args.pheno_terms.split(",") if t)

    cfg = dict(hidden=args.hidden, n_layers=2, epochs=args.epochs,
               min_epochs=args.min_epochs, patience=args.patience, dropout=0.5,
               lr=1e-3, weight_decay=5e-4, ft_epochs=args.ft_epochs,
               lambda_adv=args.lambda_adv, batch_size=32)

    t0 = time.time()
    fold_records = []
    for fold_name, tr, te in build_splits(y, sites, args.protocol, args.n_splits, 1):
        set_seed(args.seed)
        ct = ConnectivityTransformer(kind=args.conn).fit([ts_list[i] for i in tr])
        conn_all = np.zeros((len(y), n_rois, n_rois), dtype=np.float32)
        conn_all[tr] = vec_to_matrix(ct.transform([ts_list[i] for i in tr]), n_rois)
        conn_all[te] = vec_to_matrix(ct.transform([ts_list[i] for i in te]), n_rois)
        graphs = build_brain_graphs(conn_all, y, sub_ids=sub_ids, site_codes=site_codes,
                                    edge_strategy="topk", edge_param=args.edge_param)
        i_tr, i_va = inner_val_split(tr, y, frac=0.15, seed=args.seed)

        # --- stage 1: encoder as a plain graph classifier ------------------------
        encoder = build_model(args.encoder, {**cfg, "in_dim": graphs[0].x.shape[1]})
        stage1 = train_graph_level(encoder, [graphs[i] for i in i_tr],
                                   [graphs[i] for i in i_va], [graphs[i] for i in te],
                                   cfg, seed=args.seed)
        encoder = encoder.to(device)

        # --- stage 2: freeze, dump embeddings ------------------------------------
        Z = embed_all(encoder, graphs, device).numpy()

        # --- stage 3: population GNN on the embeddings ---------------------------
        pop_data = build_population_graph(Z, pheno, y, i_tr, i_va, te,
                                          pheno_terms=pheno_terms, knn=args.knn)
        pop_model = PopulationGNN(in_dim=Z.shape[1], hidden=args.hidden, n_layers=2,
                                  conv=args.pop_model.removeprefix("pop_"))
        stage3 = train_node_level(pop_model, pop_data, cfg, seed=args.seed)

        rec = {
            "fold": fold_name,
            "stage1_encoder": stage1["metrics"],
            "stage3_population": stage3["metrics"],
            "metrics": stage3["metrics"],
            "y_true": stage3["y_true"],
            "y_prob": stage3["y_prob"],
            "sub_ids": sub_ids[te].tolist(),
            "n_isolated": int(pop_data.n_isolated),
        }

        # --- stage 4: optional end-to-end fine-tune with the site head -----------
        if args.end_to_end:
            site_head = SiteDiscriminator(Z.shape[1], n_sites).to(device)
            m4, prob4 = finetune_end_to_end(
                encoder, pop_model.to(device), site_head, graphs, pop_data, y,
                i_tr, i_va, te, cfg, device, args.seed)
            rec["stage4_end_to_end"] = m4
            rec["metrics"] = m4
            rec["y_prob"] = prob4.tolist()

        fold_records.append(rec)
        log.info("%s | encoder %.3f -> population %.3f%s", fold_name,
                 stage1["metrics"]["bal_acc"], stage3["metrics"]["bal_acc"],
                 f" -> end-to-end {rec['metrics']['bal_acc']:.3f}" if args.end_to_end else "")

    public = {k: v for k, v in vars(args).items() if k != "results_dir"}
    public["model"] = "hybrid"
    run_id = config_hash(public) + f"_{args.seed}"
    blob = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": public,
        "protocol": args.protocol,
        "model": "hybrid",
        "folds": fold_records,
        "aggregate": aggregate([r["metrics"] for r in fold_records]),
        "stage1_aggregate": aggregate([r["stage1_encoder"] for r in fold_records]),
        "stage3_aggregate": aggregate([r["stage3_population"] for r in fold_records]),
        "runtime_sec": round(time.time() - t0, 1),
        "underpowered_folds": [],
    }
    path = write_json(blob, f"{args.results_dir}/{run_id}.json")
    log.info("stage1 %.3f | stage3 %.3f | reported %.3f -> %s",
             blob["stage1_aggregate"]["bal_acc_mean"],
             blob["stage3_aggregate"]["bal_acc_mean"],
             blob["aggregate"]["bal_acc_mean"], path)
    print(path)


if __name__ == "__main__":
    main()
