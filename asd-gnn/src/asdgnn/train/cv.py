from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from asdgnn.data.connectivity import ConnectivityTransformer, vec_to_matrix
from asdgnn.harmonize.combat import ComBatTransformer
from asdgnn.models.registry import model_family
from asdgnn.train.metrics import aggregate, compute_all, pooled_metrics
from asdgnn.train.splits import (
    build_splits,
    inner_val_split,
    underpowered_folds,
)
from asdgnn.utils.hashing import config_hash, git_commit
from asdgnn.utils.logging import get_logger, write_json
from asdgnn.utils.seed import set_seed

log = get_logger(__name__)


def run_cv(cfg: dict, ts_list: list[np.ndarray], pheno: pd.DataFrame,
           results_dir: str | Path = "results/runs") -> Path:
    """Orchestrate all folds x seeds for one config. Writes results/runs/<run_id>.json.

    Everything stateful — connectivity, ComBat, scaling, feature selection — is fitted
    inside the fold loop on training indices only.
    """
    t0 = time.time()
    y = pheno["y"].to_numpy(int)
    sites = pheno["SITE_ID"].to_numpy()
    site_codes = pheno["site_code"].to_numpy(int)
    sub_ids = pheno.index.to_numpy()

    protocol = cfg.get("protocol", "kfold")
    splits = build_splits(y, sites, protocol=protocol,
                          n_splits=cfg.get("n_splits", 10),
                          n_seeds=cfg.get("n_seeds", 5))
    model_name = cfg["model"]
    family = model_family(model_name)

    fold_records = []
    for fold_name, tr, te in splits:
        seed = int(cfg.get("seed", 0)) + _seed_from_fold(fold_name)
        set_seed(seed)
        rec = _run_one_fold(cfg, family, model_name, ts_list, y, site_codes, sub_ids,
                            tr, te, seed, pheno=pheno)
        rec["fold"] = fold_name
        fold_records.append(rec)
        m = rec["metrics"]
        log.info("[%s] %s bal_acc=%.3f auc=%s (n=%d)", model_name, fold_name,
                 m["bal_acc"], f"{m['auc']:.3f}" if m["auc"] is not None else "n/a",
                 m["n_test"])

    # Underscore-prefixed keys are runtime handles (the phenotype frame), not settings:
    # they must not reach the run id or the saved config.
    public_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    run_id = config_hash({**public_cfg, "protocol": protocol}) + f"_{cfg.get('seed', 0)}"
    blob = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": public_cfg,
        "protocol": protocol,
        "model": model_name,
        "folds": fold_records,
        "aggregate": aggregate([r["metrics"] for r in fold_records]),
        "pooled": pooled_metrics(fold_records),
        "runtime_sec": round(time.time() - t0, 1),
        "underpowered_folds": underpowered_folds(sites, cfg.get("min_site_n", 20))
        if protocol == "loso" else [],
    }
    path = Path(results_dir) / f"{run_id}.json"
    write_json(blob, path)
    log.info("run %s -> %s | bal_acc %.3f +/- %.3f", run_id, path,
             blob["aggregate"]["bal_acc_mean"], blob["aggregate"]["bal_acc_std"])
    return path


def _seed_from_fold(fold_name: str) -> int:
    if fold_name.startswith("seed"):
        return int(fold_name.split("_")[0].removeprefix("seed"))
    return 0


def _fit_connectivity(cfg, ts_list, tr, te):
    """Fit the connectivity estimator on TRAIN subjects only, then transform all."""
    ct = ConnectivityTransformer(kind=cfg.get("conn_kind", "correlation"),
                                 vectorize=True, fisher_z=cfg.get("fisher_z", True))
    ct.fit([ts_list[i] for i in tr])
    X_tr = ct.transform([ts_list[i] for i in tr])
    X_te = ct.transform([ts_list[i] for i in te])
    return ct, X_tr, X_te


def _harmonize(cfg, X_tr, X_te, pheno_tr, pheno_te):
    """Fit ComBat on the training fold, apply to both. Returns X unchanged when
    harmonisation is off."""
    if cfg.get("harmonize", "none") == "none":
        return X_tr, X_te
    cb = ComBatTransformer(unseen_site=cfg.get("unseen_site", "passthrough"))
    covars_tr = pheno_tr[["age", "sex", "y"]]
    # Diagnosis is deliberately zeroed at transform time for test subjects: their labels
    # are not available at inference, so they cannot enter the covariate model.
    covars_te = pheno_te[["age", "sex", "y"]].assign(y=0)
    cb.fit(X_tr, batch=pheno_tr["SITE_ID"].to_numpy(), covars=covars_tr)
    return (cb.transform(X_tr, batch=pheno_tr["SITE_ID"].to_numpy(), covars=covars_tr),
            cb.transform(X_te, batch=pheno_te["SITE_ID"].to_numpy(), covars=covars_te))


def _run_one_fold(cfg, family, model_name, ts_list, y, site_codes, sub_ids, tr, te, seed,
                  pheno=None):
    from asdgnn.models.registry import build_model

    harmonizing = cfg.get("harmonize", "none") != "none"

    if family == "classical" and not harmonizing:
        # Preferred path: the connectivity estimator, scaler and selector all live
        # inside the Pipeline, so fold-safety is structural rather than remembered.
        pipe = build_model(model_name, {**cfg, "seed": seed, "from_timeseries": True})
        from asdgnn.models.classical import decision_scores

        pipe.fit([ts_list[i] for i in tr], y[tr])
        y_prob = decision_scores(pipe, [ts_list[i] for i in te])
        return {
            "metrics": compute_all(y[te], y_prob),
            "y_true": y[te].tolist(),
            "y_prob": np.asarray(y_prob).tolist(),
            "sub_ids": sub_ids[te].tolist(),
        }

    _, X_tr, X_te = _fit_connectivity(cfg, ts_list, tr, te)
    n_rois = ts_list[0].shape[1]

    if harmonizing:
        if pheno is None:
            raise ValueError("harmonisation needs the phenotype frame for site/covariates")
        X_tr, X_te = _harmonize(cfg, X_tr, X_te, pheno.iloc[tr], pheno.iloc[te])

    if family == "classical":
        # ComBat needs the full feature matrix, so connectivity is estimated outside the
        # Pipeline here. Both are still fitted on train indices only; the scaler and
        # selector remain inside.
        from asdgnn.models.classical import decision_scores

        pipe = build_model(model_name, {**cfg, "seed": seed, "from_timeseries": False})
        pipe.fit(X_tr, y[tr])
        y_prob = decision_scores(pipe, X_te)
        return {
            "metrics": compute_all(y[te], y_prob),
            "y_true": y[te].tolist(),
            "y_prob": np.asarray(y_prob).tolist(),
            "sub_ids": sub_ids[te].tolist(),
        }

    if family == "graph":
        from asdgnn.data.graphs_brain import build_brain_graphs
        from asdgnn.train.loop import train_graph_level

        conn_tr = vec_to_matrix(X_tr, n_rois)
        conn_te = vec_to_matrix(X_te, n_rois)
        gkw = dict(node_feature=cfg.get("node_feature", "profile"),
                   edge_strategy=cfg.get("edge_strategy", "topk"),
                   edge_param=cfg.get("edge_param", 0.10),
                   use_abs=cfg.get("use_abs", True))
        g_tr = build_brain_graphs(conn_tr, y[tr], sub_ids=sub_ids[tr],
                                  site_codes=site_codes[tr], **gkw)
        g_te = build_brain_graphs(conn_te, y[te], sub_ids=sub_ids[te],
                                  site_codes=site_codes[te], **gkw)
        # Validation is carved out of TRAIN only.
        i_tr, i_va = inner_val_split(np.arange(len(tr)), y[tr], frac=cfg.get("val_frac", 0.15),
                                     seed=seed)
        model = build_model(model_name, {**cfg, "in_dim": g_tr[0].x.shape[1]})
        return train_graph_level(model, [g_tr[i] for i in i_tr], [g_tr[i] for i in i_va],
                                 g_te, cfg, seed=seed)

    if family == "population":
        return _run_population_fold(cfg, model_name, X_tr, X_te, y, tr, te, sub_ids, seed)

    if family == "braincnn":
        return _run_braincnn_fold(cfg, X_tr, X_te, y, tr, te, sub_ids, n_rois, seed)

    raise NotImplementedError(
        f"family {family!r} is trained stage-wise via scripts/02_run_experiment.py"
    )


def _run_population_fold(cfg, model_name, X_tr, X_te, y, tr, te, sub_ids, seed):
    import numpy as np

    from asdgnn.data.graphs_pop import build_population_graph, reduce_features
    from asdgnn.models.registry import build_model
    from asdgnn.train.loop import train_node_level

    pheno = cfg["_pheno"]
    X_all = np.zeros((len(y), X_tr.shape[1]), dtype=np.float32)
    X_all[tr], X_all[te] = X_tr, X_te
    # Reducer fitted on train rows only, then applied to every node.
    X_red, _ = reduce_features(X_all[tr], X_all, y[tr], method=cfg.get("reduce", "anova"),
                               k=cfg.get("reduce_k", 2000), seed=seed)
    i_tr, i_va = inner_val_split(tr, y, frac=cfg.get("val_frac", 0.15), seed=seed)
    data = build_population_graph(
        X_red, pheno, y, i_tr, i_va, te,
        sim_metric=cfg.get("sim_metric", "correlation"),
        pheno_terms=tuple(cfg.get("pheno_terms", ("sex", "site", "age"))),
        age_tol=cfg.get("age_tol", 2.0), knn=cfg.get("knn", 10),
    )
    model = build_model(model_name, {**cfg, "in_dim": X_red.shape[1]})
    rec = train_node_level(model, data, cfg, seed=seed)
    rec["sub_ids"] = sub_ids[te].tolist()
    rec["n_isolated"] = int(data.n_isolated)
    return rec


def _run_braincnn_fold(cfg, X_tr, X_te, y, tr, te, sub_ids, n_rois, seed):
    import torch

    from asdgnn.models.braincnn import BrainNetCNN
    from asdgnn.train.loop import DEFAULTS, _class_weights, _device, _probs

    c = {**DEFAULTS, **cfg}
    set_seed(seed)
    device = _device(cfg.get("device"))
    A_tr = torch.from_numpy(vec_to_matrix(X_tr, n_rois)).float()
    A_te = torch.from_numpy(vec_to_matrix(X_te, n_rois)).float()
    yt = torch.from_numpy(y[tr]).long()

    model = BrainNetCNN(n_rois=n_rois, dropout=c["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    crit = torch.nn.CrossEntropyLoss(weight=_class_weights(y[tr], device))
    ds = torch.utils.data.TensorDataset(A_tr, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=c["batch_size"], shuffle=True)

    for _ in range(c["epochs"]):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        prob = _probs(model(A_te.to(device)))
    return {
        "metrics": compute_all(y[te], prob),
        "y_true": y[te].tolist(),
        "y_prob": prob.tolist(),
        "sub_ids": sub_ids[te].tolist(),
    }
