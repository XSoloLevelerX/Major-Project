from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, train_test_split

from asdgnn.utils.logging import get_logger, read_json, write_json

log = get_logger(__name__)

Split = tuple[str, np.ndarray, np.ndarray]


def kfold_splits(y, sites, n_splits: int = 10, n_seeds: int = 5) -> Iterator[Split]:
    """Stratified k-fold over the label.

    StratifiedGroupKFold is WRONG here — grouping by site would hold out whole sites,
    which is the LOSO protocol, not this one. We stratify on y and check afterwards
    that site proportions survive.
    """
    y = np.asarray(y)
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield f"seed{seed}_fold{k}", tr, te


def loso_splits(sites, min_site_n: int = 20) -> Iterator[Split]:
    """Leave-one-site-out. Small sites are still run but flagged underpowered
    downstream (see `underpowered_folds`)."""
    sites = np.asarray(sites)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(np.zeros(len(sites)), groups=sites):
        yield str(sites[te][0]), tr, te


def underpowered_folds(sites, min_site_n: int = 20) -> list[str]:
    sites = np.asarray(sites)
    vals, counts = np.unique(sites, return_counts=True)
    return [str(v) for v, c in zip(vals, counts) if c < min_site_n]


def inner_val_split(train_idx, y, sites=None, frac: float = 0.15, seed: int = 0):
    """Carve validation OUT OF TRAIN ONLY — early stopping and HP search use this."""
    train_idx = np.asarray(train_idx)
    y_tr = np.asarray(y)[train_idx]
    strat = y_tr if np.min(np.bincount(y_tr, minlength=2)) >= 2 else None
    tr, va = train_test_split(train_idx, test_size=frac, random_state=seed, stratify=strat)
    return np.sort(tr), np.sort(va)


def site_proportion_drift(sites, tr, te) -> float:
    """Max absolute difference in per-site share between train and test. A k-fold split
    that is honest about sites keeps this small."""
    sites = np.asarray(sites)
    vals = np.unique(sites)
    p_tr = np.array([(sites[tr] == v).mean() for v in vals])
    p_te = np.array([(sites[te] == v).mean() for v in vals])
    return float(np.abs(p_tr - p_te).max())


def build_splits(y, sites, protocol: str = "kfold", n_splits: int = 10, n_seeds: int = 5,
                 min_site_n: int = 20) -> list[Split]:
    if protocol == "kfold":
        return list(kfold_splits(y, sites, n_splits=n_splits, n_seeds=n_seeds))
    if protocol == "loso":
        return list(loso_splits(sites, min_site_n=min_site_n))
    raise ValueError(f"unknown protocol {protocol!r}")


def cache_splits(splits: list[Split], sub_ids, protocol: str,
                 path: str | Path = "data/processed/splits.json") -> Path:
    """Splits are generated ONCE and shared by every model. This is what makes DeLong
    tests and paired comparisons valid across models."""
    path = Path(path)
    blob = read_json(path) if path.exists() else {}
    blob[protocol] = {
        "sub_ids": [int(s) for s in sub_ids],
        "folds": [
            {"fold": name, "train": np.asarray(tr).tolist(), "test": np.asarray(te).tolist()}
            for name, tr, te in splits
        ],
    }
    write_json(blob, path)
    log.info("cached %d %s folds -> %s", len(splits), protocol, path)
    return path


def load_splits(protocol: str, sub_ids=None,
                path: str | Path = "data/processed/splits.json") -> list[Split]:
    blob = read_json(path)
    if protocol not in blob:
        raise KeyError(f"protocol {protocol!r} not in {path}")
    entry = blob[protocol]
    if sub_ids is not None and [int(s) for s in sub_ids] != entry["sub_ids"]:
        raise ValueError(
            "cached splits were built for a different cohort ordering; rebuild them"
        )
    return [
        (f["fold"], np.asarray(f["train"], int), np.asarray(f["test"], int))
        for f in entry["folds"]
    ]
