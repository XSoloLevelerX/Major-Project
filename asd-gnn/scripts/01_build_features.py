#!/usr/bin/env python
"""Load the .1D time series, run QC, cache them, and generate the shared split file.

Connectivity is deliberately NOT built here: it is fold-dependent (tangent, partial
with shrinkage) and belongs inside the CV loop. Only the time series are cached.

    python scripts/01_build_features.py --atlas cc200
    python scripts/01_build_features.py --synthetic     # no download needed
"""
from __future__ import annotations

import argparse
from pathlib import Path

from asdgnn.data.cache import save_timeseries, ts_cache_path
from asdgnn.data.download import file_id_from_path, local_paths
from asdgnn.data.phenotype import cohort_report, load_phenotype
from asdgnn.data.timeseries import load_timeseries, qc_filter
from asdgnn.train.splits import build_splits, cache_splits, site_proportion_drift
from asdgnn.utils.logging import get_logger, write_json

log = get_logger("01_build_features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="cc200")
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--min-timepoints", type=int, default=100)
    ap.add_argument("--max-mean-fd", type=float, default=None)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        from asdgnn.data.synthetic import make_cohort

        ts, pheno = make_cohort(n_subjects=160, n_rois=30, n_sites=5, seed=0)
        filt = "synthetic"
    else:
        paths = local_paths(args.data_dir, args.atlas, args.pipeline)
        if not paths:
            raise SystemExit("no .1D files found — run scripts/00_download.py first")
        csv = next(Path(args.data_dir).rglob("Phenotypic_V1_0b_preprocessed1.csv"))
        pheno = load_phenotype(csv, [file_id_from_path(p) for p in paths])
        ts = load_timeseries(paths, pheno)
        ts, pheno, _ = qc_filter(ts, pheno, min_timepoints=args.min_timepoints,
                                 max_mean_fd=args.max_mean_fd)
        filt = "filt_noglobal"

    pheno = pheno.loc[sorted(ts)]  # the canonical ordering every module assumes
    cohort_report(pheno)
    cache = save_timeseries(ts, ts_cache_path(args.atlas, args.pipeline, filt))
    pheno.to_csv(f"data/processed/pheno_{args.atlas}_{filt}.csv")
    log.info("cached %d time series -> %s", len(ts), cache)

    y, sites = pheno["y"].to_numpy(), pheno["SITE_ID"].to_numpy()
    kfold = build_splits(y, sites, "kfold", args.n_splits, args.n_seeds)
    loso = build_splits(y, sites, "loso")
    cache_splits(kfold, pheno.index, "kfold")
    cache_splits(loso, pheno.index, "loso")

    drift = [site_proportion_drift(sites, tr, te) for _, tr, te in kfold]
    write_json({"n_subjects": len(pheno), "n_rois": int(next(iter(ts.values())).shape[1]),
                "n_kfold_folds": len(kfold), "n_loso_folds": len(loso),
                "max_site_proportion_drift": max(drift)},
               "results/qc/features.json")
    log.info("splits cached | max per-fold site-proportion drift %.3f", max(drift))


if __name__ == "__main__":
    main()
