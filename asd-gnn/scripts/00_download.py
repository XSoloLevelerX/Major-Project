#!/usr/bin/env python
"""Download ABIDE-I ROI time series and write the cohort QC report.

    python scripts/00_download.py --atlas cc200
    python scripts/00_download.py --atlas cc200 --n-subjects 40   # quick smoke run
"""
from __future__ import annotations

import argparse
from pathlib import Path

from asdgnn.data.download import fetch, file_id_from_path, local_paths
from asdgnn.data.phenotype import cohort_report, load_phenotype
from asdgnn.utils.logging import get_logger

log = get_logger("00_download")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="cc200")
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--n-subjects", type=int, default=None)
    ap.add_argument("--global-signal", action="store_true")
    ap.add_argument("--no-band-pass", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="use already-downloaded files, no network")
    args = ap.parse_args()

    band_pass = not args.no_band_pass
    if not args.skip_fetch:
        fetch(atlas=args.atlas, pipeline=args.pipeline, band_pass=band_pass,
              global_signal=args.global_signal, data_dir=args.data_dir,
              n_subjects=args.n_subjects)

    paths = local_paths(args.data_dir, args.atlas, args.pipeline, band_pass,
                        args.global_signal)
    log.info("%d local .1D files for atlas=%s", len(paths), args.atlas)

    csv = next(Path(args.data_dir).rglob("Phenotypic_V1_0b_preprocessed1.csv"), None)
    if csv is None:
        raise SystemExit("phenotype CSV not found under the data dir — did the fetch run?")
    pheno = load_phenotype(csv, [file_id_from_path(p) for p in paths])
    cohort_report(pheno)
    log.info("cohort: N=%d  ASD=%d  TC=%d  sites=%d", len(pheno), int(pheno.y.sum()),
             int((pheno.y == 0).sum()), pheno.SITE_ID.nunique())


if __name__ == "__main__":
    main()
