#!/usr/bin/env python
"""Regenerate every paper figure from results/runs/*.json.

    python scripts/04_make_figures.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asdgnn.utils.logging import get_logger
from asdgnn.viz.curves import kfold_vs_loso_scatter, per_site_bars, pr_panel, roc_panel
from asdgnn.viz.tables import load_runs

log = get_logger("04_make_figures")


def is_default_cfg(blob) -> bool:
    """The unablated configuration: no harmonisation, Pearson correlation."""
    cfg = blob.get("config", {})
    conn = cfg.get("conn_kind", cfg.get("conn", "correlation"))
    return cfg.get("harmonize", "none") == "none" and conn == "correlation"


def cfg_slug(blob) -> str:
    """A filename-safe identifier that distinguishes runs of the same model."""
    cfg = blob.get("config", {})
    parts = [blob["model"]]
    conn = cfg.get("conn_kind", cfg.get("conn", "correlation"))
    if conn and conn != "correlation":
        parts.append(str(conn).split()[0])
    if cfg.get("harmonize", "none") != "none":
        parts.append(str(cfg["harmonize"]))
        if cfg.get("unseen_site"):
            parts.append(str(cfg["unseen_site"]))
    return "_".join(parts)


def pooled(blob):
    y = np.concatenate([f["y_true"] for f in blob["folds"]])
    p = np.concatenate([f["y_prob"] for f in blob["folds"]])
    return y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()

    results_dir, fig_dir = Path(args.results_dir), Path(args.fig_dir)
    blobs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    if not blobs:
        raise SystemExit(f"no runs in {results_dir}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # One curve per MODEL only holds if we first restrict to a single comparable
    # configuration. Keying a dict on the model name across the whole grid silently keeps
    # whichever run happened to be iterated last, so the curve labelled "svm" could be the
    # ComBat one. Compare models at the default configuration; the ablations have their own
    # tables.
    for protocol in {b["protocol"] for b in blobs}:
        curves = {b["model"]: pooled(b) for b in blobs
                  if b["protocol"] == protocol and is_default_cfg(b)}
        curves = {k: v for k, v in curves.items() if len(np.unique(v[0])) == 2}
        if curves:
            written.append(roc_panel(curves, fig_dir / f"roc_{protocol}.png"))
            written.append(pr_panel(curves, fig_dir / f"pr_{protocol}.png"))

    df = load_runs(results_dir)
    if df["protocol"].nunique() > 1:
        written.append(kfold_vs_loso_scatter(df, fig_dir / "kfold_vs_loso.png"))

    # Filenames must carry the full configuration, or four SVM LOSO runs overwrite one file.
    for b in blobs:
        if b["protocol"] == "loso":
            written.append(per_site_bars(b["folds"],
                                         fig_dir / f"per_site_{cfg_slug(b)}.png"))

    for p in written:
        log.info("wrote %s", p)
    print(fig_dir.as_posix())


if __name__ == "__main__":
    main()
