from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asdgnn.data.download import file_id_from_path
from asdgnn.utils.logging import get_logger, write_json

log = get_logger(__name__)


def load_timeseries(paths: list[str | Path], pheno: pd.DataFrame) -> dict[int, np.ndarray]:
    """Load .1D ROI time series. Returns SUB_ID -> (T, R) float32."""
    fid_to_sub = {fid: sub for sub, fid in pheno["FILE_ID"].items()}
    ts: dict[int, np.ndarray] = {}
    for p in paths:
        fid = file_id_from_path(p)
        sub = fid_to_sub.get(fid)
        if sub is None:
            continue
        arr = np.loadtxt(p)  # numpy skips '#'-prefixed header lines
        if arr.ndim != 2:
            raise ValueError(f"bad shape {arr.shape} in {p}")
        ts[int(sub)] = arr.astype(np.float32)
    log.info("loaded %d time series from %d files", len(ts), len(paths))
    return ts


def qc_filter(
    ts: dict[int, np.ndarray],
    pheno: pd.DataFrame,
    min_timepoints: int = 100,
    max_mean_fd: float | None = None,
    drop_constant_rois: bool = True,
    report_path: str | Path | None = "results/qc/dropped.json",
) -> tuple[dict[int, np.ndarray], pd.DataFrame, dict]:
    """Returns filtered ts, filtered pheno, and a report of what was dropped and why."""
    dropped: dict[str, str] = {}
    kept = dict(ts)

    n_rois = {a.shape[1] for a in kept.values()}
    if len(n_rois) > 1:
        raise ValueError(f"inconsistent ROI counts across subjects: {sorted(n_rois)}")
    R = n_rois.pop() if n_rois else 0

    for sub, arr in list(kept.items()):
        if arr.shape[0] < min_timepoints:
            dropped[str(sub)] = f"too_few_timepoints({arr.shape[0]}<{min_timepoints})"
            kept.pop(sub)
            continue
        if not np.isfinite(arr).all():
            dropped[str(sub)] = "non_finite_values"
            kept.pop(sub)

    if max_mean_fd is not None:
        for sub in list(kept):
            fd = pheno["mean_fd"].get(sub, np.nan)
            if np.isfinite(fd) and fd > max_mean_fd:
                dropped[str(sub)] = f"mean_fd({fd:.3f}>{max_mean_fd})"
                kept.pop(sub)

    # A parcel with zero variance in ANY subject yields NaN correlations for that subject.
    # Brain graphs need an identical node set, so the ROI is dropped globally.
    dropped_rois: list[int] = []
    if drop_constant_rois and kept:
        bad = np.zeros(R, dtype=bool)
        for arr in kept.values():
            bad |= arr.std(axis=0) < 1e-8
        dropped_rois = np.flatnonzero(bad).tolist()
        if dropped_rois:
            good = ~bad
            kept = {s: a[:, good] for s, a in kept.items()}
            log.info("dropped %d constant ROIs -> R=%d", len(dropped_rois), int(good.sum()))

    pheno_f = pheno.loc[pheno.index.isin(kept)].copy()
    report = {
        "n_in": len(ts),
        "n_out": len(kept),
        "n_rois_in": R,
        "n_rois_out": int(next(iter(kept.values())).shape[1]) if kept else 0,
        "dropped_subjects": dropped,
        "dropped_rois": dropped_rois,
        "criteria": {
            "min_timepoints": min_timepoints,
            "max_mean_fd": max_mean_fd,
            "drop_constant_rois": drop_constant_rois,
        },
    }
    if report_path:
        write_json(report, report_path)
    log.info("QC: kept %d/%d subjects", report["n_out"], report["n_in"])
    return kept, pheno_f, report


def stack_aligned(ts: dict[int, np.ndarray], pheno: pd.DataFrame) -> list[np.ndarray]:
    """Time series as a list ordered exactly like `pheno.index` — the ordering contract
    every downstream module relies on."""
    return [ts[int(s)] for s in pheno.index]
