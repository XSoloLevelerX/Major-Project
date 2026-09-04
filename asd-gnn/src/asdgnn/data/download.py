from __future__ import annotations

import time
from pathlib import Path

from asdgnn.utils.logging import get_logger

log = get_logger(__name__)

ATLASES = ("cc200", "cc400", "aal", "ho", "dosenbach160", "ez", "tt")


def fetch(
    atlas: str = "cc200",
    pipeline: str = "cpac",
    band_pass: bool = True,
    global_signal: bool = False,
    quality_checked: bool = True,
    data_dir: str = "data/raw",
    n_subjects: int | None = None,
    max_retries: int = 5,
):
    """Download ABIDE-I ROI time series. Returns the nilearn Bunch.

    Never request 'func_preproc' — that is ~50MB/subject of 4D NIfTI we do not need.
    quality_checked=True + cpac + filt_noglobal is the canonical 871-subject cohort.
    """
    if atlas not in ATLASES:
        raise ValueError(f"unknown atlas {atlas!r}, expected one of {ATLASES}")
    from nilearn.datasets import fetch_abide_pcp

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        data_dir=data_dir,
        pipeline=pipeline,
        band_pass_filtering=band_pass,
        global_signal_regression=global_signal,
        derivatives=[f"rois_{atlas}"],
        quality_checked=quality_checked,
        verbose=1,
    )
    if n_subjects is not None:
        kwargs["n_subjects"] = n_subjects

    # The fetcher stalls mid-download fairly often; it resumes from where it stopped.
    for attempt in range(1, max_retries + 1):
        try:
            bunch = fetch_abide_pcp(**kwargs)
            log.info("fetched atlas=%s pipeline=%s", atlas, pipeline)
            return bunch
        except Exception as exc:  # noqa: BLE001 - network layer raises many types
            log.warning("fetch attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt == max_retries:
                raise
            time.sleep(5 * attempt)


def local_paths(data_dir: str = "data/raw", atlas: str = "cc200", pipeline: str = "cpac",
                band_pass: bool = True, global_signal: bool = False) -> list[Path]:
    """List already-downloaded .1D files without touching the network."""
    strategy = (
        f"{'filt' if band_pass else 'nofilt'}_{'global' if global_signal else 'noglobal'}"
    )
    root = Path(data_dir) / "ABIDE_pcp" / pipeline / strategy
    return sorted(root.glob(f"*_rois_{atlas}.1D"))


def file_id_from_path(path: str | Path) -> str:
    """'.../NYU_0051015_rois_cc200.1D' -> 'NYU_0051015'."""
    stem = Path(path).name
    return stem.split("_rois_")[0]
