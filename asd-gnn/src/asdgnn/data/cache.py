from __future__ import annotations

from pathlib import Path

import numpy as np

from asdgnn.data.connectivity import STATELESS_KINDS
from asdgnn.utils.logging import get_logger

log = get_logger(__name__)
CACHE_DIR = Path("data/cache")


def ts_cache_path(atlas: str, pipeline: str, filt: str, cache_dir=CACHE_DIR) -> Path:
    return Path(cache_dir) / f"ts_{atlas}_{pipeline}_{filt}.npz"


def save_timeseries(ts: dict[int, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in ts.items()})
    return path


def load_timeseries_cache(path: str | Path) -> dict[int, np.ndarray]:
    with np.load(path) as z:
        return {int(k): z[k].astype(np.float32) for k in z.files}


def conn_cache_path(atlas: str, kind: str, fold_hash: str = "all",
                    cache_dir=CACHE_DIR) -> Path:
    """Only stateless kinds may be cached across folds. Tangent and shrinkage-based
    partial correlation are fold-dependent; caching them by anything but fold identity
    reintroduces exactly the leak the ConnectivityTransformer removes."""
    if kind not in STATELESS_KINDS and fold_hash == "all":
        raise ValueError(
            f"refusing to cache fold-dependent kind={kind!r} without a fold hash"
        )
    return Path(cache_dir) / f"conn_{atlas}_{kind.replace(' ', '')}_{fold_hash}.npy"


def cached_array(path: str | Path, build):
    path = Path(path)
    if path.exists():
        log.info("cache hit %s", path)
        return np.load(path)
    arr = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    log.info("cache write %s %s", path, arr.shape)
    return arr
