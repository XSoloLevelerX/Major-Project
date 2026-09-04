from __future__ import annotations

import numpy as np
import pandas as pd

from asdgnn.data.cache import load_timeseries_cache, ts_cache_path

# Shared by every script. Scripts are named 00_.., 01_.. and so cannot import each
# other; anything two of them need lives here instead.


def load_cohort(atlas: str = "cc200", pipeline: str = "cpac", synthetic: bool = False,
                n_subjects: int = 160, n_rois: int = 30, n_sites: int = 5,
                seed: int = 0) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Returns (time series ordered exactly like pheno.index, phenotype frame)."""
    if synthetic:
        from asdgnn.data.synthetic import make_cohort

        ts, pheno = make_cohort(n_subjects=n_subjects, n_rois=n_rois, n_sites=n_sites,
                                seed=seed)
        pheno = pheno.loc[sorted(ts)]
        return [ts[s] for s in pheno.index], pheno

    filt = "filt_noglobal"
    ts = load_timeseries_cache(ts_cache_path(atlas, pipeline, filt))
    pheno = pd.read_csv(f"data/processed/pheno_{atlas}_{filt}.csv", index_col=0)
    pheno = pheno.loc[sorted(ts)]
    return [ts[int(s)] for s in pheno.index], pheno
