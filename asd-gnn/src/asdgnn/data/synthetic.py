from __future__ import annotations

import numpy as np
import pandas as pd

# A synthetic cohort with the same contracts as ABIDE: variable time-series lengths,
# multiple sites with additive site effects, and a weak, genuinely learnable class
# signal in a handful of ROI pairs. Used by the test suite and by the notebook's smoke
# mode so nothing depends on a 6 GB download to be verified.


def make_cohort(n_subjects: int = 120, n_rois: int = 20, n_sites: int = 4,
                signal: float = 0.35, site_effect: float = 0.5,
                t_range: tuple[int, int] = (110, 180), seed: int = 0):
    rng = np.random.RandomState(seed)
    y = rng.binomial(1, 0.5, n_subjects)
    site_idx = rng.randint(0, n_sites, n_subjects)

    # The class signal lives in a fixed set of ROI pairs, mediated by shared latent
    # factors, so it survives a correlation estimate rather than being pure noise.
    k = max(2, n_rois // 5)
    signal_rois = np.arange(k)
    site_loading = rng.randn(n_sites, n_rois)

    ts, rows = {}, []
    for i in range(n_subjects):
        T = int(rng.randint(*t_range))
        base = rng.randn(T, n_rois).astype(np.float32)
        latent = rng.randn(T, 1).astype(np.float32)
        if y[i] == 1:
            base[:, signal_rois] += signal * latent
        else:
            base[:, signal_rois] -= signal * latent
        site_latent = rng.randn(T, 1).astype(np.float32)
        base += site_effect * site_latent * site_loading[site_idx[i]].astype(np.float32)
        sub_id = 50000 + i
        ts[sub_id] = base
        rows.append({
            "SUB_ID": sub_id,
            "FILE_ID": f"SITE{site_idx[i]}_{sub_id}",
            "SITE_ID": f"SITE{site_idx[i]}",
            "y": int(y[i]),
            "age": float(rng.uniform(7, 30)),
            "sex": int(rng.binomial(1, 0.8)),
            "fiq": float(rng.normal(105, 15)),
            "eye": int(rng.binomial(1, 0.7)),
            "mean_fd": float(abs(rng.normal(0.1, 0.05))),
        })

    pheno = pd.DataFrame(rows).set_index("SUB_ID")
    pheno["site_code"] = pd.Categorical(pheno["SITE_ID"]).codes.astype(int)
    return ts, pheno


def make_conn(n: int = 40, n_rois: int = 12, seed: int = 0):
    """Symmetric, zero-diagonal connectivity matrices with labels."""
    rng = np.random.RandomState(seed)
    y = rng.binomial(1, 0.5, n)
    conn = np.zeros((n, n_rois, n_rois), dtype=np.float32)
    for i in range(n):
        A = rng.randn(n_rois, n_rois).astype(np.float32)
        A = (A + A.T) / 2
        A[:3, :3] += 1.5 * y[i]
        np.fill_diagonal(A, 0.0)
        conn[i] = A
    return conn, y
