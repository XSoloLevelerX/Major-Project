from __future__ import annotations

import numpy as np
import pandas as pd

from asdgnn.utils.logging import get_logger

log = get_logger(__name__)


def roi_to_labels(atlas: str = "cc200", kept_rois: np.ndarray | None = None) -> pd.DataFrame:
    """ROI index -> name and MNI coordinates.

    `kept_rois` must be the surviving ROI indices from timeseries.qc_filter, otherwise
    every label is shifted by however many constant parcels were dropped.
    """
    from nilearn import datasets

    if atlas == "aal":
        a = datasets.fetch_atlas_aal()
        df = pd.DataFrame({"name": a.labels})
    elif atlas == "ho":
        a = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        df = pd.DataFrame({"name": [str(x) for x in a.labels[1:]]})
    elif atlas == "dosenbach160":
        a = datasets.fetch_coords_dosenbach_2010()
        df = pd.DataFrame({"name": a.labels, "x": a.rois["x"], "y": a.rois["y"],
                           "z": a.rois["z"]})
    else:
        # CC200 / CC400 ship no label list with nilearn; parcels are numbered.
        n = {"cc200": 200, "cc400": 392}.get(atlas)
        if n is None:
            raise ValueError(f"no label source for atlas {atlas!r}")
        df = pd.DataFrame({"name": [f"{atlas.upper()}_{i:03d}" for i in range(n)]})

    df.index.name = "roi_idx"
    if kept_rois is not None:
        df = df.iloc[np.asarray(kept_rois)].reset_index(drop=True)
        df.index.name = "roi_idx"
    return df


YEO7 = ("visual", "somatomotor", "dorsal_attention", "ventral_attention",
        "limbic", "frontoparietal", "default")


def assign_yeo_networks(coords: np.ndarray) -> np.ndarray:
    """Nearest-network assignment for ROI centroids using the Yeo 7-network atlas."""
    from nilearn import datasets, image

    yeo = datasets.fetch_atlas_yeo_2011()
    img = image.load_img(yeo["thin_7"])
    data = np.asarray(img.get_fdata()).squeeze()
    inv = np.linalg.inv(img.affine)
    coords = np.asarray(coords, float)
    vox = np.round(inv[:3, :3] @ coords.T + inv[:3, [3]]).astype(int).T
    labels = []
    for v in vox:
        v = np.clip(v, 0, np.array(data.shape) - 1)
        labels.append(int(data[tuple(v)]))
    return np.asarray(labels)


def enrichment_test(top_rois, network_assignment, n_perm: int = 5000,
                    seed: int = 0) -> dict:
    """Is the top-ROI set enriched for any network beyond chance? Permutation over
    random ROI sets of the same size."""
    rng = np.random.RandomState(seed)
    net = np.asarray(network_assignment)
    top = np.asarray(top_rois, int)
    k, n = len(top), len(net)
    out = {}
    for label in np.unique(net):
        obs = int((net[top] == label).sum())
        null = np.array([
            (net[rng.choice(n, k, replace=False)] == label).sum() for _ in range(n_perm)
        ])
        out[str(label)] = {
            "observed": obs,
            "expected": float(null.mean()),
            "p": float((int((null >= obs).sum()) + 1) / (n_perm + 1)),
        }
    return out


def top_roi_table(scores: np.ndarray, stds: np.ndarray, labels: pd.DataFrame,
                  k: int = 20) -> pd.DataFrame:
    idx = np.argsort(-np.asarray(scores))[:k]
    return pd.DataFrame({
        "rank": np.arange(1, len(idx) + 1),
        "roi_idx": idx,
        "name": labels["name"].to_numpy()[idx] if "name" in labels else idx,
        "score": np.asarray(scores)[idx],
        "std_across_folds": np.asarray(stds)[idx],
    })
