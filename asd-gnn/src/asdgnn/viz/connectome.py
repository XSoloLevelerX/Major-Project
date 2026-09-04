from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def connectivity_heatmap(M, path="results/figures/conn_heatmap.png", title="",
                         vmax: float | None = None):
    M = np.asarray(M)
    v = vmax if vmax is not None else np.percentile(np.abs(M), 99)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(title=title, xlabel="ROI", ylabel="ROI")
    return _save(fig, path)


def group_difference_map(conn, y, path="results/figures/group_diff.png"):
    """Mean ASD matrix minus mean TC matrix. Descriptive only — no inference here."""
    conn = np.asarray(conn)
    y = np.asarray(y)
    D = conn[y == 1].mean(0) - conn[y == 0].mean(0)
    return connectivity_heatmap(D, path, title="mean(ASD) - mean(TC) connectivity")


def top_edges_glass_brain(edge_matrix, coords, path="results/figures/glass_brain.png",
                          n_edges: int = 50, title=""):
    """Requires ROI centroid coordinates; falls back to a heatmap when they are absent."""
    from nilearn import plotting

    M = np.asarray(edge_matrix).copy()
    if coords is None:
        return connectivity_heatmap(M, path, title=title)
    flat = np.abs(np.triu(M, 1)).ravel()
    if n_edges < flat.size:
        thr = np.partition(flat, -n_edges)[-n_edges]
        M[np.abs(M) < thr] = 0.0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    display = plotting.plot_connectome(M, coords, title=title, colorbar=True,
                                       edge_threshold=None)
    display.savefig(path, dpi=200)
    display.close()
    return Path(path)
