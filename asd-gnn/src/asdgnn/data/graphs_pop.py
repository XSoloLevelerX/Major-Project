from __future__ import annotations

import numpy as np
import pandas as pd

from asdgnn.utils.logging import get_logger

log = get_logger(__name__)


def similarity(X: np.ndarray, metric: str = "correlation") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if metric == "correlation":
        Xc = X - X.mean(axis=1, keepdims=True)
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
        S = Xc @ Xc.T
    elif metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T
    elif metric == "rbf":
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        S = np.exp(-d2 / (np.median(d2[d2 > 0]) + 1e-12))
    else:
        raise ValueError(f"unknown sim_metric {metric!r}")
    np.fill_diagonal(S, 0.0)
    return S


def phenotypic_affinity(pheno: pd.DataFrame, terms=("sex", "site", "age"),
                        age_tol: float = 2.0) -> np.ndarray:
    n = len(pheno)
    aff = np.zeros((n, n), dtype=np.float64)
    for term in terms:
        if term == "age":
            age = pheno["age"].to_numpy(float)
            aff += (np.abs(age[:, None] - age[None, :]) < age_tol).astype(float)
        elif term == "site":
            s = pheno["SITE_ID"].to_numpy()
            aff += (s[:, None] == s[None, :]).astype(float)
        else:
            v = pheno[term].to_numpy()
            aff += (v[:, None] == v[None, :]).astype(float)
    np.fill_diagonal(aff, 0.0)
    return aff


def knn_sparsify(A: np.ndarray, k: int = 10) -> np.ndarray:
    n = A.shape[0]
    k = min(k, n - 1)
    idx = np.argsort(-A, axis=1)[:, :k]
    mask = np.zeros_like(A, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    return np.where(mask, A, 0.0)


def build_population_graph(
    X: np.ndarray,
    pheno: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    sim_metric: str = "correlation",
    pheno_terms: tuple = ("sex", "site", "age"),
    age_tol: float = 2.0,
    knn: int = 10,
):
    """One graph for the whole cohort: node = subject, node-level transductive task.

    A_ij = sim(x_i, x_j) * (1[sex_i==sex_j] + 1[site_i==site_j] + 1[|age_i-age_j|<tol])

    `X` must already be dimensionality-reduced with a reducer fitted on the TRAIN MASK
    ONLY — 19,900-dim node features on 871 nodes will not train.

    Transductive is not leaky but it is close: test node FEATURES are visible, test
    LABELS are not, and `train_mask` is the only mask the loss may ever see.

    Under LOSO the `site` term makes the held-out site a disconnected component with no
    informative neighbours, so the model predicts the majority class. That is a real
    finding to report, not a bug to hide — run it both with and without the site term.
    """
    import torch
    from torch_geometric.data import Data

    X = np.asarray(X, dtype=np.float32)
    S = similarity(X, sim_metric)
    aff = phenotypic_affinity(pheno, pheno_terms, age_tol) if pheno_terms else np.ones_like(S)
    A = np.abs(S) * aff
    A = knn_sparsify(A, knn)

    src, dst = np.nonzero(A)
    data = Data(
        x=torch.from_numpy(X),
        edge_index=torch.from_numpy(np.stack([src, dst])).long(),
        edge_attr=torch.from_numpy(A[src, dst].astype(np.float32)).unsqueeze(-1),
        y=torch.from_numpy(np.asarray(y).astype(np.int64)),
    )
    n = len(y)
    for name, m in [("train_mask", train_mask), ("val_mask", val_mask),
                    ("test_mask", test_mask)]:
        t = torch.zeros(n, dtype=torch.bool)
        t[torch.as_tensor(np.asarray(m), dtype=torch.long)] = True
        setattr(data, name, t)
    data.site = torch.from_numpy(pheno["site_code"].to_numpy().astype(np.int64))

    deg = (A > 0).sum(1)
    iso = int((deg == 0).sum())
    iso_test = int((deg[np.asarray(test_mask)] == 0).sum())
    data.n_isolated = iso
    log.info(
        "population graph: %d nodes, %d edges, mean degree %.1f, %d isolated (%d in test)",
        n, int(data.edge_index.shape[1]), float(deg.mean()), iso, iso_test,
    )
    if iso_test:
        log.warning(
            "%d test nodes are isolated — with pheno_terms=%s under LOSO this is the "
            "expected disconnection behaviour, report it", iso_test, pheno_terms
        )
    return data


def reduce_features(X_train: np.ndarray, X_all: np.ndarray, y_train: np.ndarray,
                    method: str = "anova", k: int = 2000, seed: int = 0):
    """Fit the reducer on TRAIN ROWS ONLY, then apply it to every node."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif

    k = min(k, X_train.shape[1])
    if method == "anova":
        red = SelectKBest(f_classif, k=k)
    elif method == "pca":
        red = PCA(n_components=min(k, X_train.shape[0] - 1), random_state=seed)
    else:
        raise ValueError(f"unknown method {method!r}")
    red.fit(X_train, y_train)
    return red.transform(X_all), red
