from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

KINDS = ("correlation", "partial correlation", "tangent", "covariance", "precision")
STATELESS_KINDS = ("correlation", "covariance")


class ConnectivityTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible connectivity estimator so it can live inside a Pipeline
    and therefore be fold-aware.

    fit(X)        X is a LIST of (T, R) arrays for TRAINING subjects ONLY.
    transform(X)  -> (n, R*(R-1)/2) if vectorize else (n, R, R)

    'tangent' estimates a group geometric-mean covariance during fit. Fitting it on
    all subjects before cross-validation inflates accuracy by roughly 4-6 points of
    pure fantasy, and shows up as tangent beating correlation by a suspicious margin.
    'correlation' is stateless and immune.
    """

    def __init__(self, kind: str = "correlation", vectorize: bool = True,
                 fisher_z: bool = True):
        self.kind = kind
        self.vectorize = vectorize
        self.fisher_z = fisher_z

    def fit(self, X, y=None):
        from nilearn.connectome import ConnectivityMeasure

        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}, expected one of {KINDS}")
        X = [np.asarray(x, dtype=np.float64) for x in X]
        # Always estimate full matrices and vectorise ourselves: nilearn's own
        # vectoriser uses the LOWER triangle, and mixing conventions with
        # vec_to_matrix would silently transpose every ROI pair.
        self.cm_ = ConnectivityMeasure(
            kind=self.kind, vectorize=False, standardize="zscore_sample"
        )
        self.cm_.fit(X)  # TRAIN SUBJECTS ONLY
        self.n_fit_samples_ = len(X)  # asserted by tests/test_no_leakage.py
        self.n_rois_ = X[0].shape[1]
        return self

    def transform(self, X):
        X = [np.asarray(x, dtype=np.float64) for x in X]
        Z = np.asarray(self.cm_.transform(X))  # (n, R, R)
        if self.fisher_z and self.kind == "correlation":
            Z = np.arctanh(np.clip(Z, -0.999999, 0.999999))
        for i in range(len(Z)):
            np.fill_diagonal(Z[i], 0.0)
        Z = Z.astype(np.float32)
        return matrix_to_vec(Z) if self.vectorize else Z


def triu_indices(n_rois: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_rois, k=1)


def matrix_to_vec(M: np.ndarray) -> np.ndarray:
    """(R, R) -> (R(R-1)/2,) or (N, R, R) -> (N, R(R-1)/2). Upper triangle, k=1."""
    M = np.asarray(M)
    if M.ndim == 2:
        i, j = triu_indices(M.shape[0])
        return M[i, j]
    if M.ndim == 3:
        i, j = triu_indices(M.shape[1])
        return M[:, i, j]
    raise ValueError(f"expected 2-D or 3-D, got {M.shape}")


def vec_to_matrix(v: np.ndarray, n_rois: int) -> np.ndarray:
    """Exact inverse of matrix_to_vec for symmetric, zero-diagonal matrices.

    The explainability module maps feature indices back to ROI pairs through this;
    an off-by-one here silently corrupts every brain figure in the paper.
    """
    v = np.asarray(v)
    i, j = triu_indices(n_rois)
    if v.ndim == 1:
        M = np.zeros((n_rois, n_rois), dtype=v.dtype)
        M[i, j] = v
        return M + M.T
    if v.ndim == 2:
        M = np.zeros((v.shape[0], n_rois, n_rois), dtype=v.dtype)
        M[:, i, j] = v
        return M + np.transpose(M, (0, 2, 1))
    raise ValueError(f"expected 1-D or 2-D, got {v.shape}")


def vec_index_to_roi_pair(idx: int, n_rois: int) -> tuple[int, int]:
    i, j = triu_indices(n_rois)
    return int(i[idx]), int(j[idx])


def n_features(n_rois: int) -> int:
    return n_rois * (n_rois - 1) // 2
