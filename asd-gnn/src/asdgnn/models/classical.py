from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from asdgnn.data.connectivity import ConnectivityTransformer

# The Pipeline is the point: with the connectivity estimator, the scaler and the
# feature selector all INSIDE it, cross_validate refits them per fold and fold-safety
# is structural rather than a thing you have to remember. Feature selection outside
# the CV loop is the second most common leak after tangent-space fitting.


def _head(cfg: dict):
    name = cfg.get("model", "svm")
    C = float(cfg.get("C", 1.0))
    seed = int(cfg.get("seed", 0))
    if name == "svm":
        return LinearSVC(C=C, max_iter=10000, class_weight="balanced", dual="auto",
                         random_state=seed)
    if name == "svm_rbf":
        return SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced",
                   probability=True, random_state=seed)
    if name == "ridge":
        return RidgeClassifier(alpha=float(cfg.get("alpha", 1.0)),
                               class_weight="balanced", random_state=seed)
    if name == "logreg":
        return LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                                  random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 500)),
            max_depth=cfg.get("max_depth"), class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )
    if name == "mlp":
        # `hidden` is an int everywhere else in the config (the GNN width), so accept
        # both that and an explicit per-layer tuple.
        hidden = cfg.get("hidden", (64,))
        hidden = (int(hidden),) if isinstance(hidden, int) else tuple(hidden)
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=float(cfg.get("alpha", 1e-3)), max_iter=1000, random_state=seed,
        )
    raise ValueError(f"unknown classical model {name!r}")


def build_classical(cfg: dict) -> Pipeline:
    """Full fold-safe pipeline: raw time series in, class scores out."""
    steps = []
    if cfg.get("from_timeseries", True):
        steps.append(("conn", ConnectivityTransformer(
            kind=cfg.get("conn_kind", "correlation"), vectorize=True,
            fisher_z=cfg.get("fisher_z", True))))
    steps.append(("scale", StandardScaler()))
    k = cfg.get("select_k", 2000)
    if k:
        steps.append(("select", SelectKBest(f_classif, k=k)))
    steps.append(("clf", _head(cfg)))
    return Pipeline(steps)


def decision_scores(pipe: Pipeline, X):
    """Probability of class 1 where available, otherwise a min-max squashed decision
    function — AUC only needs a ranking, but the JSON schema wants [0, 1]."""
    import numpy as np

    clf = pipe[-1]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    d = pipe.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))
