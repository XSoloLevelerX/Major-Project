from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from asdgnn.utils.logging import get_logger

log = get_logger(__name__)

UNSEEN_SITE_POLICIES = ("passthrough", "transductive", "reference")


class ComBatTransformer(BaseEstimator, TransformerMixin):
    """Fit ComBat site parameters on the training folds and apply them to test.

    Covariates must preserve biological variance — age, sex, diagnosis — otherwise
    ComBat removes the signal along with the batch effect.

    Under LOSO the test site was never seen during fit, so no parameters exist for it.
    Three defensible options, selected by `unseen_site`:

      'passthrough'  leave the test site un-harmonised. Honest; measures what ComBat
                     buys on the training distribution only.
      'transductive' estimate the test site's parameters from its own unlabeled data.
                     Must be declared in the paper as transductive harmonisation.
      'reference'    map the unseen site onto the reference batch (grand mean/var).

    This limitation belongs in the Discussion, and it is the strongest argument for the
    adversarial site-invariance head, which has no such problem.
    """

    def __init__(self, unseen_site: str = "passthrough", eb: bool = True,
                 parametric: bool = True):
        if unseen_site not in UNSEEN_SITE_POLICIES:
            raise ValueError(f"unseen_site must be one of {UNSEEN_SITE_POLICIES}")
        self.unseen_site = unseen_site
        self.eb = eb
        self.parametric = parametric

    def fit(self, X, y=None, batch=None, covars=None):
        X = np.asarray(X, dtype=np.float64)
        batch = np.asarray(batch)
        self.sites_ = np.unique(batch)
        self.grand_mean_ = X.mean(axis=0)
        self.grand_std_ = X.std(axis=0) + 1e-8
        # Location/scale parameters per site on the residuals of the covariate model.
        self.beta_, resid = _fit_covariate_model(X, covars)
        self.gamma_, self.delta_ = {}, {}
        for s in self.sites_:
            m = batch == s
            r = resid[m]
            self.gamma_[s] = r.mean(axis=0)
            self.delta_[s] = r.std(axis=0) + 1e-8
        if self.eb:
            self.gamma_, self.delta_ = _empirical_bayes(self.gamma_, self.delta_)
        self.n_fit_samples_ = len(X)
        return self

    def transform(self, X, batch=None, covars=None):
        X = np.asarray(X, dtype=np.float64)
        batch = np.asarray(batch)
        fixed = _apply_covariate_model(self.beta_, covars, len(X), X.shape[1])
        resid = X - fixed
        out = resid.copy()
        unseen = set(np.unique(batch)) - set(self.sites_.tolist())
        for s in np.unique(batch):
            m = batch == s
            if s in self.gamma_:
                out[m] = (resid[m] - self.gamma_[s]) / self.delta_[s]
            elif self.unseen_site == "passthrough":
                continue
            elif self.unseen_site == "transductive":
                g, d = resid[m].mean(axis=0), resid[m].std(axis=0) + 1e-8
                out[m] = (resid[m] - g) / d
            elif self.unseen_site == "reference":
                out[m] = (resid[m] - resid[m].mean(axis=0)) / self.grand_std_
        if unseen:
            log.warning("unseen sites %s handled with policy=%s", sorted(unseen),
                        self.unseen_site)
        pooled_scale = np.mean(np.stack(list(self.delta_.values())), axis=0)
        return (out * pooled_scale + fixed).astype(np.float32)


def _design(covars: pd.DataFrame | np.ndarray | None, n: int) -> np.ndarray:
    if covars is None:
        return np.ones((n, 1))
    C = pd.get_dummies(pd.DataFrame(covars), drop_first=True).to_numpy(float)
    return np.column_stack([np.ones(n), C])


def _fit_covariate_model(X, covars):
    D = _design(covars, len(X))
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)
    return beta, X - D @ beta


def _apply_covariate_model(beta, covars, n, p):
    D = _design(covars, n)
    if D.shape[1] != beta.shape[0]:
        raise ValueError(
            f"covariate design changed between fit ({beta.shape[0]}) and "
            f"transform ({D.shape[1]}) — encode covariates with the same columns"
        )
    return D @ beta


def _empirical_bayes(gamma: dict, delta: dict):
    """Shrink per-site location/scale toward the across-site mean. Small sites benefit
    most, which is the whole point of ComBat's EB step."""
    G = np.stack(list(gamma.values()))
    gbar, tau2 = G.mean(axis=0), G.var(axis=0) + 1e-8
    n_sites = len(gamma)
    w = tau2 / (tau2 + tau2.mean() / n_sites)
    return ({s: w * g + (1 - w) * gbar for s, g in gamma.items()}, delta)


def combat_available() -> bool:
    try:
        import neuroCombat  # noqa: F401
        return True
    except ImportError:
        return False
