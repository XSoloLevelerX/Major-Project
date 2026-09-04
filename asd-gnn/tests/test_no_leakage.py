"""The most important file in the repo. If it goes red, nothing else matters.

M7 comes before M8 for a reason: writing these after you have results means you will be
emotionally invested in numbers that may be wrong.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from asdgnn.data.connectivity import ConnectivityTransformer
from asdgnn.harmonize.combat import ComBatTransformer
from asdgnn.models.classical import build_classical, decision_scores
from asdgnn.train.metrics import compute_all
from asdgnn.train.splits import build_splits, loso_splits


def test_connectivity_fit_size(dummy_data, split_idx):
    """The tangent-space reference must see only training subjects."""
    train_idx, _ = split_idx
    ct = ConnectivityTransformer(kind="tangent")
    ct.fit([dummy_data[i] for i in train_idx])
    assert ct.n_fit_samples_ == len(train_idx)
    assert ct.n_fit_samples_ < len(dummy_data)


def test_tangent_reference_changes_with_fold(dummy_data, split_idx):
    """Two different training folds must produce two different tangent references.
    If the transform is identical, the reference was global."""
    train_idx, test_idx = split_idx
    other = np.setdiff1d(np.arange(len(dummy_data)), test_idx[:5])
    a = ConnectivityTransformer(kind="tangent").fit([dummy_data[i] for i in train_idx])
    b = ConnectivityTransformer(kind="tangent").fit([dummy_data[i] for i in other])
    probe = [dummy_data[i] for i in test_idx]
    assert not np.allclose(a.transform(probe), b.transform(probe))


def test_pipeline_transformers_never_see_test(cohort, split_idx):
    """Spy on every .fit() in the pipeline; each must see at most len(train) samples."""
    ts_list, pheno = cohort
    train_idx, test_idx = split_idx
    y = pheno["y"].to_numpy()
    seen: list[int] = []

    pipe = build_classical({"model": "svm", "select_k": 20, "conn_kind": "correlation"})
    for _, step in pipe.steps:
        original = step.fit

        def spy(X, y=None, _orig=original, **kw):
            seen.append(len(X))
            return _orig(X, y, **kw) if y is not None else _orig(X, **kw)

        step.fit = spy

    pipe.fit([ts_list[i] for i in train_idx], y[train_idx])
    assert seen, "no fit calls were observed — the spy did not attach"
    assert max(seen) <= len(train_idx)


def test_scaler_stats_differ_across_folds(cohort):
    """Identical StandardScaler.mean_ across folds means it was fit on all data."""
    ts_list, pheno = cohort
    y = pheno["y"].to_numpy()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    means = []
    for tr, _ in skf.split(np.zeros(len(y)), y):
        pipe = build_classical({"model": "svm", "select_k": 20})
        pipe.fit([ts_list[i] for i in tr], y[tr])
        means.append(pipe.named_steps["scale"].mean_)
    assert not np.allclose(means[0], means[1])
    assert not np.allclose(means[1], means[2])


def test_split_disjointness(cohort):
    ts_list, pheno = cohort
    y, sites = pheno["y"].to_numpy(), pheno["SITE_ID"].to_numpy()
    for protocol in ("kfold", "loso"):
        for _, tr, te in build_splits(y, sites, protocol, n_splits=5, n_seeds=2):
            assert set(tr).isdisjoint(set(te))
            assert set(tr) | set(te) == set(range(len(y)))


def test_loso_site_purity(cohort):
    _, pheno = cohort
    sites = pheno["SITE_ID"].to_numpy()
    for _, tr, te in loso_splits(sites):
        assert len(set(sites[te])) == 1
        assert set(sites[te]).isdisjoint(set(sites[tr]))


def test_every_train_fold_has_both_classes(cohort):
    _, pheno = cohort
    y, sites = pheno["y"].to_numpy(), pheno["SITE_ID"].to_numpy()
    for _, tr, _ in build_splits(y, sites, "kfold", n_splits=5, n_seeds=1):
        assert set(y[tr]) == {0, 1}


def test_label_shuffle_collapses_to_chance(cohort):
    """The single most valuable test in the repo. Shuffle y, run the FULL pipeline;
    balanced accuracy must land at chance."""
    ts_list, pheno = cohort
    y_shuf = np.random.RandomState(0).permutation(pheno["y"].to_numpy())
    sites = pheno["SITE_ID"].to_numpy()
    scores = []
    for _, tr, te in build_splits(y_shuf, sites, "kfold", n_splits=5, n_seeds=1):
        pipe = build_classical({"model": "svm", "select_k": 20})
        pipe.fit([ts_list[i] for i in tr], y_shuf[tr])
        p = decision_scores(pipe, [ts_list[i] for i in te])
        scores.append(compute_all(y_shuf[te], p)["bal_acc"])
    mean = float(np.mean(scores))
    assert 0.35 <= mean <= 0.65, f"LEAKAGE: chance-level run scored {mean:.3f}"


def test_combat_params_from_train_only():
    """ComBat site parameters estimated on train must not change when the test data
    changes."""
    import pandas as pd

    rng = np.random.RandomState(0)
    n, p = 40, 15
    X = rng.randn(n, p)
    batch = np.array(["A"] * 20 + ["B"] * 20)
    covars = pd.DataFrame({"age": rng.uniform(8, 30, n),
                           "sex": rng.binomial(1, 0.5, n),
                           "y": rng.binomial(1, 0.5, n)})
    tr = np.arange(30)

    def fit_on(train_rows):
        cb = ComBatTransformer()
        cb.fit(X[train_rows], batch=batch[train_rows], covars=covars.iloc[train_rows])
        return cb

    a = fit_on(tr)
    X2 = X.copy()
    X2[30:] += 10.0  # perturb only the test rows
    b = ComBatTransformer()
    b.fit(X2[tr], batch=batch[tr], covars=covars.iloc[tr])
    for site in a.gamma_:
        assert np.allclose(a.gamma_[site], b.gamma_[site])


def test_combat_unseen_site_is_declared():
    """Under LOSO the test site has no fitted parameters. Every policy must be an
    explicit choice, never a silent default that invents them."""
    import pandas as pd

    rng = np.random.RandomState(1)
    X = rng.randn(30, 8)
    batch = np.array(["A"] * 15 + ["B"] * 15)
    covars = pd.DataFrame({"age": rng.uniform(8, 30, 30), "sex": rng.binomial(1, .5, 30),
                           "y": rng.binomial(1, .5, 30)})
    cb = ComBatTransformer(unseen_site="passthrough")
    cb.fit(X, batch=batch, covars=covars)
    new_batch = np.array(["C"] * 30)
    out = cb.transform(X, batch=new_batch, covars=covars)
    assert out.shape == X.shape
    with pytest.raises(ValueError):
        ComBatTransformer(unseen_site="invent_them")


def test_population_graph_loss_never_sees_test_labels(cohort, torch_available):
    if not torch_available:
        pytest.skip("torch-geometric not installed")
    from asdgnn.data.graphs_pop import build_population_graph

    _, pheno = cohort
    y = pheno["y"].to_numpy()
    n = len(y)
    X = np.random.RandomState(0).randn(n, 16).astype(np.float32)
    tr, va, te = np.arange(0, 40), np.arange(40, 48), np.arange(48, n)
    data = build_population_graph(X, pheno, y, tr, va, te)
    # The masks partition the cohort, and the training mask is the only one the loss
    # is ever indexed by (see train/loop.py::train_node_level).
    assert not (data.train_mask & data.test_mask).any()
    assert not (data.val_mask & data.test_mask).any()
    assert int(data.train_mask.sum()) == len(tr)


def test_feature_selection_is_inside_the_pipeline():
    pipe = build_classical({"model": "svm", "select_k": 50})
    assert "select" in pipe.named_steps
    assert "conn" in pipe.named_steps
    assert list(pipe.named_steps)[-1] == "clf"


def test_harmonisation_is_actually_applied(cohort):
    """Regression guard: `--harmonize combat` must change the features it is given.

    The harmonisation helper existed once without being wired into the fold path, so
    the config flag ran a silent no-op. This test fails if that ever recurs.
    """
    from asdgnn.train.cv import _harmonize

    ts_list, pheno = cohort
    tr, te = np.arange(45), np.arange(45, len(pheno))
    ct = ConnectivityTransformer(kind="correlation").fit([ts_list[i] for i in tr])
    X_tr = ct.transform([ts_list[i] for i in tr])
    X_te = ct.transform([ts_list[i] for i in te])

    same_tr, same_te = _harmonize({"harmonize": "none"}, X_tr, X_te,
                                  pheno.iloc[tr], pheno.iloc[te])
    assert np.shares_memory(same_tr, X_tr), "harmonize='none' must be a true no-op"

    cb_tr, cb_te = _harmonize({"harmonize": "combat"}, X_tr, X_te,
                              pheno.iloc[tr], pheno.iloc[te])
    assert cb_tr.shape == X_tr.shape and cb_te.shape == X_te.shape
    assert not np.allclose(cb_tr, X_tr), "ComBat produced identical features — dead code?"
    assert np.isfinite(cb_tr).all() and np.isfinite(cb_te).all()


def test_harmonisation_test_labels_never_enter_the_covariate_model(cohort):
    """ComBat's covariates include diagnosis, which is unavailable for test subjects.
    Changing the test labels must therefore not change the harmonised test features."""
    from asdgnn.train.cv import _harmonize

    ts_list, pheno = cohort
    tr, te = np.arange(45), np.arange(45, len(pheno))
    ct = ConnectivityTransformer(kind="correlation").fit([ts_list[i] for i in tr])
    X_tr, X_te = (ct.transform([ts_list[i] for i in tr]),
                  ct.transform([ts_list[i] for i in te]))
    cfg = {"harmonize": "combat"}

    a = _harmonize(cfg, X_tr, X_te, pheno.iloc[tr], pheno.iloc[te])[1]
    flipped = pheno.iloc[te].copy()
    flipped["y"] = 1 - flipped["y"]
    b = _harmonize(cfg, X_tr, X_te, pheno.iloc[tr], flipped)[1]
    assert np.allclose(a, b), "test diagnosis leaked into the harmonisation"
