from __future__ import annotations

import json

import numpy as np

from asdgnn.stats.pairing import _by_subject, _same_fold_structure, paired_comparisons


def _run(model, folds, protocol="kfold", n_test_total=None):
    return {
        "model": model, "protocol": protocol, "folds": folds,
        "aggregate": {"n_test_total": n_test_total or sum(len(f["y_true"]) for f in folds)},
    }


def _fold(name, subs, y, p):
    # Plain Python types: the real writer is utils.logging.write_json, which converts
    # numpy scalars for us. These fixtures go through json.dumps directly.
    return {"fold": name, "sub_ids": [int(s) for s in subs],
            "y_true": [int(v) for v in y], "y_prob": [float(v) for v in p],
            "metrics": {"bal_acc": float(np.mean(p)), "n_test": int(len(subs))}}


def _write(tmp_path, *runs):
    for i, r in enumerate(runs):
        (tmp_path / f"run{i}.json").write_text(json.dumps(r))
    return tmp_path


def test_by_subject_pools_folds():
    r = _run("a", [_fold("f0", [1, 2], [0, 1], [0.1, 0.9]),
                   _fold("f1", [3], [1], [0.7])])
    m = _by_subject(r)
    assert set(m) == {1, 2, 3}
    assert m[2] == (1, 0.9)


def test_by_subject_bails_when_ids_are_missing():
    f = _fold("f0", [1, 2], [0, 1], [0.1, 0.9])
    del f["sub_ids"]
    assert _by_subject(_run("a", [f])) == {}


def test_delong_pairs_on_subjects_not_fold_names(tmp_path):
    """A 2-fold run and a 4-fold run share fold names while covering different
    subjects. Concatenating by name compares predictions for different people; the
    result must be paired on subject id instead."""
    rng = np.random.RandomState(0)
    subs = np.arange(100)
    y = rng.binomial(1, 0.5, 100)
    pa, pb = rng.rand(100), rng.rand(100)

    coarse = _run("coarse", [_fold(f"seed0_fold{k}", subs[k::2], y[k::2], pa[k::2])
                             for k in range(2)])
    fine = _run("fine", [_fold(f"seed0_fold{k}", subs[k::4], y[k::4], pb[k::4])
                         for k in range(4)])
    out = paired_comparisons(_write(tmp_path, coarse, fine))

    assert "coarse_vs_fine" in out["delong"]
    assert out["delong"]["coarse_vs_fine"]["n_paired_subjects"] == 100


def test_fold_wise_ttest_refused_when_structures_differ(tmp_path):
    rng = np.random.RandomState(1)
    subs, y = np.arange(40), rng.binomial(1, 0.5, 40)
    a = _run("a", [_fold(f"seed0_fold{k}", subs[k::2], y[k::2], rng.rand(20))
                   for k in range(2)])
    b = _run("b", [_fold(f"seed0_fold{k}", subs[k::4], y[k::4], rng.rand(10))
                   for k in range(4)])
    out = paired_comparisons(_write(tmp_path, a, b))
    assert "a_vs_b" not in out["corrected_ttest"]
    assert "invalid" in out["skipped"]["a_vs_b_ttest"]


def test_fold_wise_ttest_runs_on_identical_folds(tmp_path):
    rng = np.random.RandomState(2)
    subs, y = np.arange(60), rng.binomial(1, 0.5, 60)
    folds_a = [_fold(f"seed0_fold{k}", subs[k::3], y[k::3], rng.rand(20)) for k in range(3)]
    folds_b = [_fold(f"seed0_fold{k}", subs[k::3], y[k::3], rng.rand(20)) for k in range(3)]
    out = paired_comparisons(_write(tmp_path, _run("a", folds_a), _run("b", folds_b)))
    assert "a_vs_b" in out["corrected_ttest"]
    assert "a_vs_b" in out["delong"]


def test_mismatched_labels_are_refused_not_silently_compared(tmp_path):
    subs = np.arange(30)
    a = _run("a", [_fold("f0", subs, np.zeros(30, int), np.full(30, 0.4))])
    b = _run("b", [_fold("f0", subs, np.ones(30, int), np.full(30, 0.6))])
    out = paired_comparisons(_write(tmp_path, a, b))
    assert "a_vs_b" not in out["delong"]
    assert "label mismatch" in out["skipped"]["a_vs_b"]


def test_holm_correction_is_attached(tmp_path):
    rng = np.random.RandomState(3)
    subs, y = np.arange(80), rng.binomial(1, 0.5, 80)
    runs = [_run(m, [_fold(f"seed0_fold{k}", subs[k::2], y[k::2], rng.rand(40))
                     for k in range(2)]) for m in ("a", "b", "c")]
    out = paired_comparisons(_write(tmp_path, *runs))
    assert len(out["delong"]) == 3
    assert set(out["delong_holm"]) == set(out["delong"])
    assert all(0 <= v["p_adj"] <= 1 for v in out["delong_holm"].values())


def test_same_fold_structure_detects_reordered_but_equal_folds():
    subs = [3, 1, 2]
    a = _run("a", [_fold("f0", subs, [0, 1, 0], [0.1, 0.2, 0.3])])
    b = _run("b", [_fold("f0", [1, 2, 3], [1, 0, 0], [0.4, 0.5, 0.6])])
    assert _same_fold_structure(a, b) == ["f0"]
