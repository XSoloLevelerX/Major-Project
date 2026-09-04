from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asdgnn.stats.corrected_ttest import corrected_resampled_ttest, holm_bonferroni
from asdgnn.stats.delong import delong_test
from asdgnn.utils.logging import get_logger

log = get_logger(__name__)

# Model-vs-model testing. Kept in the package rather than in the aggregation script so
# the pairing rules — the part that is easy to get quietly wrong — are unit-tested.


def _by_subject(blob) -> dict[int, tuple[int, float]]:
    """sub_id -> (y_true, y_prob), pooled across folds.

    Pairing on subject identity rather than fold name is what makes the DeLong test
    valid: two runs can share fold *names* while covering different subjects (a 5-fold
    run and a 10-fold run both have a `seed0_fold0`), and concatenating those by name
    silently compares predictions for different people.
    """
    out = {}
    for f in blob["folds"]:
        ids = f.get("sub_ids")
        if ids is None:
            return {}
        for s, yt, yp in zip(ids, f["y_true"], f["y_prob"]):
            out[int(s)] = (int(yt), float(yp))
    return out


def _same_fold_structure(ra, rb) -> list[str] | None:
    """Fold-wise tests need identical folds, not merely identically named ones."""
    fa = {f["fold"]: f for f in ra["folds"]}
    fb = {f["fold"]: f for f in rb["folds"]}
    shared = sorted(set(fa) & set(fb))
    if not shared:
        return None
    for f in shared:
        if sorted(fa[f].get("sub_ids", [])) != sorted(fb[f].get("sub_ids", [])):
            return None
    return shared


def is_default_config(blob) -> bool:
    """The unablated configuration: no harmonisation, Pearson correlation.

    Model-vs-model tests are only meaningful between runs that differ in the model and
    nothing else. Ablation axes get their own comparisons, not this one.
    """
    cfg = blob.get("config", {})
    conn = cfg.get("conn_kind", cfg.get("conn", "correlation"))
    return cfg.get("harmonize", "none") == "none" and conn == "correlation"


def paired_comparisons(results_dir: Path, protocol: str = "kfold") -> dict:
    """Every model-vs-model test the shared split file makes valid.

    Only default-configuration runs are paired. Keying on the model name alone lets a
    ComBat or tangent-space run stand in for the model it shares a name with, so
    `svm_vs_gcn` silently becomes whichever svm run happened to sort last.
    """
    runs, dropped = {}, {}
    for p in sorted(results_dir.glob("*.json")):
        blob = json.loads(p.read_text())
        if blob.get("protocol") != protocol:
            continue
        model = blob["model"]
        run_id = blob.get("run_id", p.stem)
        if not is_default_config(blob):
            dropped.setdefault(model, []).append(run_id)
            continue
        if model in runs:
            # Two default-config runs for one model is ambiguous; refuse to pick silently.
            dropped.setdefault(model, []).append(run_id)
            log.warning("%s: multiple default-config runs under %s; keeping %s",
                        model, protocol, runs[model][1])
            continue
        runs[model] = (blob, run_id)

    out: dict = {"delong": {}, "corrected_ttest": {}, "skipped": {},
                 "compared_runs": {m: rid for m, (_, rid) in sorted(runs.items())},
                 "excluded_non_default_runs": dropped}
    names = sorted(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = runs[a][0], runs[b][0]
            key = f"{a}_vs_{b}"

            sa_map, sb_map = _by_subject(ra), _by_subject(rb)
            shared_subs = sorted(set(sa_map) & set(sb_map))
            if len(shared_subs) < 10:
                out["skipped"][key] = "fewer than 10 subjects predicted by both models"
            elif any(sa_map[s][0] != sb_map[s][0] for s in shared_subs):
                out["skipped"][key] = "label mismatch between runs — different cohorts"
            else:
                y = np.array([sa_map[s][0] for s in shared_subs])
                pa = np.array([sa_map[s][1] for s in shared_subs])
                pb = np.array([sb_map[s][1] for s in shared_subs])
                try:
                    out["delong"][key] = delong_test(y, pa, pb) | {
                        "n_paired_subjects": len(shared_subs)
                    }
                except ValueError as exc:
                    out["skipped"][key] = f"DeLong: {exc}"

            shared_folds = _same_fold_structure(ra, rb)
            if shared_folds is None:
                out["skipped"][f"{key}_ttest"] = (
                    "fold structures differ — a paired fold-wise test would be invalid"
                )
                continue
            sa = [next(f for f in ra["folds"] if f["fold"] == k)["metrics"]["bal_acc"]
                  for k in shared_folds]
            sb = [next(f for f in rb["folds"] if f["fold"] == k)["metrics"]["bal_acc"]
                  for k in shared_folds]
            n_test = int(np.mean([next(f for f in ra["folds"] if f["fold"] == k)
                                  ["metrics"]["n_test"] for k in shared_folds]))
            n_train = max(1, ra["aggregate"]["n_test_total"] - n_test)
            try:
                out["corrected_ttest"][key] = corrected_resampled_ttest(
                    sa, sb, n_train=n_train, n_test=n_test
                )
            except ValueError as exc:
                # One comparison failing must never take the whole aggregation with it;
                # a single-fold run is the common case here.
                out["skipped"][f"{key}_ttest"] = f"corrected t-test: {exc}"

    for key, reason in out["skipped"].items():
        log.warning("skipped %s: %s", key, reason)

    if out["delong"]:
        out["delong_holm"] = holm_bonferroni({k: v["p"] for k, v in out["delong"].items()})
    if out["corrected_ttest"]:
        out["ttest_holm"] = holm_bonferroni(
            {k: v["p"] for k, v in out["corrected_ttest"].items()}
        )
    return out
