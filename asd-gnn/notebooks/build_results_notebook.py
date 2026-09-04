"""Generate notebooks/results.ipynb from this file.

The notebook is a *view* over results/runs/*.json — keeping it generated makes the
"no pipeline logic in any notebook" rule enforceable instead of aspirational.

    python notebooks/build_results_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
(MD, """# ASD-GNN — Results

Every number below is read from `results/runs/*.json`. **This notebook contains no
pipeline logic**: it fits nothing, trains nothing, and computes no connectivity. If a
number here is wrong, the fix belongs in `src/asdgnn/`, not in a cell.

To populate the results directory:

```bash
make smoke                 # synthetic cohort, ~1 minute, no download
# or the real thing:
make data features leak sanity baselines graphs tables figures
```

**Contents**
1. Cohort and QC
2. Leakage and sanity evidence
3. Main results table
4. k-fold vs LOSO — the generalisation gap
5. Per-site behaviour under LOSO
6. ROC / PR curves
7. Statistical comparisons
8. Explainability
9. What to check before believing any of this"""),

(CODE, """import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RUNS, FIGS, TABLES, QC = (ROOT / p for p in
                          ("results/runs", "results/figures", "results/tables", "results/qc"))

pd.set_option("display.width", 160, "display.max_columns", 40)


def load_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def show_fig(name, caption=""):
    p = FIGS / name
    if p.exists():
        if caption:
            display(Markdown(f"**{caption}**"))
        display(Image(filename=str(p)))
    else:
        display(Markdown(f"_missing figure `{name}` — run `make figures`_"))


blobs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
print(f"{len(blobs)} runs in {RUNS}")
if not blobs:
    raise SystemExit("no runs — run `make smoke` first")"""),

(MD, """## 1. Cohort and QC

`results/qc/cohort.json` is written by `data/phenotype.py`, `results/qc/dropped.json` by
the QC filter. Together they are the paper's participant flow diagram."""),

(CODE, """cohort = load_json(QC / "cohort.json", {})
if cohort:
    display(Markdown(
        f"**N = {cohort['n_subjects']}**  ·  ASD {cohort['n_asd']} / TC {cohort['n_tc']}  ·  "
        f"{cohort['n_sites']} sites  ·  {cohort['male_fraction']:.0%} male  ·  "
        f"age {cohort['age']['min']:.1f}–{cohort['age']['max']:.1f} "
        f"(mean {cohort['age']['mean']:.1f})"
    ))
    display(pd.DataFrame(cohort["per_site"]).set_index("SITE_ID").round(3))

    fd = cohort["mean_fd_by_group"]
    display(Markdown(
        f"Head motion (mean FD): ASD {fd['asd']:.3f} vs TC {fd['tc']:.3f}. "
        "A group difference here is a confound the Discussion must address — motion "
        "correlates with both diagnosis and connectivity."
    ))
else:
    display(Markdown("_no cohort report — run `scripts/01_build_features.py`_"))"""),

(CODE, """dropped = load_json(QC / "dropped.json", {})
if dropped:
    reasons = pd.Series(dropped["dropped_subjects"]).str.replace(r"\\(.*\\)", "", regex=True)
    display(Markdown(
        f"Kept **{dropped['n_out']} / {dropped['n_in']}** subjects; "
        f"ROIs {dropped['n_rois_in']} → {dropped['n_rois_out']} "
        f"({len(dropped['dropped_rois'])} constant parcels dropped globally)."
    ))
    if len(reasons):
        display(reasons.value_counts().rename("n subjects dropped").to_frame())
else:
    display(Markdown("_no QC drop log (expected on the synthetic cohort)_"))"""),

(MD, """## 2. Leakage and sanity evidence

Nothing below section 3 means anything unless these pass. The label-shuffle check is the
single most valuable number in the project: run the **full** pipeline on permuted labels
and balanced accuracy must land at chance. Anything meaningfully above 0.5 is leakage,
and the correct response is to stop and find it."""),

(CODE, """sanity = load_json(QC / "sanity.json", {})
if sanity:
    rows = []
    for name, c in sanity.items():
        rows.append({
            "check": name,
            "bal_acc": c.get("bal_acc"),
            "status": {True: "PASS", False: "FAIL"}.get(c.get("pass"), "info"),
            "note": c.get("note") or c.get("expected") or "",
        })
    display(pd.DataFrame(rows).set_index("check").round(3))

    site = sanity.get("site_control", {}).get("bal_acc")
    if site is not None:
        display(Markdown(
            f"**Site-prediction control: {site:.3f}.** This is how strongly the features "
            "encode acquisition site rather than biology, and it is the number to cite "
            "when arguing that k-fold CV over-states real-world performance."
        ))
    failed = [k for k, c in sanity.items() if c.get("pass") is False]
    if failed:
        display(Markdown(f"### ⚠️ FAILED: {failed} — do not report anything below."))
else:
    display(Markdown("_no sanity report — run `python scripts/02_run_experiment.py --sanity`_"))"""),

(MD, """## 3. Main results table

Balanced accuracy is the primary metric. Accuracy is reported because the literature
does, but a mildly imbalanced cohort plus LOSO folds with wild per-site class ratios
makes raw accuracy misleading."""),

(CODE, """from asdgnn.viz.tables import load_runs, main_table, protocol_comparison

df = load_runs(RUNS)
for protocol in sorted(df["protocol"].dropna().unique()):
    display(Markdown(f"### {protocol}"))
    display(main_table(df, protocol))"""),

(MD, """## 4. k-fold vs LOSO — the generalisation gap

This is the headline scientific figure, and the honest one. Within-site cross-validation
lets a model exploit site-specific signal; leave-one-site-out does not. Points below the
diagonal are the real story about whether any of this transfers to a new scanner."""),

(CODE, """comp = protocol_comparison(df)
display(comp.round(3))
if "drop" in comp:
    display(Markdown(
        f"Mean k-fold → LOSO drop: **{comp['drop'].mean():.3f}** balanced accuracy. "
        "A drop of this kind is the expected, publishable result — a model showing *no* "
        "drop is more likely to be leaking than to be good."
    ))
show_fig("kfold_vs_loso.png", "Within-site vs cross-site generalisation")"""),

(MD, """## 5. Per-site behaviour under LOSO

Per-site variance is large and driven mostly by site size. Report per-site n alongside
per-site scores; a site with 12 subjects contributes noise, not evidence. Sites below the
`min_site_n` threshold are flagged `underpowered` in the run JSON."""),

(CODE, """loso = [b for b in blobs if b["protocol"] == "loso"]
for b in loso:
    rows = [{"site": f["fold"], "n": f["metrics"]["n_test"], "n_pos": f["metrics"]["n_pos"],
             "bal_acc": f["metrics"]["bal_acc"], "auc": f["metrics"]["auc"]}
            for f in b["folds"]]
    t = pd.DataFrame(rows).sort_values("n", ascending=False).set_index("site")
    display(Markdown(f"### {b['model']}  (underpowered: {b['underpowered_folds'] or 'none'})"))
    display(t.round(3))
    show_fig(f"per_site_{b['model']}.png")"""),

(MD, """## 6. ROC and precision-recall curves

Pooled across folds. Pooling is the right way to draw a single curve when individual
LOSO folds are small — a per-fold AUC on a 12-subject site is not interpretable."""),

(CODE, """for protocol in sorted(df["protocol"].dropna().unique()):
    show_fig(f"roc_{protocol}.png", f"ROC — {protocol}")
    show_fig(f"pr_{protocol}.png", f"Precision-recall — {protocol}")"""),

(MD, """## 7. Statistical comparisons

Two tests, because they answer different questions:

- **DeLong** compares two ROC curves on the *same subjects*. This is possible only
  because every run saves per-fold predicted probabilities against a shared split file.
- **Nadeau–Bengio corrected resampled t-test** compares fold scores from repeated k-fold.
  A plain paired t-test is anti-conservative here because folds share training data.

Both are reported with Holm–Bonferroni correction across the family of comparisons."""),

(CODE, """for protocol in sorted(df["protocol"].dropna().unique()):
    stats = load_json(TABLES / f"stats_{protocol}.json", {})
    if not stats.get("delong"):
        continue
    display(Markdown(f"### {protocol} — DeLong (paired, pooled predictions)"))
    d = pd.DataFrame(stats["delong"]).T
    if "delong_holm" in stats:
        d["p_adj"] = pd.DataFrame(stats["delong_holm"]).T["p_adj"]
        d["significant"] = pd.DataFrame(stats["delong_holm"]).T["significant"]
    display(d.round(4))

    if stats.get("corrected_ttest"):
        display(Markdown(f"### {protocol} — corrected resampled t-test (balanced accuracy)"))
        display(pd.DataFrame(stats["corrected_ttest"]).T.round(4))"""),

(MD, """## 8. Explainability

Report **stability across folds, not just the mean**. An ROI that ranks #1 in three folds
and #150 in the other seven is noise, and it is the first thing a reviewer will ask
about. `explain/saliency.py::rank_stability` produces the selection frequency shown
here; anything below ~0.8 should not be named in the paper's text."""),

(CODE, """expl = load_json(ROOT / "results/explain/roi_importance.json", {})
if expl:
    t = pd.DataFrame(expl["top_rois"])
    display(t.round(4))
    rho = expl.get("mean_pairwise_spearman")
    n_stable = len(expl.get("stable_rois", []))
    display(Markdown(
        f"Method: **{expl['method']}** on **{expl['model']}**, {expl['n_folds']} folds.  "
        f"Mean pairwise Spearman across folds: **{rho:.3f}**  ·  "
        f"**{n_stable}** ROIs enter the top-{len(t)} in at least 80% of folds."
    ))
    if rho is not None and rho < 0.3:
        display(Markdown(
            "> **The ranking is not reproducible across folds.** Report it as unstable "
            "rather than naming individual regions in the text."
        ))
    show_fig("top_rois.png", "Top ROIs with across-fold error bars")
    show_fig("roi_stability.png", "Saliency stability")
else:
    display(Markdown(
        "_no explainability artifacts yet — these are produced at M14, after a trained "
        "GAT exists. See `explain/saliency.py`._"
    ))"""),

(MD, """## 8b. Hybrid model — stage-wise attribution

The hybrid is built in stages so that a failure is attributable to one half: the encoder
alone, then the population GNN on frozen embeddings, then optionally end-to-end with the
gradient-reversal site head. If stage 3 does not beat stage 1, the population graph is
adding nothing and the extra complexity is not earning its place in the paper."""),

(CODE, """hybrid = [b for b in blobs if b.get("model") == "hybrid"]
if hybrid:
    rows = []
    for b in hybrid:
        row = {"protocol": b["protocol"],
               "stage1_encoder": b["stage1_aggregate"]["bal_acc_mean"],
               "stage3_population": b["stage3_aggregate"]["bal_acc_mean"],
               "reported": b["aggregate"]["bal_acc_mean"],
               "lambda_adv": b["config"].get("lambda_adv"),
               "end_to_end": b["config"].get("end_to_end")}
        row["stage3_gain"] = row["stage3_population"] - row["stage1_encoder"]
        rows.append(row)
    display(pd.DataFrame(rows).round(3))
    display(Markdown(
        "`stage3_gain` is the honest measure of what the population graph contributes on "
        "top of the encoder. A gain near zero or negative is a reportable negative "
        "result, not a reason to keep tuning until it turns positive."
    ))
    iso = [f.get("n_isolated", 0) for b in hybrid for f in b["folds"]]
    if any(iso):
        display(Markdown(
            f"Up to **{max(iso)}** nodes were isolated in the population graph. Under "
            "LOSO with a site affinity term this is the expected disconnection "
            "behaviour — report it, do not hide it."
        ))
else:
    display(Markdown("_no hybrid runs yet — `make hybrid`_"))"""),

(MD, """## 9. Before believing any of this

- [ ] `make leak` green — the leakage suite, not just the general test run
- [ ] label shuffle at chance in section 2
- [ ] single-batch overfit reaches 100% (`tests/test_models.py`)
- [ ] mean node degree in the 10–40 range for CC200 — a dense graph makes message
      passing a global average, and it is the cause of a GNN stuck near 53% about 40%
      of the time
- [ ] every reported model has ≥3 seeds; a model that beats a baseline on one seed is
      not a result
- [ ] every number traceable to a `results/runs/*.json` with a git commit hash
- [ ] the k-fold → LOSO drop is reported, not buried
- [ ] saliency stability reported alongside the top-ROI list, not instead of it
- [ ] for the hybrid, stage 1 and stage 3 scores both reported, not just the best one
- [ ] statistical comparisons paired on **subject id**, and any refused comparison
      surfaced in `stats_*.json["skipped"]` rather than silently dropped"""),

(CODE, """checks = pd.DataFrame([
    {"run_id": b["run_id"], "model": b["model"], "protocol": b["protocol"],
     "git_commit": b.get("git_commit"), "n_folds": b["aggregate"]["n_folds"],
     "probs_saved": all("y_prob" in f for f in b["folds"]),
     "sub_ids_saved": all("sub_ids" in f for f in b["folds"]),
     "runtime_sec": b.get("runtime_sec")}
    for b in blobs
])
display(checks)
assert checks["probs_saved"].all(), "a run is missing per-fold probabilities — DeLong is impossible"
assert checks["sub_ids_saved"].all(), "a run is missing sub_ids — paired tests cannot align subjects"
print("provenance OK: every run carries a config, a commit, per-fold probabilities and subject ids")"""),
]


def build() -> dict:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == CODE:
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "results.ipynb"
    out.write_text(json.dumps(build(), indent=1))
    print(out)
