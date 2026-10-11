"""Generate notebooks/baselines.ipynb from this file.

A presentation notebook for the M8 classical baselines: a short live run you can
execute in front of an audience, then the full saved results. Like results.ipynb, it
implements no pipeline logic of its own; every model, split and metric comes from
src/asdgnn/.

    python notebooks/build_baseline_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
(MD, """# Baselines: SVM and logistic regression for ASD classification

**Question:** before any graph neural network, how well does a simple model do when it
treats each brain's connectivity as one flat list of numbers?

**Data:** ABIDE I, CPAC pipeline, band-pass filtered, CC200 atlas.
**Protocols:** stratified 10-fold CV (5 repeats) and leave-one-site-out (LOSO).
**Metrics:** balanced accuracy (primary), accuracy, sensitivity, specificity, F1, AUC.

Sections 1-3 run live (about a minute). Sections 4-6 read the full saved runs in
`results/runs/*.json`, produced by `make baselines`.

1. The cohort after quality control
2. What one brain looks like as features
3. **Live demo:** 5-fold SVM and logistic regression, plus the label-shuffle check
4. Full results under both protocols
5. Unseen scanners: per-site results
6. Are the differences significant?"""),

(CODE, """import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
import os
os.chdir(ROOT)  # the package reads data/ and results/ relative to the repo root

from asdgnn.data.connectivity import ConnectivityTransformer
from asdgnn.data.loading import load_cohort
from asdgnn.models.classical import build_classical, decision_scores
from asdgnn.train.metrics import compute_all
from asdgnn.train.splits import build_splits

pd.set_option("display.width", 160, "display.max_columns", 40)
FIGS = ROOT / "results/figures/presentation"


def show_fig(name, width=900):
    p = FIGS / name
    if p.exists():
        display(Image(filename=str(p), width=width))
    else:
        display(Markdown(f"_missing `{name}`: run `python scripts/07_baseline_figures.py`_"))"""),

(MD, """## 1. The cohort after quality control

QC drops subjects with fewer than 100 time points and any region that is flat (zero
variance) in any subject. Both rules are applied once, before any split, and do not
look at diagnosis."""),

(CODE, """ts_list, pheno = load_cohort("cc200")
y = pheno["y"].to_numpy(int)
sites = pheno["SITE_ID"].to_numpy()

display(Markdown(
    f"**{len(pheno)} subjects** ({y.sum()} ASD, {(y == 0).sum()} control) · "
    f"**{ts_list[0].shape[1]} regions** · **{pheno.SITE_ID.nunique()} sites** · "
    f"time points per scan: {min(t.shape[0] for t in ts_list)}–{max(t.shape[0] for t in ts_list)}"
))

dropped = json.loads((ROOT / "results/qc/dropped.json").read_text())
reasons = pd.Series(dropped["dropped_subjects"]).str.replace(r"\\(.*\\)", "", regex=True)
display(Markdown(
    f"Dropped **{dropped['n_in'] - dropped['n_out']}** subjects "
    f"({', '.join(f'{n}× {r}' for r, n in reasons.value_counts().items())}) "
    f"and **{len(dropped['dropped_rois'])}** flat regions "
    f"({dropped['n_rois_in']} → {dropped['n_rois_out']})."
))

per_site = (pheno.groupby("SITE_ID")["y"].agg(n="size", ASD="sum")
            .assign(control=lambda d: d.n - d.ASD).sort_values("n", ascending=False))
per_site.T"""),

(MD, """## 2. What one brain looks like as features

Pearson correlation between every pair of regions, Fisher z-transformed. The baselines
read only the upper triangle: 179 × 178 / 2 = 15,931 numbers per subject."""),

(CODE, """ct = ConnectivityTransformer(kind="correlation", vectorize=False).fit(ts_list[:2])
M = ct.transform(ts_list)  # correlation is stateless, so what it was fitted on is irrelevant

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
im = axes[0].imshow(M[0], cmap="RdBu_r", vmin=-1.2, vmax=1.2)
axes[0].set_title(f"Subject {pheno.index[0]} ({'ASD' if y[0] else 'control'})")
fig.colorbar(im, ax=axes[0], fraction=0.046)
d = M[y == 1].mean(0) - M[y == 0].mean(0)
lim = np.abs(d).max()
im2 = axes[1].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[1].set_title("Group mean difference: ASD − control")
fig.colorbar(im2, ax=axes[1], fraction=0.046)
for a in axes:
    a.set_xlabel("region")
    a.set_ylabel("region")
plt.tight_layout()
plt.show()

iu = np.triu_indices(M.shape[1], k=1)
print(f"features per subject: {len(iu[0]):,}")
print(f"largest |group difference|: {lim:.3f} (in Fisher-z units), so the signal is small and spread out")"""),

(MD, """## 3. Live demo: 5-fold cross-validation

The whole pipeline (connectivity → standardise → top 2,000 features by ANOVA →
classifier) is one sklearn `Pipeline`, refitted inside every fold on the training
subjects only. This is a smaller, faster version of the full runs in section 4, which
use 10 folds repeated 5 times."""),

(CODE, """splits = build_splits(y, sites, "kfold", n_splits=5, n_seeds=1)


def run(model, labels):
    rows = []
    for name, tr, te in splits:
        pipe = build_classical({"model": model, "select_k": 2000, "seed": 0})
        pipe.fit([ts_list[i] for i in tr], labels[tr])
        m = compute_all(labels[te], decision_scores(pipe, [ts_list[i] for i in te]))
        rows.append({"fold": name, **{k: m[k] for k in ("bal_acc", "acc", "sens", "spec", "f1", "auc")}})
    return pd.DataFrame(rows).set_index("fold")


live = {}
for model in ("svm", "logreg"):
    t0 = time.time()
    live[model] = run(model, y)
    print(f"{model:7s} done in {time.time() - t0:.0f}s")

summary = pd.DataFrame({m: df.mean() for m, df in live.items()}).T
summary.style.format("{:.3f}")"""),

(MD, """### The label-shuffle check

Same pipeline, but with the diagnosis labels randomly permuted. If any information from
the test subjects leaked into training, the model would still score above chance. It
should land near 0.50."""),

(CODE, """shuffled = run("logreg", np.random.RandomState(0).permutation(y))
display(Markdown(
    f"**Shuffled labels: {shuffled.bal_acc.mean():.3f}** balanced accuracy "
    f"vs **{live['logreg'].bal_acc.mean():.3f}** with real labels. "
    + ("Chance level, so there is no leakage." if 0.40 <= shuffled.bal_acc.mean() <= 0.60
       else "**Outside 0.40–0.60: investigate before trusting anything.**")
))"""),

(MD, """## 4. Full results under both protocols

Read from the saved run files: 10-fold CV repeated 5 times (50 folds) and leave-one-site-out
(19 folds, one per site). ± is the standard deviation across folds or sites."""),

(CODE, """blobs = [json.loads(p.read_text()) for p in sorted((ROOT / "results/runs").glob("*.json"))]
NAMES = {"logreg": "Logistic regression", "svm": "SVM (linear)", "rf": "Random forest", "ridge": "Ridge"}
rows = []
for b in blobs:
    a = b["aggregate"]
    rows.append({
        "model": NAMES.get(b["model"], b["model"]),
        "protocol": {"kfold": "10-fold ×5", "loso": "LOSO"}[b["protocol"]],
        **{k: f"{a[k + '_mean']:.3f} ± {a[k + '_std']:.3f}"
           for k in ("bal_acc", "acc", "sens", "spec", "f1", "auc")},
        "_sort": a["bal_acc_mean"],
    })
results = (pd.DataFrame(rows).sort_values(["protocol", "_sort"], ascending=[True, False])
           .drop(columns="_sort").set_index(["protocol", "model"]))
results"""),

(CODE, """show_fig("bal_acc_bars.png")"""),

(MD, """## 5. Unseen scanners: per-site results

Under LOSO the model is tested on a site it never saw during training. The average
barely moves, but individual sites swing widely, and small sites swing the most."""),

(CODE, """lr = next(b for b in blobs if b["model"] == "logreg" and b["protocol"] == "loso")
per = pd.DataFrame([{"site": f["fold"], "n": f["metrics"]["n_test"],
                     "bal_acc": f["metrics"]["bal_acc"], "auc": f["metrics"]["auc"]}
                    for f in lr["folds"]]).sort_values("n", ascending=False).set_index("site")
display(Markdown(
    f"Logistic regression: **{per.bal_acc.min():.1%}** ({per.bal_acc.idxmin()}) to "
    f"**{per.bal_acc.max():.1%}** ({per.bal_acc.idxmax()}) across sites; "
    f"{(per.bal_acc < 0.5).sum()} of {len(per)} sites below chance."
))
show_fig("per_site_logreg.png", width=1000)
per.T.round(3)"""),

(CODE, """sanity = json.loads((ROOT / "results/qc/sanity.json").read_text())
display(Markdown(
    f"**Site-control task: {sanity['site_control']['bal_acc']:.3f}** balanced accuracy at "
    "telling one site apart from the rest, using the same features. Scanner identity is "
    "in the data, which is why the ComBat harmonisation comparison is the next step."
))"""),

(MD, """## 6. Are the differences significant?

Every model used the same cached split file and saved a probability for every subject,
so models can be compared on the same people.

- **DeLong test:** compares two ROC curves on paired predictions.
- **Corrected resampled t-test:** compares fold scores, accounting for folds sharing
  training data (a plain paired t-test is over-confident here)."""),

(CODE, """show_fig("roc_loso.png", width=560)
for protocol in ("kfold", "loso"):
    s = json.loads((ROOT / f"results/tables/stats_{protocol}.json").read_text())
    t = pd.DataFrame(s["delong"]).T[["auc_a", "auc_b", "delta", "p"]]
    t["corrected_t_p"] = pd.DataFrame(s["corrected_ttest"]).T["p"]
    display(Markdown(f"### {protocol}"))
    display(t.style.format({"auc_a": "{:.3f}", "auc_b": "{:.3f}", "delta": "{:+.3f}",
                            "p": "{:.2g}", "corrected_t_p": "{:.2g}"}))"""),

(MD, """## Takeaways

- **Logistic regression is the strongest baseline:** about 0.65 balanced accuracy under
  both protocols. That is the bar the GNNs must clear.
- **It beats SVM by a small but significant margin** (DeLong p < 0.001 under both
  protocols).
- **Unseen scanners cost almost nothing on average**, but per-site performance ranges
  from chance to about 85%.
- **Random forest's balanced accuracy hides a bias:** it finds under half of the ASD
  cases.
- **The pipeline is honest:** shuffled labels score at chance, and the leakage test suite
  passes."""),
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
    out = Path(__file__).parent / "baselines.ipynb"
    out.write_text(json.dumps(build(), indent=1))
    print(out)
