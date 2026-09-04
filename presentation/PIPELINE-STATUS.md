# Pipeline Verification Record

**Verified:** 4 September 2026 (overnight session)
**Verified by:** end-to-end execution, not inspection — every claim below corresponds to a
command that was actually run.

This document exists so that when someone asks *"does it actually work?"* during the
presentation, the answer is a specific verified fact rather than a hopeful one.

---

## 1. Headline status

| Area | Status |
|---|---|
| Environment | ✅ Repaired and working (see §2 — it was broken) |
| Test suite | ✅ 85/85 passing |
| Lint (`ruff`) | ✅ Clean across `src`, `tests`, `scripts`, `notebooks` |
| Synthetic smoke pipeline (`make smoke`) | ✅ Green end to end, ~20 s |
| **ABIDE-I dataset** | ✅ **Downloaded — 871/871 subjects** (see §3 — it was NOT present before this session) |
| Real cohort QC + splits | ✅ Built; N=871, 403 ASD / 468 TC, 20 sites |
| Model architectures | ✅ All 9 run end to end |
| Config axes (connectivity, harmonisation, protocol) | ✅ All exercised |
| Real experiment grid | ✅ Complete — see `RESULTS-SNAPSHOT.md` |
| Bugs found and fixed | **6** (see §5) — three of them corrupted reported numbers |

---

## 2. The environment was broken — this is fixed

`pytest` could not even collect the test suite at the start of this session:

```
ImportError while loading conftest 'tests/conftest.py'
ModuleNotFoundError: No module named 'asdgnn'
```

**Cause:** the `.venv` had a stale editable install. A `__editable__.asdgnn-0.1.0.pth` file
was present and pointed at the correct `src/` path, but the package was not importable and
the venv had no `pip` (it was created by `uv`).

**Fix:** reinstalled the package into the existing venv with the tool that created it:

```bash
VIRTUAL_ENV="$PWD/.venv" uv pip install -e . --no-deps
```

**If this happens again on another machine, that is the command.** It is worth adding to the
README, because "nothing imports" is an alarming first impression for a new team member and
the cause is boring.

Verified package versions in the working venv:

| package | version |
|---|---|
| Python | 3.11.15 |
| torch | 2.13.0 |
| torch-geometric | 2.8.0.post1 |
| scikit-learn | 1.8.0 |
| nilearn | 0.14.0 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 3.0.5 |

Not installed, and **not needed**: `seaborn`, `hydra-core`, `neuroCombat`, `neuroHarmonize`.
ComBat is implemented natively in `src/asdgnn/harmonize/combat.py`, so the `neuroCombat`
dependency is optional — the pipeline does not import it on the main path.

---

## 3. The dataset was not actually ingested

**Before this session** `data/raw/` was empty. The only data present was the synthetic
cohort (`data/cache/ts_cc200_cpac_synthetic.npz`, 160 subjects × 30 ROIs × 5 fake sites), and
every artefact in `results/` had been produced from it.

This matters for the presentation: **nothing that existed before tonight was a result on
ABIDE.** It was all plumbing verification on synthetic data, which is exactly what the
synthetic cohort is for, but it is not a result.

**Now downloaded:**

```bash
python scripts/00_download.py --atlas cc200
```

- 871 `.1D` ROI time-series files, CPAC pipeline, `filt_noglobal` strategy, CC200 atlas
- ~350 MB in `data/raw/ABIDE_pcp/cpac/filt_noglobal/`
- Plus `Phenotypic_V1_0b_preprocessed1.csv`
- Fetch took ~55 minutes; the fetcher is slow but did not fail

### Cohort after QC

```bash
python scripts/01_build_features.py --atlas cc200 --min-timepoints 70
```

| | |
|---|---|
| Subjects | **871** (871/871 kept — no subject dropped) |
| ASD / TC | **403 / 468** |
| Sites | **20** (`SITE_ID` in the phenotype file; 24 download folders, some sites are split into sub-scans) |
| Male fraction | 83.5% |
| Age | 6.5 – 58.0 years (mean 16.9 ± 7.6) |
| Mean framewise displacement | **ASD 0.126 vs TC 0.094** |
| ROIs after QC | **179** (21 CC200 parcels dropped as constant across all subjects) |
| Feature dimension | 179 × 178 / 2 = **15,931** |
| k-fold splits cached | 50 (10 splits × 5 seeds) |
| LOSO folds | 20 |
| Max per-fold site-proportion drift | 0.164 |

**N=871 with 403 ASD / 468 TC is exactly the canonical benchmark subset** used by Abraham
et al. and the comparable literature. Our numbers are therefore directly comparable to
published ones — say this in the presentation.

### Two QC facts that need to be said out loud

**(a) The minimum-timepoints threshold is a real decision, not a default.**

The site scan lengths in ABIDE-I CC200 range from 78 to 296 timepoints:

| threshold | subjects kept | dropped |
|---|---|---|
| 70 (**used**) | 871 | 0 |
| 100 (the script's default) | 846 | **25 — all of OHSU** |
| 120 | 727 | 144 |

OHSU's scans are 78 timepoints. The script's default of `min_timepoints=100` would have
silently deleted **an entire site**, which costs a LOSO fold and breaks comparability with
the 871-subject convention. We used 70, which keeps every subject.

This is a defensible choice in either direction and it should be stated explicitly in the
report rather than left as a default. The counter-argument — that 78 timepoints gives a noisy
correlation estimate — is real, and the right response is a sensitivity analysis at
threshold 100 as a secondary row, not a silent drop.

**(b) 21 of the 200 CC200 parcels are constant and get dropped.**

`qc_filter` removes ROIs with zero variance, leaving R=179. This is expected for CC200 (some
parcels fall outside the acquired field of view for some subjects), and the code handles the
index shift correctly. But it means **any ROI index in our explainability output refers to the
179 surviving parcels, not to the original 200** — `roi_to_labels(kept_rois=...)` exists
precisely for this and must be passed the surviving indices when we get to Phase 5 properly.

---

## 4. What was executed, and what it proved

### Test suite — 85 passed
```bash
.venv/bin/python -m pytest tests/ -q     # 85 passed
```
Includes the 13 leakage tests in `tests/test_no_leakage.py`, run again after every fix below.

### Synthetic smoke pipeline — green, ~20 s
```bash
make smoke PY=.venv/bin/python
```
Exercises: feature build → sanity checks → SVM under k-fold and LOSO → explainability with
across-fold stability → stage-wise hybrid → table aggregation → figure generation.

Scores near 0.500 on synthetic data are the **expected and correct** outcome — the synthetic
cohort has a weak planted signal and exists to test plumbing, not to produce results.

### Every model architecture — all 9 run end to end
`gcn · gat · gin · sage · cheb · mlp_graph · braincnn · pop_gcn · pop_cheb`
Confirmed on synthetic data first, then on the real cohort.

### Every config axis exercised
- **Connectivity:** `correlation`, `partial correlation`, `tangent`
- **Harmonisation:** `none`, `combat` × all three declared unseen-site policies
  (`passthrough`, `transductive`, `reference`)
- **Protocol:** `kfold`, `loso`
- **Hybrid:** stage-wise, end-to-end (`--end-to-end`), and adversarial (`--lambda-adv`)
- **Explainability:** `attention` and `ig` (integrated gradients)

### Sanity gates on the real cohort
```bash
python scripts/02_run_experiment.py --sanity
python scripts/02_run_experiment.py --gnn-sanity
```
Results in `results/qc/sanity.json` and reproduced in `RESULTS-SNAPSHOT.md`.

### Scale and runtime

Pre-flight, on a synthetic 871 × 200 problem: SVM ~1.3 s/fold, GAT ~54 s/fold,
ComBat (871 × 19,900) 0.1 s fit and 0.1 s transform at ~139 MB per array copy, mean node
degree 19.9.

On the **real** cohort (871 × 179):

| | |
|---|---|
| SVM, 10-fold | 14 s total |
| SVM, LOSO (20 folds) | 29 s total |
| GAT, 10-fold / LOSO | 1,120 s / 2,642 s |
| ChebNet, 10-fold / LOSO | 1,469 s / 2,287 s |
| Tangent-space SVM, LOSO | 573 s (the geometric-mean fit dominates) |
| **Mean node degree, CC200 at top-10%** | **17.80** — inside the healthy 10–40 band |
| BrainNetCNN, per fold at 200 epochs | **~72 min** — see §6.4 |

Mean degree matters: a graph dense enough to make message passing a global average is the
documented cause of "accuracy stuck at 53%", and 17.8 is comfortably clear of that.

---

## 5. Bugs found and fixed

All six were found by *running* the pipeline across its full configuration space. None were
visible from reading the code, and none were caught by the existing 85-test suite — which is
the argument for running the whole grid before trusting any table.

§5.1–5.3 are crashes and a silent no-op. **§5.4–5.6 are worse: they produced wrong numbers,
wrong figures and wrong statistics without failing.** If the presentation had been built from
the tables as they stood this morning, the headline SVM LOSO figure would have been 0.577
instead of 0.631, the ComBat result would have been invisible, and every SVM significance
test would have been run against the wrong SVM.

**All three share one root cause: the model name was used as a unique key when it is not.**
With 36 runs covering 4 connectivity/harmonisation variants of the SVM, `dict[model] = run`
keeps whichever sorted last. Worth stating as a lesson in the report — it is the kind of bug
that survives a green test suite because every individual unit is correct.

### 5.1 `--conn partial` crashed — `scripts/02_run_experiment.py`

```
ValueError: unknown kind 'partial', expected one of ('correlation',
'partial correlation', 'tangent', 'covariance', 'precision')
```

The CLI shorthand was passed straight through to nilearn's `ConnectivityMeasure`, which
names the estimator `"partial correlation"`. `configs/conn/partial.yaml` already had the
correct string, so only the command-line path was broken — and the `Makefile` grid only ever
used `correlation` and `tangent`, so it never surfaced.

**Fix:** a `CONN_ALIASES` mapping applied in `build_cfg`. Verified: partial correlation now
runs under both protocols.

**Why it mattered:** partial correlation is one of the three connectivity measures in the
project scope and an ablation axis for the paper. It would have failed the first time
someone ran that ablation.

### 5.2 `--model mlp` crashed — `src/asdgnn/models/classical.py`

```
TypeError: 'int' object is not iterable
  hidden_layer_sizes=tuple(cfg.get("hidden", (64,)))
```

`--hidden` is an integer everywhere in the config (it is the GNN layer width), but
`MLPClassifier` needs a per-layer tuple. Every classical model except the MLP ignores
`hidden`, so this only appeared when the MLP was actually run.

**Fix:** accept both an int and an explicit tuple. Verified with `64`, `(64,)` and `(64, 32)`.

**Why it mattered:** the MLP is not optional. The project scope lists it as a must-have
baseline whose entire purpose is to prove the GNN gain is not just "deep learning". Without
it, a reviewer's first question has no answer.

### 5.3 `--lambda-adv` was silently a no-op — `scripts/06_hybrid.py`

The gradient-reversal site-discriminator head is only constructed and trained inside
`finetune_end_to_end`, i.e. stage 4. Passing `--lambda-adv 0.5` **without** `--end-to-end`
produced numerically identical results to `--lambda-adv 0`, with no indication why.

**Fix:** a warning at startup. Behaviour deliberately unchanged — the adversarial head
genuinely requires joint training, so this is a usability trap, not a logic error.

**Why it mattered:** sweeping λ is the core experiment for RQ4. Someone would have run the
sweep stage-wise, got a flat line, and concluded that adversarial site-invariance does not
work — which would have been a wrong conclusion drawn from a silent no-op.

### 5.4 The results tables silently averaged incomparable runs — `src/asdgnn/viz/tables.py`

**This was the most consequential bug, because it corrupted the project's headline number.**

`protocol_comparison()` pivoted on `["model", "conn_kind", "atlas"]` — omitting `harmonize`
and `unseen_site`. `pandas.pivot_table` defaults to `aggfunc="mean"`, so every SVM +
correlation run was silently averaged into one row. The published SVM LOSO figure came out as
**0.577**, which is the mean of four different experiments:

```
0.631 (no harmonisation)  0.500 (ComBat passthrough)
0.588 (ComBat transductive)  0.590 (ComBat reference)   ->  mean 0.577
```

The true no-harmonisation number is **0.631**. The `drop` column — the single number this
whole project exists to report — was computed from that fiction.

`kfold_vs_loso_scatter()` was worse: it pivoted on `model` alone, averaging across
connectivity measures *and* harmonisation. The headline figure plotted one fictitious point
per model.

**Fix:** a shared `CONFIG_KEYS = ["model", "conn_kind", "atlas", "harmonize", "unseen_site"]`
defines when two runs are comparable; both the table and the scatter now key on it, so the
scatter plots one point per *configuration*. `load_runs` now also captures `unseen_site`, and
reads the hybrid script's `conn` key so hybrid rows are no longer blank.

**Why it mattered:** it made the ComBat collapse invisible — averaged into the SVM row, a
result that turned out to be the most striking finding we have (see `RESULTS-SNAPSHOT.md` §5)
simply disappeared.

### 5.5 Figure files and ROC curves overwrote each other — `scripts/04_make_figures.py`

Same root cause: the model name was used as a unique key when it is not.

- `per_site_{model}.png` — the four SVM LOSO runs all wrote to `per_site_svm.png`. The
  surviving file was whichever was iterated last, i.e. a ComBat run masquerading as the
  headline SVM figure.
- `curves = {b["model"]: pooled(b) ...}` — a dict keyed on model name across the whole grid.
  The ROC curve labelled "svm" was silently whichever SVM run came last.

**Fix:** per-site filenames now carry a configuration slug (`per_site_svm_combat_reference.png`
and so on), and the ROC/PR panels restrict to the default configuration — no harmonisation,
Pearson correlation — so "one curve per model" is actually true rather than accidental.

**Why it mattered:** presenting `per_site_svm.png` would have shown the audience the ComBat
run's flat 0.500 bars while the speaker described the un-harmonised SVM.

---

### 5.6 Every SVM significance test used the wrong SVM — `src/asdgnn/stats/pairing.py`

`paired_comparisons` built its run table with `runs[blob["model"]] = blob`. With four SVM
k-fold runs on disk (correlation, correlation+ComBat, tangent, partial), the surviving entry
was arbitrary — in practice the **partial-correlation** SVM, pooled AUC 0.6125, rather than
the headline correlation SVM at 0.7133.

Every `*_vs_svm` DeLong result was therefore computed against a model nobody was reporting.
The distortion was large: `mlp_graph_vs_svm` under k-fold read ΔAUC −0.102 before the fix and
**−0.203** after it.

The file's own docstring says the pairing rules are "the part that is easy to get quietly
wrong" and are unit-tested — and they are, but no test covered the case of *two runs sharing
a model name*, which is exactly what a full experiment grid produces.

**Fix:** pairing now restricts to default-configuration runs (no harmonisation, Pearson
correlation) so `svm_vs_gcn` is unambiguous, refuses to silently pick when a model still has
two candidates, and records `compared_runs` (the run_id used per model) plus
`excluded_non_default_runs` in each stats file for traceability.

**Why it mattered:** §7 of `RESULTS-SNAPSHOT.md` is the statistical backbone of the talk. It
was wrong in a direction that *understated* our own strongest claim.

## 6. Known gaps — not bugs, but things to decide before Phase 5

1. **CC200 has no ROI names or centroid coordinates in `nilearn`.** Parcels come out as
   `CC200_001 … CC200_200`. This means the Yeo-7 network enrichment test and any
   "the model found the fusiform gyrus" claim **cannot be made on CC200 as things stand**.
   Options: run the interpretability analysis on **AAL** (which has anatomical labels) or
   **Dosenbach-160** (which ships coordinates), or compute CC200 parcel centroids from the
   atlas NIfTI. The atlas ablation (RQ3) should therefore come *before* the interpretability
   claim (RQ5), not after.

2. **`run_cv` rebuilds splits rather than reading `data/processed/splits.json`.**
   `build_splits` is deterministic and `test_determinism.py` covers it, so paired comparisons
   across models remain valid today. But the README states that every model reads the shared
   cached split file, and it does not. This is a latent divergence: the moment someone changes
   the split logic, the cache and the runs will silently disagree. Worth reconciling.

3. **The cached split file uses `n_seeds=5` (50 k-fold folds) but experiments run with
   `n_seeds=1` (10 folds).** Consistent across all runs, so comparisons are valid — but the
   extra 40 cached folds are unused, and the paper should state 10 folds × 1 seed, not 50.

4. **BrainNetCNN has no early stopping, and it is ~50× slower than everything else.**
   `_run_braincnn_fold` in `src/asdgnn/train/cv.py` carves out no inner validation split and
   runs the full `epochs` budget unconditionally — it ignores `min_epochs` and `patience`,
   which every other deep model honours through `train_graph_level` / `train_node_level`.

   Measured on the real cohort at the default 200 epochs: **72 minutes for a single fold**,
   which is ~12 hours for one k-fold run and ~24 hours for k-fold plus LOSO. The grid was
   stalled on it. We reran BrainNetCNN with `--epochs 30`; that number is a compute
   concession, not a tuned choice, and the reported BrainNetCNN row is therefore **not
   comparable on equal footing** with the early-stopped models. Say so if it is in the deck,
   or leave it out of the headline table.

   For reference, the first fold at 200 epochs scored bal_acc 0.639 / AUC 0.723 — the
   strongest deep-model fold we saw — so this is worth fixing properly rather than dropping.
   The fix is to give it the same `inner_val_split` + patience treatment as the graph models.

5. **Human-in-the-loop leakage remains the real risk.** No test suite catches
   hyperparameters or QC thresholds chosen after seeing test performance. ABIDE-II as a
   genuinely untouched external set (Phase 6) is the only real mitigation.

---

## 7. Commands to re-verify everything

```bash
cd asd-gnn

# if imports fail:
VIRTUAL_ENV="$PWD/.venv" uv pip install -e . --no-deps

.venv/bin/python -m pytest tests/ -q                    # 85 tests
.venv/bin/python -m ruff check src tests scripts notebooks
make smoke PY=.venv/bin/python                          # ~20 s, synthetic

# real data (already downloaded — skip 00 unless starting fresh)
.venv/bin/python scripts/00_download.py --atlas cc200
.venv/bin/python scripts/01_build_features.py --atlas cc200 --min-timepoints 70
.venv/bin/python scripts/02_run_experiment.py --sanity
.venv/bin/python scripts/02_run_experiment.py --gnn-sanity
.venv/bin/python scripts/02_run_experiment.py --model svm --protocol kfold
.venv/bin/python scripts/03_aggregate_results.py
.venv/bin/python scripts/04_make_figures.py
```

The synthetic-cohort results that were in `results/runs/` before this session have been moved
to `results/runs_synthetic/` so they cannot contaminate the real-data tables. They are still
on disk if anyone wants them.
