# Results Snapshot — ABIDE-I, CC200

**Run date:** 4 September 2026 · **Runs on disk:** 36 — grid complete
**Every number here comes from a file under `asd-gnn/results/`.** Provenance is given per
section. If a number is not in this document, it is not verified — do not put it on a slide.

**Primary metric is balanced accuracy**, mean ± std across folds. The cohort is imbalanced
(403 / 468), so plain accuracy is misleading.

---

## 1. Cohort

*Source: `results/qc/cohort.json`, `results/qc/features.json`*

| | |
|---|---|
| Subjects | **871** (871/871 passed QC — none dropped) |
| ASD / TC | **403 / 468** |
| Sites | **20** |
| Male | 83.5% |
| Age | 6.5 – 58.0 years (mean 16.9 ± 7.6) |
| Mean framewise displacement | **ASD 0.126 · TC 0.094** |
| ROIs after QC | **179** of 200 (21 constant CC200 parcels dropped) |
| Feature dimension | **15,931** (179 × 178 / 2) |
| k-fold folds used | 10 (1 seed) |
| LOSO folds | 20 |
| Max per-fold site-proportion drift | 0.164 |
| Sites with n < 20 (flagged underpowered) | CALTECH (15), CMU (11) |

> **N = 871 with 403 ASD / 468 TC is exactly the canonical benchmark subset.** Our numbers
> are directly comparable to Abraham et al. and the rest of the literature. Say this.

**The motion confound is real and present in our own data:** ASD subjects move more
(FD 0.126 vs 0.094). Do not let this come up first from the audience.

---

## 2. Sanity checks — all gates pass

*Source: `results/qc/sanity.json`*

| Check | Value | Verdict |
|---|---|---|
| **Label shuffle** (full pipeline, shuffled y) | **0.525** | ✅ PASS — collapses to chance |
| Real labels, same pipeline | **0.633** | margin **+0.108** over shuffled |
| **Connectivity fit size** | **696 / 871** | ✅ PASS — estimator never saw the test fold |
| **Mean node degree** (CC200, top-10%) | **17.80** | ✅ PASS — inside the healthy 10–40 band |
| Sex control task | 0.579 | positive control: features carry real biological signal |
| **Site control task** | **0.673** | ⚠️ **the confound, quantified** |
| Random node features ablation | 0.500 (vs GNN 0.546) | ✅ large drop — the model reads its features |
| Edge-free graph ablation | 0.511 (vs GNN 0.546) | topology contributes ≈ +0.035 here |

**The two numbers worth a sentence each on the day:**

- **Site control = 0.673.** A classifier can identify the scanner site from these features
  about as well as it can identify autism. That single number explains the entire k-fold →
  LOSO story and belongs in the discussion.
- **Label shuffle = 0.525.** The full pipeline, end to end, with the labels destroyed,
  lands at chance. This is the strongest evidence we have that we are not leaking.

---

## 3. Headline results — both protocols

*Source: `results/tables/table_kfold.md`, `table_loso.md`, `table_protocol_comparison.md`*

Balanced accuracy, mean ± std. Configuration: CC200, Pearson correlation, no harmonisation.

| Model | 10-fold CV | LOSO | Drop |
|---|---|---|---|
| **SVM (linear)** | **0.674 ± 0.060** | 0.631 ± 0.077 | 0.043 |
| MLP | 0.666 ± 0.037 | **0.643 ± 0.085** | 0.023 |
| **BrainNetCNN** | **0.658 ± 0.077** | **0.639 ± 0.087** | 0.019 |
| Ridge | 0.642 ± 0.067 | 0.589 ± 0.049 | 0.053 |
| Random Forest | 0.622 ± 0.026 | 0.625 ± 0.087 | −0.003 |
| GCN | 0.600 ± 0.053 | 0.560 ± 0.068 | 0.040 |
| **Hybrid (ours)** | 0.598 ± 0.037 | 0.569 ± 0.088 | 0.029 |
| ChebNet | 0.597 ± 0.071 | 0.593 ± 0.070 | 0.004 |
| GraphSAGE | 0.596 ± 0.077 | 0.590 ± 0.065 | 0.006 |
| GIN | 0.581 ± 0.050 | 0.572 ± 0.067 | 0.010 |
| GAT | 0.552 ± 0.050 | 0.581 ± 0.075 | −0.029 |
| Population ChebNet | 0.523 ± 0.054 | 0.520 ± 0.071 | 0.003 |
| Population GCN | 0.514 ± 0.048 | 0.508 ± 0.030 | 0.006 |
| **MLP, edge-free control** | **0.513 ± 0.029** | **0.510 ± 0.037** | 0.003 |

AUC (pooled) for the top rows: **BrainNetCNN 0.716 / 0.715** · MLP 0.735 / 0.713 ·
SVM 0.713 / 0.677 · RF 0.682 / 0.668 · ChebNet 0.648 / 0.648.

**On AUC, BrainNetCNN is the best model under both protocols** — and it is the *only* model
whose AUC does not drop at all between k-fold and LOSO (0.716 → 0.715).

### What this actually says — read carefully before writing the talk

**(a) Every number is inside the pre-registered honest band, and nothing is in "something is
wrong" territory.** Target was 68–76% k-fold and 58–68% LOSO; alarm lines were >85% k-fold
and >80% LOSO. Our best k-fold is 0.674 and our best LOSO is 0.643. We are slightly *under*
the k-fold target and comfortably inside the LOSO one.

**(b) Every model that sees the full connectivity matrix beats every model that sees a
thresholded graph.** This is the cleanest way to state our central result, and BrainNetCNN is
what makes it clean:

| | Sees | k-fold AUC | LOSO AUC |
|---|---|---|---|
| MLP | full matrix (15,931 features) | 0.735 | 0.713 |
| **BrainNetCNN** | **full matrix** (edge-to-edge convolutions) | **0.716** | **0.715** |
| SVM | full matrix | 0.713 | 0.677 |
| ChebNet (best GNN) | top-10% thresholded graph | 0.648 | 0.648 |
| GCN | thresholded graph | 0.635 | 0.568 |
| MLP, edge-free control | node features, no graph | 0.510 | 0.514 |

**This contradicts H1**, which predicted GNNs would beat SVM/Ridge under k-fold by 3–8
points. Report it as a refuted hypothesis — pre-registering it is what makes the refutation a
result rather than an excuse.

**(c) It is not that deep learning fails on this problem.** That was the obvious rival
explanation, and BrainNetCNN rules it out. BrainNetCNN is a deep model — edge-to-edge,
edge-to-node and node-to-graph convolutions — and it is statistically **indistinguishable
from the SVM** (ΔAUC +0.002, p_adj = 1.00 under k-fold; +0.038, p_adj = 0.74 under LOSO) and
from the MLP. Meanwhile it beats **every single GNN** significantly under both protocols
(ΔAUC +0.066 to +0.201, all p_adj < 0.05).

So: deep model + full matrix ≈ linear model + full matrix ≫ deep model + thresholded graph.
**The variable that separates the winners from the losers is not model class. It is whether
the model sees the whole connectivity matrix.**

**(d) The diagnosis, and it is directly testable.** Top-10% thresholding keeps ~10% of the
matrix — mean node degree 17.8 out of a possible 178. Everything else is discarded before the
GNN starts. Topology is not worthless: the GNNs beat the edge-free control (0.510 AUC) by a
wide, significant margin, so message passing over even a sparse graph is worth ~0.12 AUC. But
that gain does not recover what sparsification threw away.

**This makes the edge-density ablation the single highest-value next experiment**
(`configs/experiment/e03_edge_density.yaml`, k = 5 / 10 / 20% — already configured). The
prediction is explicit and falsifiable: **GNN performance should rise monotonically with edge
density, and approach BrainNetCNN as the graph approaches fully connected.** If it does not,
the sparsification hypothesis is wrong and the problem is the architecture. Either way we
learn something. Say this on the day — it turns a disappointing table into a concrete,
pre-registered next step.

**(e) The k-fold → LOSO drop is much smaller than the literature's.** Published work shows
collapses from ~70% to ~52–56%. Our worst honest drop is 0.053 (Ridge); Random Forest and
GAT actually score *higher* under LOSO, and **BrainNetCNN's AUC is essentially identical
across protocols (0.716 → 0.715)**. Two candidate reasons, and we should say both: our
k-fold numbers are lower than the inflated published ones *because* the pipeline is
leakage-audited, and stratifying k-fold by site (which we do) already removes some of the
optimism that random k-fold enjoys. **We are comparing an honest k-fold against an honest
LOSO, and the gap between them is correspondingly modest.** That is a finding, and arguably
a more interesting one than reproducing a big collapse.

---

## 4. Connectivity ablation (SVM)

*Source: `results/tables/table_protocol_comparison.md`*

| Connectivity | k-fold | LOSO |
|---|---|---|
| **Pearson correlation** | **0.674 ± 0.060** | **0.631 ± 0.077** |
| Tangent space | 0.626 ± 0.076 | 0.608 ± 0.083 |
| Partial correlation | 0.575 ± 0.070 | 0.572 ± 0.085 |

**Tangent space losing to plain correlation is itself a leakage-audit result.** Tangent is
usually reported as the strongest measure — but its group geometric-mean reference is
*estimated from data*, and fitting it on all subjects before cross-validation inflates
accuracy by 4–6 points. We fit it strictly inside the fold, and it does not win. That is
consistent with a large part of tangent's published advantage being fold leakage. This is a
good, quotable point, and it is worth a slide of its own if there is time.

---

## 5. Harmonisation — RQ4

*Source: `results/tables/table_protocol_comparison.md`*

SVM + CC200 + Pearson, with and without ComBat:

| Configuration | k-fold | LOSO |
|---|---|---|
| No harmonisation | **0.674 ± 0.060** | **0.631 ± 0.077** |
| ComBat, `passthrough` | 0.646 ± 0.040 | **0.500 ± 0.000** ⚠️ |
| ComBat, `transductive` | — | 0.588 ± 0.046 |
| ComBat, `reference` | — | 0.590 ± 0.045 |

**H3 predicted ComBat would improve LOSO and slightly hurt k-fold. Half right:**
it does slightly hurt k-fold (0.674 → 0.646, as predicted), and it **hurts LOSO under every
policy** (0.631 → 0.500 / 0.588 / 0.590), which the hypothesis did not predict.

**The `passthrough` collapse is the most striking single result we have, and it is fully
explainable.** Under LOSO the held-out site was never seen during ComBat's fit, so no
parameters exist for it. `passthrough` leaves that site un-harmonised while every training
site *was* harmonised — so the model is tested on a distribution it was never trained on.
It predicts one class for everyone: sensitivity 0.00, specificity 1.00, balanced accuracy
exactly 0.500 with zero variance across all 20 folds.

This is not a bug. It is the honest consequence of a declared policy, our test suite asserts
that the policy is explicit rather than accidental, and it is the single strongest argument
for the adversarial site-invariance head — which has no equivalent problem because it needs
no per-site parameters at test time. **Put this on a slide.** It is the most memorable thing
in the whole result set.

---

## 6. Per-site LOSO — SVM, the site heterogeneity story

*Source: `results/runs/*.json` (default SVM LOSO run) · figure `results/figures/per_site_svm.png`*

| Site | n | bal. acc | AUC | | Site | n | bal. acc | AUC |
|---|---|---|---|---|---|---|---|---|
| NYU | 172 | 0.678 | 0.700 | | LEUVEN_1 | 28 | 0.643 | 0.714 |
| UM_1 | 86 | 0.602 | 0.653 | | LEUVEN_2 | 28 | 0.542 | 0.609 |
| USM | 67 | 0.701 | 0.814 | | OLIN | 28 | 0.679 | 0.750 |
| UCLA_1 | 64 | 0.639 | 0.672 | | SDSU | 27 | 0.645 | 0.717 |
| PITT | 50 | 0.635 | 0.652 | | SBL | 26 | 0.685 | 0.583 |
| MAX_MUN | 46 | 0.612 | 0.593 | | **OHSU** | 25 | **0.397** | 0.500 |
| TRINITY | 44 | 0.608 | 0.608 | | STANFORD | 25 | 0.635 | 0.724 |
| YALE | 41 | 0.585 | 0.699 | | UCLA_2 | 21 | **0.759** | 0.809 |
| UM_2 | 34 | 0.679 | 0.795 | | *CALTECH* | *15* | *0.600* | *0.540* |
| KKI | 33 | 0.571 | 0.587 | | *CMU* | *11* | *0.733* | *0.867* |

*Italic sites are flagged `underpowered` (n < 20) by the pipeline and should not be
interpreted individually.*

**Range: 0.397 to 0.759 — a 36-point spread across sites, from the same model.** This is the
argument for reporting per-site n alongside per-site accuracy, and it is why a single LOSO
mean hides more than it reveals.

**OHSU is the worst site at 0.397 — below chance — and OHSU is also the site with the
shortest scans (78 timepoints vs up to 296 elsewhere).** A 78-timepoint scan gives a much
noisier correlation estimate. This ties the QC threshold decision (see
`PIPELINE-STATUS.md` §3) directly to a result, and it is a genuinely good observation to
make unprompted: it suggests scan length, not just scanner identity, drives part of the
cross-site variance.

---

## 7. Statistical tests

*Source: `results/tables/stats_kfold.json`, `stats_loso.json`* — DeLong on 871 paired
subjects, Holm-corrected across 91 comparisons per protocol.

> **Note on how these were produced.** Only *default-configuration* runs (no harmonisation,
> Pearson correlation) are paired, so `svm_vs_gcn` means what it says. An earlier version of
> the pairing code keyed on the model name alone and silently compared against whichever SVM
> run sorted last — the partial-correlation one — which understated every SVM comparison. See
> `PIPELINE-STATUS.md` §5.6. Each stats file now records `compared_runs` (the exact run_id
> used per model) and `excluded_non_default_runs`.

**Significant after Holm correction — k-fold** (47 of 91 comparisons significant):

| Comparison | ΔAUC | p_adj |
|---|---|---|
| SVM > edge-free MLP control | +0.203 | 6.4e-14 |
| BrainNetCNN > edge-free control | +0.205 | 3.9e-15 |
| SVM > population GCN / ChebNet | +0.165 / +0.164 | 3.0e-10 / 6.6e-10 |
| BrainNetCNN > GAT | +0.131 | 4.3e-08 |
| SVM > GAT | +0.129 | 2.2e-08 |
| SVM > hybrid | +0.108 | 5.6e-05 |
| BrainNetCNN > GIN / GCN / GraphSAGE | +0.088 / +0.081 / +0.078 | all < 0.05 |
| SVM > GIN / GCN / GraphSAGE | +0.086 / +0.079 / +0.076 | all < 0.05 |

**Significant after Holm correction — LOSO** (42 of 91):

| Comparison | ΔAUC | p_adj |
|---|---|---|
| BrainNetCNN > population GCN | +0.199 | 6.4e-15 |
| BrainNetCNN > edge-free control | +0.201 | 1.4e-13 |
| SVM > edge-free control | +0.163 | 3.0e-08 |
| SVM > population GCN | +0.161 | 4.4e-09 |
| BrainNetCNN > GCN | +0.147 | 9.8e-11 |
| BrainNetCNN > GIN | +0.115 | 8.4e-06 |
| SVM > GCN | +0.109 | 3.3e-05 |
| BrainNetCNN > GraphSAGE / ridge / GAT / ChebNet | +0.092 / +0.082 / +0.080 / +0.066 | all < 0.05 |

**Not significant after correction — and these matter as much as the significant ones:**

- **BrainNetCNN vs SVM:** +0.002 (k-fold, p_adj = 1.00) and +0.038 (LOSO, p_adj = 0.74)
- **BrainNetCNN vs MLP:** −0.019 (k-fold) and +0.002 (LOSO), both p_adj = 1.00
- **MLP vs SVM:** +0.021 (k-fold) and +0.036 (LOSO), both n.s.

**The honest one-line summary:** *the three full-matrix models are statistically
indistinguishable from each other, and all three significantly beat every graph-based model.*

**No GNN significantly beats any full-matrix model under either protocol**, and most are
significantly worse. ChebNet is the only GNN that is not significantly worse than the SVM
under either protocol (p_adj = 0.074 k-fold, 1.00 LOSO) — so if we keep one GNN as the
representative, it should be ChebNet, not GAT.

The corrected resampled t-test could not be run for the hybrid comparison — the pipeline
correctly **refused** it because the hybrid uses 5 folds against the others' 10, and a paired
fold-wise test across mismatched folds is invalid. That refusal is a feature and is worth
mentioning as evidence of the harness's care.

## 8. Hybrid model — RQ2

*Source: `results/runs/*.json` (hybrid runs), `runlogs/hybrid_*.log`*

| Protocol | Stage 1 (brain-graph encoder alone) | Stage 3 (+ population graph) | Reported |
|---|---|---|---|
| k-fold | 0.585 | **0.598** | 0.598 ± 0.037 |
| LOSO | 0.538 | **0.569** | 0.569 ± 0.088 |

**The hybrid does beat its own encoder** under both protocols (+0.013 k-fold, +0.031 LOSO),
which is partial support for H2 — combining topology with phenotype helps relative to
topology alone. **But it does not beat the SVM** (0.598 vs 0.674; 0.569 vs 0.631), and the
DeLong test finds no significant difference from SVM.

Note the population-graph models *alone* are near chance (0.508–0.523), so the hybrid's
population stage is contributing mainly as a refinement of the encoder embedding rather than
as a strong model in its own right.

Not yet run: the end-to-end variant and the adversarial λ sweep. Both are implemented and
tested — see `PIPELINE-STATUS.md` §5.3 for the trap that was fixed there.

---

## 9. Explainability — preliminary

*Source: `results/explain/roi_importance.json` · figures `roi_stability.png`, `top_rois.png`*

| | |
|---|---|
| Model / method | GAT attention weights, k-fold, 5 folds |
| Per-fold balanced accuracy | 0.652, 0.515, 0.590, 0.565, 0.604 |
| **Mean pairwise Spearman across folds** | **0.961** |
| ROIs in the top-k in ≥ 80% of folds | **18** |
| Top 5 ROIs | CC200_005, CC200_103, CC200_067, CC200_175, CC200_055 (all in top-k in 100% of folds) |

**The ranking is highly stable (ρ = 0.961) — report stability before ranking.** An ROI list
that reorders every fold is noise no matter how plausible it looks.

**But the ROI names are meaningless right now.** CC200 parcels have no anatomical labels or
centroid coordinates in `nilearn`, so we cannot yet say "the model found the fusiform gyrus",
and we cannot run the Yeo-7 network enrichment test. Fixing this needs AAL or Dosenbach-160,
which is why the atlas ablation (RQ3) must come before the interpretability claim (RQ5).
**Do not overclaim on this slide.**

Also note the underlying GAT is our *weakest* graph model (0.552 k-fold). A stable
explanation of a weak model is of limited value — worth re-running the saliency analysis on
ChebNet, the strongest GNN, before the final report.

---

## 10. What is still pending

| Item | Status |
|---|---|
| Edge-density ablation (k = 5/10/20%) | **Highest-value next experiment** — config already exists, and §3(d) states a falsifiable prediction for it |
| BrainNetCNN at full epoch budget | Done at **30 epochs** (0.658 / 0.639). It has no early stopping, so 200 epochs costs ~72 min/fold. Fixing that (give it the same `inner_val_split` + patience as the graph models) would make its row comparable on equal footing |
| Atlas ablation: AAL, Harvard–Oxford (RQ3) | Not run — also unblocks named-ROI interpretability |
| Hybrid end-to-end + adversarial λ sweep | Implemented, not swept |
| Yeo-7 enrichment, ADOS correlation (RQ5) | Blocked on atlas ablation |
| ABIDE-II external validation | Phase 6, not started |

---

## 11. Numbers to fill into `SLIDES.md`

| Token | Value |
|---|---|
| `{{N_SUBJECTS}}` | 871 |
| `{{N_ASD}}` / `{{N_TC}}` | 403 / 468 |
| `{{N_SITES}}` | 20 |
| `{{MALE_PCT}}` | 83.5 |
| `{{AGE_MIN}}` – `{{AGE_MAX}}` | 6.5 – 58.0 |
| `{{N_TESTS}}` | 85 |
| `{{SANITY_SHUFFLE}}` | 0.525 |
| `{{SANITY_REAL}}` | 0.633 (+0.108 over shuffled) |
| `{{SANITY_FITSIZE}}` | 696 / 871 |
| `{{SANITY_DEGREE}}` | 17.80 |
| `{{SANITY_SEX}}` | 0.579 |
| `{{SANITY_SITE}}` | 0.673 |
| `{{SANITY_RANDFEAT}}` | 0.500 (GNN 0.546) |
| `{{SANITY_EMPTY}}` | 0.511 (GNN 0.546) |
| `{{SVM_KFOLD}}` / `{{SVM_LOSO}}` / `{{SVM_DELTA}}` | 0.674 ± 0.060 / 0.631 ± 0.077 / 0.043 |
| `{{RIDGE_KFOLD}}` / `{{RIDGE_LOSO}}` / `{{RIDGE_DELTA}}` | 0.642 ± 0.067 / 0.589 ± 0.049 / 0.053 |
| `{{RF_KFOLD}}` / `{{RF_LOSO}}` / `{{RF_DELTA}}` | 0.622 ± 0.026 / 0.625 ± 0.087 / −0.003 |
| `{{GCN_KFOLD}}` / `{{GCN_LOSO}}` | 0.600 ± 0.053 / 0.560 ± 0.068 |
| `{{GAT_KFOLD}}` / `{{GAT_LOSO}}` | 0.552 ± 0.050 / 0.581 ± 0.075 |
| `{{GIN_KFOLD}}` / `{{GIN_LOSO}}` | 0.581 ± 0.050 / 0.572 ± 0.067 |
| `{{SAGE_KFOLD}}` / `{{SAGE_LOSO}}` | 0.596 ± 0.077 / 0.590 ± 0.065 |
| `{{CHEB_KFOLD}}` / `{{CHEB_LOSO}}` | 0.597 ± 0.071 / 0.593 ± 0.070 |
| `{{BRAINCNN_KFOLD}}` / `{{BRAINCNN_LOSO}}` | 0.658 ± 0.077 / 0.639 ± 0.087 |
| `{{MLPGRAPH_KFOLD}}` / `{{MLPGRAPH_LOSO}}` | 0.513 ± 0.029 / 0.510 ± 0.037 |
| `{{ROI_SPEARMAN}}` | ρ = 0.961 |
| `{{ROI_STABLE_COUNT}}` | 18 ROIs appear in the top-k in ≥80% of folds |
| `{{TORCH_VER}}` … | torch 2.13.0 · torch-geometric 2.8.0.post1 · scikit-learn 1.8.0 · nilearn 0.14.0 · numpy 2.4.6 |
