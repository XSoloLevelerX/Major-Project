# ASD Classification from Brain Connectivity using Graph Neural Networks
### End-to-End Project Scope — Major_Group ID_26271, PEC, Semester 7
**Mentor:** Prof. Amandeep Kaur · **Deliverables:** working system + research paper + report + viva

---

## 0. Read this first: the one thing that decides whether this project succeeds

There are hundreds of published ABIDE + GNN papers. Reproducing "we ran a GCN on ABIDE and got 70%" is a *guaranteed pass and a guaranteed rejection*. The papers that get accepted in 2026 are not the ones with the highest number — they are the ones that are **honest about site heterogeneity**.

Here is the uncomfortable truth of this field, and it is your opening:

- Under **random 10-fold CV**, published accuracies range from ~65% to ~90%.
- Under **leave-one-site-out (LOSO)** — training on 16 sites, testing on the 17th, i.e. what actually happens in a clinic — a competent classical ML framework got <cite index="15-1">average accuracy of 56.21%, 51.34%, 54.61%, 56.26% and 52.16% for LDA, SVM, Random Forest, MLP and KNN</cite>. That is near chance.
- The reason is documented: <cite index="18-1">machine learning performs quite poorly on combined datasets from different imaging sites, attributable to data heterogeneity such as MRI scanner suppliers, scan parameters (particularly TR), and cohort setting, wherein variability in phenotype — deviation in age and gender composition — is unavoidable</cite>.
- A 2026 analysis of ABIDE motion/IQ confounds put it bluntly: <cite index="13-1">the positive cross-validation R² one obtains under random k-fold is an artifact of site information leaking between training and test folds</cite>.

**Your project's spine, therefore:** build a rigorous, leakage-free multi-atlas GNN benchmark, report *both* k-fold and LOSO, and make site-invariance an explicit modelling objective rather than an afterthought. That single decision converts a crowded topic into a defensible contribution.

---

## 1. Problem Statement

### 1.1 Clinical framing
Autism Spectrum Disorder is diagnosed behaviourally (ADOS/ADI-R), requires trained clinicians, and in India carries a median diagnostic delay of several years. There is no objective biomarker in routine use. Resting-state fMRI reveals reproducible group-level differences in functional connectivity — notably long-range under-connectivity and local over-connectivity in the default mode network, superior temporal sulcus, insula, fusiform gyrus and cerebellum — but these differences do not translate to individual-level prediction reliably across scanners.

### 1.2 Technical framing
The brain is natively a graph: regions of interest (ROIs) as nodes, functional coupling as edges. Flattening a connectivity matrix into a vector for an SVM destroys this topology. GNNs preserve it. But existing GNN work on ABIDE (a) mostly reports optimistic within-site-mixed CV, (b) usually commits to a single brain parcellation, and (c) rarely validates that the ROIs it highlights match the ASD neuroimaging literature.

### 1.3 Formal statement

> Given a set of subjects $S = \{s_1 \dots s_N\}$, each with a preprocessed resting-state BOLD time-series matrix $T_i \in \mathbb{R}^{R \times t_i}$ over $R$ atlas-defined ROIs, together with phenotypic covariates $p_i$ (age, sex, site, eye status, FIQ) and a binary label $y_i \in \{\text{ASD}, \text{TC}\}$:
>
> **(a)** Construct a graph representation $G_i = (V, E, X)$ per subject, and/or a population graph $G_{\text{pop}}$ over all subjects.
> **(b)** Learn $f_\theta: G \rightarrow \hat{y}$ that maximises balanced accuracy and AUC.
> **(c)** Subject to the constraint that performance is measured under **leave-one-site-out** generalisation, i.e. the test site's data is never seen during training, harmonisation fitting, feature selection, or hyperparameter search.
> **(d)** And such that $f_\theta$ produces an ROI-level saliency ranking $\sigma \in \mathbb{R}^{R}$ that can be tested against independent ASD neuroimaging findings.

### 1.4 Research questions

| # | Question | Answerable by |
|---|---|---|
| RQ1 | How much of the reported GNN advantage over classical ML on ABIDE survives leave-one-site-out evaluation? | Experiment grid A |
| RQ2 | Does the graph formulation matter — individual brain graphs (graph-level classification) vs. population graphs (node-level, transductive) vs. a hybrid? | Experiment grid B |
| RQ3 | Does multi-atlas fusion beat the best single atlas, and is the gain real or an ensembling artifact? | Experiment grid C |
| RQ4 | Does explicit site harmonisation (ComBat) or adversarial site-invariance recover cross-site performance, and at what cost to within-site accuracy? | Experiment grid D |
| RQ5 | Do the ROIs the model attends to converge on the ASD literature, and do saliency scores correlate with ADOS symptom severity? | Interpretability study |

### 1.5 Hypotheses (state these up front; being wrong is publishable)
- **H1:** GNNs beat SVM/Ridge under k-fold by 3–8 points, but the margin shrinks to <3 points under LOSO.
- **H2:** The hybrid encoder (brain-graph → embedding → population-graph node feature) outperforms either formulation alone, because it uses topology *and* phenotype.
- **H3:** ComBat harmonisation improves LOSO but slightly hurts k-fold.
- **H4:** Attention weights concentrate on DMN, temporal and cerebellar ROIs above chance.

---

## 2. Contributions (what you will claim in the paper)

1. A **leakage-audited benchmark** of 4 classical baselines × 5 GNN architectures × 3 atlases × 3 connectivity measures on ABIDE I, under matched k-fold *and* LOSO protocols with identical splits released publicly.
2. A **hybrid brain-graph→population-graph architecture** with a learned subject embedding, ablated against both parent formulations.
3. A **quantified harmonisation study** (raw vs. ComBat vs. adversarial site-adversarial training) isolating how much of the k-fold→LOSO drop is scanner effect versus genuine heterogeneity.
4. **Interpretability validation**: ROI saliency maps tested for enrichment against known ASD networks, plus correlation of saliency with ADOS-Total severity.
5. A **reproducible open-source pipeline** (config-driven, seeded, one command per experiment) — this alone differentiates you from 90% of student work and is what a reviewer checks first.

You do not need all five to pass. You need 1 + 3 to have a paper. 2 and 5 are what make it good.

---

## 3. Scope Boundaries

**In scope**
- ABIDE I preprocessed (PCP), ABIDE II as an external validation set if time allows
- rs-fMRI functional connectivity only (sMRI as an optional stretch modality)
- Binary ASD vs. TC classification; ADOS severity regression as auxiliary head
- CPU-feasible preprocessing; single-GPU (Colab T4 / 16 GB) training

**Explicitly out of scope — say this in the report, it protects you**
- Raw NIfTI preprocessing from scratch (fMRIPrep/CPAC). You use PCP-derived ROI time series. Justify: standardisation and reproducibility, not laziness.
- Clinical deployment or diagnostic claims. This is a decision-support research prototype.
- Voxel-level 3D CNNs, dynamic FC / sliding-window analysis (mention as future work).
- Any collection of new human data — no ethics clearance needed, which saves you two months.

---

## 4. Dataset — exact acquisition path

### 4.1 What ABIDE is
<cite index="18-1">ABIDE consists of resting-state fMRI, structural images and phenotypic data for 539 patients with ASD and 573 typical controls, recorded at 17 locations worldwide.</cite> ABIDE II adds ~19 more sites.

### 4.2 How you actually get it — no registration, no download portal
The Preprocessed Connectomes Project derivatives are on public S3 and `nilearn` fetches them directly:

```python
from nilearn.datasets import fetch_abide_pcp

abide = fetch_abide_pcp(
    data_dir="./data/raw",
    pipeline="cpac",                 # 'ccs' | 'cpac' | 'dparsf' | 'niak'
    band_pass_filtering=True,        # 0.01–0.1 Hz
    global_signal_regression=False,  # the standard 'filt_noglobal' strategy
    derivatives=["rois_cc200", "rois_aal", "rois_ho"],
    quality_checked=True,            # -> the canonical 871-subject subset
)
```

<cite index="21-1">Possible pipelines are "ccs", "cpac", "dparsf" and "niak". Band pass filtering is optional; if true, signal is band filtered between 0.01Hz and 0.1Hz. Derivative types include rois_aal, rois_cc200, rois_cc400, rois_dosenbach160, rois_ez, rois_ho, rois_tt, alff, falff, reho, degree_binarize, eigenvector_weighted and vmhc.</cite>

**Critical:** request `rois_*` derivatives, **not** `func_preproc`. ROI time series are a few hundred MB total; the full 4-D volumes are ~50 MB *per subject* and will burn a day of bandwidth for nothing.

Phenotypic CSV (`Phenotypic_V1_0b_preprocessed1.csv`) comes with the fetch and contains `SITE_ID`, `DX_GROUP`, `AGE_AT_SCAN`, `SEX`, `FIQ`, `EYE_STATUS_AT_SCAN`, `func_mean_fd`, and ADOS/ADI/SRS subscales.

### 4.3 The canonical benchmark subset
Setting `quality_checked=True` on CPAC/filt_noglobal yields the **871-subject** cohort (≈403 ASD / 468 TC) used by Abraham et al. and nearly every comparable paper. Use this as your headline number so results are directly comparable. Report the full 1035–1112 cohort as a secondary row.

### 4.4 Atlases to run (pick three)
| Atlas | ROIs | Feature dim (upper Δ) | Why |
|---|---|---|---|
| CC200 | 200 | 19,900 | Most-used benchmark; functionally derived |
| AAL | 116 | 6,670 | Anatomical, interpretable region names |
| Harvard–Oxford | 110 | 5,995 | Probabilistic, good for cross-checking |
| *(optional)* CC400 | 392 | 76,636 | Fine-grained; heavy |

### 4.5 Known confounds you must control and report
- **Sex imbalance** — cohort is ~85% male. Report per-sex metrics; do not train a covert sex classifier.
- **Age range** — roughly 6–64 years. Include age as a covariate; consider an age-restricted sensitivity analysis.
- **Head motion** — `func_mean_fd` differs systematically between groups. Either regress it out or report accuracy stratified by motion. Reviewers *will* ask.
- **Site size imbalance** — some sites have <20 subjects; LOSO on these is statistically meaningless. Report per-site n alongside per-site accuracy.
- **Eye status** — eyes-open vs. eyes-closed protocols differ by site.

### 4.6 ABIDE II as held-out external validation
The strongest possible evaluation: train on ABIDE I, test on ABIDE II, never touching II during development. Precedent exists — one recent model reported <cite index="3-1">training on ABIDE I (n=840) and testing on ABIDE II (n=660)</cite>. If you can pull this off, it is your paper's headline figure.

---

## 5. Methodology

### 5.1 Stage 1 — Signal to connectivity

```
ROI time series (R × t)
  → [optional] scrub high-motion volumes (FD > 0.5mm), drop subjects with <100 usable frames
  → nilearn ConnectivityMeasure
  → connectivity matrix (R × R)
  → Fisher z-transform (for correlation)
  → vectorise upper triangle → feature vector
```

Three connectivity measures, all as an ablation axis:
- **Pearson correlation** — baseline, simple, noisy
- **Partial correlation** (Ledoit–Wolf shrinkage) — removes indirect paths, sparser, needs regularisation at R=200
- **Tangent space embedding** — projects covariances onto the tangent space at the group geometric mean; generally the strongest. <cite index="14-1">By tangent space embedding, the underlying functional connectivity can be achieved at group level, which to some extent reduces the difference among individuals.</cite>

> ⚠️ **Leakage trap #1:** the tangent-space reference mean must be estimated on **training folds only** and applied to test. `ConnectivityMeasure(kind='tangent')` must be inside a scikit-learn `Pipeline` and `.fit()` only on train. Getting this wrong inflates accuracy by 4–6 points and is the single most common bug in student ABIDE code.

### 5.2 Stage 2 — Three graph formulations

**(A) Individual brain graph — graph-level classification**
- Node = ROI (200 nodes for CC200)
- Node features = that ROI's row of the connectivity matrix (dim 200), optionally concatenated with MNI centroid coordinates, ALFF/ReHo, and one-hot ROI identity
- Edges = thresholded connectivity. Threshold strategies to ablate: top-k% absolute weights (k=5,10,20), kNN per node (k=5,10), absolute-value cutoff, and a learned/adaptive adjacency
- Edge features = signed connectivity weight
- Output: one label per graph → `global_mean_pool` / `TopKPooling` / `SAGPool` readout

**(B) Population graph — node-level semi-supervised classification**
- Node = **subject**; there is exactly one graph for the whole cohort
- Node features = vectorised connectivity (after dimensionality reduction: ANOVA F-test top-k, or PCA fit on train only)
- Edges = phenotypic similarity: $A_{ij} = \text{sim}(x_i, x_j) \cdot \sum_h \gamma_h(p_i^h, p_j^h)$ where $\gamma$ rewards matching sex, similar age, and (optionally) same site
- Transductive: test-node features are visible, labels are not
- This is the Parisot-style formulation; it is the classic and you must include it as a baseline

> ⚠️ **Leakage trap #2:** if you build population-graph edges using site identity and then evaluate with LOSO, the held-out site's nodes are isolated. Handle explicitly — it is a genuine methodological point worth a paragraph in the paper.

**(C) Hybrid (your proposed contribution)**
```
per-subject brain graph → GNN encoder → 64-d subject embedding
                                              ↓
                       population graph node features = [embedding ‖ phenotype]
                                              ↓
                                    population GNN → label
```
Train end-to-end or in two stages (stage-wise is easier to debug — do that first). Add a **gradient-reversal site-discriminator head** on the embedding to force site-invariance; this directly targets RQ4 and is your strongest novelty lever.

### 5.3 Stage 3 — Model zoo

**Classical baselines (must-have — a paper without these is rejected)**
- Ridge / logistic regression with L2, on vectorised FC
- Linear SVM (the field's workhorse; Abraham et al.'s reference point)
- Random Forest
- MLP (2 hidden layers) — proves the GNN gain isn't just "deep learning"

**Deep non-graph baselines**
- BrainNetCNN (edge-to-edge → edge-to-node → node-to-graph convolutions)
- 1D-CNN over the connectivity profile

**GNNs**
| Model | Why include it |
|---|---|
| GCN | Simplest message passing; the floor |
| ChebNet | Spectral, Parisot's original choice on population graphs |
| GraphSAGE | Inductive, samples neighbours; better for unseen sites |
| GAT | Learned attention → free interpretability |
| GIN | Most expressive under WL; good graph-level classifier |
| Graph Transformer | Global attention; current strong performer |

Attention-based models are the sweet spot for this problem — one 2026 study found <cite index="6-1">the network including GAT layers performed best of all architectures tested, with GCN second, and a simple feed-forward network lowest</cite>.

### 5.4 Stage 4 — Harmonisation (RQ4)
- **Baseline:** no harmonisation
- **ComBat / neuroCombat:** empirical-Bayes removal of site effects from connectivity features, preserving age/sex/diagnosis as biological covariates. Fit on training sites only.
- **Adversarial:** gradient reversal layer + site classifier on the shared embedding
- **Contrastive/invariance objectives** as a stretch — precedent exists for this being effective under LOSO

### 5.5 Stage 5 — Interpretability
- **GAT attention weights** → per-node importance, averaged over test subjects
- **Integrated Gradients / GNNExplainer** (PyG has both) → edge-level attributions
- **TopK/SAGPool selection scores** → which ROIs survive pooling
- **Validation:** map top-30 ROIs to Yeo-7 functional networks; test enrichment with a permutation test against random ROI sets. Then correlate per-subject saliency magnitude against ADOS-Total in the ASD group. A significant correlation there is a genuinely strong result.

---

## 6. Evaluation Protocol — the part reviewers actually scrutinise

### 6.1 Two protocols, always reported side by side

| Protocol | Setup | What it tells you |
|---|---|---|
| **P1: Stratified 10-fold CV** | Stratified by label **and site**; 5 seeds → 50 runs | Comparability with literature |
| **P2: Leave-one-site-out** | 17 folds (one per site), sites with n≥30 reported individually | Real generalisation |

Hyperparameters selected by **nested CV** (inner 5-fold on the training portion). Never tune on the test fold.

### 6.2 Metrics
Accuracy, **balanced accuracy** (primary — the cohort is mildly imbalanced), AUC-ROC, AUC-PR, sensitivity, specificity, F1. Always report **mean ± std across folds and seeds**. A single number with no variance is a red flag to any reviewer.

### 6.3 Statistical testing
- **DeLong test** for pairwise AUC comparison
- **Corrected resampled t-test** (Nadeau–Bengio) for CV accuracy differences — the plain t-test is invalid across overlapping folds
- **Permutation test** (500 label shuffles) to establish the empirical chance level per protocol
- Bonferroni/Holm correction across the model grid

### 6.4 Realistic targets — write these on your whiteboard

| Protocol | Honest target | "Something is wrong" territory |
|---|---|---|
| 10-fold, 871 subj, CC200 | **68–76%** acc, AUC 0.72–0.82 | >85% → you have leakage |
| Leave-one-site-out | **58–68%** acc | >80% → check your split code |
| ABIDE I → ABIDE II | **60–70%** acc | — |

For reference points in the literature: an adversarial node-edge GAT reported <cite index="1-1">74.7% accuracy on 1007 subjects across 17 sites</cite>; a shared-weight CNN benchmark reported <cite index="7-1">76.52% accuracy and AUC 0.81 on the preprocessed 871-subject ABIDE-I using nested ten-fold cross-validation</cite>; a 2026 multimodal cross-attention graph model reported <cite index="2-1">79.25% ± 4.71% on independent test data and 78.75% ± 1.56% on five-fold cross-validation</cite>. Treat anything above ~80% with scepticism unless the split protocol is fully specified.

**If you hit 72% under k-fold and 63% under LOSO with airtight methodology, you have a better paper than someone claiming 92% with a leaky pipeline.** Say so explicitly in your discussion section.

---

## 7. Codebase Architecture

```
asd-gnn/
├── configs/                     # Hydra/YAML — every experiment is a config, never a code edit
│   ├── data/{cc200,aal,ho}.yaml
│   ├── conn/{correlation,partial,tangent}.yaml
│   ├── model/{svm,ridge,rf,mlp,gcn,gat,gin,sage,cheb,braingnn,hybrid}.yaml
│   ├── protocol/{kfold,loso,abide2_holdout}.yaml
│   └── experiment/*.yaml         # compositions of the above
├── src/asdgnn/
│   ├── data/
│   │   ├── download.py           # fetch_abide_pcp wrapper, caching, checksums
│   │   ├── phenotype.py          # CSV parse, QC filters, covariate encoding
│   │   ├── connectivity.py       # ConnectivityMeasure wrappers; ALL fit-on-train-only
│   │   └── graphs.py             # brain-graph and population-graph builders → PyG Data
│   ├── models/
│   │   ├── classical.py          # sklearn pipelines
│   │   ├── gnn_graph.py          # graph-level: GCN/GAT/GIN/SAGE + pooling
│   │   ├── gnn_pop.py            # node-level: ChebNet/GCN on population graph
│   │   ├── hybrid.py             # encoder → population GNN + GRL site head
│   │   └── braincnn.py           # BrainNetCNN baseline
│   ├── harmonize/combat.py
│   ├── train/
│   │   ├── loop.py               # single train/val/test run
│   │   ├── cv.py                 # k-fold and LOSO orchestrators
│   │   └── metrics.py
│   ├── explain/                  # IG, GNNExplainer, attention extraction, ROI mapping
│   ├── stats/                    # DeLong, corrected t-test, permutation test
│   └── viz/                      # connectome plots, glass brains, ROC curves
├── scripts/
│   ├── 00_download.py  01_build_features.py  02_run_experiment.py
│   ├── 03_aggregate_results.py   04_make_figures.py
├── notebooks/                    # EDA and figure drafting ONLY — no pipeline logic
├── tests/
│   ├── test_no_leakage.py        # ← the most important file in this repo
│   ├── test_shapes.py  test_splits.py  test_determinism.py
├── results/                      # one JSON per run: config hash + seed + all metrics
├── paper/                        # LaTeX, figures, tables auto-generated from results/
├── environment.yml  README.md  Makefile
```

**Non-negotiable engineering rules**
1. Every run writes a JSON with the full resolved config, git commit hash, seed, and every metric. Tables in the paper are *generated* from `results/`, never typed by hand.
2. Seeds fixed and logged; `torch.use_deterministic_algorithms(True)` where possible.
3. Cache expensive intermediates (`.npy` of connectivity matrices) keyed by a hash of the preprocessing config.
4. Experiment tracking via Weights & Biases or MLflow from week 1, not week 12.
5. `tests/test_no_leakage.py` asserts that fitted transformers' `n_samples_seen_` never exceeds the training-fold size, and that test-set indices never appear in any `fit()` call. Run it in CI.

### 7.1 Environment
```yaml
python=3.11
pytorch, torch-geometric        # match CUDA build to Colab's
nilearn, nibabel                # neuroimaging I/O + ConnectivityMeasure
scikit-learn, scipy, numpy, pandas
neuroCombat                     # or neuroHarmonize
networkx, matplotlib, seaborn, plotly
hydra-core, wandb, pytest, black, ruff
```
Compute reality check: ROI time series ≈ few hundred MB; the heaviest single model trains in minutes on a T4. **The whole project fits on Colab free tier.** Time cost is CV × seeds × grid size, not model size. Budget for that: a full grid of 12 models × 3 atlases × 3 connectivity × 2 protocols × 5 seeds is ~1000 runs — parallelise the classical ones on CPU and queue the GNNs.

---

## 8. Execution Plan — phase by phase, with definition-of-done

### Phase 0 — Foundations (Week 1–2)
- Read the 8 core papers (§13), write a 1-page summary of each into a shared doc
- Set up repo, environment, W&B project, git workflow
- Download CC200 ROI time series for the 871 QC subset
- **DoD:** `python scripts/00_download.py` reproducibly produces 871 `.1D` files and a cleaned phenotype table; a notebook shows subject counts per site, per label, per sex, and an age histogram.

### Phase 1 — Classical baseline + evaluation harness (Week 3–4)
This comes *before* any GNN. Do not skip.
- Build connectivity matrices (Pearson first)
- Implement k-fold and LOSO splitters, metric computation, results JSON writer
- Run Ridge, SVM, RF, MLP under both protocols
- Write `test_no_leakage.py` and make it pass
- **DoD:** SVM reproduces literature-consistent numbers (~65–70% k-fold, ~55–60% LOSO). The gap between your two protocols is visible and explainable. Results table auto-generates.
- **This phase alone is a valid mid-semester presentation.**

### Phase 2 — First GNN, individual brain graphs (Week 5–6)
- Graph builder: connectivity → PyG `Data` objects; verify one graph by hand (node count, edge count, feature dims, no self-loops unless intended)
- GCN + global mean pool, then GAT
- Sanity checks before believing any number:
  - **Label-shuffle test** — accuracy must collapse to ~50%. If it doesn't, you have leakage.
  - **Single-batch overfit** — the model must reach 100% train accuracy on 32 samples. If it can't, the architecture or optimiser is broken.
  - **Random-feature test** — replace node features with noise; accuracy must drop.
- **DoD:** GCN and GAT run under both protocols; all three sanity checks documented in the report.

### Phase 3 — Population graph + full model zoo (Week 7–8)
- Population graph builder with phenotypic edge weighting; ChebNet/GCN node classification
- Add GIN, GraphSAGE, Graph Transformer, BrainNetCNN
- Sweep the graph-construction ablation (thresholding strategy × connectivity measure)
- **DoD:** the full model × atlas × connectivity grid has run at least once end to end; a results dashboard exists.

### Phase 4 — Harmonisation + hybrid model (Week 9–11)
- ComBat integrated as a fold-aware transformer
- Hybrid architecture, stage-wise first, then end-to-end
- Gradient-reversal site head; sweep the adversarial weight λ
- **DoD:** RQ4 answered with numbers. A plot of k-fold vs. LOSO accuracy for every configuration — this becomes Figure 3 of the paper.

### Phase 5 — Interpretability (Week 12–13)
- Extract attention/IG saliency; aggregate across subjects and folds
- Map to Yeo networks; permutation-test the enrichment
- Correlate saliency with ADOS-Total
- Glass-brain and circular-connectome figures
- **DoD:** a ranked ROI table with a literature-support column, and either a significant severity correlation or a documented null result.

### Phase 6 — External validation + writing (Week 14–15)
- ABIDE II held-out evaluation
- Freeze results; generate all tables/figures from `results/`
- Draft paper (§11), draft report
- **DoD:** paper draft with mentor for review; report skeleton complete.

### Phase 7 — Polish, submission, viva (Week 16+)
- Address mentor feedback, run any requested ablations
- Public GitHub repo with README, environment file, and a "reproduce Table 2" command
- Rehearse the viva against §12's question list

**If you fall behind:** cut Phase 6 external validation and the Graph Transformer first. Never cut Phase 1, `test_no_leakage.py`, or LOSO.

---

## 9. The daily working loop (train → test → modify → recheck)

```
1. Form a hypothesis          "GAT with tangent FC beats GCN with Pearson under LOSO"
2. Write it in the log         experiments.md, before running anything
3. Change ONE config knob      never two
4. Run with ≥3 seeds           a 2-point gain on one seed is noise
5. Check sanity tests still pass
6. Record result + verdict     supported / refuted / inconclusive
7. If a result surprises you   → suspect a bug before believing it
```

**Debugging heuristics specific to this project**
- Accuracy suspiciously high → check `fit()` call sites for test data; check that ComBat/tangent/feature-selection are inside the CV loop.
- Accuracy stuck at ~53% → your graph may be nearly complete (every node connected to every other), which makes message passing meaningless. Tighten the threshold.
- Train acc 100%, val acc 55% → classic small-n overfitting. Add dropout (0.3–0.5), weight decay, reduce hidden dim to 32–64, early-stop on validation AUC not loss.
- LOSO variance enormous → check per-site n. A site with 12 subjects will swing ±20 points. Report weighted and unweighted means.
- Results change between runs → unseeded shuffle in the DataLoader or non-deterministic scatter ops in PyG.

---

## 10. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Data leakage inflating results | **Critical** — invalidates everything | `test_no_leakage.py` in CI from Phase 1; all transforms inside sklearn `Pipeline`; peer code review of the CV loop |
| Results plateau near baseline | High | This is an *expected outcome* and still a paper (RQ1 is a negative-result question). Frame it as a rigour contribution from day one, not a fallback |
| S3 download failures / slow fetch | Medium | Fetch ROI derivatives only; cache to Drive; retry-with-resume wrapper; a known long-standing issue with the fetcher |
| Site confound dominates signal | Medium | This *is* RQ4 — you're studying it, not being blindsided by it |
| Scope creep (multimodal, dynamic FC, ABIDE II, SNNs) | High | Locked scope in §3; anything new goes to a "Future Work" list, not the sprint |
| Group coordination failure | Medium | Clear ownership split (§14); weekly merge; no long-lived branches |
| Mentor wants a different angle | Medium | Present §1–2 in week 1 and get explicit sign-off on the RQs before writing code |

---

## 11. Paper Outline (target: 8 pages, IEEE conference format)

1. **Abstract** — problem, gap (optimistic evaluation), method (multi-atlas GNN benchmark + hybrid site-invariant model), results (both protocols), claim.
2. **Introduction** — clinical burden, connectivity hypothesis, why graphs, the evaluation-rigour gap, contributions bulleted.
3. **Related Work** — classical ML on ABIDE; population-graph GNNs; brain-graph GNNs; harmonisation; interpretability. Organise as a comparison table with a **"protocol"** column exposing which papers report LOSO. That table is itself a contribution.
4. **Methods** — dataset & QC; connectivity estimation; three graph formulations; architectures; harmonisation; the hybrid model with a clean architecture figure.
5. **Experimental Setup** — protocols, splits, hyperparameter search, metrics, statistical tests, compute, seeds.
6. **Results** — Table 1: main comparison under both protocols. Table 2: atlas × connectivity ablation. Figure 3: k-fold vs. LOSO scatter across all configs. Figure 4: per-site LOSO accuracy vs. site n. Table 3: harmonisation ablation.
7. **Interpretability** — ROI saliency table, brain figure, enrichment test, ADOS correlation.
8. **Discussion** — what generalises and what doesn't; why the community should report LOSO; limitations (motion, sex imbalance, age range, no external clinical cohort, no causal claim).
9. **Conclusion & Future Work.**

**Venues** (mentor decides, but have a list ready): arXiv preprint first; then IEEE EMBC, IEEE ISBI, IEEE CBMS, ICPR, or MICCAI workshops (MLCN, MICCAI-GRAIL). Journals: *Brain Informatics*, *Frontiers in Neuroscience*, *Computers in Biology and Medicine*, *Biomedical Signal Processing and Control*.

---

## 12. Viva questions you must be able to answer cold

- Why a GNN rather than an SVM on the same features — and did it actually help?
- What exactly is a node, an edge, and a node feature in your formulation? (Have the answer for *both* formulations.)
- How did you threshold the connectivity matrix and why that value?
- What is tangent-space embedding and why did you fit it only on training folds?
- Why is your LOSO accuracy lower than your k-fold accuracy? Is that a failure?
- What is ComBat doing, mathematically?
- Which brain regions did the model find, and does that match the literature?
- What's your chance level, and how did you establish it empirically?
- Where could your pipeline still be leaking?
- What would you need to deploy this clinically? (Correct answer: far more than you have — prospective multi-site validation, regulatory approval, and a much higher sensitivity floor.)

---

## 13. Core reading list (read these eight before writing code)

1. Di Martino et al. (2014) — *The Autism Brain Imaging Data Exchange* — the dataset paper. **Cite it; ABIDE requires acknowledgement.**
2. Craddock et al. (2013) — Preprocessed Connectomes Project ABIDE derivatives.
3. Abraham et al. (2017), *NeuroImage* — *Deriving reproducible biomarkers from multi-site resting-state data* — the honest classical baseline and the source of the 871-subject convention.
4. Parisot et al. (2018), *Medical Image Analysis* — *Disease prediction using graph convolutional networks* — the population-graph formulation.
5. Ktena et al. (2018) — metric learning on brain connectivity graphs.
6. Li et al. (2021), *Medical Image Analysis* — **BrainGNN** — ROI-aware pooling and interpretability.
7. Kan et al. (2022) — **Brain Network Transformer** — the transformer baseline and a good critique of evaluation practice.
8. Fortin et al. (2017/2018) — ComBat harmonisation for neuroimaging.

Plus 10–15 recent (2024–2026) GNN-on-ABIDE papers for your related-work table — prioritise ones that report LOSO, and record the ones that don't.

---

## 14. Suggested group split

| Role | Owns |
|---|---|
| **Data & pipeline** | download, QC, phenotype handling, connectivity, caching, leakage tests |
| **Modelling** | GNN architectures, hybrid model, training loop, hyperparameter search |
| **Evaluation & stats** | CV orchestration, metrics, significance testing, results aggregation, figures |
| **Interpretability & writing** | saliency, ROI mapping, literature table, paper draft, report |

Everyone reviews everyone's CV/split code. Rotate a weekly "adversary" whose job is to try to find leakage in someone else's module.

---

## 15. Deliverables Checklist

- [ ] Synopsis / problem statement approved by mentor
- [ ] Literature survey (25–30 papers, with the protocol-comparison table)
- [ ] Mid-semester presentation (Phases 0–2 results)
- [ ] Reproducible public GitHub repo + README + environment file
- [ ] Results database (`results/*.json`) and auto-generated tables
- [ ] Research paper draft → mentor review → submission
- [ ] Final project report (institute format)
- [ ] Final presentation + demo (a small Streamlit/Gradio app taking a subject's ROI time series and returning prediction + ROI saliency plot makes the demo memorable)
- [ ] Viva

---

### One last framing note
When you present this to Prof. Kaur, lead with §0 and §1.4 — not the architecture diagram. The pitch is: *"the field reports optimistic numbers because it evaluates optimistically; we're going to build the benchmark that measures what actually generalises, and design a model for that objective."* That is a mentor-pleasing, paper-shaped, and honest framing, and it sits squarely in both of her named research areas.
