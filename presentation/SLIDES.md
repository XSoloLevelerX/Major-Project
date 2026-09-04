# Mid-Semester Progress Presentation — Slide-by-Slide Content

**Project:** ASD Classification from Brain Connectivity using Graph Neural Networks
**Course:** PEC, Semester 7 · Major_Group ID_26271
**Mentor:** Prof. Amandeep Kaur
**Occasion:** Mid-semester progress review

> **How to use this file.** Each `##` heading is one slide. *On the slide* is what the
> audience sees — keep it terse. *Speaker notes* is what you say. *Figure* names the
> visual asset and where it comes from. Numbers written as `{{TOKEN}}` are filled from
> `asd-gnn/results/` — see `RESULTS-SNAPSHOT.md` for the values and their provenance.
>
> Target length: 20–22 slides, 18–20 minutes, leaving 10 for questions.
> If you are cut to 10 minutes, present slides 1, 3, 4, 11, 15, 16, 18 and stop.

---

## Slide 1 — Title

**On the slide**

> ### ASD Classification from Brain Connectivity using Graph Neural Networks
> #### A leakage-audited, site-aware benchmark on ABIDE-I
>
> Major Project — Mid-Semester Progress Review
> Major_Group ID_26271 · PEC, Semester 7
>
> Mentor: Prof. Amandeep Kaur
> [Team member names]

**Speaker notes**

Keep this to fifteen seconds. Do not read the title aloud. Say instead:

*"We're classifying autism from resting-state fMRI using graph neural networks. But the
part we want to talk about today isn't the architecture — it's the evaluation. We think
that's where the interesting problem is."*

Then move on immediately. The hook is slide 3; don't spend your opening energy here.

---

## Slide 2 — The clinical problem

**On the slide**

- Autism Spectrum Disorder is diagnosed **behaviourally** — ADOS / ADI-R
- Requires trained clinicians; in India, median diagnostic delay is **several years**
- **No objective biomarker is in routine clinical use**
- Resting-state fMRI shows reproducible *group-level* differences:
  long-range under-connectivity, local over-connectivity — default mode network,
  superior temporal sulcus, insula, fusiform gyrus, cerebellum
- But group-level differences ≠ individual-level prediction

**Speaker notes**

The last bullet is the pivot and deserves a beat of silence. There is a genuine and
well-replicated group difference in the neuroimaging literature. That is not in dispute.
What does not follow — and what a decade of papers has struggled with — is turning that
into a prediction for one person in front of you.

Be careful with framing here: we are not building a diagnostic tool. We say so explicitly
on the limitations slide. This is decision-support research.

---

## Slide 3 — The gap we are actually attacking

**On the slide**

> There are hundreds of published ABIDE + GNN papers.
> Reproducing *"we ran a GCN on ABIDE and got 70%"* is a guaranteed pass — and a
> guaranteed rejection.

| Evaluation protocol | Published accuracy |
|---|---|
| Random 10-fold cross-validation | **65 – 90%** |
| Leave-one-site-out (train on 16 sites, test on the 17th) | **51 – 56%** |

- The second row is what a clinic actually faces: a new scanner, a new protocol, a new cohort
- The literature reports the first row almost exclusively
- Documented cause: *"the positive cross-validation R² one obtains under random k-fold is an
  artifact of site information leaking between training and test folds"*

**Speaker notes**

**This is the slide the whole presentation hangs on.** Slow down.

Under random k-fold, every test subject has site-mates sitting in the training set. The
model can learn "this is a Yale scanner" and get most of the way to the answer without
learning anything about autism. The reported 51–56% LOSO figures come from a competent
classical ML framework — LDA, SVM, Random Forest, MLP, KNN — evaluated honestly. That is
near chance.

If a reviewer or the mentor asks "so what's your novelty?" — this slide is the answer.
Our contribution is not a bigger number. It is an honest measurement of the gap, and a
model designed against it.

---

## Slide 4 — Problem statement

**On the slide**

Given subjects $S = \{s_1 \dots s_N\}$, each with:
- a preprocessed rs-fMRI BOLD time series $T_i \in \mathbb{R}^{R \times t_i}$ over $R$ atlas ROIs
- phenotypic covariates $p_i$ (age, sex, site, eye status, FIQ)
- a binary label $y_i \in \{\text{ASD}, \text{TC}\}$

**(a)** construct a graph $G_i = (V, E, X)$ per subject, and/or a population graph $G_\text{pop}$
**(b)** learn $f_\theta: G \rightarrow \hat{y}$ maximising balanced accuracy and AUC
**(c)** **subject to** performance being measured under leave-one-site-out generalisation —
the test site is never seen during training, harmonisation fitting, feature selection or
hyperparameter search
**(d)** and such that $f_\theta$ yields an ROI saliency ranking testable against independent
ASD neuroimaging findings

**Speaker notes**

Read (c) out loud, in full. It is the constraint that makes this project different from the
crowded field, and every design decision downstream follows from it.

Note the four verbs in (c) — training, harmonisation, feature selection, hyperparameter
search. Leakage hides in the last three far more often than the first.

---

## Slide 5 — Research questions

**On the slide**

| # | Question |
|---|---|
| **RQ1** | How much of the reported GNN advantage over classical ML survives leave-one-site-out evaluation? |
| **RQ2** | Does the graph formulation matter — individual brain graphs vs. population graphs vs. a hybrid? |
| **RQ3** | Does multi-atlas fusion beat the best single atlas — and is the gain real or an ensembling artifact? |
| **RQ4** | Does site harmonisation (ComBat / adversarial) recover cross-site performance, and at what cost within-site? |
| **RQ5** | Do the ROIs the model attends to converge on the ASD literature? |

**Hypotheses — stated up front, because being wrong is publishable**

- **H1** GNNs beat SVM/Ridge under k-fold by 3–8 points; the margin shrinks to <3 under LOSO
- **H2** The hybrid encoder beats either formulation alone
- **H3** ComBat improves LOSO but slightly hurts k-fold
- **H4** Attention concentrates on DMN, temporal and cerebellar ROIs above chance

**Speaker notes**

Emphasise that the hypotheses were registered before the experiments were run, and that
H1 predicts our own method's advantage *shrinking*. That is a deliberate choice: RQ1 is a
negative-result question, and it is still a paper. If we had designed the study so that
only a positive result was publishable, we would be under pressure to find one.

Today we have preliminary evidence bearing on H1 and H3. RQ3 and RQ5 are next semester.

---

## Slide 6 — Dataset: ABIDE-I

**On the slide**

- **ABIDE** — Autism Brain Imaging Data Exchange: rs-fMRI, structural MRI and phenotype for
  539 ASD / 573 typical controls across **17 international sites**
- We use the **Preprocessed Connectomes Project** derivatives: CPAC pipeline,
  band-pass filtered 0.01–0.1 Hz, no global signal regression (`filt_noglobal`)
- Fetched programmatically from public S3 via `nilearn` — no registration, fully reproducible
- We take **ROI time series**, not 4-D volumes: a few hundred MB instead of ~50 MB *per subject*
- `quality_checked=True` → the canonical **871-subject** benchmark subset used by Abraham
  et al. and nearly every comparable paper

**Our cohort after QC:** 871 subjects · 403 ASD / 468 TC ·
20 sites · 83.5% male · age 6.5–58.0 years

**Figure:** `C3` from `PLOT-PROMPT.md` — per-site counts, age distribution, motion by group

**Speaker notes**

Two things to stress.

First, using the PCP derivatives rather than preprocessing raw NIfTI ourselves is a
deliberate scope decision, not laziness — it is what makes our numbers directly comparable
to the published literature, and it saves two months.

Second, the 871-subject subset is a convention, not a coincidence. Using it is how we make
our numbers mean the same thing as everybody else's.

If asked about ABIDE-II: it is our planned external validation set, held out entirely and
untouched during development. Phase 6.

---

## Slide 7 — Known confounds, stated before results

**On the slide**

| Confound | Reality in ABIDE | What we do |
|---|---|---|
| **Sex imbalance** | ~85% male | Report per-sex metrics; never train a covert sex classifier |
| **Age range** | ~6 – 64 years | Age as a covariate; age-restricted sensitivity analysis planned |
| **Head motion** | `func_mean_fd` differs systematically between groups | Report accuracy stratified by motion; motion in the ComBat covariate model |
| **Site size imbalance** | Some sites have < 20 subjects | LOSO on those is statistically meaningless — we report per-site *n* alongside per-site accuracy |
| **Eye status** | Eyes-open vs. eyes-closed differs by site | Recorded as a covariate |

**Speaker notes**

Put this slide *before* the results, not in a limitations section at the end. Reviewers and
examiners will ask about motion; answering before being asked is worth a lot.

The row that will get a question is head motion. The honest position: motion differs between
groups in ABIDE, it is partially confounded with diagnosis, and no one has fully solved it.
We report stratified accuracy rather than claiming we have removed it.

---

## Slide 8 — Method: the pipeline

**On the slide**

```
ROI time series (T × R)   [T varies by site: 77 – 295]
   → connectivity estimation  (Pearson | partial | tangent)
   → Fisher-z → vectorised upper triangle   (19,900 features for CC200)
   → graph construction
   → model
   → evaluation under BOTH protocols
   → results JSON on disk → auto-generated tables and figures
```

> **Everything stateful is fitted inside the fold, on training subjects only:**
> the tangent-space reference, the scaler, the feature selector, ComBat, the PCA.

**Figure:** `C1` from `PLOT-PROMPT.md` — pipeline diagram with the fold boundary marked in red

**Speaker notes**

The callout box is the engineering thesis of the project. Say it plainly:

*"If a `.fit()` call can see a test index, the project is invalid. That is rule one in our
repository README, and we have a test suite that enforces it."*

The specific trap worth naming: tangent-space embedding estimates a group geometric mean.
Fit that on all subjects before cross-validation and you inflate accuracy by roughly four to
six points of pure fantasy. It is the single most common bug in student ABIDE code, and it
looks exactly like success.

---

## Slide 9 — Three graph formulations

**On the slide**

**(A) Individual brain graph** — graph-level classification
- Node = ROI — **179 for CC200 after QC** (21 constant parcels dropped from 200) ·
  Node features = that ROI's connectivity profile
- Edges = thresholded connectivity (top-k%, kNN, absolute cutoff — all ablated)
- One label per graph → global mean pool

**(B) Population graph** — node-level, transductive (the Parisot formulation)
- Node = **subject**; one graph for the entire cohort
- Node features = reduced connectivity vector
- Edges = phenotypic similarity: matching sex, similar age, optionally same site

**(C) Hybrid** — our proposed contribution
```
brain graph → GNN encoder → 64-d subject embedding
                                    ↓
      population node features = [embedding ‖ phenotype] → population GNN → label
```
Plus a **gradient-reversal site-discriminator head** to force site-invariance

**Speaker notes**

Have the answer ready for both formulations separately — "what is a node, what is an edge,
what is a node feature" is a guaranteed viva question and the answer is completely different
in (A) and (B).

Flag the honest methodological wrinkle in (B): if population-graph edges use site identity
and you then evaluate with LOSO, the held-out site's nodes are **disconnected** — they have
no neighbours. That is not a bug we hit by accident; it is a genuine methodological point and
it earns a paragraph in the paper. It is also the strongest argument for the adversarial head
in (C), which has no such problem.

---

## Slide 10 — Model zoo

**On the slide**

| Family | Models |
|---|---|
| **Classical baselines** (a paper without these is rejected) | Linear SVM, Ridge / logistic, Random Forest, MLP |
| **Deep non-graph** | BrainNetCNN, edge-free MLP control |
| **Graph neural networks** | GCN, ChebNet, GraphSAGE, GAT, GIN, Graph Transformer |
| **Ours** | Hybrid brain-graph → population-graph encoder, with optional site-adversarial head |

**The control that matters:** an **edge-free MLP** on the same node features.
If it matches the GNN, the topology is contributing nothing — and the paper must say so.

**Speaker notes**

The edge-free control is worth dwelling on. A large fraction of published graph-learning
results on brain data do not include it, which means they cannot distinguish "graph neural
networks help" from "deep learning on connectivity features helps." We run it as a
first-class model, not as an afterthought.

Same logic for the random-node-feature ablation on the sanity slide.

---

## Slide 11 — Evaluation protocol

**On the slide**

| | **P1: Stratified 10-fold CV** | **P2: Leave-one-site-out** |
|---|---|---|
| Setup | Stratified by label **and site** | One fold per site |
| Folds | 10 splits × 1 seed | **20 folds** (one per site) |
| Tells you | Comparability with the literature | **Real generalisation** |
| Our best | SVM 0.674 ± 0.060 | MLP 0.643 ± 0.085 |

- Hyperparameters chosen by **nested** CV — inner 5-fold on the training portion only
- Primary metric: **balanced accuracy** (the cohort is imbalanced); plus AUC-ROC, AUC-PR,
  sensitivity, specificity, F1
- Always **mean ± std across folds and seeds**. A single number with no variance is a red flag
- Significance: **DeLong** for AUC pairs · **Nadeau–Bengio corrected resampled t-test** for
  CV accuracy (the plain t-test is invalid across overlapping folds) · **permutation test**,
  500 shuffles, for the empirical chance level · Holm correction across the model grid

**Figure:** `C2` from `PLOT-PROMPT.md` — the two-protocol contrast explainer

**Speaker notes**

The corrected t-test point is worth thirty seconds because it is a genuine methodological
error that appears constantly in student work and quite often in published work. Cross-
validation folds share training data, so the accuracy differences across folds are not
independent, and the standard t-test's assumptions do not hold. Nadeau and Bengio's
correction inflates the variance estimate to account for the overlap. Our implementation
is unit-tested to be more conservative than the plain test.

---

## Slide 12 — Rigour: the leakage audit

**On the slide**

> **Two rules override everything else in our repository:**
> 1. Nothing fits on test data. Not the scaler, not the tangent reference, not the feature
>    selector, not ComBat, not the PCA.
> 2. Every result is a JSON file on disk. No numbers live in notebooks or in anyone's head.

**85 automated tests. The ones that matter most are in one file.**

- **Fold isolation** — a spy on every `.fit()` in the pipeline asserts it sees at most
  `len(train)` samples; the scaler's statistics must differ across folds
- **Split integrity** — train/test disjointness, LOSO site purity, both classes present in
  every training fold
- **Harmonisation safety** — ComBat parameters from training sites only; test diagnosis
  labels never enter the covariate model
- **The behavioural check** — shuffle the labels, run the *full* pipeline, accuracy must
  collapse to chance

**Figure:** `C4` from `PLOT-PROMPT.md` — the test-suite scoreboard

**Speaker notes**

This is the slide that separates the project from typical student work, and it is worth
being slightly proud of. Lead with the label-shuffle test: it is the single most valuable
test in the repository, because it is behavioural rather than structural. You can write a
pipeline that passes every structural check and still leaks through some path nobody thought
to assert on. If you shuffle the labels and the model still performs above chance,
information is reaching it that should not be — and you do not need to know *how* to know
that something is wrong.

If asked "where could you still be leaking?" — the honest answers are: hyperparameters
chosen by looking at test performance across many experiments over the semester, and the
QC threshold decisions we made after seeing the data. Both are human-in-the-loop leakage
that no test suite catches. Say that; it lands well.

---

## Slide 13 — Engineering: what we actually built

**On the slide**

```
asd-gnn/
  src/asdgnn/
    data/       download · phenotype · timeseries · connectivity* · graphs_brain · graphs_pop
    harmonize/  combat*
    models/     classical · gnn_graph · gnn_pop · braincnn · hybrid · registry
    train/      splits* · loop · cv · metrics
    explain/    saliency · roi_mapping
    stats/      delong · corrected_ttest · permutation
    viz/        curves · connectome · brains · tables
  scripts/      00_download → 01_build_features → 02_run_experiment
                → 03_aggregate → 04_figures → 05_explain → 06_hybrid
  tests/        test_no_leakage.py is the most important file here
```
`*` = leakage-critical

- **Config-driven:** every experiment is a config, never a code edit
- **Seeded and hashed:** each run writes its resolved config, git commit, per-fold metrics
  **and per-fold predicted probabilities** — which is what makes DeLong tests possible later
- **One shared split file:** every model reads the same splits, which is what makes paired
  statistical comparison valid
- `make smoke` reproduces the entire pipeline end to end on a synthetic cohort in ~20 seconds

**Speaker notes**

The detail worth calling out is saving per-fold predicted probabilities, not just metrics.
Re-running a thousand folds because you only saved accuracy is a genuinely miserable
afternoon, and it is the difference between being able to run a DeLong test at writing time
and not.

The `make smoke` target is our defence against a broken pipeline going unnoticed: it exercises
feature building, sanity checks, both protocols, explainability, the hybrid model, table
generation and figure generation, on synthetic data, in under a minute.

---

## Slide 14 — Sanity checks: what we verify before believing any number

**On the slide**

| Check | Expected | Observed |
|---|---|---|
| **Label shuffle** — shuffle `y`, run the full pipeline | ~0.50 | **0.525** ✅ |
| **Real labels** — same pipeline, true labels | > shuffled | 0.633 (**+0.108**) |
| **Connectivity fit size** — estimator sees train only | < N | 696 / 871 ✅ |
| **Mean node degree** — CC200 brain graphs | 10 – 40 | **17.80** ✅ |
| **Sex control task** — can the features predict sex? | well above chance | 0.579 |
| **Site control task** — can the features predict site? | **high — this is the confound** | **0.673** |
| **Random node features** ablation | large drop | 0.500 (GNN 0.546) |
| **Edge-free graph** ablation | quantifies topology's contribution | 0.511 (GNN 0.546) |

**Speaker notes**

The two rows to talk about are the control tasks.

The **sex control** is a positive control: our features should carry *some* real biological
signal, and sex is easy to decode from connectivity. If that comes out at chance, the
pipeline is broken and no autism result means anything.

The **site control** is the confound, quantified. It tells you how easily a model can
identify the scanner from the features alone. That number is the reason the k-fold to LOSO
gap exists, and it belongs in our discussion section.

Both are cheap to run and both produce paper content.

---

## Slide 15 — Results: classical baselines under both protocols

**On the slide**

**Balanced accuracy, mean ± std across folds — CC200, Pearson correlation**

| Model | 10-fold CV | Leave-one-site-out | Δ |
|---|---|---|---|
| **SVM (linear)** | **0.674 ± 0.060** | 0.631 ± 0.077 | 0.043 |
| MLP | 0.666 ± 0.037 | **0.643 ± 0.085** | 0.023 |
| BrainNetCNN | 0.658 ± 0.077 | 0.639 ± 0.087 | 0.019 |
| Ridge | 0.642 ± 0.067 | 0.589 ± 0.049 | 0.053 |
| Random Forest | 0.622 ± 0.026 | 0.625 ± 0.087 | −0.003 |

Literature reference points: SVM ~65–70% k-fold, ~55–60% LOSO
Our pre-registered "something is wrong" line: **> 85% k-fold** or **> 80% LOSO**

**Figure:** `results/figures/roc_kfold.png` and `results/figures/roc_loso.png`, side by side

**Speaker notes**

Every number is inside the honest band we wrote down in advance, and nothing is anywhere
near the alarm line. Our best k-fold is 0.674 — slightly *below* the 68–76% target — and our
best LOSO is 0.643, comfortably inside the 58–68% target. Being slightly under on k-fold is
the expected signature of a pipeline that is not leaking.

The framing to use, whatever the numbers: we are not trying to win here. We are establishing
that our harness reproduces literature-consistent classical baselines, because that is the
precondition for believing anything we say about GNNs. A baseline that is too *good* is a
worse outcome than one that is too weak — it means we are leaking.

Two rows deserve a comment. **Random Forest scores marginally higher under LOSO than under
k-fold** (−0.003 drop). Do not over-read that: the LOSO standard deviation is 0.087, so the
difference is well inside the noise. It does illustrate that with 20 folds of wildly unequal
size, LOSO estimates are noisy — which is the argument for reporting per-site n.

And the **MLP is the strongest model we have under LOSO** (0.643). A two-hidden-layer MLP on
vectorised connectivity beating every graph neural network is not the result anyone hopes
for, but it is the result, and slide 17 deals with it directly.

The framing to use, whatever the numbers: we are not trying to win here. We are establishing
that our harness reproduces literature-consistent classical baselines, because that is the
precondition for believing anything we say about GNNs. A baseline that is too *good* is a
worse outcome than one that is too weak — it means we are leaking.

---

## Slide 16 — Results: the protocol gap

**On the slide**

> ### Every point below the diagonal is the story.

**Figure:** `results/figures/kfold_vs_loso.png` — full-bleed, this slide is the figure

**Our honest k-fold → LOSO drop is 0.00 – 0.05, not the 15–20 points the literature implies.**
The one configuration that collapses is ComBat-`passthrough` (0.646 → 0.500) — and that is a
declared harmonisation policy failing, not a model failing. See slide 17b.

**Speaker notes**

This is the headline. Let the figure carry it — do not clutter this slide with a table.

**Be ready for the obvious challenge: "your gap is small, so where is the problem you claimed
on slide 3?"** The answer is the strongest thing we can say all day:

Our k-fold numbers are *already honest*. We stratify k-fold by site and we audit every
transform for leakage, so our left axis is not the inflated 85–90% the literature reports —
it is 0.55–0.67. The published collapse from ~85% to ~53% is largely a collapse *from an
inflated starting point*. When you start from a number that was never inflated, there is much
less to lose. **The gap in the literature is mostly leakage, and our small gap is the
evidence for that claim, not a contradiction of it.**

That reframing is worth rehearsing until it is fluent, because it converts what looks like a
weak result into the project's central finding.

Walk the audience through the axes once, slowly: x is what each model scores under random
10-fold, y is what the *same model, same code, same data* scores when the test site is
genuinely unseen. The dashed diagonal is where a model that generalised perfectly across
sites would sit. Then stop talking and let them look at where the points actually are.

The question you will get: *"so is your project failing?"* The answer, delivered without
defensiveness: no — this is the measurement the project exists to make. RQ1 asks how much
of the reported advantage survives honest evaluation. A gap is the finding, not the failure.
The failure mode would have been reporting only the left axis.

---

## Slide 17 — Results: GNNs and the ablation controls

**On the slide**

| Model | 10-fold CV | LOSO |
|---|---|---|
| GCN | 0.600 ± 0.053 | 0.560 ± 0.068 |
| GAT | 0.552 ± 0.050 | 0.581 ± 0.075 |
| GIN | 0.581 ± 0.050 | 0.572 ± 0.067 |
| GraphSAGE | 0.596 ± 0.077 | 0.590 ± 0.065 |
| ChebNet | 0.597 ± 0.071 | 0.593 ± 0.070 |
| **BrainNetCNN** *(full matrix, not a graph)* | **0.658 ± 0.077** | **0.639 ± 0.087** |
| Population ChebNet | 0.523 ± 0.054 | 0.520 ± 0.071 |
| Population GCN | 0.514 ± 0.048 | 0.508 ± 0.030 |
| **MLP, edge-free control** | **0.513 ± 0.029** | **0.510 ± 0.037** |
| *(reference)* SVM | *0.674 ± 0.060* | *0.631 ± 0.077* |

**The dividing line is not deep vs. classical. It is full matrix vs. thresholded graph.**

| Sees the full 15,931-edge matrix | Sees a top-10% graph |
|---|---|
| MLP · **BrainNetCNN** · SVM — AUC 0.71–0.74 | every GNN — AUC 0.57–0.65 |

1. **BrainNetCNN is deep, and it works** — statistically indistinguishable from the SVM
   (ΔAUC +0.002, p_adj = 1.00) and from the MLP, while beating **every GNN** significantly
   under both protocols. So "deep learning doesn't work here" is ruled out.
2. **No GNN beats any full-matrix model.** **H1 is refuted.**
3. **But topology is doing real work:** the edge-free control sits at AUC 0.510 on the *same
   node features*; GNNs reach 0.57–0.65. Message passing is worth ≈ 0.12 AUC over no graph.

**Speaker notes**

This is the slide where you must not get defensive. Lead with the refutation: **we predicted
GNNs would beat the classical baselines, and on our own data they do not.** We are reporting
that because we wrote the hypothesis down first.

Then give the diagnosis, because we have one, and **BrainNetCNN is what turns it from a
guess into an argument.**

The obvious objection to "our GNNs underperform" is *"maybe deep learning just doesn't work
at n=871."* BrainNetCNN rules that out. It is a deep model — edge-to-edge, edge-to-node and
node-to-graph convolutions — and it lands statistically level with the SVM and the MLP while
beating **every** GNN significantly under both protocols. Deep learning works fine here.

What separates the two groups is not model class, it is **input**. The SVM, the MLP and
BrainNetCNN all see the complete 15,931-edge connectivity matrix. Every GNN sees a graph
thresholded to the top 10% of connections — mean degree 17.8 out of a possible 178, so
roughly 90% of the matrix is discarded before the model starts. Topology is worth about
0.12 AUC over no graph at all, but that does not recover what thresholding threw away.

That makes the **edge-density ablation the single highest-value next experiment**, and it is
already configured (`configs/experiment/e03_edge_density.yaml`, k = 5 / 10 / 20%). State the
prediction out loud, because a falsifiable prediction is what makes it science rather than an
excuse: **GNN performance should rise monotonically with edge density and approach
BrainNetCNN as the graph approaches fully connected. If it does not, our hypothesis is wrong
and the problem is the architecture.** Either outcome is publishable.

Two further honest notes. **The population-graph models are at chance** (0.508–0.523), which
is a real RQ2 finding and needs investigating before we make claims about the formulation.
And be honest about sample size: 871 subjects is small for deep learning. Train accuracy at
100% with validation at 55% is the expected regime here, not a bug. We control it with
smaller hidden dimensions, higher dropout and weight decay.

If asked why GAT — the model our interpretability analysis uses — is the *weakest* GNN: it
is, and that is a problem we have already flagged. A stable explanation of a weak model is of
limited value, so the saliency analysis should be re-run on **ChebNet**, which is the only
GNN not significantly worse than the SVM under either protocol (p_adj = 0.074 and 1.00).

---

## Slide 17b — Harmonisation: our most striking result (RQ4)

**On the slide**

SVM · CC200 · Pearson correlation, with and without ComBat:

| Configuration | 10-fold CV | LOSO |
|---|---|---|
| No harmonisation | **0.674 ± 0.060** | **0.631 ± 0.077** |
| ComBat — `passthrough` | 0.646 ± 0.040 | **0.500 ± 0.000** ⚠️ |
| ComBat — `transductive` | — | 0.588 ± 0.046 |
| ComBat — `reference` | — | 0.590 ± 0.045 |

> **`passthrough` produces balanced accuracy of exactly 0.500, with zero variance, across all
> 20 sites. Sensitivity 0.00, specificity 1.00. The model predicts one class for everybody.**

**H3 said ComBat would improve LOSO and slightly hurt k-fold.**
It hurts k-fold as predicted — and it hurts LOSO under *every* policy. **H3 refuted.**

**Speaker notes**

This is the most memorable thing in the result set. Give it time.

Under LOSO, the held-out site was **never seen when ComBat was fitted**, so no site
parameters exist for it. Our three policies are the three defensible responses, and each is
explicitly declared in the config — a test in the leakage suite asserts that the choice can
never be made accidentally.

`passthrough` is the honest one: leave the unseen site un-harmonised. But that means every
*training* site was shifted and the *test* site was not, so the model is evaluated on a
distribution it never saw. It degenerates to predicting the majority class. Exactly 0.500,
zero variance — the signature of a model that has stopped discriminating at all.

**This is not a bug, and do not apologise for it.** It is the correct behaviour of a
correctly-implemented policy, and it makes a methodological point that most papers using
ComBat under cross-site evaluation never confront: *what exactly do you do about the site you
have no parameters for?* Many papers quietly fit ComBat on all sites including the test
one — which is leakage, and which would have shown up in our audit.

Land the punchline: **this is the strongest argument for the adversarial site-invariance
head.** Gradient reversal learns a site-invariant representation during training and needs no
per-site parameters at inference — so the unseen-site problem simply does not arise. That is
the λ sweep in Phase 4, and this slide is why it matters.

---

## Slide 18 — Progress against plan

**On the slide**

| Phase | Weeks | Status |
|---|---|---|
| **0 — Foundations:** repo, environment, download, cohort QC | 1–2 | ✅ Done |
| **1 — Classical baselines + evaluation harness + leakage suite** | 3–4 | ✅ Done |
| **2 — First GNNs on brain graphs, three sanity checks** | 5–6 | ✅ Done |
| **3 — Population graph + full model zoo + ablations** | 7–8 | 🔄 In progress |
| **4 — ComBat + hybrid + adversarial site head** | 9–11 | ⚙️ Implemented, not yet swept |
| **5 — Interpretability + stats + figures** | 12–13 | ⚙️ Implemented, preliminary results |
| **6 — ABIDE-II external validation + writing** | 14–15 | ⬜ Not started |
| **7 — Polish, submission, viva** | 16+ | ⬜ Not started |

**Figure:** `C5` from `PLOT-PROMPT.md` — the timeline

**Speaker notes**

The project scope document says explicitly that Phase 1 alone is a valid mid-semester
presentation. We are past that: the full model zoo and the hybrid and interpretability
machinery are implemented and tested end to end, and what remains for phases 3 through 5 is
running the experiment grid and interpreting it, not building.

If we fall behind, the pre-agreed cut order is: ABIDE-II external validation first, then the
Graph Transformer. We never cut Phase 1, the leakage suite, or LOSO.

---

## Slide 19 — Interpretability: preliminary

**On the slide**

- **GAT attention weights** → per-ROI importance, averaged across test subjects and folds
- **Integrated Gradients** → an independent attribution method, as a cross-check
- **Stability first, ranking second:** we report the mean pairwise Spearman correlation of the
  ROI ranking *across folds* before we report which ROIs won
  → observed: **ρ = 0.961**; 18 ROIs in the top-k in ≥80% of folds
- **Planned validation:** map top-30 ROIs to Yeo-7 networks, permutation-test the enrichment
  against random ROI sets, then correlate per-subject saliency magnitude with ADOS-Total
  within the ASD group

**Known limitation:** CC200 parcels ship with no anatomical names or centroid coordinates in
`nilearn` — they are numbered. Named-region and Yeo-network mapping needs AAL or
Dosenbach-160, which is why the atlas ablation (RQ3) comes before the interpretability claim.

**Figure:** `results/figures/roi_stability.png` and `results/figures/top_rois.png`

**Speaker notes**

Be disciplined here — this is where projects like ours overclaim.

An ROI ranking that changes completely from fold to fold is noise, no matter how neurologically
plausible the top of the list looks. So we report the stability statistic *first*. Only if the
ranking is stable is it worth asking whether it matches the literature.

Everything on this slide is preliminary and you should say the word "preliminary" out loud.
The enrichment test and the ADOS correlation are Phase 5 work.

---

## Slide 20 — Risks and limitations

**On the slide**

| Risk | Severity | Mitigation |
|---|---|---|
| Data leakage inflating results | **Critical** | Leakage suite in CI from Phase 1; all transforms inside sklearn `Pipeline`; peer review of the CV loop |
| Results plateau near baseline | High | *Expected outcome and still a paper* — RQ1 is a negative-result question, framed that way from day one |
| Site confound dominates signal | Medium | This **is** RQ4 — we are studying it, not blindsided by it |
| Scope creep | High | Locked scope; new ideas go to Future Work, not the sprint |
| Slow / flaky S3 fetch | Medium | ROI derivatives only; retry-with-resume; cached time series |

**Limitations we will state in the paper**

Head motion is only partially controlled · ~85% male cohort · age range 6–64 · no external
clinical cohort · **no causal claim, and no diagnostic claim** — this is a research prototype

**Speaker notes**

The second row is the one to deliver with conviction, because it is the question behind the
question when someone asks "what if it doesn't work?"

If our GNNs land near the classical baselines under LOSO, that is an answer to RQ1, not a
failed project. The field currently has an incentive to report the optimistic protocol; a
careful negative result is a contribution. We committed to that framing before we saw the
numbers, which is the only time that commitment is worth anything.

---

## Slide 21 — Next steps

**On the slide**

**Immediately (next 2 weeks)**
1. Complete the full experiment grid: 10 models × 3 connectivity measures × both protocols
2. Atlas ablation — CC200 vs. AAL vs. Harvard–Oxford (RQ3)
3. ComBat sweep, including the three declared unseen-site policies (RQ4)

**Then**
4. Hybrid model: stage-wise, then end-to-end, then the adversarial λ sweep
5. Interpretability: Yeo-network enrichment + ADOS-Total correlation (RQ5)
6. ABIDE-II as a fully held-out external test set
7. Paper draft — target: IEEE EMBC / ISBI / MICCAI workshop; arXiv preprint first

**Ask of the mentor**
- Sign-off on the research questions and the two-protocol framing
- Guidance on venue and submission timeline

**Speaker notes**

End with the ask. Presentations that end with "thank you, questions?" waste the one moment
you have the mentor's full attention on a decision.

The specific sign-off you want is on the framing — that reporting both protocols and treating
the gap as the finding is the right spine for the paper. Getting that agreed now is much
cheaper than getting it agreed after the paper is drafted.

---

## Slide 22 — Thank you / references

**On the slide**

**Core references**
1. Di Martino et al. (2014) — *The Autism Brain Imaging Data Exchange* — **dataset paper, ABIDE requires acknowledgement**
2. Craddock et al. (2013) — Preprocessed Connectomes Project ABIDE derivatives
3. Abraham et al. (2017), *NeuroImage* — the honest classical baseline; source of the 871-subject convention
4. Parisot et al. (2018), *Medical Image Analysis* — population-graph formulation
5. Ktena et al. (2018) — metric learning on brain connectivity graphs
6. Li et al. (2021), *Medical Image Analysis* — BrainGNN, ROI-aware pooling
7. Kan et al. (2022) — Brain Network Transformer
8. Fortin et al. (2017/2018) — ComBat harmonisation for neuroimaging

*Data provided by the Autism Brain Imaging Data Exchange (ABIDE) and the Preprocessed
Connectomes Project.*

**Speaker notes**

The ABIDE acknowledgement is a condition of use, not a courtesy. Keep it on the slide.

---

# Appendix — backup slides for questions

## B1 — "Why a GNN rather than an SVM on the same features?"

Flattening an $R \times R$ connectivity matrix into a vector for an SVM discards the topology
entirely — the SVM has no notion that feature 4,182 and feature 9,003 share an ROI. A GNN
preserves that structure through message passing.

**But:** whether it *helps* on this dataset is RQ1, and we report the edge-free MLP control
precisely so we can answer that question honestly rather than assume it.

## B2 — "How did you threshold the connectivity matrix, and why that value?"

Top-k% of absolute edge weights per subject, default k=10%, giving mean node degree
**17.80** ✅ on CC200. Thresholding uses the *absolute* weight, so strong negative
correlations survive; the signed weight is then preserved as the edge attribute.

We ablate k ∈ {5, 10, 20}, kNN per node with k ∈ {5, 10}, and an absolute cutoff. The failure
mode to watch is a graph so dense that message passing degenerates into a global average — the
symptom is accuracy stuck near 53%, and the first thing to print is the mean degree.

## B3 — "What is tangent-space embedding, and why fit it only on training folds?"

Covariance matrices live on a curved manifold, not in Euclidean space, so averaging or
classifying them directly is geometrically wrong. Tangent-space embedding projects each
subject's covariance onto the tangent plane at the group geometric mean, giving features that
behave Euclidean-ly and generally classify better.

The catch: that group mean is *estimated from data*. Estimate it on all subjects before
cross-validation and every test subject has contributed to its own feature representation.
That inflates accuracy by roughly 4–6 points. Our `ConnectivityTransformer` is an sklearn
transformer specifically so it lives inside a `Pipeline` and can only be fitted on training
folds — the safety is structural rather than remembered, and it is asserted by two tests.

## B4 — "What is ComBat doing, mathematically?"

ComBat models each feature as a site-specific location and scale shift on top of a covariate
model: fit the biological covariates (age, sex, diagnosis) by regression, then estimate per-site
additive and multiplicative parameters on the residuals, shrink them toward a common prior by
empirical Bayes, and remove them. Preserving the biological covariates in the model is what
stops ComBat from removing the diagnosis signal along with the batch effect.

**The LOSO problem:** the held-out site was never seen during fit, so no parameters exist for
it. We implement three declared policies — `passthrough` (leave it un-harmonised, the honest
default), `transductive` (estimate from the test site's own unlabelled data, which must be
declared as transductive harmonisation in the paper), and `reference` (map onto the grand
mean). A test asserts that each policy is explicit rather than accidental. This limitation is
the strongest argument for the adversarial site head, which has no equivalent problem.

## B5 — "What's your chance level, and how did you establish it empirically?"

Not assumed at 0.50 — measured. A permutation test with 500 label shuffles gives the empirical
null distribution per protocol, and our implementation never returns p = 0 (it reports the
correct additive-smoothed bound). The label-shuffle sanity check is the fast version of the
same idea, run before every experiment session.

This matters more under LOSO than k-fold: with 20 folds ranging from n=11 to n=172, the effective
chance level and its variance are not obvious a priori.

## B6 — "Where could your pipeline still be leaking?"

Three honest answers:
1. **Human-in-the-loop leakage** — hyperparameters and QC thresholds chosen after seeing test
   performance across a semester of experiments. No test suite catches this. The mitigation is
   a frozen final protocol and ABIDE-II as a genuinely untouched external set.
2. **The transductive population-graph setting** — test node *features* are visible by
   construction. That is inherent to the Parisot formulation, and we declare it rather than
   hide it.
3. **The `transductive` ComBat policy**, if we report it — same issue, same declaration.

## B7 — Cohort QC decision: the minimum-timepoints threshold

Scan lengths in ABIDE-I CC200 range from **78 to 296 timepoints** depending on site.

| `min_timepoints` | Subjects kept | Dropped |
|---|---|---|
| **70 (used)** | **871** | 0 |
| 100 (the script's original default) | 846 | **25 — the whole of OHSU** |
| 120 | 727 | 144 |

OHSU's scans are 78 timepoints, so a threshold of 100 deletes **an entire site**: it costs a
LOSO fold and breaks comparability with the 871-subject convention. We chose 70, which keeps
every subject, and we state the choice rather than leaving it as a default.

**The counter-argument is real and we should raise it ourselves:** 78 timepoints gives a
noisier correlation estimate than 296. And there is direct evidence for that in our own
results — **OHSU is our worst LOSO site at 0.397 balanced accuracy, below chance, with
AUC 0.500.** The right response is a sensitivity analysis at threshold 100 reported as a
secondary row, not a silent drop.

This also suggests something worth saying: part of the cross-site variance may be driven by
**scan length**, not only by scanner identity and cohort composition. That is a testable
claim and a nice thread for the discussion section.

Separately, **21 of the 200 CC200 parcels are constant across subjects and are dropped by
QC**, leaving R = 179 and a feature dimension of 15,931. This is expected for CC200 (some
parcels fall outside the acquired field of view), but it means ROI indices in our
explainability output refer to the 179 surviving parcels — `roi_to_labels(kept_rois=...)`
must be passed the surviving indices when Phase 5 is done properly.

## B8 — Reproducibility

```bash
git clone <repo> && cd asd-gnn
python -m venv .venv && source .venv/bin/activate && pip install -e ".[torch,dev]"
pytest tests/ -q          # 85 tests must pass before anything is believed
make smoke                # full pipeline on synthetic data, ~20 seconds
make data features        # ABIDE-I CC200 ROI time series + shared split file
make leak sanity          # the leakage suite and the pre-flight checks
make baselines graphs     # the experiments
make tables figures       # every table and figure, regenerated from results/runs/*.json
```

Verified working set (macOS arm64, CPU, Python 3.11): torch 2.13.0 ·
torch-geometric 2.8.0.post1 · scikit-learn 1.8.0 · nilearn 0.14.0 ·
numpy 2.4.6
