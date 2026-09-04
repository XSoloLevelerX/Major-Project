# Presentation Pack — Mid-Semester Progress Review

Everything needed to build and deliver next week's progress presentation.
Written the night of 3–4 September 2026; all numbers come from a real ABIDE-I run
completed that night.

## The four files

| File | What it is | Read it when |
|---|---|---|
| **`SLIDES.md`** | The deck, slide by slide. Every slide has *on the slide* content, *speaker notes*, and the figure it needs. 22 slides + 8 backup slides for questions. | Building the deck; rehearsing |
| **`RESULTS-SNAPSHOT.md`** | Every number, with the file it came from. This is the single source of truth for anything numeric in the deck. | Any time you are about to type a number onto a slide |
| **`PLOT-PROMPT.md`** | Copy-paste prompts for generating the visuals — a master prompt, per-figure prompts, and guardrails. Also lists which figures the pipeline already generates so you don't redraw them. | Making the figures |
| **`PIPELINE-STATUS.md`** | What was verified, how, and what was broken. Includes the three bugs found and fixed, and the known gaps. | Someone asks "does it actually work?"; planning next sprint |

## Suggested order of work

1. Read `RESULTS-SNAPSHOT.md` first — twenty minutes, and it is the substance of the talk.
2. Regenerate the pipeline figures:
   ```bash
   cd ../asd-gnn && .venv/bin/python scripts/04_make_figures.py
   ```
3. Generate the seven presentation-only figures (C1–C7) using Section C of `PLOT-PROMPT.md`.
4. Build the deck from `SLIDES.md`. Either paste Section D of `PLOT-PROMPT.md` into a
   slide-generating tool, or build it by hand — the content is already written either way.
5. Rehearse against the backup slides in the `SLIDES.md` appendix. Those cover the questions
   that are actually going to be asked.

## Status: the grid is complete

All 36 runs are on disk; tables, 24 figures and the executed notebook are regenerated from
real ABIDE-I data. Nothing is pending for the presentation.

To regenerate everything from the stored runs:

```bash
cd ../asd-gnn
.venv/bin/python scripts/03_aggregate_results.py
.venv/bin/python scripts/04_make_figures.py
.venv/bin/python notebooks/build_results_notebook.py
```

## The one-sentence version of the talk

> The published literature evaluates ASD classification optimistically — random k-fold, where
> every test subject has site-mates in training — and we have built a leakage-audited pipeline
> that reports both that protocol and leave-one-site-out, so that the gap between them becomes
> the measurement rather than the embarrassment.

## Three things not to get wrong on the day

1. **Balanced accuracy, not accuracy.** The cohort is 403 ASD / 468 TC. Every number in the
   deck is balanced accuracy, and every one carries a standard deviation across folds.
2. **Never present a synthetic-cohort number as an ABIDE result.** The synthetic cohort exists
   to test the plumbing; scores near 0.50 there are correct behaviour, not failure. The old
   synthetic runs live in `asd-gnn/results/runs_synthetic/` and are excluded from all tables.
3. **No diagnostic or clinical claims.** This is a research prototype for decision support.
   Say so on the limitations slide, unprompted.
