.PHONY: setup data features test leak sanity baselines graphs hybrid explain grid tables figures paper notebook smoke clean lint

PY ?= python

setup:
	conda env create -f environment.yml && $(PY) -m pip install -e .

data:
	$(PY) scripts/00_download.py --atlas cc200

features:
	$(PY) scripts/01_build_features.py --atlas cc200

test:
	$(PY) -m pytest tests/ -q

leak:
	$(PY) -m pytest tests/test_no_leakage.py -v

sanity:
	$(PY) scripts/02_run_experiment.py --sanity
	$(PY) scripts/02_run_experiment.py --gnn-sanity

# M8 — classical baselines under both protocols.
baselines:
	for m in svm ridge rf; do \
	  for p in kfold loso; do \
	    $(PY) scripts/02_run_experiment.py --model $$m --protocol $$p; \
	  done; \
	done

# M10 — first GNNs on individual brain graphs.
graphs:
	for m in gcn gat mlp_graph; do \
	  $(PY) scripts/02_run_experiment.py --model $$m --protocol kfold; \
	done

# M12 — the full zoo. Queue it, do not watch it.
grid:
	for m in svm ridge rf gcn gat gin sage cheb mlp_graph braincnn; do \
	  for c in correlation tangent; do \
	    for p in kfold loso; do \
	      $(PY) scripts/02_run_experiment.py --model $$m --conn $$c --protocol $$p; \
	    done; \
	  done; \
	done

# M13 — stage-wise hybrid; add --end-to-end only once stages 1-3 both look sane.
hybrid:
	$(PY) scripts/06_hybrid.py --protocol kfold
	$(PY) scripts/06_hybrid.py --protocol loso

# M14 — explainability with across-fold stability.
explain:
	$(PY) scripts/05_explain.py --model gat --method attention

tables:
	$(PY) scripts/03_aggregate_results.py

figures:
	$(PY) scripts/04_make_figures.py

paper: tables figures

notebook:
	$(PY) notebooks/build_results_notebook.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb

# End-to-end smoke run on the synthetic cohort — no download, ~1 minute.
smoke:
	$(PY) scripts/01_build_features.py --synthetic
	$(PY) scripts/02_run_experiment.py --synthetic --sanity
	$(PY) scripts/02_run_experiment.py --synthetic --model svm --protocol kfold --n-splits 5 --select-k 100
	$(PY) scripts/02_run_experiment.py --synthetic --model svm --protocol loso --select-k 100
	$(PY) scripts/05_explain.py --synthetic --method attention --n-splits 3 \
	  --epochs 25 --min-epochs 8 --patience 8 --top-k 10
	$(PY) scripts/06_hybrid.py --synthetic --n-splits 3 --epochs 25 --min-epochs 8 \
	  --patience 8
	$(PY) scripts/03_aggregate_results.py
	$(PY) scripts/04_make_figures.py

lint:
	ruff check src tests scripts notebooks

clean:
	rm -rf results/runs/*.json results/figures/*.png results/tables/* results/explain/*
