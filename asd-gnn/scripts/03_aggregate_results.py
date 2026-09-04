#!/usr/bin/env python
"""Glob results/runs/*.json into one tidy frame and emit every paper table.

    python scripts/03_aggregate_results.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from asdgnn.stats.pairing import paired_comparisons
from asdgnn.utils.logging import get_logger, write_json
from asdgnn.viz.tables import load_runs, main_table, protocol_comparison, to_latex, to_markdown

log = get_logger("03_aggregate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--out-dir", default="results/tables")
    args = ap.parse_args()

    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    df = load_runs(results_dir)
    if df.empty:
        raise SystemExit(f"no runs in {results_dir} — run scripts/02_run_experiment.py first")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_runs.csv", index=False)

    for protocol in df["protocol"].dropna().unique():
        t = main_table(df, protocol)
        to_markdown(t, out_dir / f"table_{protocol}.md")
        to_latex(t, out_dir / f"table_{protocol}.tex",
                 caption=f"Results under {protocol}", label=f"tab:{protocol}")
        write_json(paired_comparisons(results_dir, protocol),
                   out_dir / f"stats_{protocol}.json")

    comp = protocol_comparison(df)
    to_markdown(comp, out_dir / "table_protocol_comparison.md")
    log.info("%d runs -> %s", len(df), out_dir)
    print((out_dir / "all_runs.csv").as_posix())


if __name__ == "__main__":
    main()
