from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Hand-typed numbers in the paper are forbidden: every table here is produced from
# results/runs/*.json and nothing else.


def load_runs(results_dir: str | Path = "results/runs") -> pd.DataFrame:
    rows = []
    for p in sorted(Path(results_dir).glob("*.json")):
        blob = json.loads(p.read_text())
        cfg = blob.get("config", {})
        rows.append({
            "run_id": blob["run_id"],
            "path": str(p),
            "git_commit": blob.get("git_commit"),
            "model": blob.get("model", cfg.get("model")),
            "protocol": blob.get("protocol"),
            "atlas": cfg.get("atlas"),
            # scripts/06_hybrid.py writes `conn`; everything else writes `conn_kind`.
            "conn_kind": cfg.get("conn_kind", cfg.get("conn")),
            "harmonize": cfg.get("harmonize", "none"),
            # Part of the run's identity: the three LOSO policies are not comparable.
            "unseen_site": cfg.get("unseen_site", "n/a"),
            "edge_strategy": cfg.get("edge_strategy"),
            "edge_param": cfg.get("edge_param"),
            "n_folds": blob["aggregate"].get("n_folds"),
            "runtime_sec": blob.get("runtime_sec"),
            **{k: v for k, v in blob["aggregate"].items() if k.endswith(("_mean", "_std"))},
            "pooled_auc": blob.get("pooled", {}).get("auc"),
        })
    return pd.DataFrame(rows)


def _fmt(mean, std, digits=3):
    if mean is None or pd.isna(mean):
        return "--"
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if std is not None and not pd.isna(std) \
        else f"{mean:.{digits}f}"


# Two runs are the same configuration only if all of these agree.
CONFIG_KEYS = ["model", "conn_kind", "atlas", "harmonize", "unseen_site"]


def main_table(df: pd.DataFrame, protocol: str = "kfold",
               metrics=("bal_acc", "acc", "auc", "sens", "spec")) -> pd.DataFrame:
    sub = df[df["protocol"] == protocol].copy()
    out = pd.DataFrame({"model": sub["model"], "conn": sub["conn_kind"],
                        "atlas": sub["atlas"], "harmonize": sub["harmonize"],
                        "unseen_site": sub["unseen_site"]})
    for m in metrics:
        out[m] = [_fmt(a, b) for a, b in zip(sub.get(f"{m}_mean"), sub.get(f"{m}_std"))]
    return out.sort_values("bal_acc", ascending=False).reset_index(drop=True)


def protocol_comparison(df: pd.DataFrame, metric: str = "bal_acc_mean") -> pd.DataFrame:
    """k-fold vs LOSO side by side, with the drop that the paper must report."""
    # The index must carry the FULL configuration identity. Omitting harmonisation lets
    # pivot_table silently average a ComBat run together with a raw one, which quietly
    # corrupts the drop column that this whole table exists to report.
    piv = df.pivot_table(index=CONFIG_KEYS, columns="protocol", values=metric)
    if {"kfold", "loso"}.issubset(piv.columns):
        piv["drop"] = piv["kfold"] - piv["loso"]
    return piv.reset_index().sort_values("kfold", ascending=False)


def to_latex(df: pd.DataFrame, path: str | Path, caption: str = "", label: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, escape=True, caption=caption, label=label))
    return path


def to_markdown(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_markdown(index=False))
    return path
