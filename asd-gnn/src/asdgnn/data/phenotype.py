from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asdgnn.utils.logging import get_logger, write_json

log = get_logger(__name__)

MISSING_SENTINELS = (-9999, -9999.0)
KEEP = ["FILE_ID", "SITE_ID", "y", "age", "sex", "fiq", "eye", "mean_fd"]


def load_phenotype(csv_path: str | Path, file_ids: list[str] | None = None) -> pd.DataFrame:
    """Parse Phenotypic_V1_0b_preprocessed1.csv, filter to downloaded subjects,
    normalise columns, encode the label.

    Returns a DataFrame indexed by SUB_ID with columns KEEP.
    """
    df = pd.read_csv(csv_path)
    df = df.replace(list(MISSING_SENTINELS), np.nan)
    df = df[df["FILE_ID"].astype(str) != "no_filename"]

    df["y"] = (df["DX_GROUP"] == 1).astype(int)  # 1=Autism -> 1, 2=Control -> 0
    df["sex"] = (df["SEX"] == 1).astype(int)  # 1=male
    df["age"] = df["AGE_AT_SCAN"].astype(float)
    df["fiq"] = pd.to_numeric(df.get("FIQ"), errors="coerce")
    df["eye"] = pd.to_numeric(df.get("EYE_STATUS_AT_SCAN"), errors="coerce")
    df["mean_fd"] = pd.to_numeric(df.get("func_mean_fd"), errors="coerce")
    df["SITE_ID"] = df["SITE_ID"].astype(str)

    df = df.set_index("SUB_ID")
    if file_ids is not None:
        df = df[df["FILE_ID"].isin(set(file_ids))]

    out = df[KEEP].copy()
    out["site_code"] = pd.Categorical(out["SITE_ID"]).codes.astype(int)
    dup = out.index.duplicated()
    if dup.any():
        log.warning("dropping %d duplicate SUB_IDs", int(dup.sum()))
        out = out[~dup]
    return out


def cohort_report(pheno: pd.DataFrame, out_path: str | Path = "results/qc/cohort.json") -> dict:
    """n per site, n per label per site, sex ratio, age stats, mean_fd by group."""
    by_site = (
        pheno.groupby("SITE_ID")
        .agg(n=("y", "size"), n_asd=("y", "sum"), male_frac=("sex", "mean"),
             age_mean=("age", "mean"), mean_fd=("mean_fd", "mean"))
        .assign(n_tc=lambda d: d["n"] - d["n_asd"])
    )
    report = {
        "n_subjects": int(len(pheno)),
        "n_asd": int(pheno["y"].sum()),
        "n_tc": int((pheno["y"] == 0).sum()),
        "n_sites": int(pheno["SITE_ID"].nunique()),
        "male_fraction": float(pheno["sex"].mean()),
        "age": {
            "min": float(pheno["age"].min()),
            "max": float(pheno["age"].max()),
            "mean": float(pheno["age"].mean()),
            "std": float(pheno["age"].std()),
        },
        "mean_fd_by_group": {
            "asd": float(pheno.loc[pheno.y == 1, "mean_fd"].mean(skipna=True)),
            "tc": float(pheno.loc[pheno.y == 0, "mean_fd"].mean(skipna=True)),
        },
        "per_site": by_site.reset_index().to_dict(orient="records"),
    }
    write_json(report, out_path)
    log.info("cohort report -> %s (N=%d, %d sites)", out_path, report["n_subjects"],
             report["n_sites"])
    return report
