from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def config_hash(cfg: dict, keys: list[str] | None = None, n: int = 12) -> str:
    """sha1 of the subset of the config that affects a given artifact."""
    sub = cfg if keys is None else {k: cfg.get(k) for k in keys}
    blob = json.dumps(_canonical(sub), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


def git_commit(short: bool = True) -> str:
    args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"
