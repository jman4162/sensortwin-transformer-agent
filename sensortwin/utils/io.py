"""IO helpers for saving/loading generated datasets and run artifacts.

Datasets are stored as compressed ``.npz`` with arrays ``X`` ``[N, C, T]`` and ``y`` ``[N]``,
plus a sidecar ``<name>.meta.json`` holding generation parameters and per-sample event metadata.
Keeping metadata next to the arrays makes leakage audits and reproducibility checks easy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_dataset(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    """Save arrays to ``path`` (``.npz``) and metadata to a ``.meta.json`` sidecar."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default))
    return path


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load ``X``, ``y`` and metadata produced by :func:`save_dataset`."""
    path = Path(path).with_suffix(".npz")
    with np.load(path) as npz:
        X, y = npz["X"], npz["y"]
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return X, y, meta


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
