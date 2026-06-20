"""Classification metrics (roadmap v0.2, spec §12.1).

Accuracy alone is misleading on imbalanced classes, so the headline metric is **macro-F1** (equal
weight per class). We also report weighted-F1, per-class precision/recall/F1, one-vs-rest macro
AUROC, and the confusion matrix. Everything is computed against the fixed label set ``0..C-1`` so a
class absent from a particular split (e.g. a rare event in a small test fold) does not crash AUROC
or silently shift the confusion-matrix axes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute the standard metric suite. ``y_proba`` ``[N, C]`` enables AUROC (None to skip)."""
    n_classes = len(class_names)
    labels = list(range(n_classes))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    }

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_auroc": _safe_auroc(y_true, y_proba, n_classes),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
    }
    return metrics


def _safe_auroc(y_true: np.ndarray, y_proba: np.ndarray | None, n_classes: int) -> float | None:
    """One-vs-rest macro AUROC, robust to classes missing from ``y_true``.

    Restricts the average to classes actually present (sklearn cannot score a one-vs-rest column
    with no positives); returns None if probabilities weren't provided.
    """
    if y_proba is None:
        return None
    present = np.unique(y_true)
    if len(present) < 2:
        return None
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba[:, present],
                labels=present,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Write metrics to ``<path>.json`` and a flat scalar summary to ``<path>.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(metrics, indent=2, default=_json_default))

    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, type(None)))}
    for name, d in metrics.get("per_class", {}).items():
        flat[f"f1_{name}"] = d["f1"]
    with path.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in flat.items():
            w.writerow([k, v])
    return json_path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
