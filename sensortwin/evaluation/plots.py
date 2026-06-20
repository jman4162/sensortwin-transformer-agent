"""Plotting helpers for the baseline report (roadmap v0.2; spec §12, §23).

Matplotlib only (no seaborn) to stay dependency-light. Each function writes a PNG and returns its
path. Uses the non-interactive Agg backend so plots render in headless/CI contexts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sensortwin.evaluation.calibration import ReliabilityCurve  # noqa: E402


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], path: str | Path, *, normalize: bool = True
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.asarray(cm, dtype=float)
    if normalize:
        row = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row, out=np.zeros_like(mat), where=row > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_reliability(
    curve: ReliabilityCurve, path: str | Path, *, title: str = "Reliability"
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    valid = ~np.isnan(curve.bin_confidence)
    ax.plot(curve.bin_confidence[valid], curve.bin_accuracy[valid], "o-", label="model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_per_class_f1(
    per_class: dict[str, dict[str, float]], path: str | Path, *, title: str = "Per-class F1"
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(per_class)
    f1s = [per_class[n]["f1"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), f1s, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
