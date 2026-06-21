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


def plot_label_efficiency(
    curves: dict[str, dict[float, float]],
    path: str | Path,
    *,
    title: str = "Label efficiency",
    ylabel: str = "Macro-F1",
) -> Path:
    """Macro-F1 vs label fraction, one line per arm. ``curves[arm] = {fraction: score}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, by_frac in curves.items():
        fracs = sorted(by_frac)
        ax.plot([f * 100 for f in fracs], [by_frac[f] for f in fracs], "o-", label=arm)
    ax.set_xscale("log")
    ax.set_xlabel("Labeled fraction of train (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_robustness_degradation(
    curves: dict[str, dict[float, float]],
    path: str | Path,
    *,
    title: str = "Robustness degradation",
    xlabel: str = "Corruption severity",
) -> Path:
    """Macro-F1 vs corruption severity, one line per model. ``curves[model]={severity: score}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, by_sev in curves.items():
        sevs = sorted(by_sev)
        ax.plot(sevs, [by_sev[s] for s in sevs], "o-", label=model)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_attention_map(
    weights: np.ndarray,
    path: str | Path,
    *,
    channel_names: list[str] | None = None,
    title: str = "Attention (channel x patch)",
) -> Path:
    """Heatmap of per-(channel, patch) attention weights ``[C, N]`` for one sample."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(weights, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    if channel_names is not None:
        ax.set_yticks(range(len(channel_names)))
        ax.set_yticklabels(channel_names)
    ax.set_xlabel("Patch index (time)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
