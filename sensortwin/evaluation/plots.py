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


def plot_signal_gallery(
    X: np.ndarray,
    events: list[dict],
    class_names: list[str],
    path: str | Path,
    *,
    channel_names: list[str] | None = None,
) -> Path:
    """One example signal per event class (``[C, T]`` each), 8 channels stacked, event span shaded.

    ``events`` is the generator's per-sample metadata (``meta["events"]``); the first sample of each
    class is shown, with its ``[start, start+duration]`` shaded on the affected channels.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)
    cols = 2
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.0 * rows))
    axes = np.asarray(axes).reshape(-1)
    C = X.shape[1]
    spacing = 3.5
    for ci, name in enumerate(class_names):
        ax = axes[ci]
        idx = next((i for i, e in enumerate(events) if e.get("event_class") == ci), None)
        if idx is None:
            ax.set_visible(False)
            continue
        sig = X[idx]
        ev = events[idx]
        affected = set(ev.get("affected_channels") or [])
        for c in range(C):
            x = sig[c] - sig[c].mean()
            x = x / (np.abs(x).max() + 1e-6)
            ax.plot(x + c * spacing, lw=0.7, color="C0" if c in affected else "0.7")
        start, dur = ev.get("start"), ev.get("duration")
        if start is not None and dur:
            ax.axvspan(start, start + dur, color="orange", alpha=0.18)
        ax.set_title(name, fontsize=9)
        ax.set_yticks([])
        ax.set_xticks([])
    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("SensorTwin-Synth: one example per event class (affected channels in blue)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scale_comparison(
    curves: dict[str, dict[int, float]],
    path: str | Path,
    *,
    title: str = "Macro-F1 vs training-set size",
) -> Path:
    """Macro-F1 vs training size (log-x), one line per model. ``curves[model]={n: f1}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, by_n in curves.items():
        ns = sorted(by_n)
        emph = model == "transformer"
        ax.plot(
            ns,
            [by_n[n] for n in ns],
            "o-",
            label=model,
            lw=2.4 if emph else 1.4,
            zorder=3 if emph else 2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_perclass_delta(
    deltas: dict[str, float],
    path: str | Path,
    *,
    title: str = "Per-class F1 delta (transformer − best baseline)",
) -> Path:
    """Sorted diverging horizontal bars of per-class F1 deltas (green positive, red negative)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(deltas.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1))
    ax.barh(range(len(names)), vals, color=colors)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Δ F1")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_saliency_overlay(
    signal: np.ndarray,
    saliency: np.ndarray,
    path: str | Path,
    *,
    event_span: tuple[int, int] | None = None,
    channel_name: str | None = None,
    title: str = "Saliency over signal",
) -> Path:
    """One channel's signal with its (normalized) saliency shaded and the true event span marked."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sal = np.abs(np.asarray(saliency, dtype=float))
    sal = sal / (sal.max() + 1e-12)
    t = np.arange(len(signal))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(t, 0, sal, color="purple", alpha=0.25, label="saliency (|grad|, norm.)")
    sig = np.asarray(signal, dtype=float)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-12)
    sig_label = f"signal ({channel_name})" if channel_name else "signal"
    ax.plot(t, sig, color="C0", lw=0.9, label=sig_label)
    if event_span is not None:
        ax.axvspan(event_span[0], event_span[1], color="orange", alpha=0.20, label="event region")
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
