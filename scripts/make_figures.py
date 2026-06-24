"""CLI: regenerate the committed portfolio figures under ``docs/figures/`` (roadmap viz).

Four figures, all reproducible:
  * dataset_gallery.png   — one example signal per event class, event region shaded (the benchmark).
  * scale_comparison.png  — macro-F1 vs training size; the quick-demo -> colab_standard rank flip.
  * perclass_delta.png     — per-class F1 gain of the transformer over the best baseline (cnn).
  * saliency_overlay.png   — integrated-gradients saliency over a signal vs the true event region.

The first three are GPU-free (generator + recorded numbers). The saliency figure trains a small
SensorPatchTST on CPU (~1-2 min) and is skipped if torch is unavailable. Recorded numbers come from
the README quick-demo table and the 3-seed colab_standard run (see reports/research_log.md).
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

configure_omp()  # macOS-only OpenMP guard; must precede any torch import

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from sensortwin.evaluation.plots import (
    plot_perclass_delta,
    plot_saliency_overlay,
    plot_scale_comparison,
    plot_signal_gallery,
)
from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.events import CHANNELS, EVENT_CLASSES

# Macro-F1 per model at the two scales (test split). 2k = README quick-demo; 20k = 3-seed comparison.
_SCALE = {
    "transformer": {2000: 0.435, 20000: 0.900},
    "cnn": {2000: 0.504, 20000: 0.844},
    "xgboost": {2000: 0.620, 20000: 0.759},
    "logreg": {2000: 0.571, 20000: 0.697},
    "random_forest": {2000: 0.543, 20000: 0.691},
    "lstm": {2000: 0.338, 20000: 0.545},
}

# Per-class F1 delta (transformer − cnn), mean over the 3 colab_standard seeds.
_PERCLASS_DELTA = {
    "sensor_dropout": 0.135,
    "regime_shift": 0.120,
    "normal": 0.091,
    "voltage_sag": 0.082,
    "compound_fault": 0.043,
    "correlated_channel_fault": 0.032,
    "thermal_drift": 0.027,
    "current_spike": 0.011,
    "oscillatory_instability": 0.011,
    "slow_degradation": 0.009,
}


def _gallery(out: Path) -> None:
    X, _, meta = generate_dataset(GenConfig(n_samples=300, T=512, seed=0, normalize=False))
    plot_signal_gallery(
        X, meta["events"], EVENT_CLASSES, out / "dataset_gallery.png", channel_names=CHANNELS
    )
    print("  dataset_gallery.png")


def _scale(out: Path) -> None:
    plot_scale_comparison(_SCALE, out / "scale_comparison.png")
    print("  scale_comparison.png")


def _perclass(out: Path) -> None:
    plot_perclass_delta(_PERCLASS_DELTA, out / "perclass_delta.png")
    print("  perclass_delta.png")


def _saliency(out: Path) -> None:
    """Train a small transformer on CPU and overlay integrated-gradients saliency on one sample."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  saliency_overlay.png SKIPPED (torch not installed)")
        return

    from sensortwin.data.dataset import SensorArrayDataset
    from sensortwin.data.splits import make_split
    from sensortwin.data.transforms import ChannelStandardizer
    from sensortwin.evaluation.interpretability import integrated_gradients
    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.loop import class_weights, train_model
    from sensortwin.utils.seeds import set_torch_seed

    set_torch_seed(0)
    X, y, meta = generate_dataset(GenConfig(n_samples=800, T=512, seed=0, normalize=False))
    sp = make_split("random", y, {}, seed=0)
    tr, va = sp["train"], sp["val"]
    std = ChannelStandardizer().fit(X[tr])
    model: Any = SensorPatchTST(d_model=64, num_layers=2, num_heads=2)
    train_ds = SensorArrayDataset(std.transform(X[tr]), y[tr]).as_torch()
    val_ds = SensorArrayDataset(std.transform(X[va]), y[va]).as_torch()
    model, _ = train_model(
        model,
        train_ds,
        val_ds,
        epochs=5,
        weight=class_weights(y[tr], len(EVENT_CLASSES)),
        device="cpu",
    )

    # Pick a localized event (thermal_drift: affects temperature channels, has a start/duration).
    target_class = EVENT_CLASSES.index("thermal_drift")
    idx = next(
        (
            i
            for i in sp["test"]
            if meta["events"][i]["event_class"] == target_class
            and meta["events"][i]["start"] is not None
        ),
        None,
    )
    if idx is None:
        print("  saliency_overlay.png SKIPPED (no localized sample found)")
        return
    ev = meta["events"][idx]
    xs = std.transform(X[idx : idx + 1])[0]  # [C, T] numpy
    ig = np.asarray(integrated_gradients(model, xs, target_class))  # [C, T]
    ch = ev["affected_channels"][0]
    sal = np.convolve(np.abs(ig[ch]), np.ones(9) / 9, mode="same")  # smoothed attribution envelope
    plot_saliency_overlay(
        xs[ch],
        sal,
        out / "saliency_overlay.png",
        event_span=(ev["start"], ev["start"] + ev["duration"]),
        channel_name=CHANNELS[ch],
        title="Integrated-gradients saliency vs the true event region (thermal_drift)",
    )
    print("  saliency_overlay.png")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Regenerate the committed docs/figures gallery.")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--no-saliency", action="store_true", help="skip the model-training figure")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {out}/ ...")
    _gallery(out)
    _scale(out)
    _perclass(out)
    if not args.no_saliency:
        _saliency(out)
    print("done.")


if __name__ == "__main__":
    main()
