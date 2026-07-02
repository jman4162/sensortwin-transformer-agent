"""CLI: regenerate the committed portfolio figures under ``docs/figures/`` (roadmap viz).

Four figures, all reproducible:
  * dataset_gallery.png   — one example signal per event class, event region shaded (the benchmark).
  * scale_comparison.png  — macro-F1 vs training size; the quick-demo -> colab_standard rank flip.
  * perclass_delta.png     — per-class F1 gain of the transformer over the best baseline (cnn).
  * saliency_overlay.png   — integrated-gradients saliency over a signal vs the true event region.

The first three are GPU-free. The saliency figure trains a small SensorPatchTST on CPU
(~1-2 min) and is skipped if torch is unavailable. The scale and per-class figures read their
numbers from committed ``headline_comparison.json`` artifacts (written by
``scripts/compare_models.py``) rather than hardcoding them, so every plotted number traces to a
reproducible run.
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

configure_omp()  # macOS-only OpenMP guard; must precede any torch import

import argparse
import json
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

# Committed compare_models artifacts, one per dataset scale.
_HEADLINE_20K = "reports/experiment_summaries/headline_comparison.json"
_HEADLINE_2K = "reports/experiment_summaries/scale_2k/headline_comparison.json"
_MODE_SIZES = {"quick_demo": 2000, "colab_standard": 20000, "full_reproduction": 100000}


def _load_headline(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        print(f"  ({path} missing — run scripts/compare_models.py first)")
        return None
    return json.loads(p.read_text())


def _gallery(out: Path) -> None:
    X, _, meta = generate_dataset(GenConfig(n_samples=300, T=512, seed=0, normalize=False))
    plot_signal_gallery(
        X, meta["events"], EVENT_CLASSES, out / "dataset_gallery.png", channel_names=CHANNELS
    )
    print("  dataset_gallery.png")


def _scale(out: Path) -> None:
    scale: dict[str, dict[int, float]] = {}
    for path in (_HEADLINE_2K, _HEADLINE_20K):
        run = _load_headline(path)
        if run is None:
            continue
        n = _MODE_SIZES[run["meta"]["mode"]]
        for name, m in run["summary"]["models"].items():
            scale.setdefault(name, {})[n] = m["mean"]
    if not scale or min(len(v) for v in scale.values()) < 2:
        print("  scale_comparison.png SKIPPED (need runs at both scales)")
        return
    plot_scale_comparison(scale, out / "scale_comparison.png")
    print("  scale_comparison.png")


def _perclass(out: Path) -> None:
    run = _load_headline(_HEADLINE_20K)
    if run is None or "headline" not in run["summary"]:
        print("  perclass_delta.png SKIPPED (no headline comparison in the artifact)")
        return
    pc = run["summary"]["headline"]["per_class_vs_best_baseline"]
    deltas = {cls: d["delta"] for cls, d in sorted(pc.items(), key=lambda kv: -kv[1]["delta"])}
    plot_perclass_delta(deltas, out / "perclass_delta.png")
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
