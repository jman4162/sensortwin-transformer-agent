"""Smoke tests for the visualization helpers (gallery / scale / per-class / saliency)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from sensortwin.evaluation.plots import (  # noqa: E402
    plot_perclass_delta,
    plot_saliency_overlay,
    plot_scale_comparison,
    plot_signal_gallery,
)


def test_plot_signal_gallery(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 8, 64)).astype("float32")
    events = [
        {"event_class": i, "name": f"c{i}", "start": 5, "duration": 10, "affected_channels": [0, 1]}
        for i in range(10)
    ]
    out = plot_signal_gallery(X, events, [f"c{i}" for i in range(10)], tmp_path / "gallery.png")
    assert out.exists()


def test_plot_scale_comparison(tmp_path):
    curves = {"transformer": {2000: 0.43, 20000: 0.90}, "cnn": {2000: 0.50, 20000: 0.84}}
    out = plot_scale_comparison(curves, tmp_path / "scale.png")
    assert out.exists()


def test_plot_perclass_delta(tmp_path):
    out = plot_perclass_delta({"a": 0.12, "b": -0.04, "c": 0.0}, tmp_path / "delta.png")
    assert out.exists()


def test_plot_saliency_overlay(tmp_path):
    rng = np.random.default_rng(1)
    out = plot_saliency_overlay(
        rng.standard_normal(64),
        np.abs(rng.standard_normal(64)),
        tmp_path / "sal.png",
        event_span=(10, 30),
        channel_name="temperature_core",
    )
    assert out.exists()
