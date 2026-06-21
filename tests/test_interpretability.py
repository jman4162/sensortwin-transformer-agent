"""Tests for interpretability diagnostics (v0.5)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sensortwin.evaluation.interpretability import (  # noqa: E402
    attention_map,
    attention_to_time,
    integrated_gradients,
    localization_score,
    occlusion_importance,
    random_localization_baseline,
)
from sensortwin.models.transformer import SensorPatchTST  # noqa: E402
from sensortwin.simulation.events import N_CHANNELS  # noqa: E402


def _tiny():
    return SensorPatchTST(d_model=32, num_layers=2, num_heads=2)


def test_attention_map_shape_and_none_for_mean_pool():
    X = np.random.randn(5, N_CHANNELS, 128).astype(np.float32)
    am = attention_map(_tiny(), X)
    assert am.shape == (5, N_CHANNELS, 15)  # 8 channels x 15 patches
    assert (
        attention_map(SensorPatchTST(d_model=32, num_layers=2, num_heads=2, pooling="mean"), X)
        is None
    )


def test_attention_to_time_shape():
    attn = np.random.rand(8, 15)
    t = attention_to_time(attn, 128, patch_len=16, stride=8)
    assert t.shape == (8, 128)
    assert np.isfinite(t).all()


def test_integrated_gradients_shape_and_finite():
    x = np.random.randn(N_CHANNELS, 128).astype(np.float32)
    sal = integrated_gradients(_tiny(), x, target=3, steps=8)
    assert sal.shape == (N_CHANNELS, 128)
    assert np.isfinite(sal).all()


def test_occlusion_importance_keys():
    X = np.random.randn(6, N_CHANNELS, 128).astype(np.float32)
    targets = np.random.randint(0, 10, size=6)
    model = _tiny()

    def proba_fn(Xx):
        with torch.no_grad():
            return torch.softmax(model(torch.from_numpy(Xx).float()), 1).numpy()

    out = occlusion_importance(proba_fn, X, targets, n_windows=4)
    assert out["channel_drop"].shape == (N_CHANNELS,)
    assert out["window_drop"].shape == (4,)


def test_localization_score_hits_event_region():
    # Saliency concentrated exactly on the event region should score ~1.0 and beat random.
    C, T = 8, 100
    ev = {"affected_channels": [2, 3], "start": 40, "duration": 20, "name": "x"}
    sal = np.zeros((C, T))
    sal[2:4, 40:60] = 1.0
    score = localization_score(sal, ev)
    rand = random_localization_baseline(ev, C, T)
    assert score == pytest.approx(1.0)
    assert rand == pytest.approx((2 * 20) / (8 * 100))
    assert score > rand


def test_localization_none_for_unlocalized_events():
    assert (
        localization_score(
            np.ones((8, 100)), {"affected_channels": [], "start": None, "duration": None}
        )
        is None
    )
