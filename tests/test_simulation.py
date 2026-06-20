"""Tests for the synthetic simulator: shapes, labels, determinism, leakage-free metadata."""

from __future__ import annotations

import numpy as np

from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES, N_CHANNELS


def _small_cfg(**kw) -> GenConfig:
    kw.setdefault("seed", 7)
    return GenConfig(n_samples=80, T=128, **kw)


def test_shapes_and_dtypes():
    X, y, meta = generate_dataset(_small_cfg())
    assert X.shape == (80, N_CHANNELS, 128)
    assert y.shape == (80,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert meta["n_channels"] == N_CHANNELS


def test_labels_in_range_and_all_classes_reachable():
    X, y, _ = generate_dataset(GenConfig(n_samples=600, T=64, seed=1))
    assert y.min() >= 0
    assert y.max() < len(EVENT_CLASSES)
    # With 600 uniform draws over 10 classes, every class should appear.
    assert set(np.unique(y).tolist()) == set(range(len(EVENT_CLASSES)))


def test_no_nans_or_infs():
    X, _, _ = generate_dataset(_small_cfg())
    assert np.isfinite(X).all()


def test_determinism_same_seed():
    X1, y1, _ = generate_dataset(_small_cfg())
    X2, y2, _ = generate_dataset(_small_cfg())
    assert np.array_equal(y1, y2)
    assert np.allclose(X1, X2)


def test_different_seed_changes_data():
    X1, _, _ = generate_dataset(_small_cfg(seed=7))
    X2, _, _ = generate_dataset(_small_cfg(seed=8))
    assert not np.allclose(X1, X2)


def test_metadata_matches_labels():
    _, y, meta = generate_dataset(_small_cfg())
    events = meta["events"]
    assert len(events) == len(y)
    for i, ev in enumerate(events):
        assert ev["event_class"] == int(y[i])
        assert ev["name"] == EVENT_CLASSES[int(y[i])]


def test_normalization_stats_present():
    _, _, meta = generate_dataset(_small_cfg(normalize=True))
    stats = meta["channel_stats"]
    assert stats is not None
    assert len(stats["mean"]) == N_CHANNELS
    assert len(stats["std"]) == N_CHANNELS
