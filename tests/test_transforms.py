"""Tests for ChannelStandardizer leakage-safe scaling (v0.2)."""

from __future__ import annotations

import numpy as np

from sensortwin.data.transforms import ChannelStandardizer


def _toy():
    rng = np.random.default_rng(0)
    # Two splits with deliberately different scales/offsets per channel.
    X_train = rng.normal(5.0, 2.0, size=(50, 3, 64)).astype(np.float32)
    X_test = rng.normal(-1.0, 0.5, size=(20, 3, 64)).astype(np.float32)
    return X_train, X_test


def test_fit_transform_train_is_standardized():
    X_train, _ = _toy()
    std = ChannelStandardizer()
    Xt = std.fit_transform(X_train)
    # Per-channel mean ~0, std ~1 on the fit set.
    assert np.allclose(Xt.mean(axis=(0, 2)), 0.0, atol=1e-4)
    assert np.allclose(Xt.std(axis=(0, 2)), 1.0, atol=1e-3)


def test_transform_uses_train_stats_only():
    X_train, X_test = _toy()
    std = ChannelStandardizer().fit(X_train)
    Xte = std.transform(X_test)
    # Test set standardized with TRAIN stats should NOT be zero-mean (different distribution).
    assert not np.allclose(Xte.mean(axis=(0, 2)), 0.0, atol=0.1)
    # Stats are frozen from fit.
    assert std.mean_ is not None and std.mean_.shape == (1, 3, 1)


def test_transform_before_fit_raises():
    std = ChannelStandardizer()
    try:
        std.transform(np.zeros((1, 3, 4), dtype=np.float32))
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
