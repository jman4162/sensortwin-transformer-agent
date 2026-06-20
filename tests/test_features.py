"""Tests for spectral / correlation / aggregated features (v0.2)."""

from __future__ import annotations

import numpy as np

from sensortwin.features import build_feature_matrix
from sensortwin.features.correlations import correlation_feature_names, correlation_features
from sensortwin.features.spectral import spectral_feature_names, spectral_features
from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.events import CHANNELS, N_CHANNELS


def _data():
    return generate_dataset(GenConfig(n_samples=40, T=128, seed=2))


def test_spectral_shape_and_names_align():
    X, _, _ = _data()
    feats = spectral_features(X)
    names = spectral_feature_names(CHANNELS)
    assert feats.shape == (40, len(names))
    assert np.isfinite(feats).all()


def test_correlation_shape_includes_pairs_and_frozen():
    X, _, _ = _data()
    feats = correlation_features(X)
    names = correlation_feature_names(CHANNELS)
    expected = N_CHANNELS * (N_CHANNELS - 1) // 2 + N_CHANNELS  # 28 + 8
    assert feats.shape == (40, expected)
    assert len(names) == expected
    assert np.isfinite(feats).all()


def test_correlation_values_in_range():
    X, _, _ = _data()
    feats = correlation_features(X)
    n_pairs = N_CHANNELS * (N_CHANNELS - 1) // 2
    corrs = feats[:, :n_pairs]
    assert corrs.min() >= -1.0 - 1e-5 and corrs.max() <= 1.0 + 1e-5
    frozen = feats[:, n_pairs:]
    assert frozen.min() >= 0.0 and frozen.max() <= 1.0


def test_build_feature_matrix_width_matches_names():
    X, _, _ = _data()
    F, names = build_feature_matrix(X)
    assert F.shape[0] == 40
    assert F.shape[1] == len(names)
    assert F.dtype == np.float32
    assert np.isfinite(F).all()


def test_features_deterministic():
    X, _, _ = _data()
    F1, _ = build_feature_matrix(X)
    F2, _ = build_feature_matrix(X)
    assert np.array_equal(F1, F2)
