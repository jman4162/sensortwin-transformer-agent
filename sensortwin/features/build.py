"""Single entry point that assembles the full classical feature matrix (roadmap v0.2).

Concatenates statistical + spectral + cross-channel/missingness features into one ``[N, D]`` matrix
with aligned human-readable names, so baselines (and their feature-importance reports) consume one
consistent representation.
"""

from __future__ import annotations

import numpy as np

from sensortwin.features.correlations import correlation_feature_names, correlation_features
from sensortwin.features.spectral import spectral_feature_names, spectral_features
from sensortwin.features.statistical import statistical_feature_names, statistical_features
from sensortwin.simulation.events import CHANNELS


def build_feature_matrix(
    X: np.ndarray, channels: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Return ``(F[N, D] float32, feature_names)`` for raw signals ``X[N, C, T]``."""
    channels = channels or CHANNELS
    blocks = [statistical_features(X), spectral_features(X), correlation_features(X)]
    names = (
        statistical_feature_names(channels)
        + spectral_feature_names(channels)
        + correlation_feature_names(channels)
    )
    F = np.concatenate(blocks, axis=1).astype(np.float32)
    if F.shape[1] != len(names):
        raise AssertionError(f"feature/name width mismatch: {F.shape[1]} vs {len(names)}")
    return F, names
