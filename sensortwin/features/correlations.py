"""Cross-channel and missingness features for the classical baseline (roadmap v0.2).

The ``correlated_channel_fault`` event is, by construction, invisible in any single channel: it
only shows up in the *relationship* between channels. A feature model can have a fair shot at it
only if we hand it cross-channel statistics explicitly. We add:

  * the upper triangle of the channel-by-channel Pearson correlation matrix (C*(C-1)/2 feats), and
  * a per-channel "frozen fraction" proxy for the spec's "missingness fraction" — the share of
    consecutive near-equal samples, which is how ``sensor_dropout`` manifests (the generator emits
    frozen/held segments rather than NaNs).
"""

from __future__ import annotations

import numpy as np

_FROZEN_EPS = 1e-6


def _upper_tri_pairs(channels: list[str]) -> list[tuple[int, int]]:
    C = len(channels)
    return [(i, j) for i in range(C) for j in range(i + 1, C)]


def correlation_features(X: np.ndarray) -> np.ndarray:
    """Map ``[N, C, T]`` -> ``[N, C*(C-1)/2 + C]`` (pairwise correlations + frozen fractions)."""
    if X.ndim != 3:
        raise ValueError(f"expected [N, C, T], got {X.shape}")
    N, C, T = X.shape
    pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    out = np.empty((N, len(pairs) + C), dtype=np.float32)
    for n in range(N):
        # corrcoef on constant channels (e.g. a zeroed channel in the robustness sweep) divides
        # by a zero std -> NaN + a numpy warning; silence the warning and map NaN to 0 (no linear
        # relationship).
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X[n])
        corr = np.nan_to_num(corr, nan=0.0)
        out[n, : len(pairs)] = [corr[i, j] for i, j in pairs]
        # Frozen fraction: share of adjacent samples that are ~equal.
        frozen = (np.abs(np.diff(X[n], axis=1)) < _FROZEN_EPS).mean(axis=1)
        out[n, len(pairs) :] = frozen
    return out


def correlation_feature_names(channels: list[str]) -> list[str]:
    pair_names = [f"corr_{channels[i]}_{channels[j]}" for i, j in _upper_tri_pairs(channels)]
    frozen_names = [f"frozen_frac_{ch}" for ch in channels]
    return pair_names + frozen_names
