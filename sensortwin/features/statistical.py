"""Per-channel statistical features for the feature baseline (roadmap v0.2).

Produces a fixed-length descriptor per sample so a logistic-regression / random-forest baseline
can run before any deep model exists. Baselines are first-class in this project: the transformer
is only credible if it is measured against strong, simple models.
"""

from __future__ import annotations

import numpy as np

_FEATURE_NAMES = ["mean", "std", "min", "max", "slope", "lag1_autocorr"]


def _slope(x: np.ndarray) -> float:
    t = np.arange(len(x))
    t = t - t.mean()
    denom = (t**2).sum()
    return float((t * (x - x.mean())).sum() / denom) if denom > 0 else 0.0


def _lag1_autocorr(x: np.ndarray) -> float:
    xc = x - x.mean()
    denom = (xc**2).sum()
    return float((xc[:-1] * xc[1:]).sum() / denom) if denom > 1e-12 else 0.0


def statistical_features(X: np.ndarray) -> np.ndarray:
    """Map ``[N, C, T]`` -> ``[N, C * n_feats]`` of per-channel summary statistics."""
    if X.ndim != 3:
        raise ValueError(f"expected [N, C, T], got {X.shape}")
    N, C, _ = X.shape
    feats = np.empty((N, C, len(_FEATURE_NAMES)), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            x = X[n, c]
            feats[n, c] = [
                x.mean(),
                x.std(),
                x.min(),
                x.max(),
                _slope(x),
                _lag1_autocorr(x),
            ]
    return feats.reshape(N, C * len(_FEATURE_NAMES))
