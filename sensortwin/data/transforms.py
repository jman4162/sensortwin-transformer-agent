"""Signal-space transforms for the modeling pipeline (roadmap v0.2).

``ChannelStandardizer`` is the leakage-safe alternative to generating data with
``GenConfig.normalize=True`` (which z-scores using whole-dataset statistics, leaking test-set
moments into training). Fit it on the *train* split only, then transform val/test with the frozen
train statistics — the standard hygiene for honest model comparison.
"""

from __future__ import annotations

import numpy as np


class ChannelStandardizer:
    """Per-channel z-score using statistics estimated on a fit set only.

    Operates on ``[N, C, T]`` arrays; mean/std are per-channel scalars (shape ``[C]``).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> ChannelStandardizer:
        if X.ndim != 3:
            raise ValueError(f"expected [N, C, T], got {X.shape}")
        self.mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = (X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ChannelStandardizer must be fit before transform")
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
