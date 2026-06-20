"""Array-backed dataset with an optional PyTorch ``Dataset`` adapter.

Torch is an optional extra (``pip install -e ".[ml]"``). The core ``SensorArrayDataset`` works
with pure NumPy so data generation and the feature baselines never require a torch install; the
``as_torch`` adapter is only imported when the modeling layer needs it.
"""

from __future__ import annotations

import numpy as np


class SensorArrayDataset:
    """Lightweight, framework-free view over ``X[N, C, T]`` and ``y[N]`` for a set of indices."""

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray | None = None):
        if X.ndim != 3:
            raise ValueError(f"X must be [N, C, T], got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        self.X = X
        self.y = y
        self.indices = np.arange(len(X)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        j = self.indices[i]
        return self.X[j], int(self.y[j])

    def as_torch(self):  # pragma: no cover - thin adapter, exercised in the ML extra
        """Return a ``torch.utils.data.Dataset`` yielding float32 tensors. Requires the ml extra."""
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as e:  # pragma: no cover
            raise ImportError('PyTorch required: install with `pip install -e ".[ml]"`') from e

        outer = self

        class _TorchDS(Dataset):
            def __len__(self) -> int:
                return len(outer)

            def __getitem__(self, i: int):
                x, label = outer[i]
                return torch.from_numpy(np.ascontiguousarray(x)).float(), label

        return _TorchDS()
