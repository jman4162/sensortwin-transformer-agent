"""Train-time signal augmentations (roadmap v0.3, spec §11.3 augmentations).

The transformer is more overfit-prone than the baselines on synthetic data (spec model-zoo:
"more data-hungry, overfitting risk"). These light, label-preserving augmentations add diversity.
They operate on standardized ``[B, C, T]`` tensors and are applied to the **training** batch only
(never val/test) via the ``augment`` hook in ``sensortwin.training.loop.train_model``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Augment = Callable[[torch.Tensor], torch.Tensor]


def jitter(sigma: float = 0.03) -> Augment:
    """Add Gaussian noise (sigma in standardized units)."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * sigma

    return fn


def scaling(low: float = 0.8, high: float = 1.2) -> Augment:
    """Scale each (sample, channel) by a random factor in ``[low, high]``."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        factor = torch.empty(b, c, 1, device=x.device).uniform_(low, high)
        return x * factor

    return fn


def channel_dropout(p: float = 0.1) -> Augment:
    """Zero each channel independently with probability ``p`` (simulated sensor loss)."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        keep = (torch.rand(b, c, 1, device=x.device) >= p).float()
        return x * keep

    return fn


def compose(*augments: Augment) -> Augment:
    """Apply augmentations in sequence."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        for a in augments:
            x = a(x)
        return x

    return fn


def build_augment(cfg: dict | None) -> Augment | None:
    """Build an augmentation pipeline from a config block, e.g.::

        augment:
          jitter: 0.03
          scaling: [0.8, 1.2]
          channel_dropout: 0.1

    Returns None when ``cfg`` is empty/None (no augmentation).
    """
    if not cfg:
        return None
    stages: list[Augment] = []
    if "jitter" in cfg:
        stages.append(jitter(float(cfg["jitter"])))
    if "scaling" in cfg:
        lo, hi = cfg["scaling"]
        stages.append(scaling(float(lo), float(hi)))
    if "channel_dropout" in cfg:
        stages.append(channel_dropout(float(cfg["channel_dropout"])))
    return compose(*stages) if stages else None
