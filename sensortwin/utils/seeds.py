"""Deterministic seeding helpers.

Reproducibility is a non-negotiable project principle: every dataset and experiment must be
fully determined by its seed. We use ``numpy.random.SeedSequence`` to spawn independent,
collision-resistant per-sample streams from a single master seed, so generating sample ``i`` is
reproducible regardless of how many other samples were drawn.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh NumPy generator for a master seed."""
    return np.random.default_rng(seed)


def spawn_child_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Spawn ``n`` independent generators from a master ``seed``.

    Each child stream is statistically independent of the others, so per-sample generation can
    run in any order (or in parallel) and still reproduce bit-for-bit.
    """
    seq = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in seq.spawn(n)]


def set_torch_seed(seed: int) -> None:
    """Seed Python/NumPy/torch for reproducible model training.

    Guarded import: the core install has no torch, and seeding the RNGs that *do* exist should
    still work. Sets deterministic cuDNN so GPU runs are reproducible at a small speed cost.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
