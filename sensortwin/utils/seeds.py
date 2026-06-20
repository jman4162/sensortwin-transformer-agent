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
