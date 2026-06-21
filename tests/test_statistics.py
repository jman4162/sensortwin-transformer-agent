"""Tests for the significance helpers behind the agent's improvement claims (v0.7)."""

from __future__ import annotations

import numpy as np

from sensortwin.evaluation.statistics import compare_seeds, mean_std


def test_mean_std_basic():
    m, s = mean_std([0.4, 0.6])
    assert m == 0.5
    assert np.isclose(s, 0.1)


def test_compare_seeds_detects_clear_difference():
    base = [0.50, 0.51, 0.49]
    variant = [0.70, 0.71, 0.69]  # consistently ~0.2 higher
    cmp = compare_seeds(base, variant)
    assert cmp.delta > 0.15
    assert cmp.significant


def test_compare_seeds_noise_is_not_significant():
    # Tiny, sign-alternating per-seed differences: mean delta ~0 relative to spread -> no claim.
    base = [0.50, 0.55, 0.45, 0.52, 0.48]
    variant = [0.51, 0.54, 0.46, 0.51, 0.49]
    cmp = compare_seeds(base, variant)
    assert abs(cmp.delta) < 0.01
    assert not cmp.significant


def test_compare_seeds_single_seed_claims_nothing():
    cmp = compare_seeds([0.5], [0.9])
    assert cmp.n_seeds == 1
    assert not cmp.significant


def test_compare_seeds_identical_is_not_significant():
    cmp = compare_seeds([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert cmp.delta == 0.0
    assert not cmp.significant
