"""Tests for the significance helpers behind the agent's improvement claims (v0.7)."""

from __future__ import annotations

import numpy as np

from sensortwin.evaluation.statistics import compare_seeds, holm_bonferroni, mean_std


def test_mean_std_is_sample_std():
    m, s = mean_std([0.4, 0.6])
    assert m == 0.5
    assert np.isclose(s, np.std([0.4, 0.6], ddof=1))


def test_mean_std_single_value_has_undefined_spread():
    m, s = mean_std([0.7])
    assert m == 0.7
    assert np.isnan(s)


def test_compare_seeds_detects_clear_difference():
    base = [0.50, 0.51, 0.49, 0.52, 0.48]
    variant = [0.70, 0.72, 0.68, 0.73, 0.69]  # consistently ~0.2 higher, with spread
    cmp = compare_seeds(base, variant)
    assert cmp.delta > 0.15
    assert cmp.significant
    assert cmp.cohen_d is not None and cmp.cohen_d > 2.0
    assert cmp.ci_low < cmp.delta < cmp.ci_high


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


def test_zero_variance_small_n_cannot_manufacture_significance():
    # Identical nonzero deltas at n=3 used to return p=0.0/significant; the sign-test fallback
    # gives p = 2 * 0.5**3 = 0.25 — reported, but never significant below n=6.
    cmp = compare_seeds([0.50, 0.51, 0.49], [0.70, 0.71, 0.69])
    assert np.isclose(cmp.delta, 0.2)
    assert np.isclose(cmp.p_value, 0.25)
    assert not cmp.significant
    # At n=6 the same consistency is real evidence: p = 2 * 0.5**6 ~ 0.031.
    cmp6 = compare_seeds([0.5] * 6, [0.7] * 6)
    assert np.isclose(cmp6.p_value, 0.03125)
    assert cmp6.significant


def test_holm_bonferroni_rejects_stepwise():
    # m=3: smallest p tested at alpha/3, next at alpha/2, last at alpha.
    assert holm_bonferroni([0.01, 0.02, 0.04], alpha=0.05) == [True, True, True]
    assert holm_bonferroni([0.01, 0.03, 0.20], alpha=0.05) == [True, False, False]
    assert holm_bonferroni([0.30, 0.40, 0.50], alpha=0.05) == [False, False, False]


def test_holm_bonferroni_handles_nan():
    assert holm_bonferroni([0.001, float("nan")], alpha=0.05) == [True, False]
