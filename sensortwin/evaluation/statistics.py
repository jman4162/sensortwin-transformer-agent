"""Significance testing over per-seed metric runs (roadmap v0.7).

The agentic runner must not "claim improvements without statistical evidence" (spec guardrail
§13.5(5)). A single-seed macro-F1 delta is noise; this module turns several per-seed scores into a
defensible verdict. ``compare_seeds`` pairs a variant's per-seed scores against the baseline's,
reports the mean delta with a bootstrap confidence interval and a paired t-test p-value, and only
flags ``significant`` when the CI excludes zero *and* the test agrees — so a reviewer can say
"improvement" only when the evidence supports it.

Pure NumPy/SciPy (both core deps); no torch, so the reviewer's claim logic is unit-testable without
the ML extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SeedComparison:
    delta: float  # mean(variant) - mean(base)
    base_mean: float
    variant_mean: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    n_seeds: int


def mean_std(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Mean and (population) standard deviation of a per-seed score list."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def _bootstrap_ci(
    diffs: np.ndarray, *, alpha: float, n_boot: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences (robust at tiny seed counts)."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compare_seeds(
    base: list[float] | np.ndarray,
    variant: list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> SeedComparison:
    """Paired comparison of per-seed scores (same seeds, aligned by index).

    ``significant`` is True only when the bootstrap CI for the mean paired difference excludes 0 and
    the paired t-test p-value < ``alpha`` — a deliberately conservative AND so the agent does not
    overclaim. With <2 seeds no test is possible, so the result is reported as not significant.
    """
    b = np.asarray(base, dtype=float)
    v = np.asarray(variant, dtype=float)
    if b.shape != v.shape:
        raise ValueError(f"base/variant must align by seed: {b.shape} vs {v.shape}")
    diffs = v - b
    delta = float(diffs.mean())

    if len(diffs) < 2:
        # One seed: report the delta but claim nothing (no evidence of consistency).
        return SeedComparison(delta, float(b.mean()), float(v.mean()), delta, delta, 1.0, False, 1)

    if np.allclose(diffs, diffs[0]):
        # Zero-variance paired differences: the t-test is undefined. A consistent nonzero gap is
        # significant; an all-zero gap is not.
        sig = not np.isclose(delta, 0.0)
        p_value = 0.0 if sig else 1.0
        return SeedComparison(
            delta, float(b.mean()), float(v.mean()), delta, delta, p_value, sig, len(diffs)
        )

    ci_low, ci_high = _bootstrap_ci(diffs, alpha=alpha, n_boot=n_boot, seed=seed)
    p_value = float(stats.ttest_rel(v, b).pvalue)
    significant = bool((ci_low > 0 or ci_high < 0) and p_value < alpha)
    return SeedComparison(
        delta=delta,
        base_mean=float(b.mean()),
        variant_mean=float(v.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        significant=significant,
        n_seeds=len(diffs),
    )
