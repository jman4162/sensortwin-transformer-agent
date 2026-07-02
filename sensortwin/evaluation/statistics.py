"""Significance testing over per-seed metric runs (roadmap v0.7).

The agentic runner must not "claim improvements without statistical evidence" (spec guardrail
§13.5(5)). A single-seed macro-F1 delta is noise; this module turns several per-seed scores into a
defensible verdict. ``compare_seeds`` pairs a variant's per-seed scores against the baseline's and
reports the mean delta with a paired-t confidence interval, a paired t-test p-value, and the paired
Cohen's d. ``significant`` is flagged only when the CI excludes zero *and* the test agrees — so a
reviewer can say "improvement" only when the evidence supports it.

Small-sample honesty rules:
  * The CI is a paired t-interval, not a bootstrap — resampling 3-5 paired differences produces
    intervals that look precise and are not.
  * ``mean_std`` reports the sample standard deviation (ddof=1); with one seed the spread is
    undefined and returned as NaN rather than 0.
  * Identical paired differences (zero variance) fall back to an exact two-sided sign test,
    p = 2 * 0.5**n — never p = 0. At n < 6 that can never clear α = 0.05, so a run of identical
    tiny deltas cannot manufacture significance.
  * ``holm_bonferroni`` provides step-down family-wise error control for multiple comparisons
    (per-class deltas, agent ablation families).

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
    cohen_d: float | None = None  # paired effect size: delta / std(diffs, ddof=1)


def mean_std(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1) of a per-seed score list.

    With fewer than two seeds the spread is undefined and returned as NaN — format it as "n/a"
    rather than pretending a single run has zero variance.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1))


def _t_ci(diffs: np.ndarray, alpha: float) -> tuple[float, float]:
    """Two-sided paired t-interval for the mean difference."""
    n = len(diffs)
    se = diffs.std(ddof=1) / np.sqrt(n)
    half = float(stats.t.ppf(1 - alpha / 2, df=n - 1)) * se
    m = float(diffs.mean())
    return m - half, m + half


def holm_bonferroni(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> list[bool]:
    """Holm step-down rejection decisions (family-wise error control at ``alpha``).

    Sort p-values ascending; the k-th smallest is compared against ``alpha / (m - k)``; the first
    failure stops all further rejections. Returns a reject flag per input position.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    reject = np.zeros(m, dtype=bool)
    order = np.argsort(p)
    for k, idx in enumerate(order):
        if np.isnan(p[idx]) or p[idx] > alpha / (m - k):
            break
        reject[idx] = True
    return reject.tolist()


def compare_seeds(
    base: list[float] | np.ndarray,
    variant: list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> SeedComparison:
    """Paired comparison of per-seed scores (same seeds, aligned by index).

    ``significant`` is True only when the paired t-interval for the mean difference excludes 0 and
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
        # Zero-variance paired differences: the t-test is undefined. Fall back to an exact
        # two-sided sign test — n identical nonzero deltas give p = 2 * 0.5**n, which cannot
        # reach α = 0.05 below n = 6. Never p = 0, and no effect size (d is undefined at s = 0).
        nonzero = not np.isclose(delta, 0.0)
        p_value = min(2.0 * 0.5 ** len(diffs), 1.0) if nonzero else 1.0
        return SeedComparison(
            delta,
            float(b.mean()),
            float(v.mean()),
            delta,
            delta,
            p_value,
            bool(nonzero and p_value < alpha),
            len(diffs),
        )

    ci_low, ci_high = _t_ci(diffs, alpha)
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
        cohen_d=float(delta / diffs.std(ddof=1)),
    )
