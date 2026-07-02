"""Robustness probes (spec §12.2).

Every probe reports macro-F1 *degradation* from the clean baseline
(``performance_delta = metric_clean - metric_corrupted``). v0.5 adds Gaussian-noise and
short-window severity sweeps alongside the v0.2 missing-channel probe; domain shift is handled by
generating a shifted regime (see ``scripts/robustness_report.py``), not as a corruption function.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sensortwin.simulation.noise import add_gaussian_noise

# A predictor maps raw signals [N, C, T] -> hard predictions [N].
PredictFn = Callable[[np.ndarray], np.ndarray]
# A corruption maps [N, C, T] -> [N, C, T'] at a given severity level.
CorruptFn = Callable[[np.ndarray, float], np.ndarray]


def zero_channels(X: np.ndarray, channels: list[int]) -> np.ndarray:
    """Return a copy of ``X`` with the given channel indices zeroed (simulated sensor loss)."""
    out = X.copy()
    out[:, channels, :] = 0.0
    return out


def truncate_window(X: np.ndarray, keep_fraction: float, *, min_length: int = 16) -> np.ndarray:
    """Keep only the first ``keep_fraction`` of the time axis (shorter observation window).

    The patch-transformer and CNN both accept variable ``T`` (patchify / global pooling), so a
    truncated window is a valid input rather than a padded one. ``min_length`` floors the kept
    window at the transformer's default ``patch_len`` so a sweep cannot produce an input with
    zero patches.
    """
    keep = max(min_length, int(round(keep_fraction * X.shape[2])))
    keep = min(keep, X.shape[2])
    return X[:, :, :keep]


def severity_sweep(
    predict_fn: PredictFn,
    X: np.ndarray,
    y: np.ndarray,
    corrupt_fn: CorruptFn,
    severities: list[float],
    *,
    macro_f1_fn: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, float]:
    """ImageNet-C-style macro-F1 vs corruption severity (Hendrycks & Dietterich 2019).

    ``corrupt_fn(X, level)`` applies the corruption; ``severities`` are the levels to sweep. Returns
    the clean score, per-level scores (``level_<s>``), and the worst-case degradation.
    """
    clean = macro_f1_fn(y, predict_fn(X))
    per_level = {f"level_{s:g}": macro_f1_fn(y, predict_fn(corrupt_fn(X, s))) for s in severities}
    worst = min(per_level.values()) if per_level else clean
    return {
        "clean_macro_f1": float(clean),
        **{k: float(v) for k, v in per_level.items()},
        "worst_macro_f1": float(worst),
        "worst_delta": float(clean - worst),
    }


def noise_sweep(
    predict_fn: PredictFn,
    X: np.ndarray,
    y: np.ndarray,
    *,
    macro_f1_fn: Callable[[np.ndarray, np.ndarray], float],
    sigmas: list[float] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Additive Gaussian-noise severity sweep (sigma in standardized-signal units)."""
    sigmas = sigmas if sigmas is not None else [0.01, 0.02, 0.05, 0.1, 0.2]
    rng = np.random.default_rng(seed)
    return severity_sweep(
        predict_fn,
        X,
        y,
        lambda Xx, s: add_gaussian_noise(Xx, rng, s),
        sigmas,
        macro_f1_fn=macro_f1_fn,
    )


def short_window_sweep(
    predict_fn: PredictFn,
    X: np.ndarray,
    y: np.ndarray,
    *,
    macro_f1_fn: Callable[[np.ndarray, np.ndarray], float],
    keeps: list[float] | None = None,
) -> dict[str, float]:
    """Shorter-observation-window sweep (fraction of the time axis retained)."""
    keeps = keeps if keeps is not None else [0.75, 0.5, 0.25]
    return severity_sweep(
        predict_fn, X, y, lambda Xx, k: truncate_window(Xx, k), keeps, macro_f1_fn=macro_f1_fn
    )


def missing_channel_sweep(
    predict_fn: PredictFn,
    X: np.ndarray,
    y: np.ndarray,
    *,
    macro_f1_fn: Callable[[np.ndarray, np.ndarray], float],
    channels: list[int] | None = None,
) -> dict[str, float]:
    """Macro-F1 when each channel is individually zeroed, plus the worst-case delta vs clean.

    ``macro_f1_fn(y_true, y_pred) -> float`` is injected to avoid a hard dependency on the metrics
    module's full signature. Returns per-channel scores and the largest degradation observed.
    """
    n_channels = X.shape[1]
    channels = channels if channels is not None else list(range(n_channels))

    clean = macro_f1_fn(y, predict_fn(X))
    per_channel = {
        f"drop_ch{c}": macro_f1_fn(y, predict_fn(zero_channels(X, [c]))) for c in channels
    }
    worst = min(per_channel.values()) if per_channel else clean
    return {
        "clean_macro_f1": float(clean),
        **{k: float(v) for k, v in per_channel.items()},
        "worst_macro_f1": float(worst),
        "worst_delta": float(clean - worst),
    }
