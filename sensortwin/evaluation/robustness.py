"""Robustness probes (roadmap v0.2 scaffold; full suite in v0.5; spec §12.2).

v0.2 ships one probe — **missing channels** — to populate the headline robustness column of the
results table and to surface each model's dependence on individual sensors. The reported quantity
is the macro-F1 *delta* from the clean baseline (spec §12.2: ``metric_clean - metric_corrupted``).
Noise, domain-shift, short-window, and rare-event probes follow the same shape in v0.5.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# A predictor maps raw signals [N, C, T] -> hard predictions [N].
PredictFn = Callable[[np.ndarray], np.ndarray]


def zero_channels(X: np.ndarray, channels: list[int]) -> np.ndarray:
    """Return a copy of ``X`` with the given channel indices zeroed (simulated sensor loss)."""
    out = X.copy()
    out[:, channels, :] = 0.0
    return out


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
