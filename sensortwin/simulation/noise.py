"""Stochastic noise and sensor artifacts applied after event injection.

Artifacts are kept separate from events: events encode *what happened in the system*, while
artifacts encode *imperfections in measurement* (Gaussian noise, impulses, quantization,
clipping, calibration offset). Domain randomization over these is what stops a model from
overfitting superficial synthetic fingerprints (spec §8, "synthetic data pitfalls").
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(X: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    return X + rng.normal(0.0, sigma, size=X.shape)


def add_impulse_noise(
    X: np.ndarray, rng: np.random.Generator, rate: float, magnitude: float
) -> np.ndarray:
    """Sparse spike noise: a small fraction of samples get a large additive impulse."""
    mask = rng.random(X.shape) < rate
    impulses = rng.normal(0.0, magnitude, size=X.shape) * mask
    return X + impulses


def quantize(X: np.ndarray, levels: int) -> np.ndarray:
    """Uniform quantization to ``levels`` bins across each channel's observed range."""
    out = np.empty_like(X)
    for c in range(X.shape[0]):
        lo, hi = X[c].min(), X[c].max()
        if hi - lo < 1e-9:
            out[c] = X[c]
            continue
        scaled = (X[c] - lo) / (hi - lo)
        out[c] = np.round(scaled * (levels - 1)) / (levels - 1) * (hi - lo) + lo
    return out


def clip_channels(X: np.ndarray, rng: np.random.Generator, prob: float) -> np.ndarray:
    """Occasionally clip a channel to a tightened range, mimicking sensor saturation."""
    out = X.copy()
    for c in range(X.shape[0]):
        if rng.random() < prob:
            lo, hi = np.quantile(X[c], [0.05, 0.95])
            out[c] = np.clip(X[c], lo, hi)
    return out


def calibration_offset(X: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Per-channel constant additive bias (miscalibrated sensor)."""
    return X + rng.normal(0.0, scale, size=(X.shape[0], 1))


def apply_artifacts(X: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Apply the configured artifact stack in a fixed order."""
    X = add_gaussian_noise(X, rng, cfg.get("gaussian_sigma", 0.02))
    X = add_impulse_noise(X, rng, cfg.get("impulse_rate", 0.002), cfg.get("impulse_magnitude", 0.3))
    X = calibration_offset(X, rng, cfg.get("calibration_scale", 0.01))
    if rng.random() < cfg.get("clip_prob", 0.1):
        X = clip_channels(X, rng, cfg.get("clip_channel_prob", 0.25))
    if cfg.get("quantize_levels"):
        X = quantize(X, int(cfg["quantize_levels"]))
    return X
