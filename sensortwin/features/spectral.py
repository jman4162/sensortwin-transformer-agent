"""Spectral (frequency-domain) features for the classical baseline (roadmap v0.2).

Some event classes are most visible in the frequency domain: ``oscillatory_instability`` injects a
growing sinusoid, ``current_spike`` is broadband, and steady operation concentrates energy at low
frequency. We summarize each channel's one-sided power spectrum with band energies plus spectral
entropy (how spread-out the spectrum is). Pure NumPy ``rfft`` — no extra dependency.
"""

from __future__ import annotations

import numpy as np

# Fractional frequency-band edges (relative to Nyquist), so features are T-independent.
_BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]
_FEATURE_NAMES = [f"band_{lo:g}_{hi:g}" for lo, hi in _BANDS] + [
    "spectral_entropy",
    "spectral_centroid",
]


def _channel_spectral(x: np.ndarray) -> list[float]:
    # One-sided power spectrum, excluding the DC bin so band energies are drift-invariant.
    power = np.abs(np.fft.rfft(x)) ** 2
    power = power[1:]
    total = power.sum()
    n = len(power)
    if total <= 1e-12 or n == 0:
        return [0.0] * len(_BANDS) + [0.0, 0.0]

    freqs = np.linspace(0.0, 1.0, n)  # normalized 0..1 (Nyquist)
    feats = [float(power[(freqs >= lo) & (freqs < hi)].sum() / total) for lo, hi in _BANDS]

    p = power / total
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(n))  # normalized to [0, 1]
    centroid = float((freqs * p).sum())
    return feats + [entropy, centroid]


def spectral_features(X: np.ndarray) -> np.ndarray:
    """Map ``[N, C, T]`` -> ``[N, C * n_spectral]`` of per-channel spectral summaries."""
    if X.ndim != 3:
        raise ValueError(f"expected [N, C, T], got {X.shape}")
    N, C, _ = X.shape
    feats = np.empty((N, C, len(_FEATURE_NAMES)), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            feats[n, c] = _channel_spectral(X[n, c])
    return feats.reshape(N, C * len(_FEATURE_NAMES))


def spectral_feature_names(channels: list[str]) -> list[str]:
    return [f"{ch}_{f}" for ch in channels for f in _FEATURE_NAMES]
