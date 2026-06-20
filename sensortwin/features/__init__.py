"""Feature engineering for classical baselines (roadmap v0.2).

Three feature families, combined by :func:`build_feature_matrix` into one named matrix:
  * ``statistical`` — per-channel mean/std/min/max/slope/lag-1 autocorrelation,
  * ``spectral`` — per-channel FFT band energies, spectral entropy, centroid,
  * ``correlations`` — cross-channel Pearson correlations + frozen-fraction (missingness proxy).
"""

from sensortwin.features.build import build_feature_matrix
from sensortwin.features.correlations import correlation_features
from sensortwin.features.spectral import spectral_features
from sensortwin.features.statistical import statistical_features

__all__ = [
    "build_feature_matrix",
    "correlation_features",
    "spectral_features",
    "statistical_features",
]
