"""Feature engineering for classical baselines (roadmap v0.2).

Planned modules: ``statistical.py`` (mean/std/min/max/slope/autocorr/missingness),
``spectral.py`` (band energy), ``correlations.py`` (cross-channel correlation). The
``statistical`` extractor is implemented to support an early feature+LogReg sanity baseline.
"""

from sensortwin.features.statistical import statistical_features

__all__ = ["statistical_features"]
