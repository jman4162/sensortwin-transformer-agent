"""Calibration metrics (roadmap v0.2 scaffold, full study in v0.5; spec §12.3).

A scientific model should know when it does not know. These quantify the gap between a model's
confidence and its accuracy:
  * **ECE** — expected calibration error via equal-width confidence bins,
  * **Brier score** — mean squared error between predicted probability vectors and one-hot truth,
  * **reliability curve** — per-bin (confidence, accuracy) for the reliability diagram.

sklearn has no multiclass ECE, so we implement the standard top-label binning directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE: sum over bins of |accuracy - confidence| weighted by bin population."""
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Include the left edge on the first bin so confidence==lo isn't dropped.
        mask = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            mask |= confidences == 0.0
        if mask.sum() == 0:
            continue
        ece += abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.sum() / n
    return float(ece)


def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs one-hot labels (lower is better)."""
    n, n_classes = y_proba.shape
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(n), y_true] = 1.0
    return float(((y_proba - onehot) ** 2).sum(axis=1).mean())


@dataclass
class ReliabilityCurve:
    bin_confidence: np.ndarray  # mean predicted confidence per bin
    bin_accuracy: np.ndarray  # empirical accuracy per bin
    bin_count: np.ndarray  # number of samples per bin


def reliability_curve(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> ReliabilityCurve:
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            mask |= confidences == 0.0
        count.append(int(mask.sum()))
        conf.append(float(confidences[mask].mean()) if mask.any() else float("nan"))
        acc.append(float(accuracies[mask].mean()) if mask.any() else float("nan"))
    return ReliabilityCurve(np.array(conf), np.array(acc), np.array(count))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled probabilities: ``softmax(logits / T)`` (Guo et al. 2017)."""
    return _softmax(logits / temperature)


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit a single temperature T on a validation set by minimizing NLL (Guo et al. 2017).

    Post-hoc calibration: divide logits by T before softmax. T>1 softens overconfident predictions
    (the common failure under distribution shift, Ovadia et al. 2019). T is fit on val and then
    applied to test; it does not change the argmax, so accuracy is unchanged.
    """
    from scipy.optimize import minimize_scalar

    n = len(y_val)

    def nll(temp: float) -> float:
        proba = apply_temperature(logits_val, temp)
        return float(-np.log(proba[np.arange(n), y_val] + 1e-12).mean())

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)
