"""Tests for temperature scaling (v0.5)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from sensortwin.evaluation.calibration import (  # noqa: E402
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)


def _overconfident_logits(n=2000, n_classes=10, accuracy=0.7, peak=6.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    pred = y.copy()
    wrong = rng.random(n) > accuracy
    pred[wrong] = (y[wrong] + 1 + rng.integers(0, n_classes - 1, size=wrong.sum())) % n_classes
    logits = np.zeros((n, n_classes), dtype=np.float64)
    logits[np.arange(n), pred] = peak  # very peaked -> high confidence, but only ~accuracy correct
    return logits, y


def test_apply_temperature_is_distribution():
    logits, _ = _overconfident_logits()
    proba = apply_temperature(logits, 2.0)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_fit_temperature_positive_and_softens():
    logits, y = _overconfident_logits()
    t = fit_temperature(logits, y)
    assert t > 0
    # An overconfident-but-sometimes-wrong model should be softened (T > 1).
    assert t > 1.0


def test_temperature_scaling_reduces_ece():
    logits, y = _overconfident_logits()
    t = fit_temperature(logits, y)
    ece_before = expected_calibration_error(y, apply_temperature(logits, 1.0))
    ece_after = expected_calibration_error(y, apply_temperature(logits, t))
    assert ece_after < ece_before


def test_temperature_does_not_change_argmax():
    logits, _ = _overconfident_logits()
    a = apply_temperature(logits, 1.0).argmax(1)
    b = apply_temperature(logits, 3.5).argmax(1)
    assert np.array_equal(a, b)  # temperature scaling preserves predictions/accuracy
