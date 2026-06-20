"""Tests for the metrics + calibration modules (v0.2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sensortwin.evaluation.calibration import (  # noqa: E402
    brier_score_multiclass,
    expected_calibration_error,
)
from sensortwin.evaluation.metrics import classification_metrics, save_metrics  # noqa: E402

CLASSES = [f"c{i}" for i in range(4)]


def test_perfect_prediction_metrics():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    proba = np.eye(4)[y]
    m = classification_metrics(y, y, proba, CLASSES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["macro_auroc"] == 1.0


def test_macro_and_weighted_f1_differ_on_imbalance():
    # Majority class 0 perfect; rare class 3 always wrong -> macro penalized more than weighted.
    y_true = np.array([0] * 90 + [3] * 10)
    y_pred = np.array([0] * 90 + [0] * 10)
    m = classification_metrics(y_true, y_pred, None, CLASSES)
    assert m["weighted_f1"] > m["macro_f1"]
    assert m["per_class"]["c3"]["f1"] == 0.0


def test_auroc_survives_missing_class_in_split():
    # Class 3 never appears; must not crash and should still return a float.
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(4)[y]
    m = classification_metrics(y, y, proba, CLASSES)
    assert m["macro_auroc"] is not None


def test_ece_low_when_calibrated_high_when_overconfident():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 4, size=n)
    # Calibrated-ish: correct with prob = confidence.
    conf = rng.uniform(0.25, 1.0, size=n)
    proba = np.full((n, 4), 0.0)
    correct = rng.random(n) < conf
    for i in range(n):
        cls = y[i] if correct[i] else (y[i] + 1) % 4
        proba[i] = (1 - conf[i]) / 3
        proba[i, cls] = conf[i]
    ece_cal = expected_calibration_error(y, proba)

    # Overconfident: always ~0.99 on a wrong-ish guess.
    over = np.full((n, 4), 0.0025)
    over[np.arange(n), (y + 1) % 4] = 0.9925
    ece_over = expected_calibration_error(y, over)
    assert ece_over > ece_cal


def test_brier_bounds():
    y = np.array([0, 1, 2, 3])
    perfect = np.eye(4)[y]
    assert brier_score_multiclass(y, perfect) == pytest.approx(0.0)
    worst = np.eye(4)[(y + 1) % 4]
    assert brier_score_multiclass(y, worst) == pytest.approx(2.0)


def test_save_metrics_writes_json_and_csv(tmp_path):
    y = np.array([0, 1, 2, 3])
    m = classification_metrics(y, y, np.eye(4)[y], CLASSES)
    json_path = save_metrics(m, tmp_path / "m")
    assert json_path.exists()
    assert json_path.with_suffix(".csv").exists()
