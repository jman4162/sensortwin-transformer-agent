"""Tests for classical baselines + anomaly detector (v0.2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from sensortwin.features import build_feature_matrix  # noqa: E402
from sensortwin.models.baselines import (  # noqa: E402
    CLASSICAL_REGISTRY,
    feature_importance,
    make_isolation_forest,
)
from sensortwin.simulation import GenConfig, generate_dataset  # noqa: E402
from sensortwin.simulation.events import EVENT_CLASSES  # noqa: E402


def _features():
    X, y, _ = generate_dataset(GenConfig(n_samples=200, T=64, seed=4))
    F, names = build_feature_matrix(X)
    return F[:150], y[:150], F[150:], y[150:], names


def test_registry_has_three_classical():
    assert set(CLASSICAL_REGISTRY) == {"logreg", "random_forest", "xgboost"}


@pytest.mark.parametrize("name", list(CLASSICAL_REGISTRY))
def test_classical_fit_predict(name):
    F_tr, y_tr, F_te, y_te, _ = _features()
    model = CLASSICAL_REGISTRY[name]()
    model.fit(F_tr, y_tr)
    pred = model.predict(F_te)
    proba = model.predict_proba(F_te)
    assert pred.shape == (len(y_te),)
    assert proba.shape == (len(y_te), len(EVENT_CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_feature_importance_for_trees():
    F_tr, y_tr, _, _, names = _features()
    model = CLASSICAL_REGISTRY["random_forest"]()
    model.fit(F_tr, y_tr)
    imp = feature_importance(model, names, top_k=10)
    assert len(imp) == 10
    assert all(n in names for n, _ in imp)
    # Sorted descending by importance.
    vals = [v for _, v in imp]
    assert vals == sorted(vals, reverse=True)


def test_isolation_forest_scores_anomalies():
    F_tr, y_tr, F_te, y_te, _ = _features()
    det = make_isolation_forest()
    det.fit(F_tr[y_tr == 0])
    scores = det.score_samples(F_te)
    assert scores.shape == (len(y_te),)
