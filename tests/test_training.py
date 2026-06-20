"""Smoke test for the training loop (v0.2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sensortwin.data.dataset import SensorArrayDataset  # noqa: E402
from sensortwin.models.cnn import SensorCNN  # noqa: E402
from sensortwin.simulation import GenConfig, generate_dataset  # noqa: E402
from sensortwin.simulation.events import EVENT_CLASSES  # noqa: E402
from sensortwin.training.loop import class_weights, predict_proba, train_model  # noqa: E402


def _split_datasets():
    X, y, _ = generate_dataset(GenConfig(n_samples=120, T=64, seed=1, normalize=True))
    tr = SensorArrayDataset(X[:80], y[:80]).as_torch()
    va = SensorArrayDataset(X[80:100], y[80:100]).as_torch()
    te = SensorArrayDataset(X[100:], y[100:]).as_torch()
    return tr, va, te, y[:80]


def test_train_loop_runs_and_predicts():
    tr, va, te, ytr = _split_datasets()
    model = SensorCNN(widths=(8, 16))  # tiny for speed
    model, history = train_model(
        model,
        tr,
        va,
        epochs=2,
        batch_size=32,
        patience=5,
        weight=class_weights(ytr, len(EVENT_CLASSES)),
        device="cpu",
    )
    assert len(history.train_loss) >= 1
    assert all(np.isfinite(history.train_loss))

    proba = predict_proba(model, te, device="cpu")
    assert proba.shape == (20, len(EVENT_CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)  # softmax rows sum to 1


def test_class_weights_inverse_frequency():
    y = np.array([0, 0, 0, 0, 1])  # class 0 common, class 1 rare
    w = class_weights(y, 2).numpy()
    assert w[1] > w[0]  # rarer class gets larger weight
