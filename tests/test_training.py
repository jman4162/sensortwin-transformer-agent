"""Smoke test for the training loop (v0.2)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
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


def test_transformer_trains_with_full_recipe():
    """The transformer plus the v0.3 opt-in options (AdamW, label smoothing, cosine warmup,
    augmentation) runs end to end and produces a valid probability distribution."""
    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.augment import build_augment

    tr, va, te, ytr = _split_datasets()
    model = SensorPatchTST(d_model=32, num_layers=2, num_heads=2)  # tiny for speed
    model, history = train_model(
        model,
        tr,
        va,
        epochs=2,
        batch_size=32,
        patience=5,
        weight=class_weights(ytr, len(EVENT_CLASSES)),
        optimizer="adamw",
        weight_decay=0.01,
        label_smoothing=0.1,
        scheduler="cosine_warmup",
        warmup_epochs=1,
        augment=build_augment({"jitter": 0.03, "channel_dropout": 0.1}),
        device="cpu",
    )
    assert all(np.isfinite(history.train_loss))
    proba = predict_proba(model, te, device="cpu")
    assert proba.shape == (20, len(EVENT_CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_default_path_uses_plain_adam_no_schedule():
    """Regression guard: the default loop path is unchanged (plain Adam, no scheduler/smoothing)."""
    from sensortwin.training.loop import _build_optimizer, _build_scheduler

    model = SensorCNN(widths=(8, 16))
    opt = _build_optimizer(model, "adam", 1e-3, 0.0)
    assert isinstance(opt, torch.optim.Adam) and not isinstance(opt, torch.optim.AdamW)
    assert _build_scheduler(opt, None, 0, 10) is None


def test_invalid_optimizer_and_scheduler_raise():
    from sensortwin.training.loop import _build_optimizer, _build_scheduler

    model = SensorCNN(widths=(8, 16))
    with pytest.raises(ValueError, match="optimizer"):
        _build_optimizer(model, "sgd", 1e-3, 0.0)
    opt = _build_optimizer(model, "adam", 1e-3, 0.0)
    with pytest.raises(ValueError, match="scheduler"):
        _build_scheduler(opt, "step", 0, 10)
