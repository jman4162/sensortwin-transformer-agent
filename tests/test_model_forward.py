"""Forward-pass / shape tests for the deep baselines (v0.2)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sensortwin.models.cnn import SensorCNN  # noqa: E402
from sensortwin.models.lstm import SensorLSTM  # noqa: E402
from sensortwin.simulation.events import EVENT_CLASSES, N_CHANNELS  # noqa: E402


@pytest.mark.parametrize("factory", [SensorCNN, SensorLSTM])
def test_forward_shape(factory):
    model = factory()
    x = torch.randn(4, N_CHANNELS, 128)
    out = model(x)
    assert out.shape == (4, len(EVENT_CLASSES))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("factory", [SensorCNN, SensorLSTM])
def test_param_budget(factory):
    n_params = sum(p.numel() for p in factory().parameters())
    assert n_params < 500_000, f"{factory.__name__} has {n_params} params (>500k)"


def test_cnn_handles_variable_length():
    model = SensorCNN()
    for T in (64, 256, 512):
        out = model(torch.randn(2, N_CHANNELS, T))
        assert out.shape == (2, len(EVENT_CLASSES))
