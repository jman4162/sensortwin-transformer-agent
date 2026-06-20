"""Forward-pass / shape tests for the deep baselines (v0.2)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sensortwin.models.cnn import SensorCNN  # noqa: E402
from sensortwin.models.lstm import SensorLSTM  # noqa: E402
from sensortwin.models.transformer import SensorPatchTST  # noqa: E402
from sensortwin.simulation.events import EVENT_CLASSES, N_CHANNELS  # noqa: E402


@pytest.mark.parametrize("factory", [SensorCNN, SensorLSTM, SensorPatchTST])
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


def test_transformer_param_budget():
    # The transformer is larger than the conv/recurrent baselines but stays Colab-small.
    n_params = sum(p.numel() for p in SensorPatchTST().parameters())
    assert n_params < 1_500_000, f"SensorPatchTST has {n_params} params (>1.5M)"


@pytest.mark.parametrize("factory", [SensorCNN, SensorPatchTST])
def test_handles_variable_length(factory):
    model = factory()
    for T in (64, 256, 512):
        out = model(torch.randn(2, N_CHANNELS, T))
        assert out.shape == (2, len(EVENT_CLASSES))


@pytest.mark.parametrize("channel_embedding", [True, False])
@pytest.mark.parametrize("pooling", ["attention", "mean"])
def test_transformer_ablation_variants(channel_embedding, pooling):
    model = SensorPatchTST(channel_embedding=channel_embedding, pooling=pooling)
    out = model(torch.randn(3, N_CHANNELS, 128))
    assert out.shape == (3, len(EVENT_CLASSES))
    assert torch.isfinite(out).all()


def test_attention_pooling_exposes_weights():
    model = SensorPatchTST(pooling="attention")
    model(torch.randn(3, N_CHANNELS, 128))
    # 8 channels x 15 patches (T=128, patch_len=16, stride=8) = 120 tokens.
    assert model.pool.last_weights.shape == (3, 120)
    # Pooling weights are a distribution over tokens.
    assert torch.allclose(model.pool.last_weights.sum(dim=1), torch.ones(3), atol=1e-4)


def test_invalid_pooling_raises():
    with pytest.raises(ValueError, match="pooling"):
        SensorPatchTST(pooling="max")
