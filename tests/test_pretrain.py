"""Tests for masked-patch pretraining components (v0.4)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sensortwin.data.dataset import SensorArrayDataset  # noqa: E402
from sensortwin.models.transformer import SensorPatchTST  # noqa: E402
from sensortwin.simulation import GenConfig, generate_dataset  # noqa: E402
from sensortwin.simulation.events import N_CHANNELS  # noqa: E402
from sensortwin.training.pretrain import (  # noqa: E402
    _make_patch_mask,
    freeze_encoder,
    pretrain_model,
    transfer_encoder,
)

_ENCODER_PREFIXES = ("patch_proj", "channel_embedding", "encoder", "mask_token")


def _tiny():
    return SensorPatchTST(d_model=32, num_layers=2, num_heads=2)


def test_embed_encode_shapes_and_mask_token():
    m = _tiny()
    x = torch.randn(4, N_CHANNELS, 128)
    tokens, targets = m.embed(x)
    # 8 channels x 15 patches = 120 tokens; targets carry the raw patch values.
    assert tokens.shape == (4, 120, 32)
    assert targets.shape == (4, 120, m.patch_len)
    assert m.encode(tokens).shape == (4, 120, 32)
    assert m.mask_token.shape == (1, 1, 32)


def test_patch_mask_ratio_and_determinism():
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    m1 = _make_patch_mask(4, 8, 15, 0.4, g1)
    m2 = _make_patch_mask(4, 8, 15, 0.4, g2)
    assert m1.shape == (4, 8 * 15)
    assert torch.equal(m1, m2)  # seed-deterministic
    # ~40% masked, and per channel (reshape back to [B, C, N]) exactly round(15*0.4)=6.
    per_channel = m1.reshape(4, 8, 15).sum(dim=2)
    assert (per_channel == 6).all()


def test_pretrain_runs_and_returns_head():
    X, y, _ = generate_dataset(GenConfig(n_samples=40, T=128, seed=1, normalize=True))
    ds = SensorArrayDataset(X, y).as_torch()
    model, head, hist = pretrain_model(
        _tiny(), ds, ds, epochs=2, batch_size=16, warmup_epochs=0, device="cpu"
    )
    assert isinstance(head, torch.nn.Linear)
    assert head.out_features == model.patch_len
    assert all(np.isfinite(hist.val_loss))


def test_freeze_encoder_blocks_encoder_keeps_head():
    m = freeze_encoder(_tiny())
    enc = [p.requires_grad for n, p in m.named_parameters() if n.startswith(_ENCODER_PREFIXES)]
    head = [p.requires_grad for n, p in m.named_parameters() if n.startswith("head")]
    assert not any(enc)  # encoder frozen
    assert all(head)  # head trainable


def test_transfer_encoder_copies_encoder_not_head():
    src = _tiny()
    with torch.no_grad():  # perturb so equality is meaningful
        src.patch_proj.weight.add_(1.0)
    dst = _tiny()
    transfer_encoder(src.state_dict(), dst)
    assert torch.equal(src.patch_proj.weight, dst.patch_proj.weight)  # encoder transferred
    # Classification head is left at the destination's fresh init (not copied from src).
    assert not torch.equal(src.head[1].weight, dst.head[1].weight)


def test_transfer_encoder_across_channel_counts():
    """v0.6 synth(C=8) -> real(C=14): patch encoder transfers, channel embedding re-inits."""
    src = SensorPatchTST(in_channels=8, d_model=32, num_layers=2, num_heads=2)
    with torch.no_grad():
        src.patch_proj.weight.add_(1.0)
    dst = SensorPatchTST(in_channels=14, d_model=32, num_layers=2, num_heads=2)
    before = dst.channel_embedding.clone()

    transfer_encoder(src.state_dict(), dst, strict_channels=False)

    # Channel-agnostic temporal encoder copied across the channel-count change.
    assert torch.equal(src.patch_proj.weight, dst.patch_proj.weight)
    assert torch.equal(src.mask_token, dst.mask_token)
    # Channel embedding kept its C=14 shape and fresh init (cannot transfer from C=8).
    assert dst.channel_embedding.shape == (14, 32)
    assert torch.equal(dst.channel_embedding, before)


def test_transfer_encoder_strict_raises_on_channel_mismatch():
    src = SensorPatchTST(in_channels=8, d_model=32, num_layers=2, num_heads=2)
    dst = SensorPatchTST(in_channels=14, d_model=32, num_layers=2, num_heads=2)
    with pytest.raises(RuntimeError):
        transfer_encoder(src.state_dict(), dst)  # strict_channels=True (default)
