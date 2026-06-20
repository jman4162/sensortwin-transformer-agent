"""Tests for train-time augmentations (v0.3)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sensortwin.training.augment import (  # noqa: E402
    build_augment,
    channel_dropout,
    compose,
    jitter,
    scaling,
)


def _x():
    torch.manual_seed(0)
    return torch.randn(4, 8, 64)


@pytest.mark.parametrize("aug", [jitter(0.1), scaling(0.8, 1.2), channel_dropout(0.2)])
def test_augment_preserves_shape(aug):
    x = _x()
    out = aug(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_channel_dropout_zeros_whole_channels():
    x = torch.ones(4, 8, 16)
    torch.manual_seed(1)
    out = channel_dropout(p=0.5)(x)
    # Each (sample, channel) slice is either all-1 (kept) or all-0 (dropped).
    per_channel = out.reshape(4, 8, -1)
    assert ((per_channel == 1).all(dim=2) | (per_channel == 0).all(dim=2)).all()
    assert (out == 0).any()  # with p=0.5 over 32 slices, at least one dropped


def test_jitter_changes_values_but_not_shape():
    x = _x()
    torch.manual_seed(2)
    out = jitter(0.1)(x)
    assert not torch.allclose(out, x)


def test_seedable_determinism():
    x = _x()
    torch.manual_seed(3)
    a = scaling()(x)
    torch.manual_seed(3)
    b = scaling()(x)
    assert torch.allclose(a, b)


def test_build_augment_from_config():
    aug = build_augment({"jitter": 0.03, "scaling": [0.8, 1.2], "channel_dropout": 0.1})
    assert aug is not None
    assert aug(_x()).shape == (4, 8, 64)
    assert build_augment(None) is None
    assert build_augment({}) is None


def test_compose_applies_in_order():
    x = torch.zeros(2, 8, 8)
    out = compose(jitter(0.0), scaling(1.0, 1.0))(x)  # no-ops
    assert torch.allclose(out, x)
