"""Tests for robustness sweeps (v0.5)."""

from __future__ import annotations

import numpy as np

from sensortwin.evaluation.robustness import (
    noise_sweep,
    severity_sweep,
    short_window_sweep,
    truncate_window,
)


def _setup():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8, 128)).astype(np.float32)
    y = rng.integers(0, 10, size=40)
    # A predictor whose accuracy decays as the input is perturbed away from X.
    ref = X.copy()

    def predict(Xx: np.ndarray) -> np.ndarray:
        if Xx.shape == ref.shape and np.allclose(Xx, ref):
            return y  # perfect on clean
        return (y + 1) % 10  # wrong once corrupted

    def mf1(yt, yp):
        from sklearn.metrics import f1_score

        return float(f1_score(yt, yp, labels=range(10), average="macro", zero_division=0))

    return X, y, predict, mf1


def test_truncate_window_shape():
    X = np.zeros((4, 8, 100), dtype=np.float32)
    assert truncate_window(X, 0.5).shape == (4, 8, 50)
    # Floors at min_length (the transformer's default patch_len) so no model sees zero patches.
    assert truncate_window(X, 0.01).shape[2] == 16
    assert truncate_window(X, 0.01, min_length=1).shape[2] == 1
    # min_length never exceeds the available window.
    assert truncate_window(X, 0.01, min_length=500).shape[2] == 100


def test_severity_sweep_keys_and_delta():
    X, y, predict, mf1 = _setup()
    out = severity_sweep(predict, X, y, lambda Xx, s: Xx + s, [0.1, 0.5], macro_f1_fn=mf1)
    assert out["clean_macro_f1"] == 1.0
    assert "level_0.1" in out and "level_0.5" in out
    assert out["worst_delta"] >= 0.0
    assert out["worst_macro_f1"] <= out["clean_macro_f1"]


def test_noise_and_window_sweeps_run():
    X, y, predict, mf1 = _setup()
    ns = noise_sweep(predict, X, y, macro_f1_fn=mf1, sigmas=[0.05, 0.2])
    sw = short_window_sweep(predict, X, y, macro_f1_fn=mf1, keeps=[0.5])
    for out in (ns, sw):
        assert out["clean_macro_f1"] >= out["worst_macro_f1"]
        assert out["worst_delta"] >= 0.0


def test_noise_sweep_deterministic():
    X, y, predict, mf1 = _setup()
    a = noise_sweep(predict, X, y, macro_f1_fn=mf1, sigmas=[0.1], seed=3)
    b = noise_sweep(predict, X, y, macro_f1_fn=mf1, sigmas=[0.1], seed=3)
    assert a == b
