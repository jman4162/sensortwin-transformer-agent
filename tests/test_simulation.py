"""Tests for the synthetic simulator: shapes, labels, determinism, leakage-free metadata,
and the statistical signatures each event class claims to imprint."""

from __future__ import annotations

import numpy as np

from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.dynamics import base_channels, benign_activity
from sensortwin.simulation.events import (
    CURRENT,
    EVENT_CLASSES,
    N_CHANNELS,
    TEMPERATURE,
    VIBRATION,
    VOLTAGE,
    EventClass,
    inject_correlated_channel_fault,
    inject_current_spike,
    inject_thermal_drift,
)


def _small_cfg(**kw) -> GenConfig:
    kw.setdefault("seed", 7)
    return GenConfig(n_samples=80, T=128, **kw)


def test_shapes_and_dtypes():
    X, y, meta = generate_dataset(_small_cfg())
    assert X.shape == (80, N_CHANNELS, 128)
    assert y.shape == (80,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert meta["n_channels"] == N_CHANNELS


def test_labels_in_range_and_all_classes_reachable():
    X, y, _ = generate_dataset(GenConfig(n_samples=600, T=64, seed=1))
    assert y.min() >= 0
    assert y.max() < len(EVENT_CLASSES)
    # With 600 uniform draws over 10 classes, every class should appear.
    assert set(np.unique(y).tolist()) == set(range(len(EVENT_CLASSES)))


def test_no_nans_or_infs():
    X, _, _ = generate_dataset(_small_cfg())
    assert np.isfinite(X).all()


def test_determinism_same_seed():
    X1, y1, _ = generate_dataset(_small_cfg())
    X2, y2, _ = generate_dataset(_small_cfg())
    assert np.array_equal(y1, y2)
    assert np.allclose(X1, X2)


def test_different_seed_changes_data():
    X1, _, _ = generate_dataset(_small_cfg(seed=7))
    X2, _, _ = generate_dataset(_small_cfg(seed=8))
    assert not np.allclose(X1, X2)


def test_metadata_matches_labels():
    _, y, meta = generate_dataset(_small_cfg())
    events = meta["events"]
    assert len(events) == len(y)
    for i, ev in enumerate(events):
        assert ev["event_class"] == int(y[i])
        assert ev["name"] == EVENT_CLASSES[int(y[i])]


def test_normalization_stats_present():
    _, _, meta = generate_dataset(_small_cfg(normalize=True))
    stats = meta["channel_stats"]
    assert stats is not None
    assert len(stats["mean"]) == N_CHANNELS
    assert len(stats["std"]) == N_CHANNELS


def test_normalize_defaults_off():
    # Whole-dataset z-scoring leaks test statistics; the library default must not do it.
    assert GenConfig().normalize is False
    _, _, meta = generate_dataset(_small_cfg())
    assert meta["channel_stats"] is None


# --- Event-class statistical signatures ---------------------------------------------------------


def test_thermal_drift_has_upward_trend():
    rng = np.random.default_rng(0)
    for _ in range(10):
        X = base_channels(256, rng)
        before = X[TEMPERATURE[0]].copy()
        meta = inject_thermal_drift(X, rng)
        seg = slice(meta.start, meta.start + meta.duration)
        added = X[TEMPERATURE[0], seg] - before[seg]
        # The injected ramp ends near the sampled severity and rises monotonically.
        assert added[-1] > added[0]
        assert abs(added[-1] - meta.severity) < 1e-9


def test_current_spike_amplitude_matches_severity():
    rng = np.random.default_rng(1)
    for _ in range(10):
        X = base_channels(256, rng)
        before = X.copy()
        meta = inject_current_spike(X, rng)
        c = meta.affected_channels[0]
        assert abs(np.max(X[c] - before[c]) - meta.severity) < 1e-9
        assert c in CURRENT


def test_correlated_fault_preserves_segment_marginals():
    rng = np.random.default_rng(2)
    for _ in range(10):
        X = base_channels(256, rng)
        before = X.copy()
        meta = inject_correlated_channel_fault(X, rng)
        seg = slice(meta.start, meta.start + meta.duration)
        v_before, v_after = before[VOLTAGE[0], seg], X[VOLTAGE[0], seg]
        # Marginal-preserving by construction: same segment mean and std ...
        assert abs(v_after.mean() - v_before.mean()) < 1e-6
        assert abs(v_after.std() - v_before.std()) < 1e-6
        # ... but the voltage<->current relationship rotates toward positive correlation.
        i_seg = X[CURRENT[0], seg]
        corr_before = np.corrcoef(v_before, i_seg)[0, 1]
        corr_after = np.corrcoef(v_after, i_seg)[0, 1]
        assert corr_after > corr_before


def test_severity_scale_shrinks_events():
    _, y1, m1 = generate_dataset(_small_cfg())
    _, y2, m2 = generate_dataset(_small_cfg(severity_scale=0.5))
    spike = EventClass.current_spike
    sev1 = [e["severity"] for e, c in zip(m1["events"], y1, strict=True) if c == spike]
    sev2 = [e["severity"] for e, c in zip(m2["events"], y2, strict=True) if c == spike]
    assert sev1 and sev2
    assert np.mean(sev2) < np.mean(sev1)


def test_benign_activity_alters_signal_within_bounds():
    rng = np.random.default_rng(3)
    X = np.zeros((N_CHANNELS, 512))
    benign_activity(X, rng, rate=2.0, max_events=2)  # p=1 per slot: both events fire
    assert np.abs(X).max() > 0.0, "benign layer injected nothing"
    # Benign amplitudes stay below true event severities (>= 0.2 at severity_scale=1).
    assert np.abs(X).max() < 0.2 + 1e-9


def test_vibration_channel_carries_baseline_energy():
    # Vibration must not be a giveaway channel that only one class writes to.
    X, y, _ = generate_dataset(GenConfig(n_samples=400, T=128, seed=5))
    energy = (X[:, VIBRATION] ** 2).mean(axis=1)
    non_osc = energy[y != EventClass.oscillatory_instability]
    assert non_osc.mean() > 1e-3
