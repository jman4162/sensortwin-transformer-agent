"""Event ontology and injectors.

Each event class is a function that mutates the coupled base signal ``[C, T]`` to imprint a
characteristic signature, returning structured metadata (start, duration, severity, affected
channels). The classes span a deliberate difficulty gradient: some are detectable from local
single-channel features (``current_spike``), others require long-range context
(``slow_degradation``) or cross-channel reasoning (``correlated_channel_fault``). This is what
makes the benchmark discriminate between model inductive biases rather than rewarding any one.

Every injector accepts a ``scale`` factor that multiplies the sampled severity
(``GenConfig.severity_scale``): ``scale < 1`` shrinks events toward the noise floor for the
low-SNR tier of the benchmark without changing any event's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

# Channel index -> generic, scientific (non-workplace-specific) name. Order is load-bearing:
# it must match sensortwin.simulation.dynamics.base_channels.
CHANNELS: list[str] = [
    "voltage_A",
    "voltage_B",
    "current_A",
    "current_B",
    "temperature_core",
    "temperature_surface",
    "vibration_proxy",
    "ambient_load_proxy",
]
N_CHANNELS = len(CHANNELS)

VOLTAGE = (0, 1)
CURRENT = (2, 3)
TEMPERATURE = (4, 5)
VIBRATION = 6
AMBIENT = 7


class EventClass(IntEnum):
    normal = 0
    thermal_drift = 1
    voltage_sag = 2
    current_spike = 3
    sensor_dropout = 4
    oscillatory_instability = 5
    correlated_channel_fault = 6
    regime_shift = 7
    slow_degradation = 8
    compound_fault = 9


EVENT_CLASSES: list[str] = [c.name for c in EventClass]


@dataclass
class EventMeta:
    event_class: int
    name: str
    start: int | None = None
    duration: int | None = None
    severity: float | None = None
    affected_channels: list[int] = field(default_factory=list)
    components: list[str] = field(default_factory=list)  # for compound_fault

    def to_dict(self) -> dict:
        return {
            "event_class": int(self.event_class),
            "name": self.name,
            "start": self.start,
            "duration": self.duration,
            "severity": self.severity,
            "affected_channels": list(self.affected_channels),
            "components": list(self.components),
        }


def _window(T: int, rng: np.random.Generator, min_frac=0.1, max_frac=0.5) -> tuple[int, int]:
    """Sample a (start, duration) window inside [0, T)."""
    duration = int(rng.uniform(min_frac, max_frac) * T)
    duration = max(duration, 1)
    start = int(rng.integers(0, max(T - duration, 1)))
    return start, duration


def _ramp(duration: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, duration)


# --- Injectors -------------------------------------------------------------------------------
# Each takes (X, rng, scale) where X is [C, T] (mutated in place) and returns EventMeta.


def inject_normal(X: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> EventMeta:
    return EventMeta(EventClass.normal, "normal")


def inject_thermal_drift(X: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> EventMeta:
    T = X.shape[1]
    start, duration = _window(T, rng, 0.4, 0.9)
    sev = scale * rng.uniform(0.5, 2.0)
    ramp = _ramp(duration) * sev
    X[TEMPERATURE[0], start : start + duration] += ramp
    X[TEMPERATURE[1], start : start + duration] += 0.6 * ramp  # surface follows core, attenuated
    return EventMeta(
        EventClass.thermal_drift, "thermal_drift", start, duration, sev, list(TEMPERATURE)
    )


def inject_voltage_sag(X: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> EventMeta:
    T = X.shape[1]
    start, duration = _window(T, rng, 0.05, 0.25)
    sev = scale * rng.uniform(0.2, 0.6)
    ch = list(VOLTAGE) if rng.random() < 0.5 else [rng.choice(VOLTAGE)]
    for c in ch:
        X[c, start : start + duration] -= sev
    return EventMeta(EventClass.voltage_sag, "voltage_sag", start, duration, sev, ch)


def inject_current_spike(X: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> EventMeta:
    T = X.shape[1]
    start = int(rng.integers(0, T - 1))
    duration = int(rng.integers(2, max(3, int(0.03 * T))))
    sev = scale * rng.uniform(1.0, 3.0)
    c = int(rng.choice(CURRENT))
    end = min(start + duration, T)
    X[c, start:end] += sev
    return EventMeta(EventClass.current_spike, "current_spike", start, duration, sev, [c])


def inject_sensor_dropout(X: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> EventMeta:
    T = X.shape[1]
    start, duration = _window(T, rng, 0.1, 0.4)
    c = int(rng.integers(0, N_CHANNELS))
    # Frozen sensor: hold the last value before dropout for the segment.
    X[c, start : start + duration] = X[c, max(start - 1, 0)]
    return EventMeta(EventClass.sensor_dropout, "sensor_dropout", start, duration, 1.0, [c])


def inject_oscillatory_instability(
    X: np.ndarray, rng: np.random.Generator, scale: float = 1.0
) -> EventMeta:
    T = X.shape[1]
    start, duration = _window(T, rng, 0.2, 0.6)
    sev = scale * rng.uniform(0.3, 1.0)
    freq = rng.uniform(0.1, 0.4)  # cycles per sample
    growth = np.exp(np.linspace(0, rng.uniform(0.5, 2.0), duration))  # growing amplitude
    osc = sev * growth * np.sin(2 * np.pi * freq * np.arange(duration))
    X[VIBRATION, start : start + duration] += osc
    # Instability bleeds into a current channel too.
    X[CURRENT[0], start : start + duration] += 0.3 * osc
    return EventMeta(
        EventClass.oscillatory_instability,
        "oscillatory_instability",
        start,
        duration,
        sev,
        [VIBRATION, CURRENT[0]],
    )


def inject_correlated_channel_fault(
    X: np.ndarray, rng: np.random.Generator, scale: float = 1.0
) -> EventMeta:
    """Break the normal voltage<->current anti-correlation without changing marginals.

    Detectable only through the cross-channel *relationship*: a single-channel view looks normal.
    The affected voltage segment is rotated toward the current channel's fluctuation, then
    rescaled so its mean and variance match the original segment exactly — a per-channel
    statistic (std, range, spectral energy) cannot separate this class.
    """
    T = X.shape[1]
    start, duration = _window(T, rng, 0.2, 0.6)
    sev = min(scale * rng.uniform(0.3, 0.8), 1.0)  # mixing weight, capped at full rotation
    seg = slice(start, start + duration)
    v = X[VOLTAGE[0], seg]
    v_mean, v_std = v.mean(), v.std()
    i_c = X[CURRENT[0], seg] - X[CURRENT[0], seg].mean()
    i_unit = i_c / (i_c.std() + 1e-9)
    # Mix the voltage fluctuation with a component tracking (not opposing) current, then restore
    # the segment's original mean/std so marginals are unchanged.
    mixed = np.sqrt(max(1.0 - sev**2, 0.0)) * (v - v_mean) + sev * v_std * i_unit
    X[VOLTAGE[0], seg] = v_mean + mixed * (v_std / (mixed.std() + 1e-9))
    return EventMeta(
        EventClass.correlated_channel_fault,
        "correlated_channel_fault",
        start,
        duration,
        sev,
        [VOLTAGE[0], CURRENT[0]],
    )


def inject_regime_shift(X: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> EventMeta:
    T = X.shape[1]
    start = int(rng.integers(int(0.2 * T), int(0.8 * T)))
    sev = scale * rng.uniform(0.3, 1.0)
    # Step change in operating mode: shift load-coupled channels after the change point.
    for c in (*CURRENT, *VOLTAGE):
        X[c, start:] += sev * rng.uniform(-1, 1)
    return EventMeta(
        EventClass.regime_shift, "regime_shift", start, T - start, sev, [*CURRENT, *VOLTAGE]
    )


def inject_slow_degradation(
    X: np.ndarray, rng: np.random.Generator, scale: float = 1.0
) -> EventMeta:
    T = X.shape[1]
    sev = scale * rng.uniform(0.3, 1.0)
    ramp = _ramp(T) * sev
    # Gradual whole-window drift across temperature + a voltage channel.
    X[TEMPERATURE[0]] += ramp
    X[VOLTAGE[1]] -= 0.5 * ramp
    return EventMeta(
        EventClass.slow_degradation, "slow_degradation", 0, T, sev, [TEMPERATURE[0], VOLTAGE[1]]
    )


def inject_compound_fault(X: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> EventMeta:
    """Two overlapping single-events composed together."""
    components = [
        inject_voltage_sag,
        inject_current_spike,
        inject_thermal_drift,
        inject_oscillatory_instability,
    ]
    chosen = rng.choice(len(components), size=2, replace=False)
    affected: list[int] = []
    names: list[str] = []
    for idx in chosen:
        m = components[int(idx)](X, rng, scale)
        affected.extend(m.affected_channels)
        names.append(m.name)
    return EventMeta(
        EventClass.compound_fault,
        "compound_fault",
        severity=None,
        affected_channels=sorted(set(affected)),
        components=names,
    )


INJECTORS = {
    EventClass.normal: inject_normal,
    EventClass.thermal_drift: inject_thermal_drift,
    EventClass.voltage_sag: inject_voltage_sag,
    EventClass.current_spike: inject_current_spike,
    EventClass.sensor_dropout: inject_sensor_dropout,
    EventClass.oscillatory_instability: inject_oscillatory_instability,
    EventClass.correlated_channel_fault: inject_correlated_channel_fault,
    EventClass.regime_shift: inject_regime_shift,
    EventClass.slow_degradation: inject_slow_degradation,
    EventClass.compound_fault: inject_compound_fault,
}
