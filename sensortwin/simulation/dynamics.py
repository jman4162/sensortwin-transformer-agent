"""Base signal dynamics and physically-motivated channel coupling.

The eight channels are not independent. We model a small set of generative assumptions so that
some event classes are only detectable through *relationships* among channels (see the
``correlated_channel_fault`` event), not through any single channel in isolation:

  * A shared ``load_profile`` drives both current channels.
  * Voltage partially anti-correlates with current (loaded source sags under draw).
  * ``temperature_core`` integrates current through a first-order thermal lag.
  * ``temperature_surface`` lags the core temperature (a slower second lag).
  * ``ambient/load`` proxy gently biases temperature.
  * The vibration proxy reacts to high-frequency content / instability.

These are deliberately simple, documented assumptions (spec §8 educational goal): the point is a
controllable benchmark, not a faithful physics simulation.
"""

from __future__ import annotations

import numpy as np


def first_order_lag(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal first-order IIR low-pass: ``y[t] = (1-a) y[t-1] + a x[t]``.

    Models thermal/mechanical inertia. ``alpha`` in (0, 1]; smaller = slower response.
    """
    y = np.empty_like(x)
    acc = x[0]
    for t in range(len(x)):
        acc = (1.0 - alpha) * acc + alpha * x[t]
        y[t] = acc
    return y


def smooth_random_walk(T: int, rng: np.random.Generator, step: float, lag: float) -> np.ndarray:
    """A low-frequency drift term: random walk passed through a slow lag, then de-meaned."""
    walk = np.cumsum(rng.normal(0.0, step, size=T))
    walk = first_order_lag(walk, lag)
    return walk - walk.mean()


def load_profile(T: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth, mostly-positive operating-load envelope shared across coupled channels."""
    phase = np.linspace(0, rng.uniform(1.5, 4.0) * np.pi, T) + rng.uniform(0, np.pi)
    base = 1.0 + 0.3 * np.sin(phase)
    base = base + smooth_random_walk(T, rng, step=0.05, lag=0.02)
    return np.clip(base, 0.1, None)


def base_channels(T: int, rng: np.random.Generator) -> np.ndarray:
    """Construct the coupled 8-channel base tensor ``[C, T]`` before event injection/noise.

    Channel order (see :data:`sensortwin.simulation.events.CHANNELS`):
        0 voltage_A, 1 voltage_B, 2 current_A, 3 current_B,
        4 temperature_core, 5 temperature_surface, 6 vibration_proxy, 7 ambient_load_proxy
    """
    t = np.linspace(0, 1, T)
    load = load_profile(T, rng)

    # Currents track the shared load with channel-specific gain + slow drift.
    current_a = 0.8 * load + smooth_random_walk(T, rng, step=0.03, lag=0.05)
    current_b = 0.7 * load + smooth_random_walk(T, rng, step=0.03, lag=0.05)

    # Voltages sit near a nominal level and sag (anti-correlate) under current draw.
    nominal_v = rng.uniform(0.9, 1.1)
    voltage_a = nominal_v - 0.15 * current_a + smooth_random_walk(T, rng, step=0.02, lag=0.08)
    voltage_b = nominal_v - 0.12 * current_b + smooth_random_walk(T, rng, step=0.02, lag=0.08)

    # Core temperature integrates ohmic-like heating (~ current^2) through a slow thermal lag.
    heating = current_a**2 + current_b**2
    ambient = 0.2 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t) + smooth_random_walk(
        T, rng, step=0.02, lag=0.03
    )
    temp_core = first_order_lag(0.5 * heating + 0.3 * ambient, alpha=0.02)
    temp_surface = first_order_lag(temp_core, alpha=0.01)  # surface lags core

    # Vibration proxy: low-level broadband content, baseline-stable until instability events.
    vibration = 0.05 * rng.standard_normal(T)

    X = np.stack(
        [voltage_a, voltage_b, current_a, current_b, temp_core, temp_surface, vibration, ambient]
    )
    return X.astype(np.float64)
