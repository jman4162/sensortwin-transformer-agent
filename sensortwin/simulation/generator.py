"""Top-level dataset generator: composes dynamics, events, and artifacts into labeled samples.

Public entry points:
  * :func:`generate_sample` — one ``(x, meta)`` pair from a dedicated RNG.
  * :func:`generate_dataset` — a full ``(X, y, meta)`` benchmark, deterministic from one seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from sensortwin.simulation.dynamics import base_channels
from sensortwin.simulation.events import INJECTORS, N_CHANNELS, EventClass, EventMeta
from sensortwin.simulation.noise import apply_artifacts
from sensortwin.utils.seeds import spawn_child_rngs


@dataclass
class GenConfig:
    """Generation parameters. Defaults target the Colab-standard regime (spec §19)."""

    n_samples: int = 20_000
    T: int = 512
    seed: int = 0
    normalize: bool = True  # per-channel z-score using train-time stats kept in metadata
    artifacts: dict = field(
        default_factory=lambda: {
            "gaussian_sigma": 0.02,
            "impulse_rate": 0.002,
            "impulse_magnitude": 0.3,
            "calibration_scale": 0.01,
            "clip_prob": 0.1,
            "clip_channel_prob": 0.25,
            "quantize_levels": 0,  # 0 disables quantization
        }
    )
    # Class sampling weights (uniform by default; override for the rare-event split).
    class_weights: list[float] | None = None


def generate_sample(
    rng: np.random.Generator, T: int, event_class: int, artifacts: dict
) -> tuple[np.ndarray, EventMeta]:
    """Generate a single ``[C, T]`` sample for a given event class."""
    X = base_channels(T, rng)
    meta = INJECTORS[EventClass(event_class)](X, rng)
    X = apply_artifacts(X, rng, artifacts)
    return X.astype(np.float32), meta


def _sample_labels(cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    n_classes = len(EventClass)
    p = None
    if cfg.class_weights is not None:
        p = np.asarray(cfg.class_weights, dtype=float)
        p = p / p.sum()
    return rng.choice(n_classes, size=cfg.n_samples, p=p).astype(np.int64)


def generate_dataset(cfg: GenConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate the full benchmark.

    Returns ``(X, y, meta)`` where ``X`` is ``[N, C, T]`` float32, ``y`` is ``[N]`` int64, and
    ``meta`` carries the config, per-sample event metadata, and (if normalized) channel stats.
    Fully determined by ``cfg.seed``: a label-stream RNG picks classes, then one independent
    child RNG per sample generates the signal.
    """
    label_rng = np.random.default_rng(cfg.seed)
    y = _sample_labels(cfg, label_rng)

    # Independent per-sample streams (order-invariant, parallel-safe).
    rngs = spawn_child_rngs(cfg.seed + 1, cfg.n_samples)

    X = np.empty((cfg.n_samples, N_CHANNELS, cfg.T), dtype=np.float32)
    events: list[dict] = []
    for i in range(cfg.n_samples):
        x_i, meta_i = generate_sample(rngs[i], cfg.T, int(y[i]), cfg.artifacts)
        X[i] = x_i
        events.append(meta_i.to_dict())

    channel_stats = None
    if cfg.normalize:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-6
        X = (X - mean) / std
        channel_stats = {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}

    meta = {
        "config": asdict(cfg),
        "n_channels": N_CHANNELS,
        "event_classes": [c.name for c in EventClass],
        "channel_stats": channel_stats,
        "events": events,
    }
    return X, y, meta
