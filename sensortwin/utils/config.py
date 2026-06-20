"""YAML config loading with run-mode overlays.

A config file carries base fields plus a ``modes`` table (``quick_demo`` / ``colab_standard`` /
``full_reproduction``). :func:`load_synthetic_config` flattens the chosen mode onto the base and
returns a :class:`~sensortwin.simulation.generator.GenConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sensortwin.simulation.generator import GenConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def load_synthetic_config(path: str | Path, mode: str | None = None) -> GenConfig:
    raw = load_yaml(path)
    modes = raw.pop("modes", {})
    if mode is not None:
        if mode not in modes:
            raise KeyError(f"mode '{mode}' not in config; available: {sorted(modes)}")
        raw.update(modes[mode])
    # Keep only fields GenConfig understands.
    allowed = set(GenConfig.__dataclass_fields__)
    return GenConfig(**{k: v for k, v in raw.items() if k in allowed})
