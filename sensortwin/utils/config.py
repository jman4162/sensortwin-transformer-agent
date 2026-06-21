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


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a config dict to ``path`` as YAML (the counterpart to :func:`load_yaml`).

    Added in v0.7 so the agent can read a base config, patch one field, and persist the variant for
    a reproducible run. ``sort_keys=False`` keeps the human-authored field order.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def load_mode_config(path: str | Path, mode: str | None = None) -> dict[str, Any]:
    """Load a YAML config and overlay one ``modes`` entry, returning the merged dict.

    The generic counterpart to :func:`load_synthetic_config` for configs that are not a
    :class:`GenConfig` (e.g. the C-MAPSS data config). The ``modes`` table is removed from the
    returned dict; the chosen mode's keys are merged onto the base.
    """
    raw = load_yaml(path)
    modes = raw.pop("modes", {})
    if mode is not None:
        if mode not in modes:
            raise KeyError(f"mode '{mode}' not in config; available: {sorted(modes)}")
        raw.update(modes[mode])
    return raw


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
