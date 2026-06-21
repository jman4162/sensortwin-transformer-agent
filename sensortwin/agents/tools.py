"""Agent tools (spec §13.2): the small, guardrailed surface the planner/runner/reviewer call.

These are plain functions — read metrics, merge a one-variable config patch, persist it — not LLM
prompts. The destructive guardrails live here: ``write_config`` never writes into the committed
``configs/`` tree or anywhere under ``data/`` (§13.5.1-2), so the agent can neither delete data nor
clobber a baseline config.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from sensortwin.agents.schemas import GuardrailViolation
from sensortwin.utils.config import load_yaml, save_yaml

# Which YAML section each ablation knob lives in (the base config has ``model:`` and ``train:``).
MODEL_KNOBS = {"patch_len", "stride", "d_model", "num_layers", "num_heads", "dropout", "pooling"}
TRAIN_KNOBS = {"label_smoothing", "lr", "weight_decay", "optimizer", "augment", "warmup_epochs"}


def list_experiments(reports_dir: str | Path) -> list[Path]:
    """Return the per-model metrics JSON files already on disk (the agent's prior results)."""
    return sorted(Path(reports_dir).glob("metrics_*.json"))


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load one ``metrics_<name>.json`` written by :func:`evaluation.metrics.save_metrics`."""
    return json.loads(Path(path).read_text())


def compare_runs(metrics_paths: list[str | Path]) -> dict[str, float]:
    """Map ``{model_name: macro_f1}`` across saved runs (filename stem after ``metrics_``)."""
    out: dict[str, float] = {}
    for p in metrics_paths:
        p = Path(p)
        name = p.stem.replace("metrics_", "")
        out[name] = float(load_metrics(p).get("macro_f1", float("nan")))
    return out


def merge_overrides(base_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy ``base_cfg`` and place each knob in its ``model``/``train`` section."""
    patched = copy.deepcopy(base_cfg)
    for knob, value in overrides.items():
        section = "model" if knob in MODEL_KNOBS else "train"
        patched.setdefault(section, {})[knob] = value
    return patched


def _is_protected(path: Path) -> bool:
    parts = set(path.resolve().parts)
    # Never write into the committed configs tree or anywhere under data/ (guardrails §13.5.1-2).
    return "data" in parts or "configs" in parts


def write_config(base_path: str | Path, overrides: dict[str, Any], out_path: str | Path) -> Path:
    """Patch ``base_path`` with a one-variable ``overrides`` and write the variant to ``out_path``.

    Refuses to write into ``configs/`` or ``data/`` (so it can neither overwrite a baseline config
    nor touch data). Returns the written path.
    """
    if len(overrides) != 1:
        raise GuardrailViolation(
            f"one-variable patch only, got {len(overrides)}: {list(overrides)}"
        )
    out = Path(out_path)
    if _is_protected(out):
        raise GuardrailViolation(f"refusing to write a patched config into a protected path: {out}")
    patched = merge_overrides(load_yaml(base_path), overrides)
    return save_yaml(patched, out)
