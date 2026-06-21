"""Experiment planner (spec §13.4 persona 1).

Reads the baseline metrics and proposes up to ``n_experiments`` **one-variable** ablations, each
with a hypothesis and the failure mode to watch for. The default :class:`HeuristicPlanner` is
deterministic and torch-free (runs in CI with no API key); :class:`LLMPlanner` documents the
pluggable seam for a future Claude-backed planner. The spec (§13.3 v1) recommends exactly this — a
structured planner with JSON-shaped outputs — before reaching for LangGraph or an LLM.
"""

from __future__ import annotations

from typing import Any, Protocol

from sensortwin.agents.schemas import ExperimentProposal, ExperimentSpec, Guardrails
from sensortwin.agents.tools import MODEL_KNOBS, TRAIN_KNOBS

# (knob, value, hypothesis, expected_failure_mode). Ordered by the signal the planner reacts to.
_PLAYBOOK: list[tuple[str, Any, str, str]] = [
    (
        "dropout",
        0.2,
        "More dropout regularizes the small transformer, shrinking robustness deltas under "
        "corruption.",
        "Underfits at quick scale: lower clean macro-F1 with no robustness gain.",
    ),
    (
        "d_model",
        192,
        "More width gives the attention more capacity for cross-channel / compound-fault classes.",
        "Data-hungry: at quick-demo scale the extra parameters overfit and macro-F1 drops.",
    ),
    (
        "num_layers",
        6,
        "Deeper token mixing helps classes that need long-range or cross-channel context.",
        "Too data-hungry for the sample budget; trains slower and underperforms.",
    ),
    (
        "patch_len",
        32,
        "Coarser patches summarize longer context, helping long-range classes (regime/drift).",
        "Too coarse for short transients (current_spike), hurting those classes.",
    ),
    (
        "label_smoothing",
        0.0,
        "Removing label smoothing sharpens the decision boundary and may raise macro-F1.",
        "Worse calibration (higher ECE) and more overconfident errors.",
    ),
    (
        "lr",
        5e-4,
        "A lower learning rate converges more stably within the epoch budget.",
        "Underfits: too slow to converge in the allotted epochs.",
    ),
]


def _base_value(base_cfg: dict[str, Any], knob: str) -> Any:
    section = "model" if knob in MODEL_KNOBS else "train"
    return base_cfg.get(section, {}).get(knob)


class Planner(Protocol):
    def propose(
        self,
        metrics: dict[str, Any],
        base_cfg: dict[str, Any],
        guardrails: Guardrails,
        *,
        epochs: int,
        seeds: list[int],
    ) -> list[ExperimentProposal]: ...


class HeuristicPlanner:
    """Deterministic planner: ranks the playbook by what the baseline metrics say is weak."""

    def propose(
        self,
        metrics: dict[str, Any],
        base_cfg: dict[str, Any],
        guardrails: Guardrails,
        *,
        epochs: int,
        seeds: list[int],
    ) -> list[ExperimentProposal]:
        fragile = float(metrics.get("missing_channel", {}).get("worst_delta", 0.0) or 0.0)
        # If the baseline is fragile to dropped channels, prioritize regularization; else capacity.
        playbook = list(_PLAYBOOK)
        if fragile < 0.15:
            # Robustness looks fine — try capacity/resolution before more regularization.
            playbook = playbook[1:] + playbook[:1]

        proposals: list[ExperimentProposal] = []
        for knob, value, hyp, fail in playbook:
            if knob not in guardrails.allowed_knobs:
                continue
            if knob not in MODEL_KNOBS and knob not in TRAIN_KNOBS:
                continue
            if _base_value(base_cfg, knob) == value:  # not a real change
                continue
            spec = ExperimentSpec(
                overrides={knob: value}, mode="quick_demo", epochs=epochs, seeds=seeds
            )
            proposals.append(
                ExperimentProposal(
                    label=f"{knob}={value}", spec=spec, hypothesis=hyp, expected_failure_mode=fail
                )
            )
            if len(proposals) >= guardrails.n_experiments:
                break
        return proposals


class LLMPlanner:
    """Placeholder for a Claude-backed planner (optional ``agents`` extra). Same interface; would
    prompt an LLM for proposals and validate them through :class:`ExperimentProposal`. Not built in
    v0.7 — the deterministic planner is the default so the loop stays reproducible and CI-safe."""

    def propose(self, *args: Any, **kwargs: Any) -> list[ExperimentProposal]:  # pragma: no cover
        raise NotImplementedError(
            "LLMPlanner is a documented seam; use HeuristicPlanner (default) in v0.7."
        )
