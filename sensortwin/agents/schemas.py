"""Pydantic schemas for the agentic experiment runner (roadmap v0.7, spec §13).

The guardrails (spec §13.5) are enforced *structurally* here rather than by trusting prose: an
``ExperimentSpec`` cannot even be constructed with more than one changed variable, and
``Guardrails.check`` rejects out-of-bounds epochs, disallowed modes/knobs, or oversized datasets
before any training runs. This is the point of using schemas instead of free-form prompts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Transformer knobs the planner is allowed to ablate (one at a time). Anything else is rejected.
DEFAULT_KNOBS = [
    "patch_len",
    "stride",
    "d_model",
    "num_layers",
    "dropout",
    "label_smoothing",
    "lr",
    "augment",
]


class GuardrailViolation(ValueError):
    """Raised when a proposed experiment would breach a spec §13.5 guardrail."""


class ExperimentSpec(BaseModel):
    """One bounded, single-variable ablation. Construction enforces the one-variable rule (§13.5.4);
    :meth:`Guardrails.check` enforces the bounds (§13.5.3)."""

    overrides: dict[str, Any]
    mode: str = "quick_demo"
    epochs: int = Field(gt=0)
    seeds: list[int] = Field(min_length=1)

    @field_validator("overrides")
    @classmethod
    def _exactly_one_variable(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(v) != 1:
            raise ValueError(
                f"one-variable ablation: exactly one override required, got {len(v)}: {list(v)}"
            )
        return v

    @property
    def knob(self) -> str:
        return next(iter(self.overrides))

    @property
    def value(self) -> Any:
        return self.overrides[self.knob]


class ExperimentProposal(BaseModel):
    """Planner output (spec §13.4): a spec plus its hypothesis and the failure mode to watch for."""

    label: str
    spec: ExperimentSpec
    hypothesis: str
    expected_failure_mode: str


class RunResult(BaseModel):
    """Runner output (spec §13.4): per-seed scores + summary. ``failed`` runs are kept, never hidden
    (§13.5.6)."""

    label: str
    per_seed_macro_f1: list[float] = Field(default_factory=list)
    mean: float = float("nan")
    std: float = float("nan")
    ece: float | None = None
    robustness_worst_delta: float | None = None
    train_time_s: float = 0.0
    n_params: int | None = None
    metrics_path: str | None = None
    failed: bool = False
    error: str | None = None


class ReviewVerdict(BaseModel):
    """Reviewer output (spec §13.4). ``status='improvement'`` requires statistical significance
    (§13.5.5); a positive delta alone is not enough."""

    label: str
    delta_vs_base: float
    p_value: float
    significant: bool
    status: Literal["improvement", "no_change", "regression", "failed"]
    claim: str


class Guardrails(BaseModel):
    """Bounds the agent must respect (loaded from ``configs/agents/experiment_agent.yaml``)."""

    max_epochs: int = 30
    max_n_samples: int = 20_000
    allowed_modes: list[str] = Field(default_factory=lambda: ["quick_demo", "colab_standard"])
    allowed_knobs: list[str] = Field(default_factory=lambda: list(DEFAULT_KNOBS))
    n_seeds: int = 3
    n_experiments: int = 3
    alpha: float = 0.05

    def check(self, spec: ExperimentSpec, n_samples: int) -> None:
        """Raise :class:`GuardrailViolation` if ``spec`` would breach a bound."""
        if spec.epochs > self.max_epochs:
            raise GuardrailViolation(f"epochs {spec.epochs} > max {self.max_epochs}")
        if spec.mode not in self.allowed_modes:
            raise GuardrailViolation(f"mode {spec.mode!r} not in {self.allowed_modes}")
        if spec.knob not in self.allowed_knobs:
            raise GuardrailViolation(f"knob {spec.knob!r} not in allowed {self.allowed_knobs}")
        if n_samples > self.max_n_samples:
            raise GuardrailViolation(f"n_samples {n_samples} > max {self.max_n_samples}")
