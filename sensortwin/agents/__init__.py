"""Agentic experiment runner (roadmap v0.7, spec §13).

A constrained planner -> runner -> reviewer loop that reads metrics, proposes one-variable
ablations, runs bounded jobs, and writes an honest report. Guardrails (spec §13.5) are enforced by
the Pydantic schemas and ``Guardrails.check`` rather than by prose: one variable changed per
ablation, bounded epochs/dataset, never overwrite a baseline config or touch data, and no
"improvement" claim without statistical significance.

``runner`` (``ExperimentRunner``) needs the ``ml`` extra (it trains models), so it is imported
lazily by ``scripts/run_agent.py`` and not re-exported here — the planner/schemas/tools/reviewer
import with only the core dependencies.
"""

from sensortwin.agents.planner import HeuristicPlanner, LLMPlanner, Planner
from sensortwin.agents.reviewer import review, write_report
from sensortwin.agents.schemas import (
    ExperimentProposal,
    ExperimentSpec,
    Guardrails,
    GuardrailViolation,
    ReviewVerdict,
    RunResult,
)
from sensortwin.agents.tools import compare_runs, list_experiments, load_metrics, write_config

__all__ = [
    "HeuristicPlanner",
    "LLMPlanner",
    "Planner",
    "ExperimentProposal",
    "ExperimentSpec",
    "Guardrails",
    "GuardrailViolation",
    "ReviewVerdict",
    "RunResult",
    "review",
    "write_report",
    "compare_runs",
    "list_experiments",
    "load_metrics",
    "write_config",
]
