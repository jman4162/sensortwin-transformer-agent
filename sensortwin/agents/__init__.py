"""Agentic experiment runner (roadmap v0.7).

Planned: ``tools.py`` (list_experiments/load_metrics/compare_runs/write_config/run_training/...),
``planner.py``, ``runner.py``, ``reviewer.py``, ``schemas.py`` (Pydantic).
GUARDRAILS (spec §13.5): one variable changed per ablation; never delete data or overwrite
baselines; bounded jobs; no claims without statistical evidence; never hide failed runs.
Not yet implemented.
"""
