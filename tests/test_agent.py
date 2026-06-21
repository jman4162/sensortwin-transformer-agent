"""Tests for the agentic experiment runner (v0.7).

The schema/guardrail/planner/reviewer logic is pure-python (pydantic + numpy/scipy, all core deps)
and tested here without torch; the end-to-end loop is torch-gated and runs at tiny scale.
"""

from __future__ import annotations

import pytest

from sensortwin.agents.planner import HeuristicPlanner
from sensortwin.agents.reviewer import review, write_report
from sensortwin.agents.schemas import (
    ExperimentProposal,
    ExperimentSpec,
    Guardrails,
    GuardrailViolation,
    RunResult,
)
from sensortwin.agents.tools import load_metrics, merge_overrides, write_config
from sensortwin.utils.config import load_yaml

BASE_CONFIG = "configs/models/sensorpatchtst.yaml"


# --- schemas / guardrails ---------------------------------------------------------------------


def test_experiment_spec_rejects_multi_variable():
    with pytest.raises(ValueError):
        ExperimentSpec(overrides={"patch_len": 32, "d_model": 192}, epochs=5, seeds=[0])


def test_experiment_spec_accepts_single_variable():
    spec = ExperimentSpec(overrides={"patch_len": 32}, epochs=5, seeds=[0, 1])
    assert spec.knob == "patch_len"
    assert spec.value == 32


def test_guardrails_reject_out_of_bounds():
    g = Guardrails()
    with pytest.raises(GuardrailViolation):  # epochs over budget
        g.check(ExperimentSpec(overrides={"lr": 1e-4}, epochs=999, seeds=[0]), n_samples=100)
    with pytest.raises(GuardrailViolation):  # mode not allowed
        g.check(
            ExperimentSpec(overrides={"lr": 1e-4}, mode="x", epochs=5, seeds=[0]), n_samples=100
        )
    with pytest.raises(GuardrailViolation):  # knob not allowed
        g.check(ExperimentSpec(overrides={"nonsense": 1}, epochs=5, seeds=[0]), n_samples=100)
    with pytest.raises(GuardrailViolation):  # dataset too big
        g.check(ExperimentSpec(overrides={"lr": 1e-4}, epochs=5, seeds=[0]), n_samples=10**9)


def test_guardrails_accept_valid_spec():
    g = Guardrails()
    g.check(ExperimentSpec(overrides={"dropout": 0.2}, epochs=5, seeds=[0, 1]), n_samples=1500)


# --- tools ------------------------------------------------------------------------------------


def test_merge_overrides_places_knob_in_right_section():
    base = load_yaml(BASE_CONFIG)
    patched = merge_overrides(base, {"patch_len": 32})
    assert patched["model"]["patch_len"] == 32
    patched2 = merge_overrides(base, {"lr": 5e-4})
    assert patched2["train"]["lr"] == 5e-4
    assert base["model"]["patch_len"] != 32  # original untouched (deep copy)


def test_write_config_one_variable_and_refuses_protected(tmp_path):
    out = write_config(BASE_CONFIG, {"patch_len": 32}, tmp_path / "variant.yaml")
    assert load_metrics  # imported
    written = load_yaml(out)
    assert written["model"]["patch_len"] == 32

    with pytest.raises(GuardrailViolation):  # never write into a data/ path
        write_config(BASE_CONFIG, {"patch_len": 32}, tmp_path / "data" / "variant.yaml")
    with pytest.raises(GuardrailViolation):  # one-variable only
        write_config(BASE_CONFIG, {"patch_len": 32, "lr": 1e-4}, tmp_path / "v.yaml")


# --- planner ----------------------------------------------------------------------------------


def test_heuristic_planner_emits_valid_one_variable_proposals():
    metrics = {
        "per_class": {"normal": {"f1": 0.2}, "current_spike": {"f1": 0.9}},
        "missing_channel": {"worst_delta": 0.3},
    }
    base = load_yaml(BASE_CONFIG)
    g = Guardrails()
    proposals = HeuristicPlanner().propose(metrics, base, g, epochs=5, seeds=[0, 1])
    assert 0 < len(proposals) <= g.n_experiments
    for p in proposals:
        assert isinstance(p, ExperimentProposal)
        assert len(p.spec.overrides) == 1  # one-variable rule
        assert p.spec.knob in g.allowed_knobs
        assert p.hypothesis and p.expected_failure_mode


# --- reviewer ---------------------------------------------------------------------------------


def _result(label, seeds, failed=False, error=None):
    if failed:
        return RunResult(label=label, failed=True, error=error)
    import numpy as np

    return RunResult(label=label, per_seed_macro_f1=seeds, mean=float(np.mean(seeds)))


def _proposal(label):
    return ExperimentProposal(
        label=label,
        spec=ExperimentSpec(overrides={"dropout": 0.2}, epochs=1, seeds=[0, 1, 2]),
        hypothesis="h",
        expected_failure_mode="f",
    )


def test_review_flags_improvement_regression_and_failure():
    baseline = _result("baseline", [0.50, 0.50, 0.50])
    results = [
        (_proposal("better"), _result("better", [0.70, 0.70, 0.70])),
        (_proposal("worse"), _result("worse", [0.30, 0.30, 0.30])),
        (_proposal("broke"), _result("broke", [], failed=True, error="boom")),
    ]
    verdicts = {v.label: v for v in review(baseline, results)}
    assert verdicts["better"].status == "improvement"
    assert verdicts["worse"].status == "regression"
    assert verdicts["broke"].status == "failed"


def test_write_report_separates_claims_and_lists_failures(tmp_path):
    baseline = _result("baseline", [0.50, 0.50, 0.50])
    results = [
        (_proposal("better"), _result("better", [0.70, 0.70, 0.70])),
        (_proposal("broke"), _result("broke", [], failed=True, error="boom")),
    ]
    verdicts = review(baseline, results)
    out = write_report(
        baseline,
        results,
        verdicts,
        tmp_path / "agentic_ablation_report.md",
        meta={
            "planner": "HeuristicPlanner",
            "mode": "quick_demo",
            "n_samples": 200,
            "epochs": 1,
            "seeds": [0, 1, 2],
            "alpha": 0.05,
        },
    )
    text = out.read_text()
    assert "## Measured results" in text
    assert "speculative" in text.lower()  # hypotheses kept separate
    assert "broke" in text  # the failed run is not hidden
    assert "failed" in text


# --- end-to-end (torch) -----------------------------------------------------------------------


def test_agent_loop_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from scripts.run_agent import main

    main(
        [
            "--n-samples",
            "120",
            "--epochs",
            "1",
            "--seeds",
            "2",
            "--max-experiments",
            "1",
            "--device",
            "cpu",
            "--out",
            str(tmp_path),
        ]
    )
    report = tmp_path / "agentic_ablation_report.md"
    assert report.exists()
    text = report.read_text()
    assert "## Measured results" in text
    assert "speculative" in text.lower()
