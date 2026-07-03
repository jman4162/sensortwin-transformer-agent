"""CLI: the agentic experiment runner (roadmap v0.7, spec §13 / §16).

Orchestrates the planner -> runner -> reviewer loop:
  1. train the baseline and read its metrics,
  2. the planner proposes <= N one-variable ablations (each with a hypothesis + expected failure),
  3. each proposal is guardrail-checked, then run across seeds,
  4. the reviewer significance-tests every variant vs the baseline and writes
     ``reports/experiment_summaries/agentic_ablation_report.md`` — claims separated from measurements.

The planner is deterministic (no API key); the whole loop runs in CI at tiny scale. Quick-mode
numbers are wiring-grade — run with a larger ``run.n_samples`` / ``--epochs`` for real verdicts.

Example:
    python -m scripts.run_agent --mode quick_demo --epochs 1 --seeds 2 --max-experiments 1
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# macOS-only OpenMP guard; must precede torch/xgboost import.
configure_omp()

import argparse
from pathlib import Path

from sensortwin.agents.planner import HeuristicPlanner
from sensortwin.agents.reviewer import review, write_report
from sensortwin.agents.schemas import Guardrails, GuardrailViolation, RunResult
from sensortwin.utils.config import load_yaml
from sensortwin.utils.seeds import set_torch_seed


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Agentic experiment runner (planner/runner/reviewer).")
    p.add_argument("--config", default="configs/agents/experiment_agent.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--epochs", type=int, default=None, help="override run.epochs (<= max_epochs)")
    p.add_argument("--seeds", type=int, default=None, help="override guardrails.n_seeds")
    p.add_argument("--max-experiments", type=int, default=None, help="override n_experiments")
    p.add_argument("--n-samples", type=int, default=None, help="override run.n_samples")
    p.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto-detect")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    set_torch_seed(args.seed)
    cfg = load_yaml(args.config)
    guardrails = Guardrails(**cfg.get("guardrails", {}))
    if args.max_experiments is not None:
        guardrails.n_experiments = args.max_experiments
    if args.seeds is not None:
        guardrails.n_seeds = args.seeds

    run_cfg = cfg.get("run", {})
    n_samples = min(args.n_samples or run_cfg.get("n_samples", 1500), guardrails.max_n_samples)
    epochs = min(args.epochs or run_cfg.get("epochs", 6), guardrails.max_epochs)
    T = run_cfg.get("T", 256)
    seeds = list(range(guardrails.n_seeds))

    base_cfg = load_yaml(cfg.get("base_model_config", "configs/models/sensorpatchtst.yaml"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Runner imported lazily: it needs the ml extra (trains models).
    from sensortwin.agents.runner import ExperimentRunner

    print(f"Generating {n_samples} samples (T={T}) and training the baseline...")
    runner = ExperimentRunner(
        base_cfg, n_samples=n_samples, T=T, seed=args.seed, device=args.device, out_dir=str(out_dir)
    )
    baseline_result, baseline_metrics = runner.run(
        "baseline", runner.baseline_spec(epochs=epochs, seeds=seeds)
    )
    print(f"  baseline macro-F1: {baseline_result.mean:.3f} ± {baseline_result.std:.3f}")

    planner = HeuristicPlanner()
    proposals = planner.propose(baseline_metrics, base_cfg, guardrails, epochs=epochs, seeds=seeds)
    print(f"Planner proposed {len(proposals)} one-variable ablations.")

    results = []
    for prop in proposals:
        try:
            guardrails.check(prop.spec, n_samples)
        except GuardrailViolation as e:
            print(f"  [rejected] {prop.label}: {e}")
            results.append((prop, RunResult(label=prop.label, failed=True, error=str(e))))
            continue
        print(f"  running {prop.label} ...")
        result, _ = runner.run(prop.label, prop.spec)
        results.append((prop, result))
        status = "failed" if result.failed else f"{result.mean:.3f} ± {result.std:.3f}"
        print(f"    {prop.label}: {status}")

    verdicts = review(baseline_result, results, alpha=guardrails.alpha)
    # colab_standard sessions are research-grade evidence and get a committable filename;
    # quick wiring runs keep the gitignored name so they never dirty the tree.
    report_name = (
        "agentic_ablation_colab.md"
        if args.mode == "colab_standard"
        else "agentic_ablation_report.md"
    )
    report = write_report(
        baseline_result,
        results,
        verdicts,
        out_dir / report_name,
        meta={
            "planner": "HeuristicPlanner",
            "mode": args.mode,
            "n_samples": n_samples,
            "epochs": epochs,
            "seeds": seeds,
            "alpha": guardrails.alpha,
        },
    )
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
