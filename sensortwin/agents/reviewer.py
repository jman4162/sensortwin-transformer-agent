"""Research reviewer (spec §13.4 persona 3).

Turns runs into verdicts and an honest markdown report. The guardrail that matters here: a variant
is called an **improvement** only when ``statistics.compare_seeds`` finds the per-seed gain real
(§13.5.5) — a positive mean delta alone is reported as "no significant change". Every run is listed,
including regressions and failures (§13.5.6), and the report keeps the agent's *hypotheses*
(speculative) visually separate from the *measured* results (acceptance criterion §6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from sensortwin.agents.schemas import ExperimentProposal, ReviewVerdict, RunResult
from sensortwin.evaluation.statistics import compare_seeds


def review(
    baseline: RunResult, results: list[tuple[ExperimentProposal, RunResult]], *, alpha: float = 0.05
) -> list[ReviewVerdict]:
    """One :class:`ReviewVerdict` per proposal, significance-gated against the baseline seeds."""
    verdicts: list[ReviewVerdict] = []
    for _prop, res in results:
        if res.failed:
            verdicts.append(
                ReviewVerdict(
                    label=res.label,
                    delta_vs_base=float("nan"),
                    p_value=float("nan"),
                    significant=False,
                    status="failed",
                    claim=f"run did not complete: {res.error}",
                )
            )
            continue
        cmp = compare_seeds(baseline.per_seed_macro_f1, res.per_seed_macro_f1, alpha=alpha)
        status: Literal["improvement", "no_change", "regression", "failed"]
        if cmp.significant and cmp.delta > 0:
            status = "improvement"
            claim = (
                f"+{cmp.delta:.3f} macro-F1 (p={cmp.p_value:.3f}, "
                f"95% CI [{cmp.ci_low:.3f}, {cmp.ci_high:.3f}]) — significant"
            )
        elif cmp.significant and cmp.delta < 0:
            status = "regression"
            claim = f"{cmp.delta:.3f} macro-F1 (p={cmp.p_value:.3f}) — significant regression"
        else:
            status = "no_change"
            claim = (
                f"{cmp.delta:+.3f} macro-F1 (p={cmp.p_value:.3f}) — not significant at α={alpha}"
            )
        verdicts.append(
            ReviewVerdict(
                label=res.label,
                delta_vs_base=cmp.delta,
                p_value=cmp.p_value,
                significant=cmp.significant,
                status=status,
                claim=claim,
            )
        )
    return verdicts


def _fmt(r: RunResult) -> str:
    return "failed" if r.failed else f"{r.mean:.3f} ± {r.std:.3f}"


def write_report(
    baseline: RunResult,
    results: list[tuple[ExperimentProposal, RunResult]],
    verdicts: list[ReviewVerdict],
    out_path: str | Path,
    *,
    meta: dict[str, Any],
) -> Path:
    """Write ``agentic_ablation_report.md`` — measured results separated from agent hypotheses."""
    by_label = {v.label: v for v in verdicts}
    n_imp = sum(v.status == "improvement" for v in verdicts)
    lines = [
        "# Agentic ablation report (v0.7)",
        "",
        f"Planner `{meta.get('planner')}`, mode `{meta.get('mode')}`, "
        f"{meta.get('n_samples')} samples, {meta.get('epochs')} epochs, "
        f"seeds {meta.get('seeds')}, α={meta.get('alpha')}. The agent read the baseline metrics, "
        "proposed one-variable ablations, ran each across seeds, and tested each vs the baseline. "
        "An 'improvement' is claimed only when the per-seed gain is statistically significant.",
        "",
        "## Measured results",
        "",
        f"Baseline macro-F1: **{_fmt(baseline)}** (seeds {baseline.per_seed_macro_f1}).",
        "",
        "| Variant (one change) | Macro-F1 (mean ± std) | Δ vs base | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for _prop, res in results:
        v = by_label[res.label]
        delta = "—" if res.failed else f"{v.delta_vs_base:+.3f}"
        lines.append(f"| `{res.label}` | {_fmt(res)} | {delta} | {v.status} |")

    lines += [
        "",
        f"**{n_imp}** of {len(results)} proposals were significant improvements. Regressions and "
        "failed runs are listed above, not hidden.",
        "",
        "## Verdicts (measured)",
        "",
    ]
    for v in verdicts:
        lines.append(f"- `{v.label}`: **{v.status}** — {v.claim}")

    lines += [
        "",
        "## Agent proposals (hypotheses — speculative, pre-registered)",
        "",
        "These are the agent's *expectations stated before running*, kept separate from the "
        "measured verdicts above. They are not claims about reality.",
        "",
    ]
    for prop, _ in results:
        lines.append(
            f"- `{prop.label}`: {prop.hypothesis} _Expected failure:_ {prop.expected_failure_mode}"
        )

    lines += [
        "",
        "## Claims vs not-claimed",
        "",
        "- **Supported:** the agent automates the propose → run → review loop under guardrails "
        "(one variable per ablation, bounded jobs, significance-gated claims); every run reported.",
        "- **Not claimed:** the agent does not discover new science or replace human analysis; "
        "quick-mode numbers are wiring-grade (use `colab_standard` + seeds for real verdicts).",
    ]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
