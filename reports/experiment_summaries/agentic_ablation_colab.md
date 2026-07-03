# Agentic ablation report (v0.7)

Planner `HeuristicPlanner`, mode `colab_standard`, 1500 samples, 15 epochs, seeds [0, 1, 2], α=0.05. The agent read the baseline metrics, proposed one-variable ablations, ran each across seeds, and tested each vs the baseline. An 'improvement' is claimed only when the per-seed gain is statistically significant.

## Measured results

Baseline macro-F1: **0.680 ± 0.006** (seeds [0.6768126477886598, 0.6756513298710567, 0.686743925447095]).

| Variant (one change) | Macro-F1 (mean ± std) | Δ vs base | Status |
| --- | ---: | ---: | --- |
| `dropout=0.2` | 0.620 ± 0.062 | -0.060 | no_change |
| `d_model=192` | 0.658 ± 0.028 | -0.022 | no_change |
| `num_layers=6` | 0.580 ± 0.068 | -0.100 | no_change |

**0** of 3 proposals were significant improvements. Regressions and failed runs are listed above, not hidden.

## Verdicts (measured)

- `dropout=0.2`: **no_change** — -0.060 macro-F1 (p=0.204) — not significant at α=0.05 (Holm-corrected over 3 ablations)
- `d_model=192`: **no_change** — -0.022 macro-F1 (p=0.250) — not significant at α=0.05 (Holm-corrected over 3 ablations)
- `num_layers=6`: **no_change** — -0.100 macro-F1 (p=0.108) — not significant at α=0.05 (Holm-corrected over 3 ablations)

## Agent proposals (hypotheses — speculative, pre-registered)

These are the agent's *expectations stated before running*, kept separate from the measured verdicts above. They are not claims about reality.

- `dropout=0.2`: More dropout regularizes the small transformer, shrinking robustness deltas under corruption. _Expected failure:_ Underfits at quick scale: lower clean macro-F1 with no robustness gain.
- `d_model=192`: More width gives the attention more capacity for cross-channel / compound-fault classes. _Expected failure:_ Data-hungry: at quick-demo scale the extra parameters overfit and macro-F1 drops.
- `num_layers=6`: Deeper token mixing helps classes that need long-range or cross-channel context. _Expected failure:_ Too data-hungry for the sample budget; trains slower and underperforms.

## Claims vs not-claimed

- **Supported:** the agent automates the propose → run → review loop under guardrails (one variable per ablation, bounded jobs, significance-gated claims); every run reported.
- **Not claimed:** the agent does not discover new science or replace human analysis; quick-mode numbers are wiring-grade (use `colab_standard` + seeds for real verdicts).
