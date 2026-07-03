# Headline model comparison

Mode `colab_standard`, seeds [0, 1, 2, 3, 4], 30 epochs, shared augmentation recipe. Deep models at their grid-selected recipes from `configs/models/tuned` (the tuned-recipe check); classical baselines fit engineered features. Metrics on the held-out test split; mean ± sample std over seeds.

| Model | Macro-F1 (mean ± std) | ECE | Per-seed |
| --- | ---: | ---: | --- |
| transformer | 0.867 ± 0.009 | 0.087 | 0.879, 0.857, 0.869, 0.870, 0.861 |
| cnn | 0.832 ± 0.005 | 0.128 | 0.836, 0.827, 0.834, 0.838, 0.826 |
| xgboost | 0.704 ± 0.003 | 0.055 | 0.705, 0.700, 0.707, 0.705, 0.704 |
| logreg | 0.649 ± 0.006 | 0.018 | 0.655, 0.651, 0.649, 0.640, 0.652 |
| random_forest | 0.641 ± 0.004 | 0.144 | 0.645, 0.645, 0.642, 0.636, 0.638 |
| lstm | 0.560 ± 0.010 | 0.124 | 0.577, 0.561, 0.553, 0.555, 0.552 |

## Transformer vs best baseline

Best baseline by mean macro-F1: **cnn**. Paired over 5 seeds: Δ = +0.03 macro-F1, 95% t-interval [+0.03, +0.04], paired t-test p = 0.000, Cohen's d = 7.1 — **significant** at α = 0.05. Interval and p-value rest on n = 5 seeds; treat precision accordingly.

### Per-class F1 delta (transformer − best baseline, Holm-corrected)

| Class | Δ F1 | p | Significant (Holm, m=10) |
| --- | ---: | ---: | --- |
| normal | +0.03 | 0.004 | yes |
| thermal_drift | +0.00 | 0.558 | no |
| voltage_sag | +0.02 | 0.005 | yes |
| current_spike | +0.01 | 0.048 | no |
| sensor_dropout | +0.22 | 0.000 | yes |
| oscillatory_instability | +0.00 | 0.089 | no |
| correlated_channel_fault | +0.02 | 0.057 | no |
| regime_shift | +0.02 | 0.000 | yes |
| slow_degradation | -0.00 | 0.626 | no |
| compound_fault | +0.02 | 0.000 | yes |

Per-class deltas are a family of 10 comparisons; only Holm-surviving rows are claimable. Uncorrected per-class p-values at n = 5 seeds are weak evidence either way.

Regenerate: `python -m scripts.compare_models --mode colab_standard --seeds 0 1 2 3 4 --epochs 30`.
