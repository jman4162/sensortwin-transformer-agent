# Headline model comparison

Mode `colab_standard`, seeds [0, 1, 2, 3, 4], 30 epochs, shared augmentation recipe. All deep models share one training recipe family (`configs/models/*.yaml`); classical baselines fit engineered features. Metrics on the held-out test split; mean ± sample std over seeds.

| Model | Macro-F1 (mean ± std) | ECE | Per-seed |
| --- | ---: | ---: | --- |
| transformer | 0.861 ± 0.010 | 0.087 | 0.872, 0.850, 0.855, 0.872, 0.854 |
| cnn | 0.774 ± 0.010 | 0.135 | 0.760, 0.775, 0.787, 0.771, 0.779 |
| xgboost | 0.704 ± 0.003 | 0.055 | 0.705, 0.700, 0.707, 0.705, 0.704 |
| logreg | 0.649 ± 0.006 | 0.018 | 0.655, 0.651, 0.649, 0.640, 0.652 |
| random_forest | 0.641 ± 0.004 | 0.144 | 0.645, 0.645, 0.642, 0.636, 0.638 |
| lstm | 0.507 ± 0.023 | 0.105 | 0.497, 0.515, 0.543, 0.480, 0.502 |

## Transformer vs best baseline

Best baseline by mean macro-F1: **cnn**. Paired over 5 seeds: Δ = +0.09 macro-F1, 95% t-interval [+0.06, +0.11], paired t-test p = 0.001, Cohen's d = 4.6 — **significant** at α = 0.05. Interval and p-value rest on n = 5 seeds; treat precision accordingly.

### Per-class F1 delta (transformer − best baseline, Holm-corrected)

| Class | Δ F1 | p | Significant (Holm, m=10) |
| --- | ---: | ---: | --- |
| normal | +0.07 | 0.019 | no |
| thermal_drift | +0.01 | 0.189 | no |
| voltage_sag | +0.06 | 0.005 | yes |
| current_spike | +0.02 | 0.005 | yes |
| sensor_dropout | +0.35 | 0.000 | yes |
| oscillatory_instability | +0.02 | 0.015 | no |
| correlated_channel_fault | +0.20 | 0.002 | yes |
| regime_shift | +0.10 | 0.000 | yes |
| slow_degradation | +0.00 | 0.683 | no |
| compound_fault | +0.04 | 0.009 | yes |

Per-class deltas are a family of 10 comparisons; only Holm-surviving rows are claimable. Uncorrected per-class p-values at n = 5 seeds are weak evidence either way.

Regenerate: `python -m scripts.compare_models --mode colab_standard --seeds 0 1 2 3 4 --epochs 30`.
