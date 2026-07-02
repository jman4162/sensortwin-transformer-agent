# Headline model comparison

Mode `quick_demo`, seeds [0, 1, 2, 3, 4], 30 epochs, shared augmentation recipe. All deep models share one training recipe family (`configs/models/*.yaml`); classical baselines fit engineered features. Metrics on the held-out test split; mean ± sample std over seeds.

| Model | Macro-F1 (mean ± std) | ECE | Per-seed |
| --- | ---: | ---: | --- |
| transformer | 0.629 ± 0.032 | 0.077 | 0.655, 0.670, 0.613, 0.605, 0.601 |
| xgboost | 0.603 ± 0.024 | 0.125 | 0.595, 0.628, 0.571, 0.594, 0.627 |
| cnn | 0.582 ± 0.028 | 0.142 | 0.611, 0.597, 0.542, 0.566, 0.593 |
| random_forest | 0.555 ± 0.034 | 0.151 | 0.563, 0.598, 0.504, 0.554, 0.555 |
| logreg | 0.544 ± 0.028 | 0.140 | 0.546, 0.564, 0.522, 0.510, 0.577 |
| lstm | 0.384 ± 0.034 | 0.092 | 0.365, 0.427, 0.338, 0.391, 0.399 |

## Transformer vs best baseline

Best baseline by mean macro-F1: **xgboost**. Paired over 5 seeds: Δ = +0.03 macro-F1, 95% t-interval [-0.02, +0.07], paired t-test p = 0.162, Cohen's d = 0.8 — **not significant** at α = 0.05. Interval and p-value rest on n = 5 seeds; treat precision accordingly.

### Per-class F1 delta (transformer − best baseline, Holm-corrected)

| Class | Δ F1 | p | Significant (Holm, m=10) |
| --- | ---: | ---: | --- |
| normal | +0.10 | 0.108 | no |
| thermal_drift | +0.09 | 0.002 | yes |
| voltage_sag | +0.14 | 0.013 | no |
| current_spike | +0.08 | 0.031 | no |
| sensor_dropout | -0.14 | 0.098 | no |
| oscillatory_instability | +0.09 | 0.029 | no |
| correlated_channel_fault | +0.06 | 0.411 | no |
| regime_shift | -0.37 | 0.001 | yes |
| slow_degradation | -0.07 | 0.082 | no |
| compound_fault | +0.27 | 0.002 | yes |

Per-class deltas are a family of 10 comparisons; only Holm-surviving rows are claimable. Uncorrected per-class p-values at n = 5 seeds are weak evidence either way.

Regenerate: `python -m scripts.compare_models --mode quick_demo --seeds 0 1 2 3 4 --epochs 30`.
