# Model results

Synthetic dataset: 2000 samples, T=512, seed=0, leakage-safe random split (70/15/15), metrics on the held-out test split.

## Headline metrics

| Model | Macro-F1 | Weighted-F1 | Accuracy | Macro-AUROC | ECE | Brier | Missing-ch Δ | Params | Train (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logreg | 0.571 | 0.593 | 0.587 | 0.898 | 0.122 | 0.551 | 0.554 | — | 0.0 |
| xgboost | 0.620 | 0.652 | 0.660 | 0.920 | 0.114 | 0.470 | 0.302 | — | 5.0 |
| cnn | 0.504 | 0.526 | 0.540 | 0.887 | 0.082 | 0.593 | 0.479 | 54k | 35.0 |
| transformer | 0.435 | 0.463 | 0.500 | 0.859 | 0.143 | 0.659 | 0.274 | 814k | 459.1 |

Macro-F1 is the headline metric (event classes are imbalanced). Missing-ch Δ is the worst single-channel-dropout macro-F1 drop vs clean (higher = more fragile).

## Model zoo — expected strengths / weaknesses (spec §7)

| Model | Expected strength | Expected weakness |
| --- | --- | --- |
| logreg / features | Simple drift, spikes, channel correlations | Complex temporal structure |
| random_forest / xgboost | Non-linear feature interactions | No raw temporal reasoning |
| cnn | Local transients, short events | Long-range dependencies |
| lstm | Sequential dynamics | Slower, harder to optimize |

## Per-class F1 (test)

Best model: **xgboost** (macro-F1 0.620).

- Hardest classes: `sensor_dropout` (0.18), `normal` (0.26), `compound_fault` (0.42)
- Easiest classes: `oscillatory_instability` (0.88), `regime_shift` (0.90), `slow_degradation` (0.91)

## Anomaly detection (IsolationForest, normal-vs-rest)

AUROC distinguishing `normal` from all event classes: **0.696** (unsupervised; a different task from 10-way classification — included to show the contrast).

## Notes

- Feature baselines use an sklearn `StandardScaler` fit on train only; CNN/LSTM use a `ChannelStandardizer` fit on train only — no test statistics leak into training.
- These are quick-mode numbers for wiring/verification; run `--mode colab_standard` for research-grade results before drawing conclusions.
