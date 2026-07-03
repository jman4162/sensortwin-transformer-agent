# Robustness & uncertainty study (v0.5)

Mode `quick_demo`, 2 epochs. Trained on domain A; test-split metrics. Deltas are the worst-case macro-F1 drop under each corruption; domain shift = same classes, noisier regime (domain B). ECE columns: clean / shifted / shifted after temperature scaling.

| Model | Clean F1 | Shift F1 | Noise Δ | Window Δ | Missing-ch Δ | ECE clean | ECE shift | ECE shift (T) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.620 | 0.251 | 0.541 | 0.544 | 0.302 | 0.114 | 0.417 | 0.417 |
| cnn | 0.400 | 0.378 | 0.172 | 0.254 | 0.375 | 0.170 | 0.146 | 0.048 |
| transformer | 0.096 | 0.086 | 0.003 | 0.047 | 0.076 | 0.013 | 0.012 | 0.053 |
| pretrained_tf | 0.063 | 0.059 | 0.000 | 0.021 | 0.044 | 0.021 | 0.011 | 0.030 |

## Pretraining under shift (closes the v0.4 caveat)

Pretrained − scratch macro-F1: **-0.033 in-distribution**, **-0.028 under domain shift**. If the shift gap is not larger than the clean gap, pretraining is not buying extra robustness here — consistent with it partly learning the generator's regularities rather than transferable structure.

## Claims
- **Supported:** different models degrade differently under noise / window / channel loss / domain shift; temperature scaling reduces ECE.
- **Not claimed:** synthetic robustness does not imply real-world robustness; domain B is a controlled regime change, not a real deployment shift.

`quick_demo` figures for wiring/discipline, not research-grade; run `colab_standard`. See `figures/robustness_degradation.png`.
