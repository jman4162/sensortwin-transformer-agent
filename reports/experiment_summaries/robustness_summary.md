# Robustness & uncertainty study

Mode `colab_standard`, seeds [0, 1, 2], 30 epochs; mean ± sample std over seeds. Trained on domain A with the committed parity recipes; test-split metrics. Deltas are the worst-case macro-F1 drop under each corruption; domain shift = same classes, noisier regime (domain B). ECE columns: clean / shifted / shifted after temperature scaling.

The `augmented` arm trains deep models with the shared recipe including channel-dropout augmentation (deployment-realistic; comparable across models but flattering on the missing-channel probe). The `no_augment` arm removes train-time augmentation entirely.

## Arm: augmented

| Model | Clean F1 | Shift F1 | Noise Δ | Window Δ | Missing-ch Δ | ECE clean | ECE shift | ECE shift (T) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.704 ± 0.004 | 0.269 ± 0.011 | 0.589 ± 0.014 | 0.626 ± 0.003 | 0.344 ± 0.044 | 0.056 ± 0.005 | 0.457 ± 0.031 | 0.457 ± 0.031 |
| cnn | 0.774 ± 0.014 | 0.038 ± 0.008 | 0.739 ± 0.004 | 0.561 ± 0.013 | 0.537 ± 0.024 | 0.136 ± 0.002 | 0.699 ± 0.025 | 0.791 ± 0.035 |
| transformer | 0.867 ± 0.012 | 0.487 ± 0.060 | 0.848 ± 0.013 | 0.700 ± 0.014 | 0.416 ± 0.095 | 0.093 ± 0.009 | 0.238 ± 0.073 | 0.375 ± 0.063 |
| pretrained_tf | 0.871 ± 0.011 | 0.487 ± 0.013 | 0.847 ± 0.016 | 0.708 ± 0.011 | 0.510 ± 0.134 | 0.092 ± 0.007 | 0.242 ± 0.011 | 0.378 ± 0.016 |

Pretrained − scratch macro-F1: **+0.004 in-distribution**, **+0.001 under domain shift**. If the shift gap is not larger than the clean gap, pretraining is not buying extra robustness here — consistent with it partly learning the generator's regularities rather than transferable structure.

## Arm: no_augment

| Model | Clean F1 | Shift F1 | Noise Δ | Window Δ | Missing-ch Δ | ECE clean | ECE shift | ECE shift (T) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.704 ± 0.004 | 0.269 ± 0.011 | 0.589 ± 0.014 | 0.626 ± 0.003 | 0.344 ± 0.044 | 0.056 ± 0.005 | 0.457 ± 0.031 | 0.457 ± 0.031 |
| cnn | 0.810 ± 0.005 | 0.188 ± 0.018 | 0.737 ± 0.030 | 0.565 ± 0.007 | 0.746 ± 0.013 | 0.136 ± 0.005 | 0.458 ± 0.085 | 0.618 ± 0.064 |
| transformer | 0.861 ± 0.005 | 0.479 ± 0.061 | 0.829 ± 0.029 | 0.685 ± 0.002 | 0.321 ± 0.030 | 0.071 ± 0.004 | 0.272 ± 0.059 | 0.385 ± 0.062 |
| pretrained_tf | 0.866 ± 0.011 | 0.515 ± 0.029 | 0.825 ± 0.027 | 0.694 ± 0.013 | 0.336 ± 0.032 | 0.073 ± 0.001 | 0.233 ± 0.027 | 0.346 ± 0.039 |

Pretrained − scratch macro-F1: **+0.005 in-distribution**, **+0.036 under domain shift**. If the shift gap is not larger than the clean gap, pretraining is not buying extra robustness here — consistent with it partly learning the generator's regularities rather than transferable structure.

## Claims
- **Supported:** different models degrade differently under noise / window / channel loss / domain shift; temperature scaling reduces ECE.
- **Not claimed:** synthetic robustness does not imply real-world robustness; domain B is a controlled regime change, not a real deployment shift.

Regenerate: `python -m scripts.robustness_report --mode colab_standard --epochs 30 --seeds 0 1 2 --arm both`.
