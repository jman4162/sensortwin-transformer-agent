# Label efficiency (masked pretraining vs scratch)

Mode `colab_standard`, 3 seed(s), pretrain 50 / fine-tune 30 epochs. Test macro-F1, mean ± sample std over seeds. Both transformer arms select their learning rate from the same validation budget (0.001, 0.0001), so the pretrained-vs-scratch comparison is not confounded by a fixed fine-tune LR.

| Fraction | scratch | pretrained_ft | pretrained_probe | cnn | xgboost |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 0.110 ± 0.009 | 0.171 ± 0.037 | 0.108 ± 0.026 | 0.353 ± 0.027 | 0.458 ± 0.031 |
| 5% | 0.444 ± 0.103 | 0.507 ± 0.101 | 0.174 ± 0.016 | 0.486 ± 0.017 | 0.573 ± 0.007 |
| 10% | 0.697 ± 0.034 | 0.686 ± 0.032 | 0.228 ± 0.005 | 0.584 ± 0.008 | 0.614 ± 0.015 |
| 100% | 0.870 ± 0.006 | 0.866 ± 0.005 | 0.250 ± 0.015 | 0.769 ± 0.009 | 0.708 ± 0.000 |

Regenerate: `python -m scripts.label_efficiency_sweep --mode colab_standard --seeds 3 --epochs-pretrain 50 --epochs-finetune 30`.
