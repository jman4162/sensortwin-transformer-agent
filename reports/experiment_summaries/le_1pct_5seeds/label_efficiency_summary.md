# Label efficiency (masked pretraining vs scratch)

Mode `colab_standard`, 5 seed(s), pretrain 50 / fine-tune 30 epochs. Test macro-F1, mean ± sample std over seeds. Both transformer arms select their learning rate from the same validation budget (0.001, 0.0001), so the pretrained-vs-scratch comparison is not confounded by a fixed fine-tune LR.

| Fraction | scratch | pretrained_ft | pretrained_probe | cnn | xgboost |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 0.097 ± 0.016 | 0.121 ± 0.042 | 0.102 ± 0.019 | 0.364 ± 0.017 | 0.446 ± 0.033 |

Regenerate: `python -m scripts.label_efficiency_sweep --mode colab_standard --seeds 5 --epochs-pretrain 50 --epochs-finetune 30`.
