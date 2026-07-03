# Interpretability: does saliency track the injected event?

Mode `colab_standard`, seeds [0, 1, 2], 30 epochs; transformer trained with its committed parity recipe (test macro-F1 0.871 ± 0.006). On 300 correctly-classified test samples with a localized injected event (pooled across seeds), the score is the fraction of saliency mass inside the known event region (channels x time from generator metadata) vs a random-saliency baseline.

| Saliency | Localization (mean ± std over seeds) | Random baseline |
| --- | ---: | ---: |
| attention pooling | 0.110 ± 0.015 | 0.103 ± 0.009 |
| integrated gradients | 0.483 ± 0.033 | 0.103 ± 0.009 |

A localization above the random baseline means the saliency tracks the injected event better than chance. **Caveat (spec §12.5):** this is a diagnostic; attention weights are not a definitive explanation (Jain & Wallace 2019). Occlusion / integrated gradients are more faithful because they re-run or differentiate the model.

Regenerate: `python -m scripts.interpretability_report --mode colab_standard --epochs 30 --seeds 0 1 2`.
