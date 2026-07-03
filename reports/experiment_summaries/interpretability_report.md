# Interpretability report (v0.5)

Mode `quick_demo`, 3 epochs. On 27 correctly-classified test samples with a localized injected event, we measure the fraction of saliency mass landing inside the known event region (channels x time from the generator metadata), vs a random-saliency baseline.

| Saliency | Mean localization | Random baseline |
| --- | ---: | ---: |
| attention pooling | 0.139 | 0.154 |
| integrated gradients | 0.234 | 0.154 |

A localization above the random baseline means the saliency tracks the injected event better than chance. **Caveat (spec §12.5):** this is a diagnostic; attention weights are not a definitive explanation (Jain & Wallace 2019). Occlusion / integrated gradients are more faithful because they re-run or differentiate the model.

`quick_demo` figures for wiring/discipline, not research-grade; run `colab_standard`. Example attention heatmaps in `figures/`.
