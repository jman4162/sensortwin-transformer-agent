# Synthetic-to-real transfer on C-MAPSS

Mode `full`, subsets FD001,FD003, seeds [0, 1, 2], pretrain 30 / fine-tune 20 epochs; test macro-F1, mean ± sample std over seeds. Every torch arm selects its LR from the same validation budget (0.001, 0.0001), so a transfer win cannot be an artifact of a fixed low fine-tune LR handicapping scratch. Only the channel-agnostic temporal patch encoder transfers (synthetic C=8 → real C=14).

| Engine fraction | scratch | synth_pretrained | real_pretrained | xgboost |
| --- | ---: | ---: | ---: | ---: |
| 10% | 0.782 ± 0.016 | 0.792 ± 0.023 | 0.769 ± 0.040 | 0.854 ± 0.008 |
| 25% | 0.815 ± 0.035 | 0.828 ± 0.016 | 0.815 ± 0.022 | 0.879 ± 0.013 |
| 50% | 0.842 ± 0.017 | 0.858 ± 0.015 | 0.830 ± 0.033 | 0.882 ± 0.004 |
| 100% | 0.864 ± 0.009 | 0.868 ± 0.011 | 0.858 ± 0.024 | 0.899 ± 0.007 |

Transfer verdict: `synth_pretrained` ≈ `real_pretrained` > `scratch` means the synthetic encoder transferred; `synth_pretrained` ≈ `scratch` means it did not — either way the number above is the finding.

Regenerate: `python -m scripts.sim2real_transfer --raw-dir <CMAPSSData> --mode full --epochs-pretrain 30 --epochs-finetune 20 --seeds 0 1 2`.
