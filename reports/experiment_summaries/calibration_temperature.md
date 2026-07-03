# Temperature scaling (transformer)

Mode `colab_standard`, seeds [0, 1, 2], 30 epochs. One temperature fitted per seed on the validation logits; ECE measured on the held-out test split.

| Seed | T | ECE before | ECE after | Predictions changed |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.649 | 0.094 | 0.015 | 0.0% |
| 1 | 0.652 | 0.083 | 0.021 | 0.0% |
| 2 | 0.625 | 0.097 | 0.013 | 0.0% |

Mean over 3 seeds: T = 0.64 ± 0.01, ECE 0.091 ± 0.008 → 0.016 ± 0.005. Temperature scaling divides logits by a scalar, so the argmax — and every accuracy metric — is unchanged by construction.

Regenerate: `python -m scripts.calibrate_transformer --mode colab_standard --seeds 0 1 2 --epochs 30`.
