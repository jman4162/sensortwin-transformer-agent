# C-MAPSS real-data summary

Mode `full`, subsets FD001, seeds [0, 1, 2], 30 epochs; grouped-by-engine splits (reseeded per seed; no engine crosses train/test); mean ± sample std over seeds. Both deep models train with the same recipe from `configs/data/cmapss.yaml`. 3-stage health classification (healthy / degrading / critical) — a deliberate reframing of the native RUL-regression task.

| Model | Macro-F1 | ECE | Noise Δ | Short-win Δ | Missing-ch Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.877 ± 0.008 | 0.054 ± 0.020 | 0.108 ± 0.025 | 0.548 ± 0.030 | 0.108 ± 0.048 |
| cnn | 0.766 ± 0.021 | 0.088 ± 0.029 | 0.473 ± 0.175 | 0.317 ± 0.066 | 0.660 ± 0.018 |
| transformer | 0.757 ± 0.052 | 0.124 ± 0.035 | 0.172 ± 0.153 | 0.253 ± 0.174 | 0.421 ± 0.339 |

- **Supported:** the synthetic-benchmark pipeline runs on real turbofan sensors and the models can be ranked on it with seed-level error bars.
- **Not claimed:** state-of-the-art RUL/health estimation; the 3-stage binning is a classification reframing.

Regenerate: `python -m scripts.real_data_report --raw-dir <CMAPSSData> --mode full --epochs 30 --seeds 0 1 2`.
