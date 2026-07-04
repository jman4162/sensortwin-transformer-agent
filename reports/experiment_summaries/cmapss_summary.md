# C-MAPSS real-data summary

Mode `full`, subsets FD001,FD003, seeds [0, 1, 2], 30 epochs; grouped-by-engine splits (reseeded per seed; no engine crosses train/test); mean ± sample std over seeds. Both deep models train with the same recipe from `configs/data/cmapss.yaml`. 3-stage health classification (healthy / degrading / critical) — a deliberate reframing of the native RUL-regression task.

| Model | Macro-F1 | ECE | Noise Δ | Short-win Δ | Missing-ch Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.899 ± 0.007 | 0.047 ± 0.006 | 0.066 ± 0.010 | 0.571 ± 0.028 | 0.133 ± 0.016 |
| transformer | 0.869 ± 0.006 | 0.106 ± 0.013 | 0.229 ± 0.100 | 0.458 ± 0.137 | 0.478 ± 0.397 |
| cnn | 0.808 ± 0.041 | 0.085 ± 0.037 | 0.548 ± 0.098 | 0.266 ± 0.040 | 0.712 ± 0.034 |

- **Supported:** the synthetic-benchmark pipeline runs on real turbofan sensors and the models can be ranked on it with seed-level error bars.
- **Not claimed:** state-of-the-art RUL/health estimation; the 3-stage binning is a classification reframing.

Regenerate: `python -m scripts.real_data_report --raw-dir <CMAPSSData> --mode full --epochs 30 --seeds 0 1 2`.
