# Research log

A running record of experiments, hypotheses, and what actually happened — including negative
results. Newest entries first. (Spec §"keep a research log": this is what turns the repo from a
weekend demo into a mini research program.)

---

## 2026-06-19 — v0.2 baselines established

**Question.** Before building the transformer, what bar do strong non-transformer baselines set on
`SensorTwin-Synth`, and which event classes are already easy vs. hard?

**Setup.** quick_demo (2,000 samples, T=512, seed=0), leakage-safe 70/15/15 random split. Classical
models (LogReg / RandomForest / XGBoost) on engineered features (statistical + spectral +
cross-channel/missingness); CNN and BiLSTM on standardized raw signals. Feature scalers fit on train
only; CNN/LSTM use a `ChannelStandardizer` fit on train only. Headline metric: macro-F1.

**Hypothesis.** XGBoost on features would be the strongest baseline at this scale; deep models would
underperform with so little data; long-range/global events would be easy and cross-channel /
dropout events hard.

**Result (test split).**

| Model | Macro-F1 | Macro-AUROC | ECE |
| --- | ---: | ---: | ---: |
| logreg | 0.571 | 0.898 | 0.122 |
| random_forest | 0.543 | 0.915 | 0.181 |
| xgboost | **0.620** | 0.920 | 0.114 |
| cnn | 0.504 | 0.887 | 0.082 |
| lstm | 0.338 | 0.817 | 0.063 |

Best model xgboost: easiest classes `slow_degradation` (0.91), `regime_shift` (0.90),
`oscillatory_instability` (0.88); hardest `sensor_dropout` (0.18), `normal` (0.26),
`compound_fault` (0.42). IsolationForest normal-vs-rest AUROC 0.70.

**Interpretation.** Hypothesis largely confirmed. Feature+GBM leads with ~1.4k training samples;
the LSTM in particular is data-starved. The difficulty ordering matches the generator's design
intent: global/obvious signatures are learnable from summary features, while `sensor_dropout` and
`compound_fault` need either better temporal modeling or more data. Interesting wrinkle: the deep
models are *better calibrated* (lower ECE) despite lower accuracy — worth watching, not yet
explained.

**Caveats.** Quick-demo scale only; no seed averaging; single split. Not research-grade.

**Next experiments.**
1. Re-run at `colab_standard` (20k) — does the CNN overtake the feature baselines as data grows?
2. v0.3: patch transformer (`SensorPatchTST`); does it beat XGBoost specifically on
   `correlated_channel_fault` and `compound_fault`?
3. Add seed averaging (seeds 1–5) and report mean ± std before any headline claim.
