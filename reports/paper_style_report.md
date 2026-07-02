# When do patch-transformers beat strong baselines on multichannel sensor events?

*A controlled benchmark study. John Hodge, 2026. Code: [sensortwin-transformer-agent](https://github.com/jman4162/sensortwin-transformer-agent).*

## Abstract

We ask a narrow, testable question: on multichannel sensor event classification, when does a
patch-based transformer (PatchTST-style) justify its cost over strong classical and convolutional
baselines? We built a deterministic 8-channel physics-inspired generator with 10 event classes,
trained six models under one shared training recipe, and compared them at two dataset scales with
paired statistics over 5 seeds. At 20k samples the transformer reaches macro-F1 0.861 ± 0.010
versus 0.774 ± 0.010 for the strongest baseline (a CNN trained with the identical recipe):
Δ = +0.09, 95% CI [+0.06, +0.11], paired p = 0.001. The gain concentrates where the
architecture's inductive bias predicts: the largest Holm-surviving per-class deltas are
`sensor_dropout` (+0.35), the marginal-preserving cross-channel fault (+0.20), and
`regime_shift` (+0.10), while trend classes show no advantage. The transformer is also the worst
calibrated of the accurate models (ECE 0.087). Masked-patch self-supervised pretraining did not
improve label efficiency at the budgets previously tested (a matched-budget re-test is wired but
not yet run at scale). In earlier wiring-grade runs, attention weights did not localize events
better than chance while integrated gradients did. All results regenerate from committed configs
and one command per table.

## 1. Question and design principles

Transformers are routinely reported to beat baselines on time-series benchmarks under unequal
tuning budgets, which measures effort rather than architecture. This study controls that
variable. Three principles, enforced in code rather than prose:

1. **Determinism.** Every dataset and experiment is a pure function of a seed
   (`SeedSequence.spawn` per sample; an end-to-end seed→metric regression test).
2. **Parity.** All deep models read one shared recipe family (AdamW, weight decay 0.01, label
   smoothing 0.1, cosine warmup, identical augmentation) from `configs/models/*.yaml` through a
   single code path. A shared lr × capacity tuning grid (`scripts/tune_baselines.py`) selects on
   validation macro-F1 only.
3. **Claims gated by statistics.** Paired t-intervals and t-tests over seeds; Holm correction
   across the 10 per-class comparisons; paired Cohen's d; no claim from a single seed.

## 2. The benchmark

### 2.1 Generator

Eight coupled channels: two currents track a shared stochastic load; two voltages sag under
current draw; core temperature integrates ohmic-style heating (current², first-order lag);
surface temperature lags the core; a vibration proxy carries load-dependent broadband energy; an
ambient/load proxy biases temperature. Measurement artifacts (Gaussian noise σ=0.02, impulses,
calibration offsets, clipping, optional quantization) are applied after event injection, so
"what happened" and "how it was measured" are independently controllable.

Ten event classes span a difficulty gradient: local single-channel signatures (`current_spike`,
`voltage_sag`), long-range trends (`slow_degradation`, `thermal_drift`), temporal-structure
classes (`oscillatory_instability`: growing oscillation with current bleed; `sensor_dropout`:
frozen channel; `regime_shift`: persistent multi-channel step), a cross-channel-only class
(`correlated_channel_fault`), and `compound_fault` (two overlapping events).

### 2.2 Hardening against shortcuts

Three design decisions close shortcuts a classifier could otherwise exploit:

- **Benign background activity.** Every class — including `normal` — receives mild label-free
  events (smooth bumps, constant-amplitude vibration bursts, small reverting steps), so channel
  activity alone is not class evidence and `normal` is not "the class where nothing happened."
- **Marginal-preserving cross-channel fault.** `correlated_channel_fault` rotates a voltage
  segment toward the current channel's fluctuation and restores the segment's mean and standard
  deviation exactly; unit tests assert the preservation. Single-channel statistics cannot
  separate this class — only the relationship changes.
- **No giveaway channel.** The vibration baseline tracks the operating load and carries benign
  bursts, so spectral energy in that channel is not unique to `oscillatory_instability`.

A `severity_scale` knob shrinks all event severities toward the noise floor for a low-SNR tier.
Generated data carries a generator version (currently 2); numbers are comparable only within a
version.

### 2.3 Scales

`quick_demo` (2,000 samples), `colab_standard` (20,000; the headline tier), `full_reproduction`
(100,000; defined and wired via `make full-reproduction`, not yet run). All use T=512, a
70/15/15 random split, and per-channel standardization fit on the train split only.

## 3. Models

| Model | Input | Params |
| --- | --- | ---: |
| Logistic regression | ~10 engineered features/channel (stats, spectral, cross-correlations) | — |
| Random forest (300) | same features | — |
| XGBoost (300) | same features | — |
| SensorCNN | raw `[8, 512]`; 3 conv blocks + GAP | 54k |
| SensorLSTM | raw; 1-layer BiLSTM, mean-pooled | 39k |
| SensorPatchTST | raw; per-channel patches (len 16, stride 8) → 504 tokens with channel + positional embeddings → 4-layer encoder (d=128, 4 heads) → attention pooling | 815k |

The parameter gap (15–20× over the deep baselines) is reported rather than hidden; the shared
tuning grid includes one capacity step per model (CNN→210k, LSTM→530k, transformer→1.2M) so
capacity-at-parity is checkable.

All deep models train with the same recipe and the same augmentation (jitter, per-channel
scaling, channel dropout p=0.1). Because channel-dropout augmentation overlaps the
missing-channel robustness probe, robustness tables report an unaugmented arm (`--no-augment`)
alongside the default.

## 4. Headline comparison

Protocol: `python -m scripts.compare_models --mode colab_standard --seeds 0 1 2 3 4 --epochs 30`.
Per seed: regenerate data, resplit, retrain everything. Report: mean ± sample std, paired
t-interval/t-test/Cohen's d for transformer vs the best baseline, Holm-corrected per-class deltas.

### 4.1 Results at 20k (colab_standard, generator v2, 5 seeds)

| Model | Macro-F1 (mean ± std) | ECE | Per-seed |
| --- | ---: | ---: | --- |
| transformer | 0.861 ± 0.010 | 0.087 | 0.872, 0.850, 0.855, 0.872, 0.854 |
| cnn | 0.774 ± 0.010 | 0.135 | 0.760, 0.775, 0.787, 0.771, 0.779 |
| xgboost | 0.704 ± 0.003 | 0.055 | 0.705, 0.700, 0.707, 0.705, 0.704 |
| logreg | 0.649 ± 0.006 | 0.018 | 0.655, 0.651, 0.649, 0.640, 0.652 |
| random_forest | 0.641 ± 0.004 | 0.144 | 0.645, 0.645, 0.642, 0.636, 0.638 |
| lstm | 0.507 ± 0.023 | 0.105 | 0.497, 0.515, 0.543, 0.480, 0.502 |

*(Run on Apple M3 Max / torch 2.12.1 MPS, FP32; artifact:
`reports/experiment_summaries/headline_comparison.json`.)*

**Transformer vs the strongest baseline (CNN).** Paired over 5 seeds: Δ = +0.09 macro-F1, 95%
t-interval [+0.06, +0.11], paired t-test p = 0.001, paired Cohen's d = 4.6. Both models trained
under the identical recipe, so the delta measures architecture at this recipe, not tuning.

**Where the gain lives.** Six of ten per-class deltas survive Holm correction: `sensor_dropout`
+0.35, `correlated_channel_fault` +0.20, `regime_shift` +0.10, `voltage_sag` +0.06,
`compound_fault` +0.04, `current_spike` +0.02. The two largest are the classes designed to
require temporal-structure and cross-channel reasoning — `sensor_dropout` (a frozen channel among
benign background activity) and the marginal-preserving fault, which is invisible to any
single-channel statistic by construction. The trend classes show no advantage
(`slow_degradation` +0.00, p = 0.68; `thermal_drift` +0.01, p = 0.19): engineered features and
convolutions already capture monotone drifts.

### 4.2 Scale contrast: the same protocol at 2k

| Model | Macro-F1 (mean ± std) | ECE |
| --- | ---: | ---: |
| transformer | 0.629 ± 0.032 | 0.077 |
| xgboost | 0.603 ± 0.024 | 0.125 |
| cnn | 0.582 ± 0.028 | 0.142 |
| random_forest | 0.555 ± 0.034 | 0.151 |
| logreg | 0.544 ± 0.028 | 0.140 |
| lstm | 0.384 ± 0.034 | 0.092 |

*(Artifact: `reports/experiment_summaries/scale_2k/headline_comparison.json`.)*

At 2k the transformer and XGBoost-on-features are a statistical tie (Δ = +0.03, 95% t-interval
[−0.02, +0.07], p = 0.16, d = 0.8). Two things are worth stating precisely:

1. **The earlier "transformer is last at 2k" finding did not survive parity.** Under the old
   asymmetric recipes the transformer scored 0.435 at 2k, far behind XGBoost (0.620). With every
   deep model on the shared recipe, it ties for first. The dramatic rank flip previously
   attributed to data scale was mostly an artifact of unequal training budgets.
2. **What scale actually buys is the structural classes.** At 2k the transformer *loses*
   `regime_shift` by −0.37 (Holm-significant) and trails on `sensor_dropout` (−0.14, n.s.); at
   20k those become +0.10 (Holm-significant) and +0.35 (Holm-significant). The per-class sign
   flip, not the headline mean, is where the data-hunger of the attention-based architecture
   shows: its advantage on structure/cross-channel classes exists and costs data. Seed spread
   also triples at 2k (std 0.032 vs 0.010), so any single-seed 2k comparison is close to
   uninformative.

**Calibration trade-off.** Accuracy ranking is not calibration ranking: the two most accurate
models are the two worst-calibrated among the supervised set (CNN ECE 0.135, transformer 0.087)
while logistic regression is nearly calibrated out of the box (0.018). Temperature scaling
(§5.1) addresses this post hoc.

## 5. Beyond accuracy

### 5.1 Calibration

Prior single-seed evidence (generator v1): the transformer was the most accurate and worst
calibrated; one temperature parameter fit on validation logits cut ECE ≈ 0.09 → 0.02 without
changing any prediction. **[TBD: re-measure at generator v2; multi-seed.]**

### 5.2 Robustness

Noise, short-window, missing-channel, and domain-shift sweeps
(`scripts/robustness_report.py`). Prior wiring-grade signal: the strongest in-distribution
model degraded most under domain shift. **[TBD: colab_standard, ≥3 seeds, augmented and
unaugmented arms.]**

### 5.3 Interpretability

Attribution maps are scored against the generator's ground-truth event windows
(`localization_score`) with a random-placement baseline — not eyeballed. Prior wiring-grade
finding, consistent with Jain & Wallace (2019): integrated gradients localized above chance;
attention pooling weights did not. **[TBD: re-run at trained scale.]**

## 6. Negative and null results

- **Masked-patch pretraining did not improve label efficiency** at the budgets tested
  (1/5/10/100% labels, single seed, generator v1). A known confound — the pretrained arms
  fine-tuned at a lower learning rate for fewer epochs — means the honest verdict is "no help at
  this budget," not "does not transfer." **[TBD: matched-budget re-run.]**
- **Attention is not an explanation** here: pooling weights localized events below the random
  baseline in the wiring-grade run.
- **At 2k samples the transformer loses** to feature baselines — capacity without data is a
  liability, which is the expected result and reported as such.

## 7. Threats to validity

- **Synthetic data.** The generator is physics-inspired, not physics. Models may exploit
  regularities of the simulator that do not exist in real systems; the C-MAPSS pipeline
  (grouped-by-engine splits, channel-count-flexible encoder transfer) exists to check this and
  has not yet been run on the real download.
- **Residual separability.** Hardening closed the shortcuts we found (giveaway channel,
  inactive `normal`, variance-inflating "invisible" fault); others may remain. The signature
  tests document what is asserted, not everything that is true.
- **Shared recipe rather than per-model tuning.** Parity is by construction (one shared recipe), which could
  favor the model the recipe was designed around (the transformer). The shared grid
  (`make tune`) bounds this concern but was selected on validation macro-F1 at one scale.
- **Seed budget.** n=5 paired seeds detects deltas roughly ≥ 1.5 pooled standard deviations;
  smaller real effects will read as "not significant."
- **One data regime per claim.** Random splits of i.i.d. samples; no temporal drift between
  train and test except in the explicit domain-shift study.

## 8. Reproducibility

- One command per table: `make headline`, `make tune`, `make robustness-study`,
  `make label-efficiency`, `make interpretability`, `make agent`.
- `headline_comparison.json` carries every per-seed value, the device, and library versions;
  `requirements-lock.txt` pins the environment; `CITATION.cff` at the root.
- Determinism is tested end-to-end (same seed → identical trained-model metrics on CPU).
- C-MAPSS raw files are hashed at processing time and the hashes stamped into the dataset
  metadata.

## References

- Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*
  (PatchTST), ICLR 2023.
- Jain & Wallace, *Attention is not Explanation*, NAACL 2019.
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and
  Perturbations*, ICLR 2019.
- Holm, *A simple sequentially rejective multiple test procedure*, Scand. J. Statist. 1979.
- Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*
  (C-MAPSS), PHM 2008.
