# Model card — SensorPatchTST (SensorTwin)

A patch-transformer for multichannel sensor event classification, trained and evaluated on the
synthetic `SensorTwin-Synth` benchmark. This card follows the outline in the project spec (§21) and
is written to be honest about what the project does and does not establish.

## 1. Model
`SensorPatchTST`: per-channel patching (patch_len 16, stride 8) → linear patch embedding +
sinusoidal positions + learned channel embedding → 4-layer pre-norm transformer encoder
(d_model 128, 4 heads) → attention pooling → classification head. ~0.8M parameters. Input
`[B, C=8, T]`, output logits over 10 event classes. Optional masked-patch self-supervised
pretraining (v0.4). Implemented in `sensortwin/models/transformer.py`.

## 2. Intended use
A research/education artifact: a controllable benchmark for studying multichannel time-series
classification, the conditions under which a transformer's inductive bias helps versus simpler
baselines, label efficiency from self-supervision, and robustness/calibration/interpretability
methodology. For learning and portfolio demonstration.

## 3. Non-intended use
Not a production diagnostic system, not a safety-critical monitor, and not a general time-series
foundation model. Do not deploy it to make real decisions about physical equipment.

## 4. Training data
`SensorTwin-Synth`: deterministic 8-channel signals (voltage/current/temperature/vibration/ambient
proxies) with documented coupling (voltage anti-correlates with current; core temperature integrates
current through a thermal lag; surface lags core) and 10 event classes spanning a difficulty
gradient from local transients to cross-channel-only faults. Fully reproducible from a seed.

## 5. Synthetic-data limitations
The core benchmark is synthetic. Performance there does not guarantee real-world performance: the
model may exploit regularities of the generator rather than transferable structure. "Synthetic data
lets us test controlled hypotheses" — it is not real-world validation. v0.6 adds a real-data check
(§6); the synthetic headline numbers still carry this caveat.

## 6. Open-data validation
Run on **NASA C-MAPSS** turbofan data (FD001+FD003, 14 informative sensors, 6,076 windows,
200 engines), reframed as 3-stage health classification (healthy / degrading / critical) by
binning remaining-useful-life, with grouped-by-engine splits (reseeded per seed) so no engine's
windows cross train/test. 3 seeds, both deep models on one recipe; raw inputs SHA-256-pinned.
Artifacts: `cmapss_summary.json`, `sim2real_summary.json`.

- **The synthetic ordering does not port to real data.** From scratch on C-MAPSS:
  XGBoost-on-features **0.899 ± 0.007** > transformer 0.869 ± 0.006 > CNN 0.808 ± 0.041
  macro-F1. XGBoost is also the best calibrated (ECE 0.047) and degrades least under noise.
  The transformer's synthetic-benchmark advantage is conditional, not general — on this real
  task engineered features win.
- **Synthetic pretraining does not transfer.** With matched per-arm LR budgets, the
  synthetic-pretrained encoder beats scratch by +0.004 to +0.017 across engine fractions —
  never significant (paired p ≥ 0.42) — and the real-pretrained control is no better than
  scratch either. Masked-patch pretraining buys nothing here, consistent with its synthetic
  label-efficiency and robustness nulls.

**Claimed:** the pipeline runs on real sensor data with seed-level error bars, and the honest
answer is that features+GBM lead there. **Not claimed:** state-of-the-art RUL/health estimation
(the 3-stage binning is a deliberate classification reframing). The raw C-MAPSS files are a
US-government work and are not redistributed; `fetch_cmapss` verifies your download against the
pinned hashes.

## 7. Metrics
Headline metric is **macro-F1** (classes are imbalanced); also report weighted-F1, per-class
precision/recall, one-vs-rest AUROC, and the confusion matrix. Calibration via ECE, multiclass
Brier, and reliability diagrams. The research-grade comparison is `make headline`
(`scripts/compare_models.py`: 6 models × 5 seeds at `colab_standard`, shared training recipe,
paired t-statistics, Holm-corrected per-class deltas), whose committed artifact
`reports/experiment_summaries/headline_comparison.json` backs every number quoted here.
Reproduce the smaller runs with `make baselines` / `make transformer` / `make robustness-study`;
real-data studies with `make real-data` / `make sim2real` (after the C-MAPSS download).

## 8. Robustness
Evaluated as macro-F1 degradation from clean under: Gaussian-noise severity sweep, shorter
observation windows, missing/zeroed channels, and a domain shift (a noisier measurement regime),
at colab_standard × 3 seeds in augmented and unaugmented arms (`make robustness-full`; artifact:
`robustness_summary.json`). Measured: the transformer degrades least under domain shift (0.49
vs the CNN's 0.19 under domain B, unaugmented) and most under extreme Gaussian noise
(worst-case delta 0.85 vs XGBoost's 0.59 at σ=0.2). Channel-dropout augmentation flatters the
CNN's missing-channel numbers (0.54 vs 0.75 delta) but not the transformer's. Absolute
robustness is not claimed beyond the synthetic benchmark.

## 9. Calibration
ECE / Brier / reliability are reported on clean and shifted test sets. Post-hoc **temperature
scaling** (one parameter fit on validation) is provided; as is standard, models tend to be more
overconfident under distribution shift, and temperature scaling reduces ECE without changing
accuracy. Accuracy and calibration rankings differ: in the 5-seed headline run the two most
accurate models are also the worst-calibrated supervised ones (transformer ECE 0.087, CNN 0.135)
while logistic regression is nearly calibrated out of the box (0.018). A single temperature fit
per seed (T = 0.64 ± 0.01 on validation logits) cuts the transformer's test ECE from
0.091 ± 0.008 to **0.016 ± 0.005** with zero predictions changed (3 seeds, `make calibration`;
artifact: `reports/experiment_summaries/calibration_temperature.json`). Most accurate and
well-calibrated after a one-parameter fit — **in distribution only**: the same temperature
*worsens* ECE under domain shift for every deep model (transformer 0.24 → 0.38; see
`robustness_summary.json`). Recalibrate after any regime change.

## 10. Known failure modes
- The transformer's data-hunger shows **per class, not in the headline mean**: at 2k (5 seeds,
  shared recipe) it statistically ties XGBoost-on-features overall but *loses* `regime_shift` by
  −0.37 (Holm-significant); at 20k it beats the strongest baseline (CNN, identical recipe) by
  +0.09 macro-F1 (p=0.001) and the same class flips to +0.10. Its largest 20k wins are the
  structure/cross-channel classes (`sensor_dropout` +0.35, `correlated_channel_fault` +0.20).
  An earlier claim that it was far behind at 2k (0.435) came from baselines trained without its
  recipe and is superseded (research log, 2026-07-02).
- **Per-model tuning shrinks the margin to +0.03** (tuned-recipe check, 5 seeds: transformer
  0.867 ± 0.009 vs grid-tuned 211k CNN 0.832 ± 0.005; still significant, wins every seed). The
  remaining reliable advantage is almost entirely `sensor_dropout` (+0.22); the tuned CNN
  recovers most of `correlated_channel_fault`. Quote +0.03, not +0.09, when the question is
  architecture rather than shared-budget behavior.
- Masked-pretraining **did not improve label efficiency under matched budgets** (per-arm LR
  selection, 3 seeds, `label_efficiency_summary.json`): pretrained-vs-scratch is +0.06 at 1%
  labels (paired p = 0.13), never significant at any fraction, and the frozen linear probe
  plateaus at 0.25 macro-F1 even with all labels. Below 10% labels XGBoost-on-features and the
  CNN beat both transformer arms outright. Fourth consistent pretraining null (with the
  robustness and sim2real studies); use engineered features, not pretraining, when labels are
  scarce here.
- Engineered features + gradient boosting are **surprisingly strong** on simple drift/spike events.
- **Extreme noise breaks the transformer first**: at 10× the training noise floor its macro-F1
  drops by 0.85 — worse than every baseline — despite jitter augmentation. Its shift robustness
  does not generalize to heavy in-distribution corruption.
- **Attention weights localize events at chance level** (0.110 vs random 0.103, 3 seeds,
  300 events) while integrated gradients reach 0.483: use gradient/perturbation attributions,
  never the pooling weights, when asking "where did the model look."
- The hardest classes at trained scale are `sensor_dropout` and `correlated_channel_fault`,
  both confused with `normal` (see `docs/figures/confusion_matrix.png`) — the two classes the
  task-hardening deliberately made shortcut-free.

## 11. Ethical / safety considerations
The data is synthetic, so there is no personal or sensitive information. The main risk is
**overclaiming**: treating synthetic accuracy as real-world capability, or attention weights as
explanations. The project deliberately separates supported claims from non-claims and documents
negative results. Attention weights are **not** treated as definitive explanations (Jain & Wallace
2019); occlusion and integrated gradients are used as more faithful diagnostics, checked against the
generator's known event regions.

## 12. Reproducibility
Deterministic from a seed end to end (`set_torch_seed`, leakage-safe splits, train-only
standardization, and an end-to-end seed→metric regression test). Quality gates (`make check`:
ruff + mypy + pytest) run in CI on Python 3.10/3.12. Install with `pip install -e ".[ml,dev]"`;
`requirements-lock.txt` pins the exact environment behind the committed numbers; regenerate any
study with its `make` target. The headline artifacts
(`headline_comparison.{json,md}`, with per-seed values, device, and library versions) are
committed alongside this card; the private design spec and the wiring-grade per-run reports are
gitignored. Datasets carry a generator version; results are comparable only within one version
(current: v2).
