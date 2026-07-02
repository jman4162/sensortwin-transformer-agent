# Glossary

Terms as used in this repo. Cross-references point at the module that implements each one.

## Task and data

- **Event class** — one of the 10 labels a sample carries (`normal`, `thermal_drift`,
  `voltage_sag`, `current_spike`, `sensor_dropout`, `oscillatory_instability`,
  `correlated_channel_fault`, `regime_shift`, `slow_degradation`, `compound_fault`).
  Injectors live in `sensortwin/simulation/events.py`.
- **Channel** — one of the 8 coupled sensor streams (`voltage_A/B`, `current_A/B`,
  `temperature_core/surface`, `vibration_proxy`, `ambient_load_proxy`). Coupling is defined in
  `sensortwin/simulation/dynamics.py`: currents track a shared load, voltage sags under current
  draw, core temperature integrates ohmic-style heating.
- **Benign activity** — mild, label-free background events (bumps, constant-amplitude vibration
  bursts, small reverting steps) injected into every class so that "some channel moved" is not,
  by itself, class evidence. `dynamics.benign_activity`, configured by
  `GenConfig.benign_activity`.
- **Severity scale** — a multiplier on every event's sampled severity. `severity_scale < 1`
  shrinks events toward the noise floor: the low-SNR tier of the benchmark.
- **Artifacts** — measurement imperfections applied after event injection (Gaussian noise,
  impulses, calibration offset, clipping, quantization). Events describe the system; artifacts
  describe the sensors. `sensortwin/simulation/noise.py`.
- **Run mode** — a named dataset size: `quick_demo` (2k samples, wiring-grade), `colab_standard`
  (20k, the headline tier), `full_reproduction` (100k). Defined in `configs/synthetic/base.yaml`.
- **Generator version** — an integer stamped into dataset metadata, bumped whenever a code change
  alters generated data for a fixed seed. Results are comparable only within one version.
- **Leakage** — any flow of test-set information into training. Guarded here by: split-disjointness
  asserts (`data/splits.py`), train-only standardization (`ChannelStandardizer`), grouped
  by-engine splits for C-MAPSS, and `GenConfig.normalize=False` by default.

## Models

- **Feature baselines** — logistic regression, random forest, XGBoost on ~10 engineered features
  per channel (statistics, spectral summaries, cross-channel correlations).
  `sensortwin/features/`, `models/baselines.py`.
- **SensorCNN / SensorLSTM** — the deep baselines: stacked 1-D convolutions with global average
  pooling; a bidirectional LSTM with mean pooling.
- **SensorPatchTST** — the main model (PatchTST-inspired). Each channel is cut into overlapping
  patches; each patch becomes a token carrying positional and channel embeddings; a transformer
  encoder attends over all channel-patch tokens jointly; attention pooling summarizes them for
  the classifier. `models/transformer.py`.
- **Patch / patch_len / stride** — a window of `patch_len` consecutive timesteps (default 16),
  taken every `stride` steps (default 8), per channel. 8 channels x 63 patches = 504 tokens at
  T=512.
- **Channel embedding** — a learned per-channel vector added to that channel's patch tokens, so
  attention can tell voltage tokens from vibration tokens.
- **Attention pooling** — a single learned query that softmax-weights the encoder's output tokens
  into one vector; the weights are inspectable but are not an explanation.
- **Masked-patch pretraining** — self-supervised stage: hide a fraction of patch tokens behind a
  learned mask token, reconstruct the hidden patches with MSE, keep the encoder.
  `training/pretrain.py`.
- **Linear probe** — freeze the pretrained encoder, train only the classifier head. Measures
  representation quality separately from fine-tuning capacity.
- **Training-parity recipe** — all deep models train with the same family of settings (AdamW,
  weight decay, label smoothing, cosine warmup, identical augmentation) read from
  `configs/models/*.yaml`, so comparisons measure architecture rather than tuning budget.

## Evaluation

- **Macro-F1** — the headline metric: F1 averaged over classes with equal weight, so rare classes
  count as much as common ones.
- **AUROC** — area under the ROC curve, one-vs-rest, macro-averaged over classes.
- **ECE (expected calibration error)** — average gap between predicted confidence and empirical
  accuracy over confidence bins. Low ECE means "80% confident" is right about 80% of the time.
- **Brier score** — mean squared error between the predicted probability vector and the one-hot
  label; punishes confident wrong answers.
- **Reliability curve** — accuracy plotted against confidence per bin; the visual counterpart
  of ECE.
- **Temperature scaling** — post-hoc calibration: divide logits by a scalar T fitted on the
  validation split. Changes confidence, never the argmax. `evaluation/calibration.py`.
- **Severity sweep** — macro-F1 as a corruption (noise, missing channels, shorter windows) grows
  in strength; the robustness analogue of an accuracy number. `evaluation/robustness.py`.
- **Missing-channel delta** — worst-case macro-F1 drop when one channel is zeroed at test time.
  Note: the shared training recipe includes channel-dropout augmentation, which overlaps this
  probe; the `--no-augment` arm exists to separate the two.
- **Domain shift** — train on one measurement regime (`domain_a`), test on a noisier one
  (`domain_b`); `configs/synthetic/domain_shift.yaml`.
- **Occlusion / integrated gradients** — attribution methods: mask a region and watch the logit
  move; integrate gradients along a path from a baseline input. `evaluation/interpretability.py`.
- **Localization score** — overlap between an attribution map and the generator's ground-truth
  event window, compared against a random-placement baseline. Quantifies whether saliency points
  at the event instead of merely looking plausible.

## Statistics

- **Per-seed run** — one complete train/evaluate cycle where the seed controls data generation,
  splitting, initialization, and shuffling. Reported results aggregate several seeds.
- **Paired comparison** — comparing two models on the same seeds and differencing per seed, which
  removes the shared seed-to-seed variance. `evaluation/statistics.compare_seeds`.
- **t-interval** — the confidence interval for the mean paired difference based on the Student-t
  distribution; used instead of a bootstrap because resampling 3-5 differences produces intervals
  that look precise and are not.
- **Cohen's d (paired)** — mean difference divided by the standard deviation of the differences;
  an effect size that a raw delta does not convey.
- **Sign-test fallback** — when all paired differences are identical the t-test is undefined;
  the exact two-sided sign test gives p = 2 x 0.5^n, which cannot reach 0.05 below n = 6. This
  replaces an old shortcut that reported p = 0.
- **Holm correction** — step-down control of the family-wise error rate when several hypotheses
  are tested at once (10 per-class deltas, or an agent's ablation family). A single comparison
  needs no correction; ten comparisons at α = 0.05 each do.
- **Significance gate** — the agent may call a variant an "improvement" only when the CI excludes
  zero *and* the (Holm-corrected) p-value clears α. `agents/reviewer.py`.

## Agent

- **Planner / runner / reviewer** — the three roles of the constrained experiment loop: propose
  one-variable ablations (Pydantic-validated), execute them across seeds, and issue
  significance-gated verdicts. `sensortwin/agents/`.
- **One-variable rule** — each proposal may change exactly one config knob, enforced in the
  schema, so any effect is attributable.
- **Guardrails** — the schema bounds every knob, the tools refuse writes outside the run
  directory, jobs are bounded, and failed runs appear in the report. Spec §13.5.
