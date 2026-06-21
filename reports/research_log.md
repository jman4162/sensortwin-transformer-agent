# Research log

A running record of experiments, hypotheses, and what actually happened — including negative
results. Newest entries first. (Spec §"keep a research log": this is what turns the repo from a
weekend demo into a mini research program.)

---

## 2026-06-20 — v0.5 robustness, calibration, interpretability

**Question.** How do the models degrade under corruption and domain shift, are they calibrated, and
does the transformer's saliency actually localize the injected events?

**Setup.** quick_demo, trained on domain A; evaluated on the test split under Gaussian-noise / short-
window / missing-channel sweeps and on a noisier domain B. ECE clean vs shift, before/after
temperature scaling. Interpretability: attention vs integrated-gradients saliency, scored against the
generator's known event region (`localization_score`) vs a random baseline.

**Result (quick_demo wiring).** Robustness: xgboost clean macro-F1 0.62 but collapses to 0.25 under
domain shift; CNN is the most shift-robust (0.40 → 0.38); transformers undertrained at 2 epochs
(~0.06–0.10). Temperature scaling cut the CNN's shifted ECE 0.15 → 0.05. Interpretability (27
localized test events): **integrated gradients localized the event (0.234) above the random baseline
(0.154); attention pooling did not (0.139, below random).**

**Interpretation.** Two honest findings, both expected: (1) the strongest in-distribution model
(xgboost) is the *least* robust to domain shift, while the CNN trades accuracy for robustness — a
real model-selection tension. (2) The faithful, perturbation/gradient-based saliency tracks the
ground-truth event, but **attention weights do not** — concrete evidence for the project's standing
caveat that attention is a diagnostic, not an explanation (Jain & Wallace 2019). Temperature scaling
behaves as theory predicts.

**Caveats.** quick_demo + few epochs: transformers undertrained, single seed; numbers are wiring-
grade. Domain B is a controlled regime change, not a real deployment shift.

**Next.** v0.6 open-dataset (NASA) for the synthetic-to-real check; re-run the studies at
`colab_standard` once the transformer is properly trained, and confirm whether attention localization
improves with training (it may, but IG remains the more faithful tool).

---

## 2026-06-20 — v0.4 masked pretraining + label-efficiency (wired)

**Question.** Does masked-patch self-supervised pretraining improve `SensorPatchTST`'s sample
efficiency — i.e. beat a from-scratch transformer most when labels are scarce (spec §12.4)?

**Setup.** Pretrain once on the unlabeled train split (40% per-channel patch masking, BERT-style
learnable mask token, MSE on masked patches). Then a 5-arm sweep across {1, 5, 10, 100}% label
fractions: scratch transformer, pretrained+fine-tuned, pretrained+linear-probe, CNN, XGBoost-
features — all on the same split, scored by test macro-F1.

**Hypothesis.** Pretraining helps most at 1–10% labels; the gap to scratch closes by 100% (the
spec's documented "what did not work" pattern). XGBoost stays the strong low-data anchor.

**Result.** Pipeline verified end to end on quick_demo (pretrain MSE decreases; all arms train and
score). **No conclusion drawn:** at quick-demo scale with the 2-epoch wiring budget the transformer
arms are barely trained (macro-F1 < 0.1), so the curve is dominated by XGBoost and the
pretrain-vs-scratch deltas are within noise. The label-efficiency question needs `colab_standard`
(20k) with the full pretrain/fine-tune epoch budget and several seeds.

**Interpretation.** This phase delivers the *method and the honest measurement apparatus*, not a
verdict. Per the model-zoo note, any gain may partly reflect the encoder learning the synthetic
generator's regularities rather than transferable structure; a domain-shift test (v0.5/v0.6) is the
real check.

**Next.** Run `make label-efficiency` at `colab_standard` with 3–5 seeds; report the curve with
error bars and state plainly whether pretraining helped, at which fractions, and where it did not.

---

## 2026-06-19 — v0.3 SensorPatchTST vs. baselines

**Question.** Does the patch-transformer beat the strong v0.2 baselines on `SensorTwin-Synth`, and
at what data scale does its inductive bias start to pay off?

**Setup.** Same quick_demo data/split as v0.2 (2,000 samples, T=512, seed=0, 70/15/15).
`SensorPatchTST`: per-channel patching (patch_len=16, stride=8 → 8×63 tokens), learned channel
embedding, sinusoidal positions, 4-layer pre-norm encoder (d_model=128, 4 heads), attention pooling.
Trained with AdamW + weight decay 0.01, label smoothing 0.1, cosine-warmup, and jitter/scaling/
channel-dropout augmentation. 814k params.

**Hypothesis.** At ~1.4k training samples the transformer would *trail* the cheaper baselines (it is
the most data-hungry model); its advantage on cross-channel/compound events should only appear at
larger scale.

**Result (test split, quick_demo).** Macro-F1: xgboost 0.620 > cnn 0.504 > **transformer 0.435**.
Transformer ECE 0.143, train time 459 s on CPU (vs 5 s for xgboost). It does not win any class here.

**Interpretation.** Hypothesis confirmed: more capacity + less data = worse, and the augmentation /
regularization did not close the gap at this scale. This is the intended "transformer is not the
hero by default" result — reported honestly rather than tuned away. The fair test is `colab_standard`
(20k) and, later, masked pretraining (v0.4) for label efficiency.

**Caveats.** Single seed, quick-demo scale, CPU timings. Not a verdict on the architecture.

**Next.** (1) Re-run at `colab_standard` with seed averaging. (2) v0.4 masked-patch pretraining →
does pretraining + fine-tuning beat supervised-from-scratch at low label fractions? (3) Read the
`make ablate` output: did channel-embedding / attention-pooling help as designed?

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
