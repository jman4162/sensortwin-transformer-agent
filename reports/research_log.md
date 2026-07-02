# Research log

A running record of experiments, hypotheses, and what actually happened — including negative
results. Newest entries first. (Spec §"keep a research log": this is what turns the repo from a
weekend demo into a mini research program.)

---

## 2026-07-02 — the fair headline: parity + hardened task, 5 seeds, both scales

**Question.** With every deep model on the identical recipe and the shortcut channels closed
(generator v2), does the transformer still beat the baselines — and does the "loses at 2k, wins
at 20k" scale story survive?

**Setup.** `make headline` (`scripts/compare_models.py`): 6 models × 5 seeds (0–4) at
`colab_standard` (20k) and the same protocol at 2k. Per seed: regenerate, resplit, retrain.
30 epochs, shared recipe from `configs/models/*.yaml`. Apple M3 Max, torch 2.12.1
(20k on MPS, 2k on CPU), FP32. Artifacts:
`reports/experiment_summaries/headline_comparison.json` and `scale_2k/headline_comparison.json`.

**Result — 20k (test macro-F1, mean ± sample std).** transformer **0.861 ± 0.010** >
cnn 0.774 ± 0.010 > xgboost 0.704 ± 0.003 > logreg 0.649 ± 0.006 > random_forest 0.641 ± 0.004 >
lstm 0.507 ± 0.023. Transformer vs cnn: **Δ = +0.09, 95% t-interval [+0.06, +0.11], paired
p = 0.001, Cohen's d = 4.6**. Holm-surviving per-class gains: `sensor_dropout` +0.35,
`correlated_channel_fault` +0.20, `regime_shift` +0.10, `voltage_sag` +0.06, `compound_fault`
+0.04, `current_spike` +0.02. No gain on the trend classes (`slow_degradation` +0.00,
`thermal_drift` +0.01, n.s.).

**Result — 2k (same protocol).** transformer 0.629 ± 0.032 ≈ xgboost 0.603 ± 0.024 (Δ = +0.03,
CI [−0.02, +0.07], p = 0.16: a tie) > cnn 0.582 > random_forest 0.555 > logreg 0.544 >
lstm 0.384. Per class the transformer *loses* `regime_shift` −0.37 (Holm-significant) and wins
`compound_fault` +0.27, `thermal_drift` +0.09.

**Interpretation.** Three findings, all cleaner than the superseded v1 numbers:
(1) The parity fix strengthened, not weakened, the headline — +0.09 vs the old +0.056 — while
the hardened task lowered everyone's absolute scores (the old 0.900 was partly shortcut
learning). (2) The old "transformer last at 2k (0.435)" result was a training-budget artifact;
under parity it ties for first at 2k. What data actually buys is the structural classes: the
`regime_shift` delta flips sign from −0.37 to +0.10 between 2k and 20k. (3) The gains
concentrate exactly where the architecture's inductive bias predicts — the marginal-preserving
cross-channel fault (+0.20) is now detectable only through channel relationships, and the
transformer is the model that exploits them.

**Caveats.** Synthetic only; one shared recipe (designed around the transformer — the shared
lr × capacity grid in `make tune` exists to bound this and has not yet been run at scale);
n = 5 seeds; MPS/CPU FP32, not the T4 path. Calibration trade-off persists (transformer ECE
0.087; temperature-scaling re-measurement at generator v2 in progress).

**Next.** Fill the calibration re-measurement into the model card; matched-budget
label-efficiency re-run (`ARM_LRS` selection now removes the fine-tune confound); robustness
with and without the `--no-augment` arm; `make tune` at colab_standard.

---

## 2026-07-02 — fairness audit: training parity, hardened task, honest statistics

**Question.** A principal-level review of the repo asked: do the headline claims survive scrutiny?
Three did not. (1) The deep baselines trained with plain Adam and no augmentation while the
transformer got AdamW + weight decay + label smoothing + cosine warmup + augmentation — the
`configs/models/{cnn,lstm}.yaml` recipes existed but were never read. The +0.056 headline measured
tuning budget, not architecture. (2) The transformer's channel-dropout augmentation trains it on
the exact corruption the missing-channel robustness probe applies. (3) Several synthetic classes
were separable by shortcuts: only one class ever wrote to the vibration channel, `normal` was the
only class where nothing happened, and `correlated_channel_fault` inflated `voltage_A` variance
despite claiming to be invisible to single-channel statistics.

**Changes.**

- *Parity:* all deep models now load one shared recipe family from `configs/models/*.yaml`
  through a single code path; `scripts/tune_baselines.py` gives every model the same
  lr x capacity grid, selected on validation macro-F1 only. `--no-augment` adds the
  unaugmented arm so the robustness tables can separate "trained on the probe" from "robust."
- *Task hardening (generator v2):* benign label-free background activity in every class,
  load-tracking vibration baseline, a marginal-preserving `correlated_channel_fault` (segment
  mean/std preserved exactly; only the cross-channel relationship rotates), and a
  `severity_scale` knob for a low-SNR tier. Signature tests now assert each class's intended
  statistical fingerprint.
- *Statistics:* paired t-intervals replace bootstrap CIs (resampling 3 paired differences
  produces precision that is not there); sample std (ddof=1); paired Cohen's d; an exact
  sign-test fallback for zero-variance differences (the old code reported p=0.0 as significant);
  Holm correction over per-class deltas and the agent's ablation family.
- *Reproducibility:* `scripts/compare_models.py` + `make headline` reproduce the headline table
  in one command and write committable `headline_comparison.{json,md}` with per-seed values,
  device, and library versions; `requirements-lock.txt` pins the environment; an end-to-end
  seed→metric determinism test guards the chain; C-MAPSS inputs are SHA-256-stamped.

**Consequence.** Every number reported before this date was produced by generator v1 under
asymmetric training and is **superseded**. The 2026-06-23 "+0.056, p=0.006" comparison should be
read as historical record, not as a current claim. The fair 5-seed re-run at colab_standard is in
progress; its numbers replace the old headline wherever they land — including if the transformer's
lead shrinks or disappears, which would itself answer the project's research question honestly.

**Caveats.** The shared recipe is the transformer's spec-default recipe applied to everyone —
parity by construction, not per-model tuning; `make tune` documents the grid protocol for
refinement. Early signals from the re-run suggest the parity CNN is stronger than previously
reported, which was the point of the exercise.

**Next.** Land the 5-seed fair headline (`make headline`), reconcile README / model card to it,
then re-run calibration, label efficiency (with matched fine-tune budgets), robustness, and
interpretability at colab_standard.

---

## 2026-06-25 — colab_standard follow-ups: calibration win, pretraining null

**Question.** (1) Does temperature scaling fix the transformer's overconfidence at scale? (2) Does
masked-patch pretraining improve label efficiency at colab_standard?

**Setup.** colab_standard (20k), T4 GPU. Calibration: train the transformer (1 seed, 15 epochs),
fit one temperature on the validation logits (`evaluation/calibration.fit_temperature`), re-measure
ECE on test. Label efficiency: `scripts/label_efficiency_sweep` at a reduced first-run budget —
1 seed, pretrain 30 epochs / fine-tune 15 epochs — over label fractions {1, 5, 10, 100}%.

**Result — calibration.** ECE **0.092 → 0.017 at T=0.62**; argmax is invariant, so macro-F1 is
unchanged. The transformer's overconfidence is corrected by a single parameter.

**Result — label efficiency (test macro-F1).**

| Fraction | scratch | pretrained_ft | pretrained_probe | cnn | xgboost |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 0.100 | 0.124 | 0.035 | 0.352 | 0.429 |
| 5% | 0.260 | 0.129 | 0.097 | 0.520 | 0.604 |
| 10% | 0.519 | 0.219 | 0.124 | 0.613 | 0.664 |
| 100% | 0.893 | 0.744 | 0.168 | 0.832 | 0.764 |

**Interpretation.** Calibration: a clean win — "most accurate **and** well-calibrated after a
one-parameter fit." Label efficiency: masked pretraining **did not help** — `pretrained_ft` ≤
`scratch` at every fraction except a hair at 1%, and the frozen linear probe is near-useless. Feature
baselines (xgboost, cnn) dominate at 1–10% — the data-hungry story again.

**Caveats.** Label efficiency is single-seed and under a reduced budget. A real confound: the
pretrained arms fine-tune at lr 1e-4 for only 15 epochs while `scratch` uses lr 1e-3, so they are
likely undertrained — the 100% gap (0.744 vs 0.893) is the tell. This is "did not help *at this
budget*," not "pretraining doesn't transfer."

**Next.** Fair re-run: `--epochs-finetune 30`+ (ideally a matched fine-tune LR) with ≥3 seeds before
concluding. The recipe (40% masking, MSE reconstruction) may also simply not transfer on this
synthetic benchmark — itself a legitimate finding once the budget is matched.

---

## 2026-06-23 — colab_standard comparison: transformer overtakes the baselines at scale

**Question.** Does `SensorPatchTST` overtake the baselines at `colab_standard` scale, and on which
classes — the headline question open since v0.3?

**Setup.** Full slate (logreg / random_forest / xgboost / cnn / lstm / transformer) at colab_standard
(20k, T=512), **3 seeds (0,1,2)**, 15 epochs, Tesla T4 with auto-AMP. Each seed is an independent
draw (data + split + init); leakage-safe 70/15/15; `ChannelStandardizer` fit on train. Run via
`notebooks/04_colab_standard_comparison.ipynb`; aggregated with `evaluation/statistics.py`.

**Result (test, mean ± std over 3 seeds).**

| Model | Macro-F1 | Macro-AUROC | ECE |
| --- | ---: | ---: | ---: |
| transformer | 0.900 ± 0.003 | 0.990 ± 0.000 | 0.095 ± 0.006 |
| cnn | 0.844 ± 0.004 | 0.983 ± 0.001 | 0.023 ± 0.002 |
| xgboost | 0.759 ± 0.004 | 0.967 ± 0.000 | 0.042 ± 0.004 |
| logreg | 0.697 ± 0.006 | 0.950 ± 0.001 | 0.019 ± 0.003 |
| random_forest | 0.691 ± 0.003 | 0.950 ± 0.000 | 0.185 ± 0.003 |
| lstm | 0.545 ± 0.022 | 0.902 ± 0.008 | 0.056 ± 0.009 |

Transformer vs best baseline (cnn): **+0.056 macro-F1, 95% CI [0.048, 0.061], paired p=0.006**.
Per-class gain (transformer − cnn) is largest on `sensor_dropout` +0.135, `regime_shift` +0.120,
`normal` +0.092; all reported per-class deltas are ≥ 0.

**Interpretation.** The quick-demo ranking reverses: at 2k features+GBM led and the transformer was
last (0.435); at 20k the deep models overtake and the transformer significantly tops the strongest
baseline. The transformer's edge over the cnn is real but modest (+0.056); the larger effect is
deep-vs-features. Its gains concentrate on the hard / cross-channel classes its inductive bias
targets — the benchmark working as designed. **Trade-off:** the transformer is the most accurate but
the least calibrated (ECE 0.095 vs cnn 0.023); random_forest is worst (0.185). Accuracy ranking ≠
calibration ranking — a case for the v0.5 temperature scaling.

**Caveats.** Synthetic data — not real-world validation. 15 epochs may not be fully converged (more
could lift the deep models further). Single GPU / AMP; 3 seeds, one split family.

**Next.** Temperature-scale the transformer (v0.5 `fit_temperature`) to close the calibration gap;
run the label-efficiency and robustness studies at `colab_standard` (the remaining wiring-grade
items).

---

## 2026-06-22 — colab_standard GPU run: transformer at scale

**Question.** Does `SensorPatchTST`'s inductive bias pay off at `colab_standard` scale — the open
question since v0.3, where the transformer trailed the baselines at quick-demo?

**Setup.** colab_standard (20,000 samples, T=512, seed=0), default `configs/models/
sensorpatchtst.yaml`, 15 epochs, on a Tesla T4 with auto mixed precision and a pinned multi-worker
DataLoader. Leakage-safe 70/15/15 split; `ChannelStandardizer` fit on train only.

**Result.** Validation macro-F1 0.909; **test macro-F1 0.905, macro-AUROC 0.991**. The same model at
quick-demo (2k) scored 0.435 — roughly a 2x jump from 10x the data.

**Interpretation.** Confirms the data-hungry hypothesis: with enough data the transformer's capacity
flips the quick-demo result. **Not claimed:** that it *beats the baselines* at this scale — the
`colab_standard` baselines (XGBoost-on-features, CNN, LSTM) have not been run, and they may also rise
at 20k. The project's headline question ("does the transformer overtake the baselines at scale, and on
which classes") needs the full slate at the same scale with seed averaging.

**Caveats.** Single seed (0); transformer-only; one split. AMP is seed-deterministic on a fixed GPU
but not FP32-bit-identical, so expect small run-to-run variation across hardware.

**Next.** Run `train_baseline --mode colab_standard --models logreg,xgboost,cnn,lstm,transformer`
over ≥3 seeds for the apples-to-apples table, then the label-efficiency and robustness studies at
`colab_standard` (wiring-grade until now). Once those land, update the README Results headline.

---

## 2026-06-21 — v0.7 agentic experiment runner + Colab GPU readiness

**Question.** Can a constrained agent run useful one-variable ablations and write an honest failure
analysis — without overclaiming — and is the training path ready for a real GPU run?

**Setup.** A deterministic planner → runner → reviewer loop (`sensortwin/agents/`): the planner reads
baseline metrics and proposes ≤3 one-variable ablations (hypothesis + expected failure mode), the
runner trains each across seeds via the existing `train_model`, and the reviewer significance-tests
each variant vs the baseline (`evaluation/statistics.py`, paired t-test + bootstrap CI) before
calling anything an improvement. Guardrails (spec §13.5) are enforced by Pydantic schemas. GPU path:
auto mixed-precision + pinned/worker DataLoader when CUDA is detected (CPU stays bit-identical),
`--device`/`--no-amp` flags, a macOS-only OMP guard, and a Colab notebook.

**Result.** The loop runs end to end and the guardrails hold. On a quick-demo wiring run the planner
proposed `d_model=192`; the reviewer measured −0.032 macro-F1 (p=0.51) and reported **no_change**, not
an improvement — which is exactly the proposal's own pre-registered "expected failure" (the wider
model is data-hungry and overfits at small scale). CPU training is numerically unchanged (the
existing training/pretrain tests pass bit-for-bit); AMP/workers engage only on CUDA.

**Interpretation.** The agent automates the *workflow* and refuses to overclaim — the honest outcome
the spec asks for. The point demonstrated here is the discipline (one variable per ablation, bounded
jobs, significance-gated claims, every run reported, claims separated from measurements), not a new
result. Real verdicts need `colab_standard` with the full seed budget.

**Caveats.** Quick-demo scale; the planner is a fixed heuristic (LLM backend is a documented seam,
not built). AMP trades FP32-bit-identity for ~1.5-2x GPU speed while staying seed-deterministic.

**Next.** Run `make agent` and the Colab notebook at `colab_standard` for research-grade numbers;
optionally backfill seed error-bars into the v0.3-v0.6 sweeps now that `statistics.py` exists.

---

## 2026-06-21 — v0.6 open-data validation on NASA C-MAPSS

**Question.** Does the benchmark pipeline port to *real* multichannel sensor data, and does
masked-patch pretraining on synthetic data transfer to it — or did it just learn the generator?

**Setup.** A C-MAPSS adapter (`data/cmapss.py`) windows the 14 informative turbofan sensors and bins
remaining-useful-life into 3 health stages (healthy / degrading / critical). A grouped-by-engine
split prevents windows from one engine straddling train/test. Two runners: `real_data_report.py`
trains XGBoost-features / CNN / SensorPatchTST from scratch on real data with the same metrics +
robustness sweeps; `sim2real_transfer.py` pretrains the masked-patch encoder on synthetic (C=8),
transfers the channel-agnostic temporal encoder to C-MAPSS (C=14, channel embedding re-initialized),
and compares it against a real-pretrained upper bound, a from-scratch lower bound, and XGBoost across
label fractions.

**Result.** Method and apparatus delivered and CI-verified: the adapter, grouped split, and
channel-flexible `transfer_encoder(strict_channels=False)` are covered by unit tests on a synthetic
C-MAPSS fixture (no download needed), and `make check` is green. **No real-data numbers are claimed
yet** — the raw C-MAPSS files are not committed, so the studies run once the dataset is downloaded
and pointed at via `--raw-dir`, at `--mode full` for research-grade figures.

**Interpretation.** This phase makes the synthetic-to-real check *runnable* and closes the model
card's open item; the verdict (does synthetic-pretraining transfer? does the synthetic ordering of
models hold on real turbofan data?) waits on the actual run. The honest framings are baked in: only
the temporal encoder transfers across the channel-count change, and the 3-stage binning is a
deliberate classification reframing of a regression benchmark.

**Caveats.** C-MAPSS is natively RUL regression; binning is a modeling choice. FD001 (+FD003) only;
the six-operating-condition subsets (FD002/FD004) need condition-aware normalization, deferred.

**Next.** Download C-MAPSS, run both studies at `--mode full`, and report whether synthetic-encoder
transfer beats scratch and approaches the real-pretrained bound. Then v0.7: the agentic runner.

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
