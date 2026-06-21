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
Implemented in v0.6 on **NASA C-MAPSS** turbofan data (14 informative sensors), reframed as 3-stage
health classification (healthy / degrading / critical) by binning remaining-useful-life, with a
grouped-by-engine split so no engine's windows cross train/test. Two studies ship:

- `scripts/real_data_report.py` runs the same slate (XGBoost-features, CNN, SensorPatchTST) **from
  scratch on real data** with the same metrics and robustness sweeps — does the harness and the
  synthetic finding port?
- `scripts/sim2real_transfer.py` pretrains the masked-patch encoder on synthetic data and transfers
  the channel-agnostic *temporal* encoder to C-MAPSS (synthetic C=8 → real C=14, so the channel
  embedding is re-initialized), against a real-pretrained upper bound and a from-scratch lower bound.

**Claimed:** the synthetic pipeline runs on real sensor data and the models can be ranked there.
**Not claimed:** state-of-the-art RUL/health estimation (C-MAPSS is natively a regression benchmark;
the 3-stage binning is a deliberate classification reframing), or full encoder transfer (only the
temporal patch encoder transfers across the channel-count change). The raw C-MAPSS files are a
US-government work and are not redistributed here; run the studies after downloading them, and at
`--mode full` for research-grade numbers rather than the quick-mode wiring figures.

## 7. Metrics
Headline metric is **macro-F1** (classes are imbalanced); also report weighted-F1, per-class
precision/recall, one-vs-rest AUROC, and the confusion matrix. Calibration via ECE, multiclass
Brier, and reliability diagrams. Quick-demo numbers are sanity figures only; run `colab_standard`
for research-grade results. Reproduce with `make baselines` / `make transformer` /
`make robustness-study`; real-data studies with `make real-data` / `make sim2real` (after the
C-MAPSS download).

## 8. Robustness
Evaluated as macro-F1 degradation from clean under: Gaussian-noise severity sweep, shorter
observation windows, missing/zeroed channels, and a domain shift (a noisier measurement regime).
`make robustness-study`. Models degrade differently by inductive bias; absolute robustness depends
on data scale and is not claimed beyond the synthetic benchmark.

## 9. Calibration
ECE / Brier / reliability are reported on clean and shifted test sets. Post-hoc **temperature
scaling** (one parameter fit on validation) is provided; as is standard, models tend to be more
overconfident under distribution shift, and temperature scaling reduces ECE without changing
accuracy.

## 10. Known failure modes
- The transformer is **data-hungry**: at quick-demo scale (~1.4k train) it trails XGBoost-on-features
  and the CNN; its inductive bias is expected to pay off only at larger scale.
- Masked-pretraining benefit is **inconclusive at small scale** and may partly reflect learning the
  generator's regularities; the fair test is `colab_standard` with seed averaging and the
  domain-shift comparison.
- Engineered features + gradient boosting are **surprisingly strong** on simple drift/spike events.
- The hardest classes are `sensor_dropout`, `normal`, and `compound_fault`.

## 11. Ethical / safety considerations
The data is synthetic, so there is no personal or sensitive information. The main risk is
**overclaiming**: treating synthetic accuracy as real-world capability, or attention weights as
explanations. The project deliberately separates supported claims from non-claims and documents
negative results. Attention weights are **not** treated as definitive explanations (Jain & Wallace
2019); occlusion and integrated gradients are used as more faithful diagnostics, checked against the
generator's known event regions.

## 12. Reproducibility
Deterministic from a seed end to end (`set_torch_seed`, leakage-safe splits, train-only
standardization). Quality gates (`make check`: ruff + mypy + pytest) run in CI on Python 3.10/3.12.
Install with `pip install -e ".[ml,dev]"`; regenerate any study with its `make` target. The private
design spec and the run-generated study reports are gitignored; this card and the code are committed.
