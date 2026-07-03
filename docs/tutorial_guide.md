# Tutorial guide

A walkthrough of the repo, ordered the way the project was built: data, models, evaluation,
agent. Commands assume a virtualenv with `pip install -e ".[ml,dev]"` (macOS/CPU is fine for
everything at `quick_demo` scale; use a GPU or Apple-silicon `--device mps` for
`colab_standard`). Unfamiliar terms are defined in [glossary.md](glossary.md).

## 1. Generate and look at the data

```bash
python -m scripts.generate_synthetic --mode quick_demo        # -> data/synth_quick_demo.npz
```

Everything about a dataset is determined by its seed and config: `GenConfig` in
`sensortwin/simulation/generator.py`. Three things are worth internalizing before training
anything:

- The 8 channels are coupled (currents share a load, voltage sags under draw, temperature
  integrates heating) — see `simulation/dynamics.py`. Some event classes only exist in those
  relationships.
- Every class, including `normal`, receives mild benign background activity, so a model cannot
  win by asking "did any channel move?"
- `meta["events"]` records each sample's ground truth (class, start, duration, severity,
  affected channels). The interpretability study scores attribution maps against it.

`notebooks/03_transformer_training_colab.ipynb` starts with a signal gallery — one example per
class — worth a look before anything else.

## 2. Train the baselines first

```bash
python -m scripts.train_baseline --mode quick_demo --epochs 10
# -> reports/experiment_summaries/baseline_results.md
```

This trains feature baselines (logistic regression, random forest, XGBoost on engineered
features) and the deep baselines (CNN, LSTM), and evaluates all of them: macro-F1, per-class
F1, AUROC, ECE, Brier, and a missing-channel robustness delta. Add the transformer with
`--models logreg,xgboost,cnn,lstm,transformer`.

Two design decisions matter here:

- **Parity.** Every deep model reads its recipe from `configs/models/*.yaml`, and those recipes
  are one shared family (AdamW, weight decay, label smoothing, cosine warmup, identical
  augmentation). A comparison under unequal tuning budgets measures effort, not architecture.
- **Leakage hygiene.** Standardization is fit on the train split only; the generator's own
  whole-dataset normalization is off by default.

## 3. The headline comparison

```bash
make headline-quick        # wiring check: 3 seeds at 2k, minutes on CPU
make headline              # the real thing: 5 seeds x 6 models at 20k (GPU/MPS, hours)
```

`scripts/compare_models.py` is the one command behind the README's headline table. Per seed it
regenerates the data, resplits, retrains every model; across seeds it reports mean ± sample
std, a paired t-interval and t-test for transformer-vs-best-baseline, Cohen's d, and
Holm-corrected per-class deltas. Output: `reports/experiment_summaries/headline_comparison.{md,json}`
with every per-seed number, the device, and library versions.

To re-tune the shared recipe rather than trust the committed one:

```bash
make tune                  # same lr x capacity grid for every deep model, selected on val macro-F1
```

## 4. Pretraining and label efficiency

```bash
make label-efficiency        # wiring-grade
make label-efficiency-full   # research-grade: 3 seeds, matched LR budgets (GPU/MPS)
```

Masked-patch pretraining (`training/pretrain.py`) hides 40% of each channel's patches behind a
learned mask token and reconstructs them. The sweep then compares scratch training,
fine-tuning, and a linear probe at 1/5/10/100% of labels, against CNN and XGBoost at the same
label budgets. The research log records a null result at the budgets tried so far — pretraining
has not yet beaten scratch training here — and the confound that explains why that verdict is
still provisional.

## 5. Robustness, calibration, interpretability

```bash
make robustness-study      # noise / short-window / missing-channel / domain-shift sweeps
make interpretability      # attention, occlusion, integrated gradients + localization score
make robustness-full       # research-grade: 3 seeds, augmented + no-augment arms
make interpretability-full # research-grade: 3 seeds; also writes the README confusion matrix
```

The evaluation philosophy: accuracy at one operating point is the least interesting number.
`evaluation/robustness.py` measures degradation curves; `evaluation/calibration.py` measures
whether confidence means anything (and fixes it with temperature scaling);
`evaluation/interpretability.py` checks attribution maps against the generator's ground truth
instead of eyeballing them.

## 6. Real data (optional)

Download NASA C-MAPSS (not redistributed here), then:

```bash
make real-data RAW_DIR=/path/to/CMAPSSData
make sim2real              # synthetic-pretrained encoder -> C-MAPSS fine-tune
```

`fetch_cmapss.py` prints the SHA-256 of each raw file and stamps it into the processed
dataset's metadata; pin the hashes in `configs/data/cmapss.yaml` after your first download.
Splits are grouped by engine so no engine appears in both train and test.

## 7. The agent

```bash
make agent
```

The agentic runner (`sensortwin/agents/`) automates one loop: read baseline metrics, propose
one-variable ablations (Pydantic schema enforces the one-variable rule and value bounds), run
each across seeds, and issue verdicts through the significance gate — an "improvement" claim
requires the Holm-corrected paired test to clear α, not a bigger mean. Read
`agentic_ablation_report.md` afterwards; failed runs and regressions are listed, not hidden.

## 8. Extending the benchmark

- **New event class**: add an injector to `simulation/events.py` (take `(X, rng, scale)`,
  return `EventMeta`), register it in `INJECTORS` and `EventClass`, and add a signature test to
  `tests/test_simulation.py` asserting the statistical fingerprint you intended.
- **New model**: implement `[B, C, T] -> [B, num_classes]`, add a config under
  `configs/models/` using the shared recipe family, and wire it into
  `scripts/train_baseline.DEEP_CONFIGS`.
- **New robustness probe**: add a corruption function to `evaluation/robustness.py` and sweep
  it with `severity_sweep`.

Every change should keep the three project invariants: deterministic from a seed, baselines
get the same budget as the hero model, and negative results are reported.
