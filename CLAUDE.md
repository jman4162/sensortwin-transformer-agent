# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repository is **greenfield**. The only content so far is `BACKGROUND-SPEC-GITIGNORE.md`,
a detailed design spec authored by the project owner. No source code, `pyproject.toml`,
tests, or git history exist yet. When implementing, you are building from the spec, not
modifying existing code.

`BACKGROUND-SPEC-GITIGNORE.md` is the **source of truth** for scope, architecture, naming,
and acceptance criteria. The filename indicates it is a private planning doc meant to be
gitignored — do not commit it to a public remote, and do not reproduce its contents in
public-facing files (README, reports) verbatim.

## What this project is

`sensortwin-transformer-agent` — a reproducible, research-style benchmark for **multichannel
sensor event classification**. It is a portfolio/educational artifact, not a production
system. Three layers, built strictly in this order (do not start with the agent or a large
model):

1. **Simulation**: synthetic 8-channel, physics-inspired time-series generator with 10 known
   event classes (`normal`, `thermal_drift`, `voltage_sag`, `current_spike`, `sensor_dropout`,
   `oscillatory_instability`, `correlated_channel_fault`, `regime_shift`, `slow_degradation`,
   `compound_fault`). Deterministic given a seed.
2. **Modeling**: classical/feature baselines, CNN, LSTM, then the main model `SensorPatchTST`
   (PatchTST-inspired: per-channel patching + channel embeddings + transformer encoder over
   channel-patch tokens + attention pooling). Plus masked-patch self-supervised pretraining.
3. **Agentic runner**: a constrained planner/runner/reviewer loop that reads metrics, proposes
   one-variable ablations, patches configs, launches runs, and writes markdown reports. It
   automates the experiment workflow; it is not the core novelty.

## Architecture conventions (from the spec)

- **Tensor layout:** model input is `x: FloatTensor[B, C, T]` with optional `mask: BoolTensor[B, C, T]`.
  Defaults: `C = 8`, `T = 512`. Patchify with `patch_len ∈ {16, 32}`, `stride ∈ {8, 16}`.
- **Default transformer config:** `d_model=128`, `num_layers=4`, `num_heads=4`, `dropout=0.1`,
  `mlp_ratio=4`, GELU. Keep it Colab-friendly.
- **Config-driven:** YAML configs under `configs/` (PyYAML or OmegaConf), separated into
  `synthetic/`, `models/`, `agents/`. Notebooks use small defaults; configs hold the larger
  research-reproduction defaults. Support named run modes (`quick_demo`, `colab_standard`,
  `full_reproduction`).
- **Package layout:** code lives in `sensortwin/` with subpackages `simulation/`, `data/`,
  `features/`, `models/`, `training/`, `evaluation/`, `agents/`, `utils/`. CLI entrypoints in
  `scripts/` (`generate_synthetic.py`, `train_baseline.py`, `train_transformer.py`,
  `pretrain_transformer.py`, `evaluate.py`, `run_agent.py`). See spec §14 for the full tree.
- **Stack:** Python + PyTorch + NumPy + scikit-learn + Matplotlib; PyTest, Ruff, Black. Pydantic
  for agent tool/result schemas. Keep dependencies light; avoid heavy frameworks early.

## Non-negotiable project principles

These are the point of the project — preserve them in any implementation:

- **Reproducibility:** every dataset and experiment must be deterministic given a seed. Save
  metrics as JSON/CSV and confusion matrices to disk.
- **Baselines are first-class:** never present `SensorPatchTST` as the hero without strong
  baselines (feature+LogReg/RF/GBM, CNN, LSTM, anomaly detector). The research question is
  *when* the transformer's inductive biases help, not that it always wins.
- **Evaluation beyond accuracy:** report macro-F1 (emphasized due to class imbalance), per-class
  precision/recall, AUROC, calibration (ECE, Brier, reliability), and robustness deltas under
  noise / missing channels / domain shift / short windows / rare events.
- **Honest reporting:** document negative results and failure modes explicitly. Keep a
  `reports/research_log.md`. In public docs, separate "supported claims" from "not claimed."
  Do not overclaim attention as explanation, or call this a "foundation model."
- **Agent guardrails (spec §13.5):** the agent must not delete data, overwrite baseline results,
  run unbounded jobs, change more than one variable per ablation, claim improvements without
  statistical evidence, or hide failed experiments. Use Pydantic schemas, not free-form prompts;
  validate generated configs before running.

## Writing style: avoid AI slop

Before finishing any writing artifact (README, `reports/**`, future `model_card.md` and
`paper_style_report.md`, docstrings, commit messages, PR bodies), scrub it for the patterns in
`AI_WRITING_SLOP_Guide.md` (gitignored, repo root). This applies to generated prose too: the report
strings in `scripts/train_baseline.py` produce a committed `.md`, so edit them, not just the output.

Top tells to avoid:
- Puffery and significance-inflation ("stands as a testament", "plays a vital role", "marking a
  pivotal moment", "ever-evolving landscape").
- Trailing "-ing" superficial analyses (", highlighting its importance", ", underscoring the role").
- AI-vocabulary words used reflexively: delve, crucial, pivotal, robust, seamless, leverage, foster,
  garner, intricate, showcase, underscore, tapestry, vibrant, realm.
- Rule-of-three padding, gratuitous em-dashes, "Despite its challenges…" outros, and vague
  attributions ("experts say", "studies show") with no concrete source.

Prefer specific, falsifiable statements over editorializing. State the result; do not narrate its
significance.

## Commands

Install: `pip install -e ".[dev]"` (core + tooling) or `".[ml,dev]"` (adds torch/sklearn/xgboost/
matplotlib — needed for models, training, evaluation, and the baseline script).

- Quality gates (all green; enforced in CI): `make check` = `ruff check .` + `mypy sensortwin scripts` + `pytest`.
  Format with `black .` (CI runs `black --check .`). Single test: `pytest tests/test_metrics.py::test_brier_bounds`.
- Generate data: `make data` or `python -m scripts.generate_synthetic --mode quick_demo`.
- Train + evaluate baselines: `make baselines` or `python -m scripts.train_baseline --mode quick_demo --epochs 10`
  → writes `reports/experiment_summaries/baseline_results.md` (+ metrics JSON/CSV and figures, gitignored).

**macOS gotcha (already handled):** torch and xgboost each bundle an OpenMP runtime and
segfault/deadlock when used in one process. `tests/conftest.py` and the top of
`scripts/train_baseline.py` set `OMP_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK=TRUE` before importing
either, and the sklearn/xgboost models use `n_jobs=1`. Keep these when adding code that touches both.

## Implemented so far

- **v0.1** — `simulation/` generator, `data/` (dataset, splits, `ChannelStandardizer`), `features/statistical.py`, `utils/`.
- **v0.2** — `features/` (spectral, correlations, `build_feature_matrix`), `models/` (baselines.py:
  LogReg/RF/XGBoost + IsolationForest; cnn.py; lstm.py), `training/loop.py`, `evaluation/`
  (metrics, calibration, robustness, plots), `scripts/train_baseline.py`.
- **v0.3** — `models/transformer.py` (`SensorPatchTST`), `training/augment.py` + opt-in `train_model`
  args (AdamW/weight-decay/label-smoothing/cosine-warmup/augment; defaults keep baselines unchanged),
  `configs/models/sensorpatchtst.yaml`, `scripts/ablate_transformer.py`. The transformer plugs into
  `train_baseline`'s `_run_deep` (run with `--models transformer`).

CI (`.github/workflows/ci.yml`) runs a fast `lint` job (core+dev: ruff/black/mypy) and a `test` job
(installs `ml` extra so baselines/metrics are exercised) on Python 3.10 + 3.12. Library submodules
that need the `ml` extra are intentionally not imported in package `__init__.py` files, so the core
package imports without torch/sklearn; tests gate those with `pytest.importorskip`.
