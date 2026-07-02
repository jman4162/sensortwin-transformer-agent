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
- **v0.4** — `training/pretrain.py` (masked-patch `pretrain_model` + `freeze_encoder`/
  `transfer_encoder`), `SensorPatchTST.embed`/`encode`/`mask_token` seams, `configs/models/
  sensorpatchtst_pretrain.yaml`, `scripts/label_efficiency_sweep.py` (5-arm sweep:
  scratch / pretrained-ft / pretrained-probe / cnn / xgboost across {1,5,10,100}% labels;
  `make label-efficiency`).
- **v0.5** — `evaluation/robustness.py` (noise/short-window severity sweeps), `calibration.py`
  (temperature scaling), `interpretability.py` (attention map, occlusion, integrated gradients,
  `localization_score` vs generator ground truth), `loop.predict_logits`,
  `configs/synthetic/domain_shift.yaml`, `scripts/robustness_report.py` +
  `scripts/interpretability_report.py` (`make robustness-study` / `make interpretability`), and the
  committed `reports/model_card.md`.
- **v0.6** — open-data validation on NASA C-MAPSS: `data/cmapss.py` (`load_cmapss` →
  `(X[N,14,T], y, meta)`, RUL binned to 3 health stages), `splits.make_split("grouped", ...)`
  (by-engine, no leakage), `pretrain.transfer_encoder(..., strict_channels=False)` (transfers the
  channel-agnostic temporal encoder across C=8→14), `utils/config.load_mode_config`,
  `configs/data/cmapss.yaml`, `scripts/fetch_cmapss.py` + `scripts/real_data_report.py`
  (`make real-data`) + `scripts/sim2real_transfer.py` (`make sim2real`). Raw C-MAPSS `.txt` is **not
  committed** (US-gov work; download and pass `--raw-dir`); tests run on a synthetic C-MAPSS fixture,
  so CI needs no download. The committed artifacts are the model-card §6 update + research-log entry,
  not the gitignored auto reports.
- **v0.7** — agentic experiment runner (Layer 3) + Colab GPU readiness. Agent: `agents/schemas.py`
  (Pydantic guardrails — one-variable rule + bounds), `tools.py` (`write_config`/`merge_overrides`,
  refuses to touch `configs/`/`data/`), `planner.py` (`Planner` Protocol + deterministic
  `HeuristicPlanner`, `LLMPlanner` stub), `runner.py` (`ExperimentRunner`, reuses `train_model`),
  `reviewer.py` (significance-gated verdicts via `evaluation/statistics.py`), `scripts/run_agent.py`
  (`make agent`), `configs/agents/experiment_agent.yaml`. GPU: `utils/runtime.py` `configure_omp()`
  (Darwin-only OMP guard, replaces the unconditional blocks), auto-AMP + pinned/worker DataLoader in
  `train_model`/`pretrain_model` (`None`=auto-by-device; CPU path bit-identical), `--device`/`--no-amp`
  flags on the deep scripts, `utils/config.save_yaml`, `notebooks/03_transformer_training_colab.ipynb`.
  The auto `agentic_ablation_report.md` is gitignored; agent/schema/stats tests are torch-free, with a
  tiny torch-gated end-to-end loop test.
- **Results + viz** — committed portfolio figures live in `docs/figures/` (not gitignored),
  regenerated by `scripts/make_figures.py` from committed `headline_comparison.json` artifacts and
  rendered by `evaluation/plots.py` helpers (`plot_signal_gallery`/`plot_scale_comparison`/
  `plot_perclass_delta`/`plot_saliency_overlay`); the README "At a glance" section embeds them.
- **v0.8 (2026-07-02) — fairness overhaul.** All numbers produced before this date are superseded
  (see the research-log entry of the same date): the deep baselines had trained without the
  transformer's recipe (dead `configs/models/{cnn,lstm}.yaml`), and the generator had shortcut
  channels. Now: `_build_deep_model` in `scripts/train_baseline.py` loads every deep model's config
  through one path (shared AdamW/label-smoothing/cosine-warmup/augment recipe family; `--no-augment`
  arm); generator v2 (`GENERATOR_VERSION` in `simulation/generator.py`) adds benign background
  activity, a load-tracking vibration baseline, a marginal-preserving `correlated_channel_fault`,
  and `severity_scale`; `GenConfig.normalize` defaults to False. `evaluation/statistics.py` uses
  paired t-intervals + `holm_bonferroni` + Cohen's d (no bootstrap, no p=0 shortcut; sample std).
  New: `scripts/compare_models.py` (`make headline` = colab_standard x 5 seeds → committable
  `headline_comparison.{json,md}`), `scripts/tune_baselines.py` (`make tune`, shared lr x capacity
  grid), `make full-reproduction`, `requirements-lock.txt`, `CITATION.cff`,
  `docs/{glossary,tutorial_guide}.md`, `reports/paper_style_report.md`. The headline numbers in
  README/model_card must always trace to the committed `headline_comparison.json`.

CI (`.github/workflows/ci.yml`) runs a fast `lint` job (core+dev: ruff/black/mypy) and a `test` job
(installs `ml` extra so baselines/metrics are exercised) on Python 3.10 + 3.12. Library submodules
that need the `ml` extra are intentionally not imported in package `__init__.py` files, so the core
package imports without torch/sklearn; tests gate those with `pytest.importorskip`.
