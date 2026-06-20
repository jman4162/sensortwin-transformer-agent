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

1. **Simulation** — synthetic 8-channel, physics-inspired time-series generator with 10 known
   event classes (`normal`, `thermal_drift`, `voltage_sag`, `current_spike`, `sensor_dropout`,
   `oscillatory_instability`, `correlated_channel_fault`, `regime_shift`, `slow_degradation`,
   `compound_fault`). Deterministic given a seed.
2. **Modeling** — classical/feature baselines, CNN, LSTM, then the main model `SensorPatchTST`
   (PatchTST-inspired: per-channel patching + channel embeddings + transformer encoder over
   channel-patch tokens + attention pooling). Plus masked-patch self-supervised pretraining.
3. **Agentic runner** — a constrained planner/runner/reviewer loop that reads metrics, proposes
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

## Commands

No build/test tooling exists yet. Once scaffolded, expect (per spec; confirm against the actual
`pyproject.toml`/`Makefile` when they exist):

- Tests: `pytest` (single test: `pytest tests/test_model_forward.py::test_name`)
- Lint/format: `ruff check .` and `black .`
- Generate data / train / evaluate / run agent: the `scripts/*.py` entrypoints above.

When you add real tooling, update this section with the verified commands.
