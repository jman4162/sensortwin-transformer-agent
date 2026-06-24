# SensorTwin Transformer Agent

A reproducible, research-style benchmark for **multichannel sensor event classification**:
synthetic physics-inspired data, patch-based transformers, self-supervised pretraining,
robustness/calibration evaluation, and an agentic experiment runner.

> Status: **v0.7 done.** Implemented: the synthetic benchmark (Layer 1); the baseline + evaluation
> suite (Layer 2); the `SensorPatchTST` patch-transformer + ablations; masked-patch pretraining with
> a label-efficiency sweep; the robustness / calibration / interpretability study with a
> [model card](reports/model_card.md); open-dataset validation on NASA C-MAPSS (the same slate on
> real turbofan sensors, plus a synthetic→real encoder-transfer test); and the constrained agentic
> experiment runner (Layer 3) with GPU/Colab support. All three layers are in place.

## Why this matters

Many scientific and engineering systems emit structured multichannel time-series: voltage,
current, temperature, vibration, telemetry. This project builds a *controllable* benchmark for
learning from such signals. Because we own the data-generating process, we can test specific
hypotheses (does a transformer's inductive bias actually help on cross-channel or long-range events
vs. a CNN or feature baseline?) rather than reward memorizing synthetic artifacts.

The project is built in three layers, **in order** (the agent comes last, on purpose):

1. **Simulation**: synthetic 8-channel generator, 10 event classes, deterministic from a seed.
2. **Modeling**: strong baselines (features, CNN, LSTM) + `SensorPatchTST` + masked pretraining.
3. **Agentic runner**: a *constrained* planner/runner/reviewer loop that orchestrates ablations.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + test tooling; add ".[ml]" for the modeling layer

make test                        # run the suite
make data                        # generate the quick-demo dataset to data/synth_quick_demo.npz
make baselines                   # train + evaluate baselines (needs ".[ml]"), writes the report
```

Train and evaluate the baselines directly (requires the `ml` extra — `pip install -e ".[ml]"`):

```bash
python -m scripts.train_baseline --mode quick_demo --epochs 10
# -> reports/experiment_summaries/baseline_results.md  (+ metrics JSON/CSV and figures)
```

Generate a dataset directly:

```bash
python -m scripts.generate_synthetic --mode quick_demo          # 2k samples
python -m scripts.generate_synthetic --mode colab_standard \
    --out data/synth_colab                                      # 20k samples
```

Use it in code:

```python
from sensortwin.simulation import generate_dataset, GenConfig
from sensortwin.data import make_split

X, y, meta = generate_dataset(GenConfig(n_samples=2000, T=512, seed=0))   # X: [N, 8, T]
splits = make_split("random", y, meta, seed=0)                            # leakage-checked
```

## Dataset: `SensorTwin-Synth`

Each sample is an 8-channel series composed as
`base_dynamics + load_profile + channel_coupling + event_signature + noise + artifacts`.
Channels are coupled by documented assumptions (voltage anti-correlates with current; core
temperature integrates current through a thermal lag; surface temperature lags core), so some
event classes are detectable *only* through cross-channel relationships.

The 10 event classes span an intentional difficulty gradient: local transients
(`current_spike`), long-range drift (`slow_degradation`), and cross-channel-only faults
(`correlated_channel_fault`). This lets the benchmark discriminate between model inductive biases.

Run modes (spec §19): `quick_demo` (2k), `colab_standard` (20k), `full_reproduction` (100k).

## Results

At `colab_standard` scale (20,000 samples, leakage-safe 70/15/15 split, test set), over **3 seeds**
(mean ± std), produced by [`notebooks/04_colab_standard_comparison.ipynb`](notebooks/04_colab_standard_comparison.ipynb)
on a T4 GPU:

| Model | Macro-F1 | Macro-AUROC | ECE |
| --- | ---: | ---: | ---: |
| **transformer** | **0.900 ± 0.003** | 0.990 ± 0.000 | 0.095 ± 0.006 |
| cnn | 0.844 ± 0.004 | 0.983 ± 0.001 | 0.023 ± 0.002 |
| xgboost | 0.759 ± 0.004 | 0.967 ± 0.000 | 0.042 ± 0.004 |
| logreg | 0.697 ± 0.006 | 0.950 ± 0.001 | 0.019 ± 0.003 |
| random_forest | 0.691 ± 0.003 | 0.950 ± 0.000 | 0.185 ± 0.003 |
| lstm | 0.545 ± 0.022 | 0.902 ± 0.008 | 0.056 ± 0.009 |

At 20k the ordering **flips** relative to small scale: the deep models overtake feature+GBM, and
`SensorPatchTST` tops the strongest baseline (cnn) by **+0.056 macro-F1 (95% CI [0.048, 0.061],
p=0.006**, paired over seeds — `evaluation/statistics.py`). The transformer's gains land on the
hard, cross-channel / long-context classes its inductive bias targets: vs the cnn it gains
`sensor_dropout` +0.135, `regime_shift` +0.120, `normal` +0.092. One honest trade-off: the
transformer is the **most accurate but least calibrated** model here (ECE 0.095 vs cnn 0.023) —
post-hoc temperature scaling (`evaluation/calibration.py`) is the fix. This is the result a
*controllable* benchmark is built to surface: the transformer's inductive bias pays off at scale,
not at the small-data scale below.

### Small-scale contrast (quick-demo, 2k — wiring-grade)

The same slate at 2k (`make baselines` / `make transformer`) shows the opposite ranking, because the
deep models are data-starved:

| Model | Macro-F1 | Macro-AUROC | ECE | Params |
| --- | ---: | ---: | ---: | ---: |
| **xgboost** | **0.620** | 0.920 | 0.114 | — |
| logreg | 0.571 | 0.898 | 0.122 | — |
| random_forest | 0.543 | 0.915 | 0.181 | — |
| cnn | 0.504 | 0.887 | 0.082 | 54k |
| transformer | 0.435 | 0.859 | 0.143 | 814k |
| lstm | 0.338 | 0.817 | 0.063 | 39k |

At 2k the **feature + gradient-boosting baseline (xgboost) leads** and the 814k-param transformer
trails — the most data-hungry model, far below the scale where its inductive bias pays off. The
contrast between the two tables is the benchmark's whole point. The §17 ablations (patch size,
channel embedding, pooling) run via `make ablate`.

## Real-data validation (v0.6)

The synthetic numbers above are a controlled testbed, not evidence of real-world performance. v0.6
runs the same pipeline on **NASA C-MAPSS** turbofan data — 14 informative sensors, reframed as
3-stage health classification (healthy / degrading / critical) by binning remaining-useful-life. The
raw files are a US-government work (NASA PCoE) and are not committed; download them and point
`--raw-dir` at the folder:

```bash
python -m scripts.fetch_cmapss --raw-dir /path/to/CMAPSSData --mode quick_demo
python -m scripts.real_data_report --data data/cmapss_FD001 --mode quick_demo   # baselines on real data
python -m scripts.sim2real_transfer --data data/cmapss_FD001 --mode quick_demo  # synthetic->real transfer
```

The split is **grouped by engine** (no engine's windows cross train/test). `sim2real_transfer`
pretrains the masked-patch encoder on synthetic data and transfers the channel-agnostic *temporal*
encoder to C-MAPSS (the 8→14 channel mismatch means the channel embedding is re-learned), against a
real-pretrained upper bound and a from-scratch lower bound.

- **Supported:** the synthetic-data pipeline (windowing, features, model classes, evaluation) runs
  on real sensor data, and models can be ranked on it.
- **Not claimed:** these are not state-of-the-art RUL estimates; the 3-stage binning is a deliberate
  classification reframing, and only the temporal encoder (not channel identity) transfers.

## Agentic experiment runner (v0.7)

A constrained planner → runner → reviewer loop (Layer 3) that automates the experiment workflow — it
is deliberately *not* the novelty. It reads the baseline metrics, proposes one-variable ablations
(each with a hypothesis and an expected failure mode), runs each across seeds, and writes a report.

```bash
python -m scripts.run_agent --epochs 6 --seeds 3 --max-experiments 3   # -> agentic_ablation_report.md
```

Guardrails (spec §13.5) are enforced by Pydantic schemas, not prose: an experiment can't be
constructed with more than one changed variable; `Guardrails.check` rejects out-of-bounds
epochs/modes/dataset sizes; `write_config` refuses to overwrite a committed config or touch `data/`;
and the reviewer calls a variant an **improvement only when the per-seed gain is statistically
significant** (`evaluation/statistics.py`). Every run — including regressions and failures — is
reported, and the agent's *hypotheses* are kept visually separate from the *measured* results. The
planner is deterministic (no API key, runs in CI); an LLM backend is a documented pluggable seam.

## Run on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/sensortwin-transformer-agent/blob/master/notebooks/03_transformer_training_colab.ipynb)

[`notebooks/03_transformer_training_colab.ipynb`](notebooks/03_transformer_training_colab.ipynb)
installs the package, generates `colab_standard` (20k), and trains `SensorPatchTST` on a GPU. Mixed
precision and a pinned multi-worker DataLoader **switch on automatically when CUDA is detected**; on
CPU the path is unchanged and bit-identical, so tests stay deterministic. Pass `--device` /
`--no-amp` to the training scripts for explicit control.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/sensortwin-transformer-agent/blob/master/notebooks/04_colab_standard_comparison.ipynb)
&nbsp;[`notebooks/04_colab_standard_comparison.ipynb`](notebooks/04_colab_standard_comparison.ipynb)
runs the full model slate at `colab_standard` over several seeds, aggregates to mean ± std, and
significance-tests the transformer against the strongest baseline — the apples-to-apples comparison
behind the Results section. It writes a downloadable `summary.md`/`summary.json`.

## Project principles

- **Reproducible**: every dataset/experiment is deterministic given its seed.
- **Baselines are first-class**: the transformer is only credible measured against strong
  simple models.
- **Beyond accuracy**: macro-F1 (headline), per-class P/R, AUROC, calibration, robustness deltas.
- **Honest**: negative results and failure modes are documented; claims separated from
  speculation.

## Roadmap

| Version | Scope | Status |
| --- | --- | --- |
| v0.1 | Synthetic 8-channel benchmark, splits, tests | **done** |
| v0.2 | Feature + CNN + LSTM baselines, metrics, calibration | **done** |
| v0.3 | `SensorPatchTST` classifier + ablations | **done** |
| v0.4 | Masked-patch pretraining, label-efficiency | **done** |
| v0.5 | Robustness, calibration, interpretability + model card | **done** |
| v0.6 | NASA C-MAPSS open-data adaptation + synthetic→real transfer | **done** |
| v0.7 | Agentic experiment runner + Colab GPU readiness | **done** |

## License

MIT — see [LICENSE](LICENSE).
