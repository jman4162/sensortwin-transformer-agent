# SensorTwin Transformer Agent

A reproducible, research-style benchmark for **multichannel sensor event classification** —
synthetic physics-inspired data, patch-based transformers, self-supervised pretraining,
robustness/calibration evaluation, and an agentic experiment runner.

> Status: **v0.1 in progress** — the synthetic benchmark (Layer 1) is implemented. Modeling
> (Layer 2) and the agentic runner (Layer 3) are on the roadmap below.

## Why this matters

Many scientific and engineering systems emit structured multichannel time-series: voltage,
current, temperature, vibration, telemetry. This project builds a *controllable* benchmark for
learning from such signals. Because we own the data-generating process, we can test specific
hypotheses — does a transformer's inductive bias actually help on cross-channel or long-range
events vs. a CNN or feature baseline? — rather than memorizing synthetic artifacts.

The project is built in three layers, **in order** (the agent comes last, on purpose):

1. **Simulation** — synthetic 8-channel generator, 10 event classes, deterministic from a seed.
2. **Modeling** — strong baselines (features, CNN, LSTM) + `SensorPatchTST` + masked pretraining.
3. **Agentic runner** — a *constrained* planner/runner/reviewer loop that orchestrates ablations.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + test tooling; add ".[ml]" for the modeling layer

make test                        # run the suite
make data                        # generate the quick-demo dataset to data/synth_quick_demo.npz
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

The 10 event classes span an intentional difficulty gradient — local transients
(`current_spike`), long-range drift (`slow_degradation`), and cross-channel-only faults
(`correlated_channel_fault`) — so the benchmark discriminates between model inductive biases.

Run modes (spec §19): `quick_demo` (2k), `colab_standard` (20k), `full_reproduction` (100k).

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
| v0.1 | Synthetic 8-channel benchmark, splits, tests | **in progress** |
| v0.2 | Feature + CNN + LSTM baselines, metrics | planned |
| v0.3 | `SensorPatchTST` classifier + ablations | planned |
| v0.4 | Masked-patch pretraining, label-efficiency | planned |
| v0.5 | Robustness, calibration, interpretability | planned |
| v0.6 | NASA battery / C-MAPSS open-data adaptation | planned |
| v0.7 | Agentic experiment runner + report | planned |

## License

MIT — see [LICENSE](LICENSE).
