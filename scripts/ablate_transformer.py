"""CLI: SensorPatchTST ablation campaign (roadmap v0.3, spec §16 / §17).

Runs the three experiment-matrix ablations, each changing exactly **one** variable from the base
config (spec ablation discipline): patch size (16 vs 32), channel embedding (on vs off), and pooling
(attention vs mean). Every variant is trained with the same recipe and scored on the held-out test
split (macro-F1, ECE, missing-channel delta), then compared against the base model. Writes
``reports/experiment_summaries/transformer_ablations.md``.

Example:
    python -m scripts.ablate_transformer --mode quick_demo --epochs 8
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# Precede torch import (macOS-only OpenMP guard; see scripts/train_baseline.py / tests/conftest.py).
configure_omp()

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.calibration import expected_calibration_error
from sensortwin.evaluation.metrics import classification_metrics
from sensortwin.evaluation.robustness import missing_channel_sweep
from sensortwin.simulation import generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config, load_yaml
from sensortwin.utils.seeds import set_torch_seed

# (label, one-variable override vs base, hypothesis)
ABLATIONS: list[tuple[str, dict[str, Any], str]] = [
    ("base (patch16, chan-emb, attn-pool)", {}, "Reference model."),
    (
        "patch32",
        {"patch_len": 32, "stride": 16},
        "Coarser patches: fewer tokens and less temporal detail; expect a small macro-F1 change and "
        "faster training.",
    ),
    (
        "no_channel_emb",
        {"channel_embedding": False},
        "Without channel identity the encoder cannot tell channels apart; cross-channel events "
        "should degrade.",
    ),
    (
        "mean_pool",
        {"pooling": "mean"},
        "Mean pooling weights all patch tokens equally; expect a drop vs learned attention pooling.",
    ),
]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=range(len(EVENT_CLASSES)), average="macro", zero_division=0)
    )


def _train_eval(
    model_overrides: dict[str, Any],
    base_cfg: dict,
    splits,
    X,
    y,
    standardizer,
    epochs: int,
    device=None,
) -> dict[str, Any]:
    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.augment import build_augment
    from sensortwin.training.loop import class_weights, predict_proba, train_model

    tr, va, te = splits["train"], splits["val"], splits["test"]
    model_cfg = {**base_cfg.get("model", {}), **model_overrides}
    model: Any = SensorPatchTST(**model_cfg)
    n_params = sum(p.numel() for p in model.parameters())

    tcfg = dict(base_cfg.get("train", {}))
    keys = (
        "optimizer",
        "lr",
        "weight_decay",
        "label_smoothing",
        "scheduler",
        "warmup_epochs",
        "batch_size",
        "patience",
    )
    train_kwargs: dict[str, Any] = {k: tcfg[k] for k in keys if k in tcfg}
    augment = build_augment(tcfg.get("augment"))
    if augment is not None:
        train_kwargs["augment"] = augment

    train_ds = SensorArrayDataset(standardizer.transform(X[tr]), y[tr]).as_torch()
    val_ds = SensorArrayDataset(standardizer.transform(X[va]), y[va]).as_torch()
    test_ds = SensorArrayDataset(standardizer.transform(X[te]), y[te]).as_torch()

    t0 = time.perf_counter()
    model, _ = train_model(
        model,
        train_ds,
        val_ds,
        epochs=epochs,
        weight=class_weights(y[tr], len(EVENT_CLASSES)),
        device=device,
        **train_kwargs,
    )
    train_time = time.perf_counter() - t0

    y_proba = predict_proba(model, test_ds, device=device)
    y_pred = y_proba.argmax(1)
    metrics = classification_metrics(y[te], y_pred, y_proba, EVENT_CLASSES)

    def predict_raw(Xc: np.ndarray) -> np.ndarray:
        ds = SensorArrayDataset(standardizer.transform(Xc), y[te]).as_torch()
        return predict_proba(model, ds, device=device).argmax(1)

    sweep = missing_channel_sweep(predict_raw, X[te], y[te], macro_f1_fn=_macro_f1)
    return {
        "macro_f1": metrics["macro_f1"],
        "ece": expected_calibration_error(y[te], y_proba),
        "missing_delta": sweep["worst_delta"],
        "n_params": n_params,
        "train_s": round(train_time, 1),
    }


def _write_report(results: dict[str, dict], out_dir: Path, mode: str, epochs: int) -> Path:
    base = results["base (patch16, chan-emb, attn-pool)"]
    lines = [
        "# SensorPatchTST ablations (v0.3)",
        "",
        f"Mode `{mode}`, {epochs} epochs, held-out test split. Each ablation changes one variable "
        "from the base model. Macro-F1 is the headline metric; Missing-ch Δ is the worst "
        "single-channel-dropout macro-F1 drop.",
        "",
        "| Variant | Macro-F1 | Δ vs base | ECE | Missing-ch Δ | Params | Train (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _override, _hyp in ABLATIONS:
        m = results[label]
        delta = m["macro_f1"] - base["macro_f1"]
        lines.append(
            f"| {label} | {m['macro_f1']:.3f} | {delta:+.3f} | {m['ece']:.3f} | "
            f"{m['missing_delta']:.3f} | {m['n_params'] / 1000:.0f}k | {m['train_s']:.1f} |"
        )
    lines += ["", "## Hypotheses and outcomes", ""]
    for label, _override, hyp in ABLATIONS:
        if label.startswith("base"):
            continue
        delta = results[label]["macro_f1"] - base["macro_f1"]
        lines.append(f"- **{label}** ({delta:+.3f} macro-F1 vs base): {hyp}")
    lines += [
        "",
        f"These are `{mode}` figures for wiring/discipline, not research-grade conclusions; "
        "run `colab_standard` with multiple seeds before interpreting small deltas.",
    ]
    out = out_dir / "transformer_ablations.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SensorPatchTST ablation campaign.")
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--model-config", default="configs/models/sensorpatchtst.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto-detect")
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    set_torch_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_synthetic_config(args.config, mode=args.mode)
    cfg.seed = args.seed
    cfg.normalize = False
    print(f"Generating {cfg.n_samples} samples (T={cfg.T}, seed={cfg.seed})...")
    X, y, _ = generate_dataset(cfg)
    splits = make_split("random", y, {}, seed=args.seed)
    standardizer = ChannelStandardizer().fit(X[splits["train"]])
    base_cfg = load_yaml(args.model_config)

    results: dict[str, dict] = {}
    for label, override, _hyp in ABLATIONS:
        print(f"--- {label} ---")
        results[label] = _train_eval(
            override, base_cfg, splits, X, y, standardizer, args.epochs, args.device
        )
        print(f"  macro-F1={results[label]['macro_f1']:.3f}")

    report = _write_report(results, out_dir, args.mode, args.epochs)
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
