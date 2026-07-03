"""CLI: temperature-scaling study for the transformer (multi-seed).

Per seed: train SensorPatchTST with its committed recipe, fit one temperature on the validation
logits (never test), and report test ECE before/after plus the fraction of predictions changed
(always 0 for temperature scaling — dividing logits by a scalar preserves the argmax).

Writes ``calibration_temperature.{json,md}`` — committable, so the model card's calibration
claim traces to an artifact.

Example:
    python -m scripts.calibrate_transformer --mode colab_standard --seeds 0 1 2 --device mps
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# Must precede any torch import (macOS OpenMP clash; see scripts/train_baseline.py).
configure_omp()

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_baseline import _build_deep_model
from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from sensortwin.evaluation.statistics import mean_std
from sensortwin.simulation import generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config
from sensortwin.utils.seeds import set_torch_seed


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    from sensortwin.training.loop import class_weights, predict_logits, train_model

    set_torch_seed(seed)
    cfg = load_synthetic_config(args.config, mode=args.mode)
    cfg.seed = seed
    cfg.normalize = False
    print(f"[seed {seed}] generating {cfg.n_samples} samples...")
    X, y, _ = generate_dataset(cfg)
    sp = make_split("random", y, {}, seed=seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    std = ChannelStandardizer().fit(X[tr])

    model, kw = _build_deep_model("transformer")
    model, _ = train_model(
        model,
        SensorArrayDataset(std.transform(X[tr]), y[tr]).as_torch(),
        SensorArrayDataset(std.transform(X[va]), y[va]).as_torch(),
        epochs=args.epochs,
        weight=class_weights(y[tr], len(EVENT_CLASSES)),
        device=args.device,
        **kw,
    )
    val_ds = SensorArrayDataset(std.transform(X[va]), y[va]).as_torch()
    test_ds = SensorArrayDataset(std.transform(X[te]), y[te]).as_torch()
    logits_val = predict_logits(model, val_ds, device=args.device)
    logits_te = predict_logits(model, test_ds, device=args.device)

    temp = fit_temperature(logits_val, y[va])
    proba_before = _softmax(logits_te)
    proba_after = apply_temperature(logits_te, temp)
    rec = {
        "seed": seed,
        "temperature": round(float(temp), 3),
        "ece_before": round(float(expected_calibration_error(y[te], proba_before)), 4),
        "ece_after": round(float(expected_calibration_error(y[te], proba_after)), 4),
        "argmax_unchanged_frac": float((proba_before.argmax(1) == proba_after.argmax(1)).mean()),
    }
    print(f"[seed {seed}] {rec}")
    return rec


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Temperature-scaling study for SensorPatchTST.")
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    import platform

    import torch

    results = [run_seed(s, args) for s in args.seeds]
    meta = {
        "mode": args.mode,
        "epochs": args.epochs,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_temperature.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2) + "\n"
    )

    t_m, t_s = mean_std([r["temperature"] for r in results])
    b_m, b_s = mean_std([r["ece_before"] for r in results])
    a_m, a_s = mean_std([r["ece_after"] for r in results])
    lines = [
        "# Temperature scaling (transformer)",
        "",
        f"Mode `{args.mode}`, seeds {args.seeds}, {args.epochs} epochs. One temperature fitted "
        "per seed on the validation logits; ECE measured on the held-out test split.",
        "",
        "| Seed | T | ECE before | ECE after | Predictions changed |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        changed = f"{(1 - r['argmax_unchanged_frac']) * 100:.1f}%"
        lines.append(
            f"| {r['seed']} | {r['temperature']:.3f} | {r['ece_before']:.3f} | "
            f"{r['ece_after']:.3f} | {changed} |"
        )
    lines += [
        "",
        f"Mean over {len(results)} seeds: T = {t_m:.2f} ± {t_s:.2f}, "
        f"ECE {b_m:.3f} ± {b_s:.3f} → {a_m:.3f} ± {a_s:.3f}. Temperature scaling divides logits "
        "by a scalar, so the argmax — and every accuracy metric — is unchanged by construction.",
        "",
        "Regenerate: `python -m scripts.calibrate_transformer --mode "
        f"{args.mode} --seeds {' '.join(str(s) for s in args.seeds)} --epochs {args.epochs}`.",
    ]
    report = out_dir / "calibration_temperature.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
