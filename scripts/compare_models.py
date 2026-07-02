"""CLI: the headline multi-seed model comparison, reproducible with one command.

For each seed this regenerates the dataset, refits the leakage-safe split/standardizer, trains
every requested model, and evaluates on the held-out test split. Across seeds it reports
mean ± sample std macro-F1, a paired comparison of the transformer against the best baseline
(t-interval + paired t-test + Cohen's d via ``evaluation.statistics``), and per-class F1 deltas
with Holm correction across the 10 classes.

Writes ``headline_comparison.{json,md}`` — both are committable, so every number in the README
headline traces back to an artifact this command regenerates.

Examples
--------
    python -m scripts.compare_models --mode quick_demo --epochs 5 --seeds 0 1 2
    python -m scripts.compare_models --mode colab_standard --seeds 0 1 2 3 4   # the headline
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# Must precede any torch/xgboost import (macOS OpenMP clash; see scripts/train_baseline.py).
configure_omp()

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from scripts.train_baseline import DEEP_MODELS, _build_deep_model
from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.calibration import expected_calibration_error
from sensortwin.evaluation.statistics import compare_seeds, holm_bonferroni, mean_std
from sensortwin.features import build_feature_matrix
from sensortwin.models.baselines import CLASSICAL_REGISTRY
from sensortwin.simulation import generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config
from sensortwin.utils.seeds import set_torch_seed

DEFAULT_MODELS = ["logreg", "random_forest", "xgboost", "cnn", "lstm", "transformer"]
N_CLASSES = len(EVENT_CLASSES)


def _eval(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    y_pred = y_proba.argmax(1)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=range(N_CLASSES), average="macro", zero_division=0)
        ),
        "per_class_f1": f1_score(
            y_true, y_pred, labels=range(N_CLASSES), average=None, zero_division=0
        ).tolist(),
        "ece": float(expected_calibration_error(y_true, y_proba)),
    }


def _run_seed(
    seed: int,
    models: list[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Train + evaluate every model for one seed; return per-model metric dicts."""
    set_torch_seed(seed)
    cfg = load_synthetic_config(args.config, mode=args.mode)
    cfg.seed = seed
    cfg.normalize = False
    print(f"[seed {seed}] generating {cfg.n_samples} samples (T={cfg.T})...")
    X, y, _ = generate_dataset(cfg)

    sp = make_split("random", y, {}, seed=seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    standardizer = ChannelStandardizer().fit(X[tr])

    needs_features = any(m in CLASSICAL_REGISTRY for m in models)
    if needs_features:
        F_train, _ = build_feature_matrix(X[tr])
        F_test, _ = build_feature_matrix(X[te])

    out: dict[str, dict[str, Any]] = {}
    for name in models:
        t0 = time.perf_counter()
        if name in CLASSICAL_REGISTRY:
            model = CLASSICAL_REGISTRY[name]()
            model.fit(F_train, y[tr])
            metrics = _eval(y[te], model.predict_proba(F_test))
        elif name in DEEP_MODELS:
            from sensortwin.training.loop import class_weights, predict_proba, train_model

            model, train_kwargs = _build_deep_model(name, use_augment=not args.no_augment)
            train_ds = SensorArrayDataset(standardizer.transform(X[tr]), y[tr]).as_torch()
            val_ds = SensorArrayDataset(standardizer.transform(X[va]), y[va]).as_torch()
            test_ds = SensorArrayDataset(standardizer.transform(X[te]), y[te]).as_torch()
            amp = False if args.no_amp else None
            model, _ = train_model(
                model,
                train_ds,
                val_ds,
                epochs=args.epochs,
                weight=class_weights(y[tr], N_CLASSES),
                device=args.device,
                amp=amp,
                **train_kwargs,
            )
            metrics = _eval(y[te], predict_proba(model, test_ds, device=args.device))
        else:
            raise SystemExit(f"unknown model '{name}'")
        metrics["train_time_s"] = round(time.perf_counter() - t0, 1)
        out[name] = metrics
        print(f"[seed {seed}] {name}: macro-F1={metrics['macro_f1']:.3f}")
    return out


def _aggregate(
    per_seed: dict[int, dict[str, dict[str, Any]]], models: list[str], alpha: float
) -> dict[str, Any]:
    """Mean ± std per model plus the paired transformer-vs-best-baseline comparison."""
    seeds = sorted(per_seed)
    summary: dict[str, Any] = {"seeds": seeds, "models": {}}
    for name in models:
        scores = [per_seed[s][name]["macro_f1"] for s in seeds]
        m, sd = mean_std(scores)
        summary["models"][name] = {
            "per_seed_macro_f1": scores,
            "mean": m,
            "std": sd,
            "ece_mean": float(np.mean([per_seed[s][name]["ece"] for s in seeds])),
        }

    if "transformer" in models and len(models) > 1 and len(seeds) >= 2:
        baselines = [n for n in models if n != "transformer"]
        best = max(baselines, key=lambda n: summary["models"][n]["mean"])
        cmp = compare_seeds(
            summary["models"][best]["per_seed_macro_f1"],
            summary["models"]["transformer"]["per_seed_macro_f1"],
            alpha=alpha,
        )
        pc: dict[str, Any] = {}
        p_values = []
        for k, cls in enumerate(EVENT_CLASSES):
            c = compare_seeds(
                [per_seed[s][best]["per_class_f1"][k] for s in seeds],
                [per_seed[s]["transformer"]["per_class_f1"][k] for s in seeds],
                alpha=alpha,
            )
            pc[cls] = {"delta": c.delta, "p_value": c.p_value}
            p_values.append(c.p_value)
        reject = holm_bonferroni(p_values, alpha=alpha)
        for cls, rej in zip(EVENT_CLASSES, reject, strict=True):
            pc[cls]["significant_holm"] = rej
        summary["headline"] = {
            "best_baseline": best,
            "delta": cmp.delta,
            "ci_low": cmp.ci_low,
            "ci_high": cmp.ci_high,
            "p_value": cmp.p_value,
            "cohen_d": cmp.cohen_d,
            "significant": cmp.significant,
            "per_class_vs_best_baseline": pc,
        }
    return summary


def _write_md(summary: dict[str, Any], out_dir: Path, meta: dict[str, Any]) -> Path:
    seeds = summary["seeds"]
    lines = [
        "# Headline model comparison",
        "",
        f"Mode `{meta['mode']}`, seeds {seeds}, {meta['epochs']} epochs, "
        f"{'no augmentation' if meta['no_augment'] else 'shared augmentation recipe'}. "
        "All deep models share one training recipe family (`configs/models/*.yaml`); "
        "classical baselines fit engineered features. Metrics on the held-out test split; "
        "mean ± sample std over seeds.",
        "",
        "| Model | Macro-F1 (mean ± std) | ECE | Per-seed |",
        "| --- | ---: | ---: | --- |",
    ]
    ranked = sorted(summary["models"].items(), key=lambda kv: -kv[1]["mean"])
    for name, m in ranked:
        std = "n/a" if np.isnan(m["std"]) else f"{m['std']:.3f}"
        per_seed = ", ".join(f"{v:.3f}" for v in m["per_seed_macro_f1"])
        lines.append(f"| {name} | {m['mean']:.3f} ± {std} | {m['ece_mean']:.3f} | {per_seed} |")

    if "headline" in summary:
        h = summary["headline"]
        n = len(seeds)
        verdict = "significant" if h["significant"] else "not significant"
        lines += [
            "",
            "## Transformer vs best baseline",
            "",
            f"Best baseline by mean macro-F1: **{h['best_baseline']}**. Paired over {n} seeds: "
            f"Δ = {h['delta']:+.2f} macro-F1, 95% t-interval [{h['ci_low']:+.2f}, "
            f"{h['ci_high']:+.2f}], paired t-test p = {h['p_value']:.3f}"
            + (f", Cohen's d = {h['cohen_d']:.1f}" if h["cohen_d"] is not None else "")
            + f" — **{verdict}** at α = 0.05. "
            f"Interval and p-value rest on n = {n} seeds; treat precision accordingly.",
            "",
            "### Per-class F1 delta (transformer − best baseline, Holm-corrected)",
            "",
            "| Class | Δ F1 | p | Significant (Holm, m=10) |",
            "| --- | ---: | ---: | --- |",
        ]
        for cls, d in h["per_class_vs_best_baseline"].items():
            sig = "yes" if d["significant_holm"] else "no"
            lines.append(f"| {cls} | {d['delta']:+.2f} | {d['p_value']:.3f} | {sig} |")
        lines += [
            "",
            "Per-class deltas are a family of 10 comparisons; only Holm-surviving rows are "
            "claimable. Uncorrected per-class p-values at n = "
            f"{n} seeds are weak evidence either way.",
        ]

    lines += [
        "",
        "Regenerate: `python -m scripts.compare_models --mode "
        f"{meta['mode']} --seeds {' '.join(str(s) for s in seeds)} --epochs {meta['epochs']}`.",
    ]
    out = out_dir / "headline_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Multi-seed headline model comparison.")
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated subset")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out)
    ckpt_dir = out_dir / "per_seed"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed checkpoints: a multi-hour run killed partway resumes instead of restarting.
    # A checkpoint is reused only if it covers the requested models at the same mode/epochs.
    per_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in args.seeds:
        ckpt = ckpt_dir / f"seed_{seed}.json"
        if ckpt.exists():
            saved = json.loads(ckpt.read_text())
            if saved["meta"] == {"mode": args.mode, "epochs": args.epochs} and all(
                m in saved["results"] for m in models
            ):
                print(f"[seed {seed}] reusing checkpoint {ckpt}")
                per_seed[seed] = saved["results"]
                continue
        per_seed[seed] = _run_seed(seed, models, args)
        ckpt.write_text(
            json.dumps(
                {"meta": {"mode": args.mode, "epochs": args.epochs}, "results": per_seed[seed]},
                indent=2,
            )
        )

    import platform

    import torch

    summary = _aggregate(per_seed, models, args.alpha)
    meta = {
        "mode": args.mode,
        "epochs": args.epochs,
        "no_augment": args.no_augment,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    (out_dir / "headline_comparison.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "per_seed": per_seed}, indent=2) + "\n"
    )
    report = _write_md(summary, out_dir, meta)
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
