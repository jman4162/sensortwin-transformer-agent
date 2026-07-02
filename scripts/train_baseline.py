"""CLI: train and evaluate the v0.2 baselines, then write a results report.

End-to-end (spec §16 v0.2 deliverable):
  generate/load data -> leakage-safe split -> fit classical (LogReg/RF/XGBoost) on engineered
  features + train deep (CNN/LSTM) on standardized signals -> evaluate on the held-out test split
  (metrics + calibration + missing-channel robustness) -> save JSON/CSV/figures + a markdown report.

Examples
--------
    python -m scripts.train_baseline --mode quick_demo --epochs 5
    python -m scripts.train_baseline --models logreg,cnn --mode quick_demo --out reports/run1
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# Must precede any torch/xgboost import: on macOS their bundled OpenMP runtimes clash
# (segfault/deadlock). Gated to Darwin so Linux/Colab keeps multi-threaded feature extraction.
configure_omp()

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.calibration import (
    brier_score_multiclass,
    expected_calibration_error,
    reliability_curve,
)
from sensortwin.evaluation.metrics import classification_metrics, save_metrics
from sensortwin.evaluation.plots import plot_confusion_matrix, plot_per_class_f1, plot_reliability
from sensortwin.evaluation.robustness import missing_channel_sweep
from sensortwin.features import build_feature_matrix
from sensortwin.models.baselines import (
    CLASSICAL_REGISTRY,
    feature_importance,
    make_isolation_forest,
)
from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config, load_yaml
from sensortwin.utils.io import load_dataset
from sensortwin.utils.seeds import set_torch_seed

DEEP_MODELS = {"cnn", "lstm", "transformer"}
# Default comparison set excludes the transformer so quick runs stay fast; request it with
# --models transformer (or include it explicitly).
DEFAULT_MODELS = ["logreg", "random_forest", "xgboost", "cnn", "lstm"]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=range(len(EVENT_CLASSES)), average="macro", zero_division=0)
    )


def _evaluate_predictions(
    name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    predict_raw: Callable[[np.ndarray], np.ndarray],
    X_test_raw: np.ndarray,
    out_dir: Path,
    train_time: float,
    n_params: int | None,
) -> dict[str, Any]:
    """Compute the full metric suite for one model and write its artifacts."""
    metrics = classification_metrics(y_test, y_pred, y_proba, EVENT_CLASSES)
    if y_proba is not None:
        metrics["ece"] = expected_calibration_error(y_test, y_proba)
        metrics["brier"] = brier_score_multiclass(y_test, y_proba)
    sweep = missing_channel_sweep(predict_raw, X_test_raw, y_test, macro_f1_fn=_macro_f1)
    metrics["missing_channel"] = sweep
    metrics["train_time_s"] = round(train_time, 2)
    metrics["n_params"] = n_params

    save_metrics(metrics, out_dir / f"metrics_{name}")
    fig_dir = out_dir / "figures"
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), EVENT_CLASSES, fig_dir / f"cm_{name}.png"
    )
    plot_per_class_f1(
        metrics["per_class"], fig_dir / f"f1_{name}.png", title=f"{name} per-class F1"
    )
    if y_proba is not None:
        plot_reliability(
            reliability_curve(y_test, y_proba),
            fig_dir / f"reliability_{name}.png",
            title=f"{name} reliability",
        )
    return metrics


def _run_classical(
    name: str, F_train, y_train, F_test, X_test_raw, y_test, feat_names, out_dir
) -> dict[str, Any]:
    model = CLASSICAL_REGISTRY[name]()
    t0 = time.perf_counter()
    model.fit(F_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(F_test)
    y_proba = model.predict_proba(F_test)

    def predict_raw(Xc: np.ndarray) -> np.ndarray:
        return model.predict(build_feature_matrix(Xc)[0])

    metrics = _evaluate_predictions(
        name, y_test, y_pred, y_proba, predict_raw, X_test_raw, out_dir, train_time, None
    )
    metrics["feature_importance"] = feature_importance(model, feat_names)
    return metrics


DEEP_CONFIGS = {
    "cnn": "configs/models/cnn.yaml",
    "lstm": "configs/models/lstm.yaml",
    "transformer": "configs/models/sensorpatchtst.yaml",
}


def _deep_model_class(name: str) -> type:
    if name == "cnn":
        from sensortwin.models.cnn import SensorCNN

        return SensorCNN
    if name == "lstm":
        from sensortwin.models.lstm import SensorLSTM

        return SensorLSTM
    from sensortwin.models.transformer import SensorPatchTST

    return SensorPatchTST


def _build_deep_model(name: str, *, use_augment: bool = True) -> tuple[Any, dict[str, Any]]:
    """Return ``(model, train_kwargs)`` for a deep model.

    Every deep model reads its architecture and training recipe from ``configs/models/*.yaml``
    through this one code path, and the committed configs share a single recipe family
    (AdamW / weight decay / label smoothing / cosine-warmup / identical augmentation stack) —
    so cross-model comparisons measure architecture, not training budget.

    ``use_augment=False`` strips train-time augmentation: the no-augmentation arm for robustness
    studies, where ``channel_dropout`` augmentation would otherwise train every model on the same
    corruption the missing-channel sweep probes.
    """
    from sensortwin.training.augment import build_augment

    if name not in DEEP_CONFIGS:
        raise SystemExit(f"unknown deep model '{name}'")
    cfg = load_yaml(DEEP_CONFIGS[name])
    model = _deep_model_class(name)(**cfg.get("model", {}))
    tcfg = dict(cfg.get("train", {}))
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
    if use_augment:
        augment = build_augment(tcfg.get("augment"))
        if augment is not None:
            train_kwargs["augment"] = augment
    return model, train_kwargs


def _run_deep(
    name: str,
    Xtr_std,
    ytr,
    Xva_std,
    yva,
    Xte_std,
    X_test_raw,
    yte,
    standardizer,
    epochs,
    out_dir,
    device=None,
    amp=None,
    use_augment=True,
) -> dict[str, Any]:
    import torch

    from sensortwin.training.loop import class_weights, predict_proba, train_model

    model, train_kwargs = _build_deep_model(name, use_augment=use_augment)
    n_params = sum(p.numel() for p in model.parameters())
    train_ds = SensorArrayDataset(Xtr_std, ytr).as_torch()
    val_ds = SensorArrayDataset(Xva_std, yva).as_torch()
    test_ds = SensorArrayDataset(Xte_std, yte).as_torch()

    t0 = time.perf_counter()
    model, _ = train_model(
        model,
        train_ds,
        val_ds,
        epochs=epochs,
        weight=class_weights(ytr, len(EVENT_CLASSES)),
        device=device,
        amp=amp,
        **train_kwargs,
    )
    train_time = time.perf_counter() - t0

    y_proba = predict_proba(model, test_ds, device=device)
    y_pred = y_proba.argmax(1)

    def predict_raw(Xc: np.ndarray) -> np.ndarray:
        ds = SensorArrayDataset(standardizer.transform(Xc), yte).as_torch()
        return predict_proba(model, ds, device=device).argmax(1)

    with torch.no_grad():
        return _evaluate_predictions(
            name, yte, y_pred, y_proba, predict_raw, X_test_raw, out_dir, train_time, n_params
        )


def _write_report(
    results: dict[str, dict],
    anomaly: dict | None,
    out_dir: Path,
    cfg: GenConfig,
    augmented: bool = True,
) -> Path:
    """Assemble the markdown results report (spec §16 deliverable, §18 table, §7 model-zoo)."""
    lines = [
        "# Model results",
        "",
        f"Synthetic dataset: {cfg.n_samples} samples, T={cfg.T}, seed={cfg.seed}, "
        "leakage-safe random split (70/15/15), metrics on the held-out test split.",
        "",
        "## Headline metrics",
        "",
        "| Model | Macro-F1 | Weighted-F1 | Accuracy | Macro-AUROC | ECE | Brier | Missing-ch Δ | Params | Train (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in results.items():
        params = f"{m['n_params']/1000:.0f}k" if m.get("n_params") else "—"
        lines.append(
            f"| {name} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} | {m['accuracy']:.3f} | "
            f"{_fmt(m.get('macro_auroc'))} | {_fmt(m.get('ece'))} | {_fmt(m.get('brier'))} | "
            f"{_fmt(m['missing_channel']['worst_delta'])} | {params} | {m['train_time_s']:.1f} |"
        )

    lines += [
        "",
        "Macro-F1 is the headline metric (event classes are imbalanced). "
        "Missing-ch Δ is the worst single-channel-dropout macro-F1 drop vs clean (higher = more fragile).",
        "",
        "## Model zoo — expected strengths / weaknesses (spec §7)",
        "",
        "| Model | Expected strength | Expected weakness |",
        "| --- | --- | --- |",
        "| logreg / features | Simple drift, spikes, channel correlations | Complex temporal structure |",
        "| random_forest / xgboost | Non-linear feature interactions | No raw temporal reasoning |",
        "| cnn | Local transients, short events | Long-range dependencies |",
        "| lstm | Sequential dynamics | Slower, harder to optimize |",
        "",
        "## Per-class F1 (test)",
        "",
    ]
    # Hardest/easiest classes from the best model by macro-F1.
    best = max(results.items(), key=lambda kv: kv[1]["macro_f1"])
    pc = best[1]["per_class"]
    ranked = sorted(pc.items(), key=lambda kv: kv[1]["f1"])
    lines.append(f"Best model: **{best[0]}** (macro-F1 {best[1]['macro_f1']:.3f}).")
    lines.append("")
    lines.append("- Hardest classes: " + ", ".join(f"`{n}` ({d['f1']:.2f})" for n, d in ranked[:3]))
    lines.append(
        "- Easiest classes: " + ", ".join(f"`{n}` ({d['f1']:.2f})" for n, d in ranked[-3:])
    )

    if anomaly is not None:
        lines += [
            "",
            "## Anomaly detection (IsolationForest, normal-vs-rest)",
            "",
            f"AUROC distinguishing `normal` from all event classes: **{anomaly['auroc']:.3f}** "
            "(unsupervised; a different task from 10-way classification — included to show the contrast).",
        ]

    lines += [
        "",
        "## Notes",
        "",
        "- Feature baselines use an sklearn `StandardScaler` fit on train only; CNN/LSTM use a "
        "`ChannelStandardizer` fit on train only — no test statistics leak into training.",
        "- All deep models share one training recipe family (AdamW, weight decay, label "
        "smoothing, cosine-warmup, identical augmentation) read from `configs/models/*.yaml`, "
        "so deep-model comparisons hold the training budget fixed. Classical baselines fit "
        "engineered features on clean signals; augmentation does not apply to them.",
    ]
    if augmented:
        lines.append(
            "- Deep models trained **with** the shared augmentation, which includes channel "
            "dropout — the same corruption the missing-channel sweep probes. The Missing-ch Δ "
            "column is comparable across deep models but flatters all of them; rerun with "
            "`--no-augment` for the unaugmented arm."
        )
    else:
        lines.append(
            "- Deep models trained **without** augmentation (`--no-augment`): the "
            "no-augmentation arm for robustness studies."
        )
    lines.append(
        "- These are quick-mode numbers for wiring/verification; run `--mode colab_standard` "
        "for research-grade results before drawing conclusions."
    )
    out = out_dir / "baseline_results.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _run_anomaly(F_train, y_train, F_test, y_test) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    detector = make_isolation_forest()
    detector.fit(F_train[y_train == 0])  # learn "normal" only
    scores = -detector.score_samples(F_test)  # higher = more anomalous
    is_anomaly = (y_test != 0).astype(int)
    return {"auroc": float(roc_auc_score(is_anomaly, scores))}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train and evaluate SensorTwin v0.2 baselines.")
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--data", default=None, help="reuse a saved .npz instead of generating")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated subset")
    p.add_argument("--epochs", type=int, default=None, help="override deep-model epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-anomaly", action="store_true")
    p.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto-detect")
    p.add_argument(
        "--no-amp", action="store_true", help="force FP32 (default: AMP auto-on when CUDA)"
    )
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="train deep models without augmentation (the no-augmentation arm for "
        "robustness studies; default recipes include channel dropout, which overlaps "
        "with the missing-channel probe)",
    )
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)
    amp = False if args.no_amp else None

    set_torch_seed(args.seed)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data — generate with normalize=False so standardization is fit on train only.
    if args.data:
        X, y, _ = load_dataset(args.data)
        cfg = GenConfig(n_samples=len(y))
    else:
        cfg = load_synthetic_config(args.config, mode=args.mode)
        cfg.seed = args.seed
        cfg.normalize = False
        print(f"Generating {cfg.n_samples} samples (T={cfg.T}, seed={cfg.seed})...")
        X, y, _ = generate_dataset(cfg)

    # 2. Split + standardizers (fit on train only).
    sp = make_split("random", y, {}, seed=args.seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    standardizer = ChannelStandardizer().fit(X[tr])
    X_test_raw = X[te]

    F_train, feat_names = build_feature_matrix(X[tr])
    F_test, _ = build_feature_matrix(X[te])

    results: dict[str, dict] = {}
    for name in models:
        print(f"--- {name} ---")
        if name in CLASSICAL_REGISTRY:
            results[name] = _run_classical(
                name, F_train, y[tr], F_test, X_test_raw, y[te], feat_names, out_dir
            )
        elif name in DEEP_MODELS:
            epochs = args.epochs if args.epochs is not None else 30
            results[name] = _run_deep(
                name,
                standardizer.transform(X[tr]),
                y[tr],
                standardizer.transform(X[va]),
                y[va],
                standardizer.transform(X[te]),
                X_test_raw,
                y[te],
                standardizer,
                epochs,
                out_dir,
                args.device,
                amp,
                not args.no_augment,
            )
        else:
            raise SystemExit(f"unknown model '{name}'")
        print(f"  macro-F1={results[name]['macro_f1']:.3f}")

    anomaly = None if args.no_anomaly else _run_anomaly(F_train, y[tr], F_test, y[te])
    report = _write_report(results, anomaly, out_dir, cfg, augmented=not args.no_augment)
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
