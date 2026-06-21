"""CLI: run the benchmark slate on real NASA C-MAPSS data (roadmap v0.6, spec §16).

Question (A): does the harness and the synthetic finding *port to real data*? We train the same
models — XGBoost-on-features, CNN, SensorPatchTST — from scratch on C-MAPSS (3-stage health
classification), with the same metrics (macro-F1 headline, per-class P/R, AUROC, ECE) and the same
robustness sweeps (noise / short-window / missing-channel). No synthetic-to-real weight transfer here
(that is ``scripts/sim2real_transfer.py``); this is the apples-to-apples real-data baseline.

The split is **grouped by engine** so windows from one engine never straddle train/test.

Examples
--------
    python -m scripts.fetch_cmapss --raw-dir /path/to/CMAPSSData --mode quick_demo
    python -m scripts.real_data_report --data data/cmapss_FD001 --mode quick_demo --epochs 5
    python -m scripts.real_data_report --raw-dir /path/to/CMAPSSData --mode quick_demo --epochs 5
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# macOS-only OpenMP guard; must precede torch/xgboost import. See CLAUDE.md.
configure_omp()

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from sensortwin.data.cmapss import load_cmapss, load_cmapss_subsets
from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.calibration import brier_score_multiclass, expected_calibration_error
from sensortwin.evaluation.metrics import classification_metrics, save_metrics
from sensortwin.evaluation.plots import plot_confusion_matrix, plot_per_class_f1
from sensortwin.evaluation.robustness import (
    missing_channel_sweep,
    noise_sweep,
    short_window_sweep,
)
from sensortwin.features import build_feature_matrix
from sensortwin.models.baselines import feature_importance, make_xgboost
from sensortwin.utils.config import load_mode_config
from sensortwin.utils.io import load_dataset
from sensortwin.utils.seeds import set_torch_seed

DEFAULT_MODELS = ["xgboost", "cnn", "transformer"]


def _macro_f1_fn(n_classes: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(
            f1_score(y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0)
        )

    return fn


def _evaluate(
    name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    predict_raw: Callable[[np.ndarray], np.ndarray],
    X_test_raw: np.ndarray,
    class_names: list[str],
    out_dir: Path,
    train_time: float,
    n_params: int | None,
) -> dict[str, Any]:
    macro_f1 = _macro_f1_fn(len(class_names))
    metrics = classification_metrics(y_test, y_pred, y_proba, class_names)
    metrics["ece"] = expected_calibration_error(y_test, y_proba)
    metrics["brier"] = brier_score_multiclass(y_test, y_proba)
    metrics["missing_channel"] = missing_channel_sweep(
        predict_raw, X_test_raw, y_test, macro_f1_fn=macro_f1
    )
    metrics["noise"] = noise_sweep(predict_raw, X_test_raw, y_test, macro_f1_fn=macro_f1)
    metrics["short_window"] = short_window_sweep(
        predict_raw, X_test_raw, y_test, macro_f1_fn=macro_f1
    )
    metrics["train_time_s"] = round(train_time, 2)
    metrics["n_params"] = n_params

    save_metrics(metrics, out_dir / f"metrics_cmapss_{name}")
    fig_dir = out_dir / "figures"
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), class_names, fig_dir / f"cmapss_cm_{name}.png"
    )
    plot_per_class_f1(
        metrics["per_class"],
        fig_dir / f"cmapss_f1_{name}.png",
        title=f"{name} per-class F1 (C-MAPSS)",
    )
    return metrics


def _run_xgboost(Xtr_raw, ytr, Xte_raw, yte, channel_names, class_names, out_dir) -> dict[str, Any]:
    F_train, feat_names = build_feature_matrix(Xtr_raw, channels=channel_names)
    F_test, _ = build_feature_matrix(Xte_raw, channels=channel_names)
    model = make_xgboost()
    t0 = time.perf_counter()
    model.fit(F_train, ytr)
    train_time = time.perf_counter() - t0

    y_proba = model.predict_proba(F_test)

    def predict_raw(Xc: np.ndarray) -> np.ndarray:
        return model.predict(build_feature_matrix(Xc, channels=channel_names)[0])

    metrics = _evaluate(
        "xgboost",
        yte,
        y_proba.argmax(1),
        y_proba,
        predict_raw,
        Xte_raw,
        class_names,
        out_dir,
        train_time,
        None,
    )
    metrics["feature_importance"] = feature_importance(model, feat_names)
    return metrics


def _run_deep(
    name,
    in_channels,
    num_classes,
    arch,
    train_kwargs,
    Xtr_std,
    ytr,
    Xva_std,
    yva,
    Xte_std,
    Xte_raw,
    yte,
    standardizer,
    epochs,
    class_names,
    out_dir,
    device=None,
) -> dict[str, Any]:
    import torch

    from sensortwin.models.cnn import SensorCNN
    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.loop import class_weights, predict_proba, train_model

    if name == "cnn":
        model: Any = SensorCNN(in_channels=in_channels, num_classes=num_classes)
    elif name == "transformer":
        model = SensorPatchTST(in_channels=in_channels, num_classes=num_classes, **arch)
    else:
        raise SystemExit(f"unknown deep model '{name}'")
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
        weight=class_weights(ytr, num_classes),
        device=device,
        **train_kwargs,
    )
    train_time = time.perf_counter() - t0

    y_proba = predict_proba(model, test_ds, device=device)

    def predict_raw(Xc: np.ndarray) -> np.ndarray:
        ds = SensorArrayDataset(standardizer.transform(Xc), yte[: len(Xc)]).as_torch()
        return predict_proba(model, ds, device=device).argmax(1)

    with torch.no_grad():
        return _evaluate(
            name,
            yte,
            y_proba.argmax(1),
            y_proba,
            predict_raw,
            Xte_raw,
            class_names,
            out_dir,
            train_time,
            n_params,
        )


def _load_data(args) -> tuple[np.ndarray, np.ndarray, dict]:
    data_cfg = load_mode_config(args.config, mode=args.mode).get("data", {})
    if args.data:
        return load_dataset(args.data)
    if not args.raw_dir:
        raise SystemExit(
            "provide --data <npz> (run scripts.fetch_cmapss first) or --raw-dir <CMAPSS folder>"
        )
    subsets = [s.strip() for s in str(data_cfg.get("subset", "FD001")).split(",") if s.strip()]
    kwargs = {
        "channels": data_cfg.get("channels"),
        "window": data_cfg.get("window", 48),
        "stride": data_cfg.get("stride", 12),
        "rul_bins": tuple(data_cfg.get("rul_bins", (30, 70))),
        "rul_cap": data_cfg.get("rul_cap", 125),
        "class_names": data_cfg.get("class_names"),
    }
    if len(subsets) == 1:
        return load_cmapss(args.raw_dir, subsets[0], **kwargs)
    return load_cmapss_subsets(args.raw_dir, subsets, **kwargs)


def _write_report(results, out_dir, meta, class_names) -> Path:
    lines = [
        "# C-MAPSS real-data results (v0.6)",
        "",
        f"Source `{meta.get('source', 'cmapss')}`, {meta['n_channels']} sensors, "
        f"window T={meta.get('window')}, {len(class_names)}-stage health classification "
        f"({' / '.join(class_names)}). Grouped-by-engine split (no engine crosses train/test); "
        "metrics on the held-out test engines.",
        "",
        "| Model | Macro-F1 | Weighted-F1 | Accuracy | Macro-AUROC | ECE | Noise Δ | Short-win Δ | Missing-ch Δ | Params | Train (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in results.items():
        params = f"{m['n_params']/1000:.0f}k" if m.get("n_params") else "—"
        lines.append(
            f"| {name} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} | {m['accuracy']:.3f} | "
            f"{_fmt(m.get('macro_auroc'))} | {_fmt(m.get('ece'))} | "
            f"{_fmt(m['noise']['worst_delta'])} | {_fmt(m['short_window']['worst_delta'])} | "
            f"{_fmt(m['missing_channel']['worst_delta'])} | {params} | {m['train_time_s']:.1f} |"
        )
    lines += [
        "",
        "Δ columns are the worst-case macro-F1 drop vs clean under each corruption (higher = more "
        "fragile). Macro-F1 is the headline (the health classes are imbalanced).",
        "",
        "## Claims",
        "",
        "- **Supported:** the synthetic-data pipeline (windowing, features, the same model classes, "
        "the same evaluation) runs on *real* C-MAPSS sensor data, and the models can be ranked on it.",
        "- **Not claimed:** absolute numbers here are not state-of-the-art RUL/health estimation. "
        "C-MAPSS is natively a regression benchmark; the 3-stage binning is a deliberate "
        "classification reframing for an apples-to-apples comparison with the synthetic task.",
        "",
        "These are quick-mode wiring figures; run `--mode full` (more windows, both failure modes) "
        "with more epochs before drawing conclusions.",
    ]
    out = out_dir / "cmapss_results.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the benchmark slate on real C-MAPSS data.")
    p.add_argument("--data", default=None, help="processed .npz from scripts.fetch_cmapss")
    p.add_argument("--raw-dir", default=None, help="raw C-MAPSS folder (build on the fly)")
    p.add_argument("--config", default="configs/data/cmapss.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto-detect")
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    set_torch_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    X, y, meta = _load_data(args)
    class_names = meta["event_classes"]
    channel_names = meta.get("channel_names")
    in_channels = X.shape[1]
    print(
        f"Loaded C-MAPSS: X={X.shape}, {len(class_names)} classes, "
        f"{len(set(meta['groups']))} engines"
    )

    sp = make_split("grouped", y, meta, seed=args.seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    standardizer = ChannelStandardizer().fit(X[tr])
    Xte_raw = X[te]

    full = load_mode_config(args.config, mode=args.mode)
    arch = full.get("model", {})
    train_block = full.get("train", {})
    loop_keys = (
        "optimizer",
        "lr",
        "weight_decay",
        "label_smoothing",
        "scheduler",
        "warmup_epochs",
        "batch_size",
        "patience",
    )
    transformer_kwargs = {k: train_block[k] for k in loop_keys if k in train_block}

    results: dict[str, dict] = {}
    for name in models:
        print(f"--- {name} ---")
        if name == "xgboost":
            results[name] = _run_xgboost(
                X[tr], y[tr], Xte_raw, y[te], channel_names, class_names, out_dir
            )
        else:
            results[name] = _run_deep(
                name,
                in_channels,
                len(class_names),
                arch,
                transformer_kwargs if name == "transformer" else {},
                standardizer.transform(X[tr]),
                y[tr],
                standardizer.transform(X[va]),
                y[va],
                standardizer.transform(Xte_raw),
                Xte_raw,
                y[te],
                standardizer,
                args.epochs,
                class_names,
                out_dir,
                args.device,
            )
        print(f"  macro-F1={results[name]['macro_f1']:.3f}")

    report = _write_report(results, out_dir, meta, class_names)
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
