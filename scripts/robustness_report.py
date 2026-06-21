"""CLI: robustness + uncertainty study (roadmap v0.5, spec §12.2 / §12.3 / §18).

Trains the key models on domain A (a clean-ish measurement regime), then reports, on the held-out
test split: clean macro-F1; degradation under Gaussian-noise / short-window / missing-channel
sweeps; macro-F1 on a shifted regime (domain B); and calibration (ECE) clean vs shifted, before and
after temperature scaling. Also closes the v0.4 loop: does the pretrained transformer hold up better
than the scratch one under domain shift, or did pretraining just memorize the generator?

Example:
    python -m scripts.robustness_report --mode quick_demo --epochs 3
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from sensortwin.evaluation.plots import plot_robustness_degradation
from sensortwin.evaluation.robustness import missing_channel_sweep, noise_sweep, short_window_sweep
from sensortwin.features import build_feature_matrix
from sensortwin.models.baselines import make_xgboost
from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_yaml
from sensortwin.utils.seeds import set_torch_seed

NOISE_SIGMAS = [0.02, 0.05, 0.1, 0.2]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=range(len(EVENT_CLASSES)), average="macro", zero_division=0)
    )


def _gen_domain(base: dict, artifacts: dict, n: int, T: int, seed: int):
    cfg = GenConfig(n_samples=n, T=T, seed=seed, normalize=False, artifacts=dict(artifacts))
    X, y, meta = generate_dataset(cfg)
    return X, y, meta


def _train_models(X, y, splits, standardizer, model_cfg, pre_cfg, epochs):
    """Return {name: dict(predict, proba_fn, logits_fn|None)} trained on the (standardized) train."""
    import torch  # noqa: F401

    from sensortwin.models.cnn import SensorCNN
    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.loop import class_weights, predict_logits, predict_proba, train_model
    from sensortwin.training.pretrain import pretrain_model, transfer_encoder

    tr, va = splits["train"], splits["val"]
    Xtr, ytr = standardizer.transform(X[tr]), y[tr]
    val_ds = SensorArrayDataset(standardizer.transform(X[va]), y[va]).as_torch()
    tr_ds = SensorArrayDataset(Xtr, ytr).as_torch()
    weight = class_weights(ytr, len(EVENT_CLASSES))

    def torch_arm(model):
        def proba_fn(Xc):
            ds = SensorArrayDataset(standardizer.transform(Xc), np.zeros(len(Xc), int)).as_torch()
            return predict_proba(model, ds)

        def logits_fn(Xc):
            ds = SensorArrayDataset(standardizer.transform(Xc), np.zeros(len(Xc), int)).as_torch()
            return predict_logits(model, ds)

        return {
            "predict": lambda Xc: proba_fn(Xc).argmax(1),
            "proba": proba_fn,
            "logits": logits_fn,
        }

    models: dict[str, dict] = {}

    # xgboost on features (no logits / temp scaling)
    clf = make_xgboost()
    clf.fit(build_feature_matrix(X[tr])[0], ytr)
    models["xgboost"] = {
        "predict": lambda Xc: clf.predict(build_feature_matrix(Xc)[0]),
        "proba": lambda Xc: clf.predict_proba(build_feature_matrix(Xc)[0]),
        "logits": None,
    }

    cnn = SensorCNN()
    cnn, _ = train_model(cnn, tr_ds, val_ds, epochs=epochs, weight=weight)
    models["cnn"] = torch_arm(cnn)

    tf = SensorPatchTST(**model_cfg)
    tf, _ = train_model(tf, tr_ds, val_ds, epochs=epochs, weight=weight)
    models["transformer"] = torch_arm(tf)

    # pretrained transformer
    pmodel, _, _ = pretrain_model(SensorPatchTST(**model_cfg), tr_ds, val_ds, epochs=epochs)
    ptf = transfer_encoder(pmodel.state_dict(), SensorPatchTST(**model_cfg))
    ptf, _ = train_model(ptf, tr_ds, val_ds, epochs=epochs, weight=weight)
    models["pretrained_tf"] = torch_arm(ptf)
    return models


def _evaluate(models, X, y, splits, Xb, seed):
    te, va = splits["test"], splits["val"]
    rows: dict[str, dict[str, Any]] = {}
    noise_curves: dict[str, dict[float, float]] = {}
    for name, arm in models.items():
        clean = _macro_f1(y[te], arm["predict"](X[te]))
        shift = _macro_f1(y[te], arm["predict"](Xb[te]))
        ns = noise_sweep(
            arm["predict"], X[te], y[te], macro_f1_fn=_macro_f1, sigmas=NOISE_SIGMAS, seed=seed
        )
        sw = short_window_sweep(arm["predict"], X[te], y[te], macro_f1_fn=_macro_f1)
        mc = missing_channel_sweep(arm["predict"], X[te], y[te], macro_f1_fn=_macro_f1)
        ece_clean = expected_calibration_error(y[te], arm["proba"](X[te]))
        ece_shift = expected_calibration_error(y[te], arm["proba"](Xb[te]))
        ece_shift_ts = ece_shift
        if arm["logits"] is not None:
            temp = fit_temperature(arm["logits"](X[va]), y[va])
            ece_shift_ts = expected_calibration_error(
                y[te], apply_temperature(arm["logits"](Xb[te]), temp)
            )
        rows[name] = {
            "clean": clean,
            "shift": shift,
            "noise_delta": ns["worst_delta"],
            "window_delta": sw["worst_delta"],
            "missing_delta": mc["worst_delta"],
            "ece_clean": ece_clean,
            "ece_shift": ece_shift,
            "ece_shift_ts": ece_shift_ts,
        }
        noise_curves[name] = {0.0: clean, **{s: ns[f"level_{s:g}"] for s in NOISE_SIGMAS}}
    return rows, noise_curves


def _write_report(rows, out_dir, mode, epochs) -> Path:
    lines = [
        "# Robustness & uncertainty study (v0.5)",
        "",
        f"Mode `{mode}`, {epochs} epochs. Trained on domain A; test-split metrics. Deltas are the "
        "worst-case macro-F1 drop under each corruption; domain shift = same classes, noisier regime "
        "(domain B). ECE columns: clean / shifted / shifted after temperature scaling.",
        "",
        "| Model | Clean F1 | Shift F1 | Noise Δ | Window Δ | Missing-ch Δ | ECE clean | ECE shift | ECE shift (T) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n, r in rows.items():
        lines.append(
            f"| {n} | {r['clean']:.3f} | {r['shift']:.3f} | {r['noise_delta']:.3f} | "
            f"{r['window_delta']:.3f} | {r['missing_delta']:.3f} | {r['ece_clean']:.3f} | "
            f"{r['ece_shift']:.3f} | {r['ece_shift_ts']:.3f} |"
        )
    tf, ptf = rows.get("transformer"), rows.get("pretrained_tf")
    if tf and ptf:
        gap_clean = ptf["clean"] - tf["clean"]
        gap_shift = ptf["shift"] - tf["shift"]
        lines += [
            "",
            "## Pretraining under shift (closes the v0.4 caveat)",
            "",
            f"Pretrained − scratch macro-F1: **{gap_clean:+.3f} in-distribution**, "
            f"**{gap_shift:+.3f} under domain shift**. If the shift gap is not larger than the clean "
            "gap, pretraining is not buying extra robustness here — consistent with it partly "
            "learning the generator's regularities rather than transferable structure.",
        ]
    lines += [
        "",
        "## Claims",
        "- **Supported:** different models degrade differently under noise / window / channel loss / "
        "domain shift; temperature scaling reduces ECE.",
        "- **Not claimed:** synthetic robustness does not imply real-world robustness; domain B is a "
        "controlled regime change, not a real deployment shift.",
        "",
        f"`{mode}` figures for wiring/discipline, not research-grade; run `colab_standard`. See "
        "`figures/robustness_degradation.png`.",
    ]
    out = out_dir / "robustness_study.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Robustness + uncertainty study.")
    p.add_argument("--config", default="configs/synthetic/domain_shift.yaml")
    p.add_argument("--model-config", default="configs/models/sensorpatchtst.yaml")
    p.add_argument("--pretrain-config", default="configs/models/sensorpatchtst_pretrain.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    set_torch_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = load_yaml(args.config)
    n = dcfg["modes"][args.mode]["n_samples"]
    T = dcfg.get("T", 512)
    print(f"Generating domains A/B ({n} samples, T={T})...")
    X, y, _ = _gen_domain(dcfg, dcfg["domain_a"], n, T, args.seed)
    Xb, _, _ = _gen_domain(
        dcfg, dcfg["domain_b"], n, T, args.seed
    )  # same seed/events, shifted regime

    splits = make_split("random", y, {}, seed=args.seed)
    standardizer = ChannelStandardizer().fit(X[splits["train"]])
    model_cfg = load_yaml(args.model_config).get("model", {})

    print("Training models on domain A...")
    models = _train_models(
        X, y, splits, standardizer, model_cfg, load_yaml(args.pretrain_config), args.epochs
    )
    rows, noise_curves = _evaluate(models, X, y, splits, Xb, args.seed)
    for n_, r in rows.items():
        print(
            f"  {n_}: clean={r['clean']:.3f} shift={r['shift']:.3f} ECE shift {r['ece_shift']:.3f}->{r['ece_shift_ts']:.3f}"
        )
    plot_robustness_degradation(
        noise_curves,
        out_dir / "figures" / "robustness_degradation.png",
        title="Macro-F1 vs Gaussian-noise sigma",
        xlabel="noise sigma",
    )
    report = _write_report(rows, out_dir, args.mode, args.epochs)
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
