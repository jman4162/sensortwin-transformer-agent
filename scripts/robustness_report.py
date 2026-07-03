"""CLI: robustness + uncertainty study (roadmap v0.5, spec §12.2 / §12.3 / §18).

Trains the key models on domain A (a clean-ish measurement regime), then reports, on the held-out
test split: clean macro-F1; degradation under Gaussian-noise / short-window / missing-channel
sweeps; macro-F1 on a shifted regime (domain B); and calibration (ECE) clean vs shifted, before and
after temperature scaling. Also closes the v0.4 loop: does the pretrained transformer hold up better
than the scratch one under domain shift, or did pretraining just memorize the generator?

Deep models train with their committed parity recipes (`scripts/train_baseline._build_deep_model`).
Because that recipe includes channel-dropout augmentation — the same corruption the
missing-channel sweep probes — the study runs two arms: ``augmented`` (deployment-realistic,
comparable across models) and ``no_augment`` (separates "robust" from "trained on the probe").

Multi-seed: pass ``--seeds 0 1 2``; per-(arm, seed) checkpoints under ``per_seed/`` make a killed
run resumable. Writes committable ``robustness_summary.{json,md}`` (mean ± sample std over seeds).

Example:
    python -m scripts.robustness_report --mode quick_demo --epochs 3 --seeds 0 1
    python -m scripts.robustness_report --mode colab_standard --epochs 30 --seeds 0 1 2 --arm both
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# macOS-only OpenMP guard; must precede torch/xgboost import.
configure_omp()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from scripts.train_baseline import _build_deep_model
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
from sensortwin.evaluation.statistics import mean_std
from sensortwin.features import build_feature_matrix
from sensortwin.models.baselines import make_xgboost
from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_yaml
from sensortwin.utils.seeds import set_torch_seed

NOISE_SIGMAS = [0.02, 0.05, 0.1, 0.2]
MODEL_ORDER = ["xgboost", "cnn", "transformer", "pretrained_tf"]
METRIC_KEYS = (
    "clean",
    "shift",
    "noise_delta",
    "window_delta",
    "missing_delta",
    "ece_clean",
    "ece_shift",
    "ece_shift_ts",
)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=range(len(EVENT_CLASSES)), average="macro", zero_division=0)
    )


def _gen_domain(artifacts: dict, n: int, T: int, seed: int):
    cfg = GenConfig(n_samples=n, T=T, seed=seed, normalize=False, artifacts=dict(artifacts))
    X, y, meta = generate_dataset(cfg)
    return X, y, meta


def _train_models(X, y, splits, standardizer, epochs, device=None, use_augment=True):
    """Return {name: dict(predict, proba_fn, logits_fn|None)} trained on the (standardized) train.

    CNN and both transformer arms use their committed parity recipes via ``_build_deep_model``;
    ``use_augment=False`` is the no-augmentation arm.
    """
    import torch  # noqa: F401

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
            return predict_proba(model, ds, device=device)

        def logits_fn(Xc):
            ds = SensorArrayDataset(standardizer.transform(Xc), np.zeros(len(Xc), int)).as_torch()
            return predict_logits(model, ds, device=device)

        return {
            "predict": lambda Xc: proba_fn(Xc).argmax(1),
            "proba": proba_fn,
            "logits": logits_fn,
        }

    models: dict[str, dict] = {}

    # xgboost on features (no logits / temp scaling; augmentation does not apply)
    clf = make_xgboost()
    clf.fit(build_feature_matrix(X[tr])[0], ytr)
    models["xgboost"] = {
        "predict": lambda Xc: clf.predict(build_feature_matrix(Xc)[0]),
        "proba": lambda Xc: clf.predict_proba(build_feature_matrix(Xc)[0]),
        "logits": None,
    }

    cnn, cnn_kwargs = _build_deep_model("cnn", use_augment=use_augment)
    cnn, _ = train_model(
        cnn, tr_ds, val_ds, weight=weight, device=device, **(cnn_kwargs | {"epochs": epochs})
    )
    models["cnn"] = torch_arm(cnn)

    tf, tf_kwargs = _build_deep_model("transformer", use_augment=use_augment)
    tf, _ = train_model(
        tf, tr_ds, val_ds, weight=weight, device=device, **(tf_kwargs | {"epochs": epochs})
    )
    models["transformer"] = torch_arm(tf)

    # pretrained transformer: masked-patch pretrain, then fine-tune with the same parity recipe
    fresh, _ = _build_deep_model("transformer", use_augment=False)
    pmodel, _, _ = pretrain_model(fresh, tr_ds, val_ds, epochs=epochs, device=device)
    ptf, ptf_kwargs = _build_deep_model("transformer", use_augment=use_augment)
    ptf = transfer_encoder(pmodel.state_dict(), ptf)
    ptf, _ = train_model(
        ptf, tr_ds, val_ds, weight=weight, device=device, **(ptf_kwargs | {"epochs": epochs})
    )
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


def _run_arm_seed(arm_name: str, seed: int, args) -> tuple[dict, dict]:
    """Train + evaluate every model for one (arm, seed); returns (rows, noise_curves)."""
    set_torch_seed(seed)
    dcfg = load_yaml(args.config)
    n = dcfg["modes"][args.mode]["n_samples"]
    T = dcfg.get("T", 512)
    print(f"[{arm_name} seed {seed}] generating domains A/B ({n} samples, T={T})...")
    X, y, _ = _gen_domain(dcfg["domain_a"], n, T, seed)
    Xb, _, _ = _gen_domain(dcfg["domain_b"], n, T, seed)  # same seed/events, shifted regime

    splits = make_split("random", y, {}, seed=seed)
    standardizer = ChannelStandardizer().fit(X[splits["train"]])
    models = _train_models(
        X,
        y,
        splits,
        standardizer,
        args.epochs,
        args.device,
        use_augment=(arm_name == "augmented"),
    )
    rows, curves = _evaluate(models, X, y, splits, Xb, seed)
    for n_, r in rows.items():
        print(
            f"[{arm_name} seed {seed}] {n_}: clean={r['clean']:.3f} shift={r['shift']:.3f} "
            f"ECE shift {r['ece_shift']:.3f}->{r['ece_shift_ts']:.3f}"
        )
    return rows, curves


def _aggregate(per_seed: dict[int, dict]) -> dict[str, dict[str, Any]]:
    """{model: {metric: {mean, std, per_seed}}} over seeds."""
    seeds = sorted(per_seed)
    agg: dict[str, dict[str, Any]] = {}
    for model in per_seed[seeds[0]]:
        agg[model] = {}
        for key in METRIC_KEYS:
            vals = [per_seed[s][model][key] for s in seeds]
            m, sd = mean_std(vals)
            agg[model][key] = {"mean": m, "std": sd, "per_seed": vals}
    return agg


def _fmt(cell: dict[str, Any]) -> str:
    sd = "" if np.isnan(cell["std"]) else f" ± {cell['std']:.3f}"
    return f"{cell['mean']:.3f}{sd}"


def _write_summary(arms: dict[str, dict], out_dir: Path, meta: dict) -> Path:
    (out_dir / "robustness_summary.json").write_text(
        json.dumps({"meta": meta, "arms": arms}, indent=2) + "\n"
    )
    lines = [
        "# Robustness & uncertainty study",
        "",
        f"Mode `{meta['mode']}`, seeds {meta['seeds']}, {meta['epochs']} epochs; mean ± sample "
        "std over seeds. Trained on domain A with the committed parity recipes; test-split "
        "metrics. Deltas are the worst-case macro-F1 drop under each corruption; domain shift = "
        "same classes, noisier regime (domain B). ECE columns: clean / shifted / shifted after "
        "temperature scaling.",
        "",
        "The `augmented` arm trains deep models with the shared recipe including channel-dropout "
        "augmentation (deployment-realistic; comparable across models but flattering on the "
        "missing-channel probe). The `no_augment` arm removes train-time augmentation entirely.",
    ]
    for arm_name, agg in arms.items():
        lines += [
            "",
            f"## Arm: {arm_name}",
            "",
            "| Model | Clean F1 | Shift F1 | Noise Δ | Window Δ | Missing-ch Δ | ECE clean | ECE shift | ECE shift (T) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in MODEL_ORDER:
            if name not in agg:
                continue
            r = agg[name]
            lines.append(
                f"| {name} | {_fmt(r['clean'])} | {_fmt(r['shift'])} | {_fmt(r['noise_delta'])} | "
                f"{_fmt(r['window_delta'])} | {_fmt(r['missing_delta'])} | {_fmt(r['ece_clean'])} | "
                f"{_fmt(r['ece_shift'])} | {_fmt(r['ece_shift_ts'])} |"
            )
        tf, ptf = agg.get("transformer"), agg.get("pretrained_tf")
        if tf and ptf:
            gap_clean = ptf["clean"]["mean"] - tf["clean"]["mean"]
            gap_shift = ptf["shift"]["mean"] - tf["shift"]["mean"]
            lines += [
                "",
                f"Pretrained − scratch macro-F1: **{gap_clean:+.3f} in-distribution**, "
                f"**{gap_shift:+.3f} under domain shift**. If the shift gap is not larger than "
                "the clean gap, pretraining is not buying extra robustness here — consistent "
                "with it partly learning the generator's regularities rather than transferable "
                "structure.",
            ]
    lines += [
        "",
        "## Claims",
        "- **Supported:** different models degrade differently under noise / window / channel "
        "loss / domain shift; temperature scaling reduces ECE.",
        "- **Not claimed:** synthetic robustness does not imply real-world robustness; domain B "
        "is a controlled regime change, not a real deployment shift.",
        "",
        "Regenerate: `python -m scripts.robustness_report --mode "
        f"{meta['mode']} --epochs {meta['epochs']} --seeds "
        f"{' '.join(str(s) for s in meta['seeds'])} --arm {meta['arm']}`.",
    ]
    out = out_dir / "robustness_summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Robustness + uncertainty study.")
    p.add_argument("--config", default="configs/synthetic/domain_shift.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument(
        "--arm",
        choices=["augmented", "no_augment", "both"],
        default="augmented",
        help="train deep models with the shared augmentation, without it, or run both arms",
    )
    p.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto-detect")
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    ckpt_dir = out_dir / "per_seed"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    arm_names = ["augmented", "no_augment"] if args.arm == "both" else [args.arm]

    arms: dict[str, dict] = {}
    last_curves: dict = {}
    for arm_name in arm_names:
        per_seed: dict[int, dict] = {}
        for seed in args.seeds:
            ckpt = ckpt_dir / f"robustness_{arm_name}_seed_{seed}.json"
            if ckpt.exists():
                saved = json.loads(ckpt.read_text())
                if saved["meta"] == {"mode": args.mode, "epochs": args.epochs}:
                    print(f"[{arm_name} seed {seed}] reusing checkpoint {ckpt}")
                    per_seed[seed] = saved["rows"]
                    continue
            rows, curves = _run_arm_seed(arm_name, seed, args)
            per_seed[seed] = rows
            last_curves = curves
            ckpt.write_text(
                json.dumps(
                    {"meta": {"mode": args.mode, "epochs": args.epochs}, "rows": rows}, indent=2
                )
            )
        arms[arm_name] = _aggregate(per_seed)

    if last_curves:
        plot_robustness_degradation(
            last_curves,
            out_dir / "figures" / "robustness_degradation.png",
            title="Macro-F1 vs Gaussian-noise sigma (last seed)",
            xlabel="noise sigma",
        )

    import platform

    import torch

    meta = {
        "mode": args.mode,
        "epochs": args.epochs,
        "seeds": list(args.seeds),
        "arm": args.arm,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    report = _write_summary(arms, out_dir, meta)
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
