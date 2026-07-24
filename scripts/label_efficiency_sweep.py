"""CLI: label-efficiency sweep (roadmap v0.4, spec §12.4 / §16).

Pretrains SensorPatchTST once with masked-patch reconstruction on the unlabeled train split, then
compares five arms across label fractions {1, 5, 10, 100}% on the held-out test split:
  scratch transformer | pretrained+fine-tuned | pretrained+linear-probe | CNN | XGBoost-features.

The question (spec §12.4): does self-supervised pretraining buy accuracy when labels are scarce?
Expected and reported honestly: pretraining helps most at low fractions and the gap narrows at 100%.

Example:
    python -m scripts.label_efficiency_sweep --mode quick_demo --epochs-pretrain 5 --epochs-finetune 5
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# macOS-only OpenMP guard; must precede torch/xgboost import.
configure_omp()

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.calibration import expected_calibration_error
from sensortwin.evaluation.plots import plot_label_efficiency
from sensortwin.features import build_feature_matrix
from sensortwin.models.baselines import make_xgboost
from sensortwin.simulation import generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config, load_yaml
from sensortwin.utils.seeds import set_torch_seed

ARMS = ["scratch", "pretrained_ft", "pretrained_probe", "cnn", "xgboost"]
DEFAULT_FRACTIONS = [0.01, 0.05, 0.10, 1.0]
# Per-arm learning-rate candidates, selected on validation macro-F1. Both transformer arms get
# the same two-point budget so "pretraining did not help" cannot be an artifact of the fine-tune
# arm training at a fixed lower LR than scratch (the confound in the 2026-06-25 run).
ARM_LRS = (1e-3, 1e-4)
_LOOP_KEYS = (
    "optimizer",
    "lr",
    "weight_decay",
    "label_smoothing",
    "scheduler",
    "warmup_epochs",
    "batch_size",
    "patience",
)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=range(len(EVENT_CLASSES)), average="macro", zero_division=0)
    )


def _loop_kwargs(block: dict, epochs: int) -> dict[str, Any]:
    from sensortwin.training.augment import build_augment

    kw: dict[str, Any] = {k: block[k] for k in _LOOP_KEYS if k in block}
    kw["epochs"] = epochs
    aug = build_augment(block.get("augment"))
    if aug is not None:
        kw["augment"] = aug
    return kw


def _stratified_subset(y: np.ndarray, idx: np.ndarray, fraction: float, rng) -> np.ndarray:
    """Sample ``fraction`` of ``idx`` per class (>=1 per present class), so every class survives 1%."""
    if fraction >= 1.0:
        return idx
    keep = []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        n = max(1, round(fraction * len(members)))
        keep.append(rng.choice(members, size=n, replace=False))
    return np.concatenate(keep)


def _eval_torch(model, test_ds, y_test, device=None) -> dict[str, float]:
    from sensortwin.training.loop import predict_proba

    proba = predict_proba(model, test_ds, device=device)
    return {
        "macro_f1": _macro_f1(y_test, proba.argmax(1)),
        "ece": expected_calibration_error(y_test, proba),
    }


def _run_fraction(
    fraction: float,
    seed: int,
    X,
    y,
    splits,
    standardizer,
    model_cfg,
    pretrained_state,
    sup_block,
    ft_block,
    epochs_ft,
    device=None,
) -> dict[str, dict[str, float]]:
    import torch  # noqa: F401

    from scripts.train_baseline import _build_deep_model
    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.loop import class_weights, train_model
    from sensortwin.training.pretrain import freeze_encoder, transfer_encoder

    rng = np.random.default_rng(seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    sub = _stratified_subset(y, tr, fraction, rng)

    Xsub, ysub = standardizer.transform(X[sub]), y[sub]
    val_ds = SensorArrayDataset(standardizer.transform(X[va]), y[va]).as_torch()
    test_ds = SensorArrayDataset(standardizer.transform(X[te]), y[te]).as_torch()
    sub_ds = SensorArrayDataset(Xsub, ysub).as_torch()
    weight = class_weights(ysub, len(EVENT_CLASSES))

    def train_best_lr(build_model, block) -> tuple[Any, float]:
        """Train at each candidate LR; keep the best by validation macro-F1 (never test).

        Both transformer arms get this identical budget, removing the old confound where the
        pretrained arms fine-tuned at a fixed lower LR than scratch and were undertrained.
        """
        best_model, best_val, best_lr = None, -1.0, ARM_LRS[0]
        for lr in ARM_LRS:
            kw = _loop_kwargs(block, epochs_ft) | {"lr": lr}
            m, hist = train_model(build_model(), sub_ds, val_ds, weight=weight, device=device, **kw)
            if hist.best_val_macro_f1 > best_val:
                best_model, best_val, best_lr = m, hist.best_val_macro_f1, lr
        return best_model, best_lr

    out: dict[str, dict[str, float]] = {}

    # 1. scratch transformer (supervised recipe, per-arm LR selected on val)
    m, lr = train_best_lr(lambda: SensorPatchTST(**model_cfg), sup_block)
    out["scratch"] = _eval_torch(m, test_ds, y[te], device) | {"lr": lr}

    # 2. pretrained + full fine-tune (same LR budget as scratch)
    m, lr = train_best_lr(
        lambda: transfer_encoder(pretrained_state, SensorPatchTST(**model_cfg)), ft_block
    )
    out["pretrained_ft"] = _eval_torch(m, test_ds, y[te], device) | {"lr": lr}

    # 3. pretrained + linear probe (frozen encoder, same LR budget)
    m, lr = train_best_lr(
        lambda: freeze_encoder(transfer_encoder(pretrained_state, SensorPatchTST(**model_cfg))),
        ft_block,
    )
    out["pretrained_probe"] = _eval_torch(m, test_ds, y[te], device) | {"lr": lr}

    # 4. CNN baseline (its committed parity recipe, not the bare training loop)
    m, cnn_kwargs = _build_deep_model("cnn")
    m, _ = train_model(
        m, sub_ds, val_ds, weight=weight, device=device, **(cnn_kwargs | {"epochs": epochs_ft})
    )
    out["cnn"] = _eval_torch(m, test_ds, y[te], device)

    # 5. XGBoost on engineered features
    clf = make_xgboost()
    clf.fit(build_feature_matrix(X[sub])[0], ysub)
    proba = clf.predict_proba(build_feature_matrix(X[te])[0])
    out["xgboost"] = {
        "macro_f1": _macro_f1(y[te], proba.argmax(1)),
        "ece": expected_calibration_error(y[te], proba),
    }
    return out


def _write_report(curves, out_dir, mode, fractions, seeds, epochs_pre, epochs_ft) -> Path:
    lines = [
        "# Label efficiency (v0.4)",
        "",
        f"Mode `{mode}`, {seeds} seed(s), pretrain {epochs_pre} epochs / fine-tune {epochs_ft} "
        "epochs. Masked-patch pretraining on the unlabeled train split, then each arm trained on a "
        "stratified label subset and scored (macro-F1) on the held-out test split.",
        "",
        "| Fraction | " + " | ".join(ARMS) + " |",
        "| --- | " + " | ".join("---:" for _ in ARMS) + " |",
    ]
    for f in fractions:
        row = " | ".join(f"{curves[a][f]:.3f}" for a in ARMS)
        lines.append(f"| {f * 100:g}% | {row} |")
    lines += [
        "",
        "`pretrained_ft` = pretrained encoder, full fine-tune; `pretrained_probe` = frozen encoder + "
        "linear head. The label-efficiency question: does `pretrained_ft` beat `scratch` most at the "
        "smallest fractions, with the gap closing toward 100%?",
        "",
        "## Claims",
        "",
        "- **Supported:** this measures whether masked pretraining improves sample efficiency on "
        "*this synthetic benchmark*.",
        "- **Not claimed:** this is not a foundation model, and synthetic-pretraining gains may partly "
        "reflect learning the generator's regularities rather than transferable structure.",
        "",
        f"These are `{mode}` figures for wiring/discipline, not research-grade; run `colab_standard` "
        "with several seeds before interpreting the curve. See the macro-F1-vs-fraction plot in "
        "`figures/label_efficiency.png`.",
    ]
    out = out_dir / "label_efficiency.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _write_summary(acc, out_dir: Path, args, fractions, epochs_pre, epochs_ft) -> Path:
    """Committable ``label_efficiency_summary.{json,md}``: per-seed values, mean ± sample std,
    and run metadata — the artifact behind any pretraining verdict in the public docs."""
    import json
    import platform

    import torch

    from sensortwin.evaluation.statistics import mean_std

    meta = {
        "mode": args.mode,
        "seeds": args.seeds,
        "base_seed": args.seed,
        "epochs_pretrain": epochs_pre,
        "epochs_finetune": epochs_ft,
        "arm_lrs": list(ARM_LRS),
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    table = {
        a: {
            f"{f:g}": {
                "per_seed": acc[a][f],
                "mean": mean_std(acc[a][f])[0],
                "std": mean_std(acc[a][f])[1],
            }
            for f in fractions
        }
        for a in ARMS
    }
    (out_dir / "label_efficiency_summary.json").write_text(
        json.dumps({"meta": meta, "arms": table}, indent=2) + "\n"
    )

    def _fmt(cell):
        sd = cell["std"]
        return f"{cell['mean']:.3f}" if np.isnan(sd) else f"{cell['mean']:.3f} ± {sd:.3f}"

    lines = [
        "# Label efficiency (masked pretraining vs scratch)",
        "",
        f"Mode `{meta['mode']}`, {meta['seeds']} seed(s), pretrain {epochs_pre} / fine-tune "
        f"{epochs_ft} epochs. Test macro-F1, mean ± sample std over seeds. Both transformer arms "
        f"select their learning rate from the same validation budget {ARM_LRS}, so the "
        "pretrained-vs-scratch comparison is not confounded by a fixed fine-tune LR.",
        "",
        "| Fraction | " + " | ".join(ARMS) + " |",
        "| --- | " + " | ".join("---:" for _ in ARMS) + " |",
    ]
    for f in fractions:
        row = " | ".join(_fmt(table[a][f"{f:g}"]) for a in ARMS)
        lines.append(f"| {f * 100:g}% | {row} |")
    lines += [
        "",
        "Regenerate: `python -m scripts.label_efficiency_sweep --mode "
        f"{meta['mode']} --seeds {meta['seeds']} --epochs-pretrain {epochs_pre} "
        f"--epochs-finetune {epochs_ft}`.",
    ]
    out = out_dir / "label_efficiency_summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Label-efficiency sweep (masked pretraining vs scratch)."
    )
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--model-config", default="configs/models/sensorpatchtst.yaml")
    p.add_argument("--pretrain-config", default="configs/models/sensorpatchtst_pretrain.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--epochs-pretrain", type=int, default=None)
    p.add_argument("--epochs-finetune", type=int, default=None)
    p.add_argument("--fractions", default=None, help="comma-separated, e.g. 0.01,0.05,0.1,1.0")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto-detect")
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    set_torch_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = (
        [float(x) for x in args.fractions.split(",")] if args.fractions else DEFAULT_FRACTIONS
    )

    model_cfg = load_yaml(args.model_config).get("model", {})
    sup_block = load_yaml(args.model_config).get("train", {})
    pre_cfg = load_yaml(args.pretrain_config)
    pre_block, ft_block = pre_cfg.get("pretrain", {}), pre_cfg.get("finetune", {})

    cfg = load_synthetic_config(args.config, mode=args.mode)
    cfg.seed = args.seed
    cfg.normalize = False
    print(f"Generating {cfg.n_samples} samples (T={cfg.T}, seed={cfg.seed})...")
    X, y, _ = generate_dataset(cfg)
    splits = make_split("random", y, {}, seed=args.seed)
    standardizer = ChannelStandardizer().fit(X[splits["train"]])

    # Pretrain once on the unlabeled (standardized) train split.
    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.pretrain import pretrain_model

    epochs_pre = (
        args.epochs_pretrain if args.epochs_pretrain is not None else pre_block.get("epochs", 50)
    )
    epochs_ft = (
        args.epochs_finetune if args.epochs_finetune is not None else ft_block.get("epochs", 30)
    )
    # Checkpoints: a Colab disconnect or kill resumes instead of restarting (the 100% fraction
    # alone is hours of GPU). The pretrained encoder is cached to disk; each (seed, fraction)
    # cell is cached as JSON. Reused only when the run parameters match.
    import json as _json

    import torch

    ckpt_dir = out_dir / "per_seed"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_meta = {
        "mode": args.mode,
        "epochs_pretrain": epochs_pre,
        "epochs_finetune": epochs_ft,
        "arm_lrs": list(ARM_LRS),
        "base_seed": args.seed,
        "fractions": [float(f) for f in fractions],
    }
    pt_path = ckpt_dir / "le_pretrained_state.pt"
    pt_meta_path = ckpt_dir / "le_pretrained_meta.json"

    if (
        pt_path.exists()
        and pt_meta_path.exists()
        and _json.loads(pt_meta_path.read_text()) == ckpt_meta
    ):
        print(f"Reusing pretrained encoder checkpoint {pt_path}")
        pretrained_state = torch.load(pt_path, map_location="cpu", weights_only=True)
    else:
        print(f"Pretraining (masked reconstruction) for {epochs_pre} epochs...")
        pre_ds = SensorArrayDataset(
            standardizer.transform(X[splits["train"]]), y[splits["train"]]
        ).as_torch()
        val_ds = SensorArrayDataset(
            standardizer.transform(X[splits["val"]]), y[splits["val"]]
        ).as_torch()
        pre_kwargs = {k: pre_block[k] for k in _LOOP_KEYS + ("mask_ratio",) if k in pre_block}
        from sensortwin.training.augment import build_augment

        aug = build_augment(pre_block.get("augment"))
        pmodel, _, phist = pretrain_model(
            SensorPatchTST(**model_cfg),
            pre_ds,
            val_ds,
            epochs=epochs_pre,
            augment=aug,
            device=args.device,
            **pre_kwargs,
        )
        pretrained_state = copy.deepcopy({k: v.cpu() for k, v in pmodel.state_dict().items()})
        torch.save(pretrained_state, pt_path)
        pt_meta_path.write_text(_json.dumps(ckpt_meta, indent=2))
        print(f"  pretrain val MSE: {phist.val_loss[0]:.4f} -> {phist.val_loss[-1]:.4f}")

    # Accumulate macro-F1 per (arm, fraction) over seeds.
    acc: dict[str, dict[float, list[float]]] = {a: {f: [] for f in fractions} for a in ARMS}
    for s in range(args.seeds):
        for f in fractions:
            cell = ckpt_dir / f"le_seed{args.seed + s}_frac{f:g}.json"
            if cell.exists():
                saved = _json.loads(cell.read_text())
                if saved["meta"] == ckpt_meta:
                    print(f"--- seed {s} fraction {f * 100:g}% (reusing checkpoint) ---")
                    for a in ARMS:
                        acc[a][f].append(saved["res"][a]["macro_f1"])
                    continue
            print(f"--- seed {s} fraction {f * 100:g}% ---")
            res = _run_fraction(
                f,
                args.seed + s,
                X,
                y,
                splits,
                standardizer,
                model_cfg,
                pretrained_state,
                sup_block,
                ft_block,
                epochs_ft,
                args.device,
            )
            cell.write_text(_json.dumps({"meta": ckpt_meta, "res": res}, indent=2))
            for a in ARMS:
                acc[a][f].append(res[a]["macro_f1"])
                print(f"  {a}: macro-F1={res[a]['macro_f1']:.3f}")

    curves = {a: {f: float(np.mean(acc[a][f])) for f in fractions} for a in ARMS}
    plot_label_efficiency(curves, out_dir / "figures" / "label_efficiency.png")
    report = _write_report(curves, out_dir, args.mode, fractions, args.seeds, epochs_pre, epochs_ft)
    summary = _write_summary(acc, out_dir, args, fractions, epochs_pre, epochs_ft)
    print(f"\nReports written to {report} and {summary}")


if __name__ == "__main__":
    main()
