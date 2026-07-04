"""CLI: synthetic-to-real representation transfer on C-MAPSS (roadmap v0.6, spec §16).

Question (B): does masked-patch self-supervised pretraining on the *synthetic* benchmark learn
structure that transfers to *real* sensor data, or did it just memorize the generator? This is the
honest test the model card flags. Synthetic has 8 channels and C-MAPSS has 14, so the channel
embedding cannot transfer — we transfer the channel-agnostic temporal patch encoder
(``patch_proj`` + transformer ``encoder`` + ``mask_token``) via
``transfer_encoder(..., strict_channels=False)`` and re-learn channel identity on the target.

Four arms, scored by test macro-F1 across label fractions of the *training engines*:
  scratch | synth-pretrained (transfer) | real-pretrained (transfer, control) | xgboost-features.

If synth-pretrained ~ real-pretrained > scratch, the synthetic encoder transferred. If
synth-pretrained ~ scratch < real-pretrained, the synthetic structure did not transfer — also a
real, reportable finding. quick-mode numbers are wiring-grade; run `--mode full` for real ones.

Example:
    python -m scripts.sim2real_transfer --data data/cmapss_FD001 --mode quick_demo \
        --epochs-pretrain 5 --epochs-finetune 5
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# macOS-only OpenMP guard; must precede torch/xgboost import. See CLAUDE.md.
configure_omp()

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from sensortwin.data.cmapss import load_cmapss, load_cmapss_subsets
from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.plots import plot_label_efficiency
from sensortwin.features import build_feature_matrix
from sensortwin.models.baselines import make_xgboost
from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.utils.config import load_mode_config
from sensortwin.utils.io import load_dataset
from sensortwin.utils.seeds import set_torch_seed

ARMS = ["scratch", "synth_pretrained", "real_pretrained", "xgboost"]
DEFAULT_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
SYNTH_CHANNELS = 8
# Per-arm learning-rate candidates, selected on validation macro-F1 (matched budgets).
ARM_LRS = (1e-3, 1e-4)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(
        f1_score(y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0)
    )


def _engine_subset(groups: np.ndarray, idx: np.ndarray, fraction: float, rng) -> np.ndarray:
    """Sample a ``fraction`` of the engines present in ``idx`` and return all their window indices."""
    if fraction >= 1.0:
        return idx
    engines = np.unique(groups[idx])
    n = max(1, round(fraction * len(engines)))
    chosen = set(rng.choice(engines, size=n, replace=False).tolist())
    return idx[np.isin(groups[idx], list(chosen))]


def _load_real(args) -> tuple[np.ndarray, np.ndarray, dict]:
    data_cfg = load_mode_config(args.config, mode=args.mode).get("data", {})
    if args.data:
        return load_dataset(args.data)
    if not args.raw_dir:
        raise SystemExit("provide --data <npz> (run scripts.fetch_cmapss) or --raw-dir <folder>")
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


def _pretrain_state(model, train_ds, val_ds, epochs, device=None) -> dict:
    from sensortwin.training.pretrain import pretrain_model

    pmodel, _, hist = pretrain_model(
        model, train_ds, val_ds, epochs=epochs, warmup_epochs=min(2, epochs), device=device
    )
    print(f"  pretrain val MSE: {hist.val_loss[0]:.4f} -> {hist.val_loss[-1]:.4f}")
    return copy.deepcopy(pmodel.state_dict())


def _run_fraction(
    fraction,
    seed,
    X,
    y,
    groups,
    tr,
    va,
    te,
    standardizer,
    arch,
    num_classes,
    synth_state,
    real_state,
    ft_kwargs,
    epochs_ft,
    device=None,
) -> dict[str, float]:
    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.loop import class_weights, predict_proba, train_model
    from sensortwin.training.pretrain import transfer_encoder

    rng = np.random.default_rng(seed)
    sub = _engine_subset(groups, tr, fraction, rng)
    sub_ds = SensorArrayDataset(standardizer.transform(X[sub]), y[sub]).as_torch()
    val_ds = SensorArrayDataset(standardizer.transform(X[va]), y[va]).as_torch()
    test_ds = SensorArrayDataset(standardizer.transform(X[te]), y[te]).as_torch()
    weight = class_weights(y[sub], num_classes)
    in_channels = X.shape[1]

    def _make() -> Any:
        return SensorPatchTST(in_channels=in_channels, num_classes=num_classes, **arch)

    def _score(model) -> float:
        return _macro_f1(y[te], predict_proba(model, test_ds, device=device).argmax(1), num_classes)

    def _train_best_lr(build_model) -> Any:
        """Each arm selects its LR from the same validation budget — a fixed low fine-tune LR
        would handicap scratch and could fake a transfer win (the label-efficiency confound,
        mirrored)."""
        best_model, best_val = None, -1.0
        for lr in ARM_LRS:
            m, hist = train_model(
                build_model(),
                sub_ds,
                val_ds,
                weight=weight,
                epochs=epochs_ft,
                device=device,
                **(ft_kwargs | {"lr": lr}),
            )
            if hist.best_val_macro_f1 > best_val:
                best_model, best_val = m, hist.best_val_macro_f1
        return best_model

    out: dict[str, float] = {}
    out["scratch"] = _score(_train_best_lr(_make))
    out["synth_pretrained"] = _score(
        _train_best_lr(lambda: transfer_encoder(synth_state, _make(), strict_channels=False))
    )
    out["real_pretrained"] = _score(
        _train_best_lr(lambda: transfer_encoder(real_state, _make(), strict_channels=False))
    )

    clf = make_xgboost()
    clf.fit(build_feature_matrix(X[sub])[0], y[sub])
    proba = clf.predict_proba(build_feature_matrix(X[te])[0])
    out["xgboost"] = _macro_f1(y[te], proba.argmax(1), num_classes)
    return out


def _write_report(curves, out_dir, mode, fractions, epochs_pre, epochs_ft) -> Path:
    lines = [
        "# Synthetic-to-real transfer on C-MAPSS (v0.6)",
        "",
        f"Mode `{mode}`, pretrain {epochs_pre} epochs / fine-tune {epochs_ft} epochs. Masked-patch "
        "pretraining of the temporal patch encoder; the channel embedding is re-initialized for the "
        "real channel count (synthetic C=8 -> C-MAPSS C=14). Each arm trained on a fraction of the "
        "training *engines* and scored (macro-F1) on the held-out test engines.",
        "",
        "| Engine fraction | " + " | ".join(ARMS) + " |",
        "| --- | " + " | ".join("---:" for _ in ARMS) + " |",
    ]
    for f in fractions:
        row = " | ".join(f"{curves[a][f]:.3f}" for a in ARMS)
        lines.append(f"| {f * 100:g}% | {row} |")
    lines += [
        "",
        "`synth_pretrained` transfers the encoder pretrained on the synthetic benchmark; "
        "`real_pretrained` transfers one pretrained on unlabeled C-MAPSS (the upper-bound control); "
        "`scratch` is random init. The transfer question: does `synth_pretrained` beat `scratch` and "
        "approach `real_pretrained`, most at the smallest fractions?",
        "",
        "## Claims",
        "",
        "- **Supported:** this measures whether a synthetic-pretrained *temporal* encoder helps on "
        "real C-MAPSS, against a real-pretrained upper bound and a scratch lower bound.",
        "- **Not claimed:** only the patch encoder transfers (the channel embedding is re-learned); a "
        "null or negative result is reported honestly, not tuned away. Synthetic accuracy is not real "
        "capability.",
        "",
        f"These are `{mode}` wiring figures; run `--mode full` with more epochs before interpreting. "
        "See `figures/sim2real_transfer.png`.",
    ]
    out = out_dir / "sim2real_transfer.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _run_one_seed(seed, X, y, meta, fractions, arch, args) -> dict[str, dict[float, float]]:
    """Pretrain (synthetic + real) and run all fractions for one seed."""
    from sensortwin.models.transformer import SensorPatchTST

    num_classes = len(meta["event_classes"])
    groups = np.asarray(meta["groups"])
    set_torch_seed(seed)
    sp = make_split("grouped", y, meta, seed=seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    standardizer = ChannelStandardizer().fit(X[tr])
    ft_kwargs = {"optimizer": "adamw", "weight_decay": 0.01, "label_smoothing": 0.1}

    # Pretrain on synthetic (C=8) with the SAME patch geometry so the patch encoder transfers.
    print(f"[seed {seed}] pretraining on synthetic (C=8)...")
    syn_X, syn_y, _ = generate_dataset(
        GenConfig(n_samples=args.synth_samples, T=args.synth_T, seed=seed, normalize=False)
    )
    syn_std = ChannelStandardizer().fit(syn_X)
    syn_n = len(syn_y)
    syn_tr, syn_va = np.arange(int(0.85 * syn_n)), np.arange(int(0.85 * syn_n), syn_n)
    syn_train_ds = SensorArrayDataset(syn_std.transform(syn_X[syn_tr]), syn_y[syn_tr]).as_torch()
    syn_val_ds = SensorArrayDataset(syn_std.transform(syn_X[syn_va]), syn_y[syn_va]).as_torch()
    synth_state = _pretrain_state(
        SensorPatchTST(in_channels=SYNTH_CHANNELS, num_classes=num_classes, **arch),
        syn_train_ds,
        syn_val_ds,
        args.epochs_pretrain,
        args.device,
    )

    # Pretrain on unlabeled real C-MAPSS train (the transfer upper bound).
    print(f"[seed {seed}] pretraining on real C-MAPSS train (C=14)...")
    real_train_ds = SensorArrayDataset(standardizer.transform(X[tr]), y[tr]).as_torch()
    real_val_ds = SensorArrayDataset(standardizer.transform(X[va]), y[va]).as_torch()
    real_state = _pretrain_state(
        SensorPatchTST(in_channels=X.shape[1], num_classes=num_classes, **arch),
        real_train_ds,
        real_val_ds,
        args.epochs_pretrain,
        args.device,
    )

    curves: dict[str, dict[float, float]] = {a: {} for a in ARMS}
    for f in fractions:
        print(f"--- [seed {seed}] engine fraction {f * 100:g}% ---")
        res = _run_fraction(
            f,
            seed,
            X,
            y,
            groups,
            tr,
            va,
            te,
            standardizer,
            arch,
            num_classes,
            synth_state,
            real_state,
            ft_kwargs,
            args.epochs_finetune,
            args.device,
        )
        for a in ARMS:
            curves[a][f] = res[a]
            print(f"  {a}: macro-F1={res[a]:.3f}")
    return curves


def _write_summary(per_seed, out_dir: Path, meta_run: dict, fractions) -> Path:
    import json

    from sensortwin.evaluation.statistics import mean_std

    seeds = sorted(per_seed)
    table = {
        a: {
            f"{f:g}": {
                "per_seed": [per_seed[s][a][f] for s in seeds],
                "mean": mean_std([per_seed[s][a][f] for s in seeds])[0],
                "std": mean_std([per_seed[s][a][f] for s in seeds])[1],
            }
            for f in fractions
        }
        for a in ARMS
    }
    (out_dir / "sim2real_summary.json").write_text(
        json.dumps({"meta": meta_run, "arms": table}, indent=2) + "\n"
    )

    def _f(c):
        return f"{c['mean']:.3f}" if np.isnan(c["std"]) else f"{c['mean']:.3f} ± {c['std']:.3f}"

    lines = [
        "# Synthetic-to-real transfer on C-MAPSS",
        "",
        f"Mode `{meta_run['mode']}`, subsets {meta_run['subsets']}, seeds {meta_run['seeds']}, "
        f"pretrain {meta_run['epochs_pretrain']} / fine-tune {meta_run['epochs_finetune']} epochs; "
        "test macro-F1, mean ± sample std over seeds. Every torch arm selects its LR from the "
        f"same validation budget {ARM_LRS}, so a transfer win cannot be an artifact of a fixed "
        "low fine-tune LR handicapping scratch. Only the channel-agnostic temporal patch encoder "
        "transfers (synthetic C=8 → real C=14).",
        "",
        "| Engine fraction | " + " | ".join(ARMS) + " |",
        "| --- | " + " | ".join("---:" for _ in ARMS) + " |",
    ]
    for f in fractions:
        row = " | ".join(_f(table[a][f"{f:g}"]) for a in ARMS)
        lines.append(f"| {f * 100:g}% | {row} |")
    lines += [
        "",
        "Transfer verdict: `synth_pretrained` ≈ `real_pretrained` > `scratch` means the "
        "synthetic encoder transferred; `synth_pretrained` ≈ `scratch` means it did not — "
        "either way the number above is the finding.",
        "",
        "Regenerate: `python -m scripts.sim2real_transfer --raw-dir <CMAPSSData> --mode "
        f"{meta_run['mode']} --epochs-pretrain {meta_run['epochs_pretrain']} "
        f"--epochs-finetune {meta_run['epochs_finetune']} --seeds "
        f"{' '.join(str(s) for s in meta_run['seeds'])}`.",
    ]
    out = out_dir / "sim2real_summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Synthetic-to-real encoder transfer on C-MAPSS.")
    p.add_argument("--data", default=None, help="processed C-MAPSS .npz from scripts.fetch_cmapss")
    p.add_argument("--raw-dir", default=None, help="raw C-MAPSS folder (build on the fly)")
    p.add_argument("--config", default="configs/data/cmapss.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--epochs-pretrain", type=int, default=20)
    p.add_argument("--epochs-finetune", type=int, default=15)
    p.add_argument("--fractions", default=None, help="comma-separated, e.g. 0.1,0.25,0.5,1.0")
    p.add_argument("--synth-samples", type=int, default=800)
    p.add_argument("--synth-T", type=int, default=96)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto-detect")
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = (
        [float(x) for x in args.fractions.split(",")] if args.fractions else DEFAULT_FRACTIONS
    )

    X, y, meta = _load_real(args)
    num_classes = len(meta["event_classes"])
    print(
        f"Loaded C-MAPSS: X={X.shape}, {num_classes} classes, "
        f"{len(np.unique(np.asarray(meta['groups'])))} engines"
    )
    arch = load_mode_config(args.config, mode=args.mode).get("model", {})

    per_seed = {s: _run_one_seed(s, X, y, meta, fractions, arch, args) for s in args.seeds}

    curves = per_seed[args.seeds[-1]]
    plot_label_efficiency(
        curves,
        out_dir / "figures" / "sim2real_transfer.png",
        title="Synthetic-to-real transfer (C-MAPSS)",
    )
    report = _write_report(
        curves, out_dir, args.mode, fractions, args.epochs_pretrain, args.epochs_finetune
    )

    import platform

    import torch

    meta_run = {
        "mode": args.mode,
        "subsets": str(load_mode_config(args.config, mode=args.mode).get("data", {}).get("subset")),
        "seeds": list(args.seeds),
        "epochs_pretrain": args.epochs_pretrain,
        "epochs_finetune": args.epochs_finetune,
        "raw_sha256": meta.get("raw_sha256", {}),
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    summary = _write_summary(per_seed, out_dir, meta_run, fractions)
    print(f"\nReports written to {report} and {summary}")


if __name__ == "__main__":
    main()
