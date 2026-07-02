"""CLI: shared hyperparameter grid for the deep models, selected on validation macro-F1.

Every deep model gets the same tuning budget: 3 learning rates x 2 capacities = 6 runs each,
trained on the train split and selected on the validation split only (the test split is never
touched here). This is the documented tuning protocol behind the committed
``configs/models/*.yaml`` recipes — without it, "the transformer wins" is indistinguishable
from "the transformer was tuned and the baselines were not."

Writes ``tuning_results.{json,md}`` with the full grid so the selection is auditable.

Examples
--------
    python -m scripts.tune_baselines --mode quick_demo --epochs 10
    python -m scripts.tune_baselines --mode colab_standard --models cnn,lstm --seed 0
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# Must precede any torch import (macOS OpenMP clash; see scripts/train_baseline.py).
configure_omp()

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.train_baseline import DEEP_CONFIGS, _deep_model_class
from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.simulation import generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config, load_yaml
from sensortwin.utils.seeds import set_torch_seed

GRID_LRS = (3e-4, 1e-3, 3e-3)
# One capacity step per model, chosen to close (part of) the parameter gap to the ~0.8M
# transformer: cnn 54k -> ~210k, lstm 39k -> ~530k, transformer 0.8M -> ~1.2M.
CAPACITY_VARIANTS: dict[str, dict[str, dict[str, Any]]] = {
    "cnn": {"base": {}, "large": {"widths": [64, 128, 256]}},
    "lstm": {"base": {}, "large": {"hidden_size": 128, "num_layers": 2}},
    "transformer": {"base": {}, "large": {"d_model": 192}},
}
RECIPE_KEYS = (
    "optimizer",
    "weight_decay",
    "label_smoothing",
    "scheduler",
    "warmup_epochs",
    "batch_size",
    "patience",
)


def _grid_for(name: str) -> list[dict[str, Any]]:
    return [{"lr": lr, "capacity": cap} for lr in GRID_LRS for cap in CAPACITY_VARIANTS[name]]


def run_grid(
    name: str,
    Xtr,
    ytr,
    Xva,
    yva,
    *,
    epochs: int,
    device: str | None,
    amp: bool | None,
) -> list[dict[str, Any]]:
    """Train the full grid for one model; return per-cell val macro-F1 records."""
    from sensortwin.training.augment import build_augment
    from sensortwin.training.loop import class_weights, train_model

    cfg = load_yaml(DEEP_CONFIGS[name])
    base_model_kwargs = dict(cfg.get("model", {}))
    tcfg = dict(cfg.get("train", {}))
    train_kwargs: dict[str, Any] = {k: tcfg[k] for k in RECIPE_KEYS if k in tcfg}
    augment = build_augment(tcfg.get("augment"))
    if augment is not None:
        train_kwargs["augment"] = augment

    train_ds = SensorArrayDataset(Xtr, ytr).as_torch()
    val_ds = SensorArrayDataset(Xva, yva).as_torch()
    weight = class_weights(ytr, len(EVENT_CLASSES))

    records = []
    for cell in _grid_for(name):
        model_kwargs = base_model_kwargs | CAPACITY_VARIANTS[name][cell["capacity"]]
        model = _deep_model_class(name)(**model_kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.perf_counter()
        _, history = train_model(
            model,
            train_ds,
            val_ds,
            epochs=epochs,
            lr=cell["lr"],
            weight=weight,
            device=device,
            amp=amp,
            **train_kwargs,
        )
        rec = {
            "model": name,
            "lr": cell["lr"],
            "capacity": cell["capacity"],
            "n_params": n_params,
            "val_macro_f1": round(history.best_val_macro_f1, 4),
            "best_epoch": history.best_epoch,
            "train_time_s": round(time.perf_counter() - t0, 1),
        }
        print(
            f"  {name} lr={cell['lr']:g} capacity={cell['capacity']}: "
            f"val macro-F1={rec['val_macro_f1']:.4f} ({rec['n_params']/1000:.0f}k params)"
        )
        records.append(rec)
    return records


def _write_report(records: list[dict[str, Any]], out_dir: Path, meta: dict[str, Any]) -> Path:
    (out_dir / "tuning_results.json").write_text(
        json.dumps({"meta": meta, "grid": records}, indent=2) + "\n"
    )
    lines = [
        "# Deep-model tuning grid (validation macro-F1)",
        "",
        f"Mode `{meta['mode']}`, seed {meta['seed']}, {meta['epochs']} epochs. Selection on the "
        "validation split only; the test split is never read here. Every model gets the same "
        f"budget: {len(GRID_LRS)} learning rates x 2 capacities.",
        "",
        "| Model | LR | Capacity | Params | Val macro-F1 | Best epoch |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in records:
        star = ""
        best = max(
            (x for x in records if x["model"] == r["model"]), key=lambda x: x["val_macro_f1"]
        )
        if r is best:
            star = " **(selected)**"
        lines.append(
            f"| {r['model']} | {r['lr']:g} | {r['capacity']}{star} | {r['n_params']/1000:.0f}k "
            f"| {r['val_macro_f1']:.4f} | {r['best_epoch']} |"
        )
    lines += [
        "",
        "Commit each model's selected cell to its `configs/models/*.yaml` so the committed "
        "recipes trace back to this grid.",
    ]
    out = out_dir / "tuning_results.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tune deep models on a shared lr/capacity grid.")
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--models", default="cnn,lstm,transformer")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)
    amp = False if args.no_amp else None

    set_torch_seed(args.seed)
    cfg = load_synthetic_config(args.config, mode=args.mode)
    cfg.seed = args.seed
    cfg.normalize = False
    print(f"Generating {cfg.n_samples} samples (T={cfg.T}, seed={cfg.seed})...")
    X, y, _ = generate_dataset(cfg)

    sp = make_split("random", y, {}, seed=args.seed)
    tr, va = sp["train"], sp["val"]
    standardizer = ChannelStandardizer().fit(X[tr])
    Xtr, Xva = standardizer.transform(X[tr]), standardizer.transform(X[va])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"--- {name} ---")
        records += run_grid(
            name, Xtr, y[tr], Xva, y[va], epochs=args.epochs, device=args.device, amp=amp
        )

    meta = {"mode": args.mode, "seed": args.seed, "epochs": args.epochs}
    report = _write_report(records, out_dir, meta)
    print(f"\nGrid written to {report}")


if __name__ == "__main__":
    main()
