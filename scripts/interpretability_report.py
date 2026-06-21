"""CLI: interpretability diagnostics for SensorPatchTST (roadmap v0.5, spec §12.5).

Trains the transformer, then on correctly-classified test samples computes attention maps and
integrated-gradients saliency, saves a few attention heatmaps, and reports a faithfulness check:
does the saliency land on the generator's KNOWN event region more than random? (localization_score
vs a random-saliency baseline). Attention is a diagnostic, not an explanation.

Example:
    python -m scripts.interpretability_report --mode quick_demo --epochs 5
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.interpretability import (
    attention_map,
    attention_to_time,
    integrated_gradients,
    localization_score,
    random_localization_baseline,
)
from sensortwin.evaluation.plots import plot_attention_map
from sensortwin.simulation import generate_dataset
from sensortwin.simulation.events import CHANNELS, EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config, load_yaml
from sensortwin.utils.seeds import set_torch_seed


def _write_report(att_scores, ig_scores, rand_scores, n_used, out_dir, mode, epochs) -> Path:
    def _m(xs):
        return float(np.mean(xs)) if xs else float("nan")

    lines = [
        "# Interpretability report (v0.5)",
        "",
        f"Mode `{mode}`, {epochs} epochs. On {n_used} correctly-classified test samples with a "
        "localized injected event, we measure the fraction of saliency mass landing inside the known "
        "event region (channels x time from the generator metadata), vs a random-saliency baseline.",
        "",
        "| Saliency | Mean localization | Random baseline |",
        "| --- | ---: | ---: |",
        f"| attention pooling | {_m(att_scores):.3f} | {_m(rand_scores):.3f} |",
        f"| integrated gradients | {_m(ig_scores):.3f} | {_m(rand_scores):.3f} |",
        "",
        "A localization above the random baseline means the saliency tracks the injected event better "
        "than chance. **Caveat (spec §12.5):** this is a diagnostic; attention weights are not a "
        "definitive explanation (Jain & Wallace 2019). Occlusion / integrated gradients are more "
        "faithful because they re-run or differentiate the model.",
        "",
        f"`{mode}` figures for wiring/discipline, not research-grade; run `colab_standard`. Example "
        "attention heatmaps in `figures/`.",
    ]
    out = out_dir / "interpretability_report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Interpretability diagnostics for SensorPatchTST.")
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--model-config", default="configs/models/sensorpatchtst.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=40)
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    set_torch_seed(args.seed)
    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    from sensortwin.models.transformer import SensorPatchTST
    from sensortwin.training.loop import class_weights, predict_proba, train_model

    cfg = load_synthetic_config(args.config, mode=args.mode)
    cfg.seed = args.seed
    cfg.normalize = False
    print(f"Generating {cfg.n_samples} samples...")
    X, y, meta = generate_dataset(cfg)
    splits = make_split("random", y, {}, seed=args.seed)
    standardizer = ChannelStandardizer().fit(X[splits["train"]])
    model_cfg = load_yaml(args.model_config).get("model", {})

    tr, va, te = splits["train"], splits["val"], splits["test"]
    train_ds = SensorArrayDataset(standardizer.transform(X[tr]), y[tr]).as_torch()
    val_ds = SensorArrayDataset(standardizer.transform(X[va]), y[va]).as_torch()
    print(f"Training transformer ({args.epochs} epochs)...")
    model: Any = SensorPatchTST(**model_cfg)
    model, _ = train_model(
        model, train_ds, val_ds, epochs=args.epochs, weight=class_weights(y[tr], len(EVENT_CLASSES))
    )

    # Correctly-classified test samples that have a localized event.
    Xte_std = standardizer.transform(X[te])
    preds = predict_proba(model, SensorArrayDataset(Xte_std, y[te]).as_torch()).argmax(1)
    T = X.shape[2]
    att_scores, ig_scores, rand_scores = [], [], []
    saved = 0
    n_used = 0
    for i, gi in enumerate(te):
        if preds[i] != y[gi] or n_used >= args.max_samples:
            continue
        ev = meta["events"][gi]
        rand = random_localization_baseline(ev, len(CHANNELS), T)
        if rand is None:  # no localized region (normal / compound / whole-window)
            continue
        xi = Xte_std[i]
        am = attention_map(model, xi[None])  # [1, C, N]
        if am is not None:
            att_t = attention_to_time(am[0], T, model.patch_len, model.stride)
            s = localization_score(att_t, ev)
            if s is not None:
                att_scores.append(s)
            if saved < 4:
                plot_attention_map(
                    am[0],
                    out_dir / "figures" / f"attn_{ev['name']}_{saved}.png",
                    channel_names=CHANNELS,
                    title=f"Attention — {ev['name']}",
                )
                saved += 1
        ig = integrated_gradients(model, xi, int(y[gi]), steps=16)
        s_ig = localization_score(ig, ev)
        if s_ig is not None:
            ig_scores.append(s_ig)
        rand_scores.append(rand)
        n_used += 1

    report = _write_report(
        att_scores, ig_scores, rand_scores, n_used, out_dir, args.mode, args.epochs
    )
    print(
        f"attention loc={np.mean(att_scores):.3f} IG loc={np.mean(ig_scores):.3f} "
        f"random={np.mean(rand_scores):.3f} over {n_used} samples"
    )
    print(f"Report written to {report}")


if __name__ == "__main__":
    main()
