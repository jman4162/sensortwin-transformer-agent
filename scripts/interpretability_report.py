"""CLI: interpretability diagnostics for SensorPatchTST (roadmap v0.5, spec §12.5).

Trains the transformer (committed parity recipe), then on correctly-classified test samples
computes attention maps and integrated-gradients saliency, saves a few attention heatmaps, and
reports a faithfulness check: does the saliency land on the generator's KNOWN event region more
than random? (localization_score vs a random-saliency baseline). Attention is a diagnostic, not
an explanation.

Multi-seed: pass ``--seeds 0 1 2``; per-seed checkpoints make a killed run resumable. Writes a
committable ``interpretability_summary.{json,md}`` (mean ± sample std over seeds) and, from the
first seed, the test confusion matrix to ``docs/figures/confusion_matrix.png`` (the README
figure — this run trains exactly the model the README describes).

Example:
    python -m scripts.interpretability_report --mode quick_demo --epochs 5 --seeds 0
    python -m scripts.interpretability_report --mode colab_standard --epochs 30 --seeds 0 1 2
"""

from __future__ import annotations

from sensortwin.utils.runtime import configure_omp

# macOS-only OpenMP guard; must precede torch import.
configure_omp()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train_baseline import _build_deep_model
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
from sensortwin.evaluation.plots import plot_attention_map, plot_confusion_matrix
from sensortwin.evaluation.statistics import mean_std
from sensortwin.simulation import generate_dataset
from sensortwin.simulation.events import CHANNELS, EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config
from sensortwin.utils.seeds import set_torch_seed


def _run_seed(seed: int, args, out_dir: Path) -> dict:
    """Train the transformer for one seed; return localization scores + test macro-F1/CM."""
    from sklearn.metrics import confusion_matrix, f1_score

    from sensortwin.training.loop import class_weights, predict_proba, train_model

    set_torch_seed(seed)
    cfg = load_synthetic_config(args.config, mode=args.mode)
    cfg.seed = seed
    cfg.normalize = False
    print(f"[seed {seed}] generating {cfg.n_samples} samples...")
    X, y, meta = generate_dataset(cfg)
    splits = make_split("random", y, {}, seed=seed)
    standardizer = ChannelStandardizer().fit(X[splits["train"]])

    tr, va, te = splits["train"], splits["val"], splits["test"]
    train_ds = SensorArrayDataset(standardizer.transform(X[tr]), y[tr]).as_torch()
    val_ds = SensorArrayDataset(standardizer.transform(X[va]), y[va]).as_torch()
    print(f"[seed {seed}] training transformer ({args.epochs} epochs)...")
    model, kw = _build_deep_model("transformer")
    model, _ = train_model(
        model,
        train_ds,
        val_ds,
        weight=class_weights(y[tr], len(EVENT_CLASSES)),
        device=args.device,
        **(kw | {"epochs": args.epochs}),
    )

    # Correctly-classified test samples that have a localized event.
    Xte_std = standardizer.transform(X[te])
    preds = predict_proba(
        model, SensorArrayDataset(Xte_std, y[te]).as_torch(), device=args.device
    ).argmax(1)
    macro_f1 = float(
        f1_score(y[te], preds, labels=range(len(EVENT_CLASSES)), average="macro", zero_division=0)
    )
    # Attribution helpers run on CPU tensors and take the concrete SensorPatchTST.
    tf: Any = model.cpu().eval()
    if seed == args.seeds[0]:
        # The README confusion matrix comes from this run's first seed (the exact model the
        # README describes: parity recipe, this mode).
        cm = confusion_matrix(y[te], preds, labels=range(len(EVENT_CLASSES)))
        plot_confusion_matrix(cm, EVENT_CLASSES, Path("docs/figures/confusion_matrix.png"))
        print(f"[seed {seed}] confusion matrix -> docs/figures/confusion_matrix.png")

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
        am = attention_map(tf, xi[None])  # [1, C, N]
        if am is not None:
            att_t = attention_to_time(am[0], T, tf.patch_len, tf.stride)
            s = localization_score(att_t, ev)
            if s is not None:
                att_scores.append(s)
            if saved < 4 and seed == args.seeds[0]:
                plot_attention_map(
                    am[0],
                    out_dir / "figures" / f"attn_{ev['name']}_{saved}.png",
                    channel_names=CHANNELS,
                    title=f"Attention — {ev['name']}",
                )
                saved += 1
        ig = integrated_gradients(tf, xi, int(y[gi]), steps=16)
        s_ig = localization_score(ig, ev)
        if s_ig is not None:
            ig_scores.append(s_ig)
        rand_scores.append(rand)
        n_used += 1

    rec = {
        "seed": seed,
        "macro_f1": round(macro_f1, 4),
        "n_samples": n_used,
        "attention_loc": round(float(np.mean(att_scores)), 4) if att_scores else None,
        "ig_loc": round(float(np.mean(ig_scores)), 4) if ig_scores else None,
        "random_loc": round(float(np.mean(rand_scores)), 4) if rand_scores else None,
    }
    print(f"[seed {seed}] {rec}")
    return rec


def _fmt_ms(vals: list[float]) -> str:
    m, sd = mean_std(vals)
    return f"{m:.3f}" if np.isnan(sd) else f"{m:.3f} ± {sd:.3f}"


def _write_summary(results: list[dict], out_dir: Path, meta: dict) -> Path:
    (out_dir / "interpretability_summary.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2) + "\n"
    )
    att = [r["attention_loc"] for r in results if r["attention_loc"] is not None]
    ig = [r["ig_loc"] for r in results if r["ig_loc"] is not None]
    rnd = [r["random_loc"] for r in results if r["random_loc"] is not None]
    n_total = sum(r["n_samples"] for r in results)
    lines = [
        "# Interpretability: does saliency track the injected event?",
        "",
        f"Mode `{meta['mode']}`, seeds {meta['seeds']}, {meta['epochs']} epochs; transformer "
        f"trained with its committed parity recipe (test macro-F1 "
        f"{_fmt_ms([r['macro_f1'] for r in results])}). On {n_total} correctly-classified test "
        "samples with a localized injected event (pooled across seeds), the score is the "
        "fraction of saliency mass inside the known event region (channels x time from generator "
        "metadata) vs a random-saliency baseline.",
        "",
        "| Saliency | Localization (mean ± std over seeds) | Random baseline |",
        "| --- | ---: | ---: |",
        f"| attention pooling | {_fmt_ms(att)} | {_fmt_ms(rnd)} |",
        f"| integrated gradients | {_fmt_ms(ig)} | {_fmt_ms(rnd)} |",
        "",
        "A localization above the random baseline means the saliency tracks the injected event "
        "better than chance. **Caveat (spec §12.5):** this is a diagnostic; attention weights "
        "are not a definitive explanation (Jain & Wallace 2019). Occlusion / integrated "
        "gradients are more faithful because they re-run or differentiate the model.",
        "",
        "Regenerate: `python -m scripts.interpretability_report --mode "
        f"{meta['mode']} --epochs {meta['epochs']} --seeds "
        f"{' '.join(str(s) for s in meta['seeds'])}`.",
    ]
    out = out_dir / "interpretability_summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Interpretability diagnostics for SensorPatchTST.")
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="reports/experiment_summaries")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    ckpt_dir = out_dir / "per_seed"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        ckpt = ckpt_dir / f"interpretability_seed_{seed}.json"
        if ckpt.exists():
            saved = json.loads(ckpt.read_text())
            if saved["meta"] == {"mode": args.mode, "epochs": args.epochs}:
                print(f"[seed {seed}] reusing checkpoint {ckpt}")
                results.append(saved["rec"])
                continue
        rec = _run_seed(seed, args, out_dir)
        results.append(rec)
        ckpt.write_text(
            json.dumps({"meta": {"mode": args.mode, "epochs": args.epochs}, "rec": rec}, indent=2)
        )

    import platform

    import torch

    meta = {
        "mode": args.mode,
        "epochs": args.epochs,
        "seeds": list(args.seeds),
        "max_samples": args.max_samples,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    report = _write_summary(results, out_dir, meta)
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
