"""CLI: generate the SensorTwin-Synth dataset to a compressed ``.npz`` + metadata sidecar.

Examples
--------
    python -m scripts.generate_synthetic --mode quick_demo
    python -m scripts.generate_synthetic --config configs/synthetic/base.yaml \\
        --mode colab_standard --out data/synth_colab
"""

from __future__ import annotations

import argparse
from collections import Counter

from sensortwin.simulation import generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.config import load_synthetic_config
from sensortwin.utils.io import save_dataset


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate the SensorTwin-Synth dataset.")
    p.add_argument("--config", default="configs/synthetic/base.yaml")
    p.add_argument("--mode", default="quick_demo", help="run mode defined in the config's 'modes'")
    p.add_argument("--out", default=None, help="output path stem (default: data/synth_<mode>)")
    args = p.parse_args(argv)

    cfg = load_synthetic_config(args.config, mode=args.mode)
    out = args.out or f"data/synth_{args.mode}"

    print(f"Generating {cfg.n_samples} samples (T={cfg.T}, seed={cfg.seed})...")
    X, y, meta = generate_dataset(cfg)

    path = save_dataset(out, X, y, meta)
    counts = Counter(int(v) for v in y)
    print(f"Saved X={X.shape} y={y.shape} -> {path}")
    print("Class distribution:")
    for cls_idx, name in enumerate(EVENT_CLASSES):
        print(f"  {cls_idx:>2} {name:<26} {counts.get(cls_idx, 0)}")


if __name__ == "__main__":
    main()
