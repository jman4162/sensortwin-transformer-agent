"""CLI: process a manually-downloaded NASA C-MAPSS dataset into the benchmark format (v0.6).

The raw C-MAPSS ``.txt`` files are a US-government work (NASA Prognostics Center of Excellence) and
are not redistributed here. Download them, then:

    python -m scripts.fetch_cmapss --raw-dir /path/to/CMAPSSData --mode quick_demo

This reads the configured subset(s), windows them into 3-stage health-classification samples, and
caches ``data/cmapss_<subset>.npz`` (+ a ``.meta.json`` sidecar) for the training scripts. Nothing
is downloaded automatically and nothing under ``data/`` is committed (gitignored).

Source: https://www.nasa.gov/intelligent-systems-division/  (PCoE prognostics data repository).
"""

from __future__ import annotations

import argparse
from collections import Counter

from sensortwin.data.cmapss import load_cmapss, load_cmapss_subsets
from sensortwin.utils.config import load_mode_config
from sensortwin.utils.io import save_dataset


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Process NASA C-MAPSS into the benchmark format.")
    p.add_argument("--raw-dir", required=True, help="folder with train_FD00x.txt files")
    p.add_argument("--config", default="configs/data/cmapss.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--out", default=None, help="output .npz path (default data/cmapss_<subset>)")
    args = p.parse_args(argv)

    cfg = load_mode_config(args.config, mode=args.mode).get("data", {})
    subsets = [s.strip() for s in str(cfg.get("subset", "FD001")).split(",") if s.strip()]
    kwargs = {
        "channels": cfg.get("channels"),
        "window": cfg.get("window", 48),
        "stride": cfg.get("stride", 12),
        "rul_bins": tuple(cfg.get("rul_bins", (30, 70))),
        "rul_cap": cfg.get("rul_cap", 125),
        "class_names": cfg.get("class_names"),
    }

    if len(subsets) == 1:
        X, y, meta = load_cmapss(args.raw_dir, subsets[0], **kwargs)
    else:
        X, y, meta = load_cmapss_subsets(args.raw_dir, subsets, **kwargs)

    out = args.out or f"data/cmapss_{'_'.join(subsets)}"
    path = save_dataset(out, X, y, meta)

    classes = meta["event_classes"]
    counts = Counter(int(v) for v in y)
    balance = ", ".join(f"{classes[c]}={counts.get(c, 0)}" for c in range(len(classes)))
    print(
        f"Wrote {path}  |  X={X.shape}  C={meta['n_channels']}  "
        f"engines={len(set(meta['groups']))}  windows={len(y)}\n  class balance: {balance}"
    )


if __name__ == "__main__":
    main()
