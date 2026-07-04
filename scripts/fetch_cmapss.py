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
import hashlib
from collections import Counter
from pathlib import Path

from sensortwin.data.cmapss import load_cmapss, load_cmapss_subsets
from sensortwin.utils.config import load_cmapss_data_cfg, load_mode_config
from sensortwin.utils.io import save_dataset


def _hash_raw_files(raw_dir: str, subsets: list[str], expected: dict[str, str]) -> dict[str, str]:
    """SHA-256 every input file for provenance; verify against config-declared hashes if any.

    The raw files are not redistributed, so this repo cannot ship authoritative hashes. Instead,
    the hash of every input is stamped into the processed dataset's ``.meta.json`` — any result
    built from it traces back to exact input bytes. After your first download, copy the printed
    hashes into ``raw_sha256:`` in ``configs/data/cmapss.yaml`` to lock later runs to the same
    files.
    """
    hashes: dict[str, str] = {}
    for subset in subsets:
        name = f"train_{subset}.txt"
        path = Path(raw_dir) / name
        if not path.exists():
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = got
        want = expected.get(name)
        if want and got != want:
            raise SystemExit(
                f"checksum mismatch for {path}:\n  expected {want}\n  got      {got}\n"
                "The file differs from the download this repo's config was locked to. "
                "Re-download it, or update raw_sha256 in configs/data/cmapss.yaml."
            )
        print(f"sha256 {name}: {got}" + ("  (verified)" if want else ""))
    return hashes


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Process NASA C-MAPSS into the benchmark format.")
    p.add_argument("--raw-dir", required=True, help="folder with train_FD00x.txt files")
    p.add_argument("--config", default="configs/data/cmapss.yaml")
    p.add_argument("--mode", default="quick_demo")
    p.add_argument("--out", default=None, help="output .npz path (default data/cmapss_<subset>)")
    args = p.parse_args(argv)

    full_cfg = load_mode_config(args.config, mode=args.mode)
    cfg = load_cmapss_data_cfg(args.config, mode=args.mode)
    subsets = [s.strip() for s in str(cfg.get("subset", "FD001")).split(",") if s.strip()]
    raw_hashes = _hash_raw_files(args.raw_dir, subsets, full_cfg.get("raw_sha256") or {})
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

    meta["raw_sha256"] = raw_hashes  # provenance: processed data traces to exact input bytes
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
