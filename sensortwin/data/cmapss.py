"""NASA C-MAPSS turbofan adapter (roadmap v0.6, spec §16 open-data validation).

Brings a *real* multichannel sensor dataset through the same ``(X[N, C, T], y, meta)`` contract as
the synthetic generator, so the existing baselines, standardizer, metrics, and robustness probes run
unchanged. C-MAPSS is natively a remaining-useful-life (RUL) regression benchmark; we reframe it as
3-stage health classification (healthy / degrading / critical) by binning the RUL at each window's
last step. This is the synthetic-to-real check the model card flags as not-yet-done.

File format (per subset FD001-FD004): whitespace-delimited ``train_FD00x.txt`` with 26 columns:
unit number, time-in-cycles, 3 operational settings, then 21 sensor measurements. We keep the 14
informative sensors (the other 7 are constant within a subset and carry no signal). Windows from one
engine must never split across train/test, so callers must use ``make_split("grouped", ...)`` on
``meta["groups"]`` (the per-window engine id).

The raw ``.txt`` files are a US-government work (NASA Prognostics Center of Excellence) and are not
redistributed in this repo; download them and point ``--raw-dir`` at the folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Sensors constant within a subset (1,5,6,10,16,18,19) carry no signal and are dropped; the
# remaining 14 are the standard informative set used across the C-MAPSS literature.
DEFAULT_SENSORS: list[int] = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
HEALTH_CLASSES: list[str] = ["healthy", "degrading", "critical"]

# Column layout: 0=unit, 1=cycle, 2..4=settings, 5..25=sensors 1..21. Sensor k (1-based) = col 4+k.
_FIRST_SENSOR_COL = 5


def _bin_rul(rul: int, rul_bins: tuple[int, int]) -> int:
    """Map a remaining-useful-life value to a health class (0=healthy, 1=degrading, 2=critical)."""
    low, high = rul_bins
    if rul >= high:
        return 0
    if rul >= low:
        return 1
    return 2


def load_cmapss(
    raw_dir: str | Path,
    subset: str = "FD001",
    *,
    channels: list[int] | None = None,
    window: int = 48,
    stride: int = 12,
    rul_bins: tuple[int, int] = (30, 70),
    rul_cap: int = 125,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one C-MAPSS subset as windowed 3-stage health-classification samples.

    Returns ``(X[N, C, T], y[N], meta)`` with ``X`` raw (un-normalized) float32, ``y`` in
    ``0..len(class_names)-1``, and ``meta["groups"]`` the per-window engine id.
    """
    channels = channels or DEFAULT_SENSORS
    class_names = class_names or HEALTH_CLASSES
    path = Path(raw_dir) / f"train_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"C-MAPSS file not found: {path}. Download the dataset (NASA PCoE prognostics "
            "repository) and pass its folder via --raw-dir; raw .txt files are not committed."
        )

    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] < _FIRST_SENSOR_COL + 21:
        raise ValueError(f"unexpected C-MAPSS layout in {path}: {raw.shape}, expected >=26 cols")

    units = raw[:, 0].astype(int)
    cycles = raw[:, 1].astype(int)
    sensor_cols = [4 + k for k in channels]  # sensor k (1-based) -> column 4+k
    sensors = raw[:, sensor_cols].astype(np.float32)

    windows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    ruls: list[int] = []
    for u in np.unique(units):
        m = units == u
        order = np.argsort(cycles[m])  # ensure ascending cycle order
        s = sensors[m][order]  # [n_cycles, C]
        c = cycles[m][order]
        max_cycle = int(c.max())
        n = len(c)
        for start in range(0, n - window + 1, stride):
            end = start + window
            rul_last = min(int(max_cycle - c[end - 1]), rul_cap)
            windows.append(s[start:end].T)  # [C, window]
            labels.append(_bin_rul(rul_last, rul_bins))
            groups.append(int(u))
            ruls.append(rul_last)

    if not windows:
        raise ValueError(
            f"no windows produced from {path} (window={window} longer than every engine's history?)"
        )

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    meta: dict[str, Any] = {
        "event_classes": list(class_names),
        "n_channels": len(channels),
        "channel_names": [f"sensor_{k}" for k in channels],
        "groups": [int(g) for g in groups],
        "rul": [int(r) for r in ruls],
        "source": f"cmapss_{subset}",
        "window": window,
        "stride": stride,
        "rul_cap": rul_cap,
        "rul_bins": list(rul_bins),
    }
    return X, y, meta


def load_cmapss_subsets(
    raw_dir: str | Path,
    subsets: list[str],
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and concatenate several subsets, offsetting engine ids so ``groups`` stay unique.

    Subsets must share the same channel selection and window length (they do by default), so the
    stacked ``X`` has a consistent ``[N, C, T]`` shape.
    """
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    all_groups: list[int] = []
    all_rul: list[int] = []
    sources: list[str] = []
    offset = 0
    base_meta: dict[str, Any] = {}
    for sub in subsets:
        X, y, meta = load_cmapss(raw_dir, sub, **kwargs)
        Xs.append(X)
        ys.append(y)
        all_groups.extend(int(g) + offset for g in meta["groups"])
        all_rul.extend(meta["rul"])
        sources.append(meta["source"])
        offset = max(all_groups) + 1
        base_meta = meta
    meta_out = dict(base_meta)
    meta_out["groups"] = all_groups
    meta_out["rul"] = all_rul
    meta_out["source"] = "+".join(sources)
    return np.concatenate(Xs), np.concatenate(ys), meta_out
