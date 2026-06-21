"""Split strategies (spec §7.4).

Beyond a plain random split, the benchmark's scientific value comes from *stress* splits that
probe generalization rather than memorization. v0.1 ships the random split and the leakage-safe
scaffolding for the rest; severity/domain splits consume the per-sample event metadata produced
by the generator.

A split returns index arrays into the dataset, never copies, so callers control materialization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SplitIndices = dict[str, np.ndarray]


def _check_no_leakage(splits: SplitIndices) -> None:
    """Assert train/val/test index sets are disjoint (a core reproducibility guard)."""
    seen: set[int] = set()
    for name, idx in splits.items():
        s = set(idx.tolist())
        overlap = seen & s
        if overlap:
            raise ValueError(f"Split '{name}' leaks {len(overlap)} indices into another split")
        seen |= s


def random_split(
    y: np.ndarray, seed: int, fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> SplitIndices:
    """Stratified-ish random split by shuffling indices (per-class balance approx. preserved)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    splits = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    _check_no_leakage(splits)
    return splits


def grouped_split(
    groups: np.ndarray, seed: int, fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> SplitIndices:
    """Split by group id so no group's members straddle two splits (e.g. windows from one engine).

    A plain random split would leak: overlapping windows from the same engine would appear in both
    train and test, inflating scores. Here the *unique groups* are partitioned, then expanded back
    to sample indices. Asserts both index- and group-level disjointness.
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    perm = rng.permutation(unique)
    n = len(unique)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    group_sets = {
        "train": set(perm[:n_train].tolist()),
        "val": set(perm[n_train : n_train + n_val].tolist()),
        "test": set(perm[n_train + n_val :].tolist()),
    }
    splits = {name: np.where(np.isin(groups, list(gset)))[0] for name, gset in group_sets.items()}
    _check_no_leakage(splits)
    seen: set[int] = set()
    for name, gset in group_sets.items():
        if seen & gset:
            raise ValueError(f"Split '{name}' shares groups with another split")
        seen |= gset
    return splits


def severity_split(meta_events: list[dict[str, Any]], threshold: float, seed: int) -> SplitIndices:
    """Train on mild events, test on severe ones (severity >= ``threshold``).

    Samples without a severity (e.g. ``normal``, ``compound_fault``) are split randomly so both
    sides keep a baseline class distribution.
    """
    rng = np.random.default_rng(seed)
    mild, severe, neutral = [], [], []
    for i, ev in enumerate(meta_events):
        sev = ev.get("severity")
        if sev is None:
            neutral.append(i)
        elif sev >= threshold:
            severe.append(i)
        else:
            mild.append(i)
    neutral_perm = rng.permutation(neutral)
    cut = int(0.85 * len(neutral_perm))
    splits = {
        "train": np.array(mild + list(neutral_perm[:cut]), dtype=int),
        "test": np.array(severe + list(neutral_perm[cut:]), dtype=int),
    }
    _check_no_leakage(splits)
    return splits


def make_split(kind: str, y: np.ndarray, meta: dict, seed: int, **kwargs) -> SplitIndices:
    """Dispatch to a named split strategy."""
    if kind == "random":
        return random_split(y, seed, **kwargs)
    if kind == "grouped":
        if "groups" not in meta:
            raise KeyError("grouped split requires meta['groups'] (one group id per sample)")
        return grouped_split(np.asarray(meta["groups"]), seed, **kwargs)
    if kind == "severity":
        return severity_split(meta["events"], kwargs.get("threshold", 1.0), seed)
    raise NotImplementedError(
        f"Split '{kind}' not implemented yet. Planned (spec §7.4): seed, domain, "
        "missing-channel, rare-event."
    )
