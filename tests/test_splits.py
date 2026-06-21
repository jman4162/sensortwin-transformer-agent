"""Tests for split strategies, focused on the grouped (by-engine) split (v0.6)."""

from __future__ import annotations

import numpy as np
import pytest

from sensortwin.data.splits import grouped_split, make_split


def _groups(n_per_group=7, n_groups=20) -> np.ndarray:
    return np.repeat(np.arange(n_groups), n_per_group)


def test_grouped_split_no_index_or_group_leakage():
    groups = _groups()
    y = np.zeros(len(groups), dtype=np.int64)
    sp = make_split("grouped", y, {"groups": groups}, seed=0)

    # Every window assigned exactly once, splits cover everything.
    all_idx = np.concatenate([sp["train"], sp["val"], sp["test"]])
    assert sorted(all_idx.tolist()) == list(range(len(groups)))

    # No engine id appears in two splits.
    g_train, g_val, g_test = (set(groups[sp[k]].tolist()) for k in ("train", "val", "test"))
    assert g_train & g_val == set()
    assert g_train & g_test == set()
    assert g_val & g_test == set()


def test_grouped_split_is_deterministic():
    groups = _groups()
    a = grouped_split(groups, seed=3)
    b = grouped_split(groups, seed=3)
    for k in ("train", "val", "test"):
        assert np.array_equal(a[k], b[k])


def test_grouped_split_requires_groups_meta():
    y = np.zeros(10, dtype=np.int64)
    with pytest.raises(KeyError):
        make_split("grouped", y, {}, seed=0)
