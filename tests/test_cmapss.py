"""Tests for the NASA C-MAPSS adapter (v0.6).

Uses a tiny synthetic C-MAPSS-shaped fixture (a few fake engines, 26 columns) written to a temp
dir, so the suite runs in CI without the real download.
"""

from __future__ import annotations

import numpy as np

from sensortwin.data.cmapss import DEFAULT_SENSORS, load_cmapss, load_cmapss_subsets


def _write_fixture(path, engine_lengths) -> None:
    """Write train_FD001.txt with 26 columns: unit, cycle, 3 settings, 21 sensors."""
    rng = np.random.default_rng(0)
    rows = []
    for unit, length in enumerate(engine_lengths, start=1):
        for cycle in range(1, length + 1):
            settings = rng.normal(size=3)
            sensors = rng.normal(size=21) + cycle * 0.01  # mild drift so windows differ
            rows.append([unit, cycle, *settings, *sensors])
    np.savetxt(path / "train_FD001.txt", np.asarray(rows))


def _n_windows(length, window, stride) -> int:
    return max(0, (length - window) // stride + 1)


def test_load_cmapss_shapes_and_meta(tmp_path):
    lengths = [120, 90, 150]
    _write_fixture(tmp_path, lengths)
    window, stride = 20, 10
    X, y, meta = load_cmapss(tmp_path, "FD001", window=window, stride=stride)

    expected = sum(_n_windows(length, window, stride) for length in lengths)
    assert X.shape == (expected, len(DEFAULT_SENSORS), window)
    assert X.dtype == np.float32
    assert y.shape == (expected,)
    assert set(np.unique(y)).issubset({0, 1, 2})
    assert len(meta["groups"]) == expected
    assert set(meta["groups"]) == {1, 2, 3}
    assert meta["n_channels"] == len(DEFAULT_SENSORS)
    assert len(meta["channel_names"]) == len(DEFAULT_SENSORS)


def test_load_cmapss_health_classes_vary(tmp_path):
    # A long engine yields healthy early windows and critical late ones, so >1 class appears.
    _write_fixture(tmp_path, [150])
    _, y, _ = load_cmapss(tmp_path, "FD001", window=20, stride=10)
    assert len(np.unique(y)) >= 2
    assert y[-1] == 2  # last window of an engine is at RUL 0 -> critical


def test_load_cmapss_rul_monotone_within_engine(tmp_path):
    _write_fixture(tmp_path, [120])
    _, _, meta = load_cmapss(tmp_path, "FD001", window=20, stride=10)
    rul = np.asarray(meta["rul"])
    assert np.all(np.diff(rul) <= 0)  # later windows have less remaining life


def test_load_cmapss_missing_file_raises(tmp_path):
    try:
        load_cmapss(tmp_path, "FD001")
    except FileNotFoundError as e:
        assert "raw-dir" in str(e) or "not found" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError for a missing C-MAPSS file")


def test_load_cmapss_subsets_offsets_groups(tmp_path):
    _write_fixture(tmp_path, [100, 100])  # used as both "FD001" and (aliased) second subset
    # Reuse the same file by symlinking name FD003 to the same content.
    (tmp_path / "train_FD003.txt").write_bytes((tmp_path / "train_FD001.txt").read_bytes())
    X, y, meta = load_cmapss_subsets(tmp_path, ["FD001", "FD003"], window=20, stride=10)
    groups = meta["groups"]
    # Engine ids from the second subset are offset so they do not collide with the first.
    assert len(set(groups)) == 4
    assert X.shape[0] == len(y) == len(groups)


def test_cmapss_mode_overrides_reach_the_data_block():
    """Regression: the `full` mode's subset/stride live at the mode's top level, but consumers
    read the nested data block — load_cmapss_data_cfg must merge them. Before the fix every
    'full' run silently used FD001 at the default stride."""
    from sensortwin.utils.config import load_cmapss_data_cfg

    full = load_cmapss_data_cfg("configs/data/cmapss.yaml", mode="full")
    assert full["subset"] == "FD001,FD003"
    assert full["stride"] == 6
    quick = load_cmapss_data_cfg("configs/data/cmapss.yaml", mode="quick_demo")
    assert quick["subset"] == "FD001"
    assert quick["stride"] == 24
    assert quick["window"] == 48  # nested data keys survive the merge
