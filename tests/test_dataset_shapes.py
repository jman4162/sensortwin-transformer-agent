"""Tests for dataset wrapping, splits (leakage), features, and IO round-trips."""

from __future__ import annotations

import numpy as np

from sensortwin.data import make_split
from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.features import statistical_features
from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.events import N_CHANNELS
from sensortwin.utils.io import load_dataset, save_dataset


def _data():
    return generate_dataset(GenConfig(n_samples=120, T=64, seed=3))


def test_array_dataset_indexing():
    X, y, _ = _data()
    ds = SensorArrayDataset(X, y, indices=np.arange(10))
    assert len(ds) == 10
    x0, label0 = ds[0]
    assert x0.shape == (N_CHANNELS, 64)
    assert isinstance(label0, int)


def test_random_split_no_leakage_and_covers_all():
    X, y, meta = _data()
    splits = make_split("random", y, meta, seed=0)
    all_idx = np.concatenate([splits["train"], splits["val"], splits["test"]])
    assert len(np.unique(all_idx)) == len(y)  # disjoint + complete


def test_severity_split_disjoint():
    X, y, meta = _data()
    splits = make_split("severity", y, meta, seed=0, threshold=1.0)
    train, test = set(splits["train"].tolist()), set(splits["test"].tolist())
    assert train.isdisjoint(test)
    assert len(train) > 0 and len(test) > 0


def test_statistical_features_shape():
    X, y, _ = _data()
    feats = statistical_features(X)
    assert feats.shape[0] == len(y)
    assert feats.shape[1] == N_CHANNELS * 6
    assert np.isfinite(feats).all()


def test_io_roundtrip(tmp_path):
    X, y, meta = _data()
    path = save_dataset(tmp_path / "ds", X, y, meta)
    X2, y2, meta2 = load_dataset(path)
    assert np.array_equal(X, X2)
    assert np.array_equal(y, y2)
    assert meta2["event_classes"] == meta["event_classes"]
