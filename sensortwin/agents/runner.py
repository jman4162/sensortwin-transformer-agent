"""Experiment runner (spec §13.4 persona 2).

Generates the dataset once, then trains ``SensorPatchTST`` for each one-variable spec across the
requested seeds, reusing the audited ``train_model`` / ``classification_metrics`` / ``save_metrics``
seams. Returns a :class:`RunResult` (per-seed macro-F1 + summary) per spec. A spec that raises is
captured as ``failed=True`` rather than dropped, so the reviewer can report it (guardrail §13.5.6).

This module imports torch and is exercised by the torch-gated end-to-end test, not the pure-python
schema/planner tests.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from sensortwin.agents.schemas import ExperimentSpec, RunResult
from sensortwin.agents.tools import merge_overrides
from sensortwin.data.dataset import SensorArrayDataset
from sensortwin.data.splits import make_split
from sensortwin.data.transforms import ChannelStandardizer
from sensortwin.evaluation.calibration import expected_calibration_error
from sensortwin.evaluation.metrics import classification_metrics, save_metrics
from sensortwin.evaluation.robustness import missing_channel_sweep
from sensortwin.evaluation.statistics import mean_std
from sensortwin.simulation import GenConfig, generate_dataset
from sensortwin.simulation.events import EVENT_CLASSES
from sensortwin.utils.seeds import set_torch_seed

_LOOP_KEYS = (
    "optimizer",
    "lr",
    "weight_decay",
    "label_smoothing",
    "scheduler",
    "warmup_epochs",
    "batch_size",
    "patience",
)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(y_true, y_pred, labels=range(len(EVENT_CLASSES)), average="macro", zero_division=0)
    )


class ExperimentRunner:
    """Holds the (fixed) dataset + split and trains one spec at a time."""

    def __init__(
        self,
        base_cfg: dict[str, Any],
        *,
        n_samples: int,
        T: int = 256,
        seed: int = 0,
        device: str | None = None,
        out_dir: str | None = None,
    ) -> None:
        self.base_cfg = base_cfg
        self.device = device
        self.out_dir = out_dir
        cfg = GenConfig(n_samples=n_samples, T=T, seed=seed, normalize=False)
        X, y, _ = generate_dataset(cfg)
        self.X, self.y = X, y
        self.splits = make_split("random", y, {}, seed=seed)
        self.standardizer = ChannelStandardizer().fit(X[self.splits["train"]])

    def _train_kwargs(self, train_cfg: dict[str, Any]) -> dict[str, Any]:
        from sensortwin.training.augment import build_augment

        kw: dict[str, Any] = {k: train_cfg[k] for k in _LOOP_KEYS if k in train_cfg}
        aug = build_augment(train_cfg.get("augment"))
        if aug is not None:
            kw["augment"] = aug
        return kw

    def run(self, label: str, spec: ExperimentSpec) -> tuple[RunResult, dict[str, Any]]:
        """Train ``spec`` across its seeds; return ``(RunResult, representative_full_metrics)``."""
        from sensortwin.models.transformer import SensorPatchTST
        from sensortwin.training.loop import class_weights, predict_proba, train_model

        merged = merge_overrides(self.base_cfg, spec.overrides)
        model_cfg = merged.get("model", {})
        train_kwargs = self._train_kwargs(merged.get("train", {}))

        tr, va, te = self.splits["train"], self.splits["val"], self.splits["test"]
        Xtr, Xva, Xte = (
            self.standardizer.transform(self.X[tr]),
            self.standardizer.transform(self.X[va]),
            self.standardizer.transform(self.X[te]),
        )
        val_ds = SensorArrayDataset(Xva, self.y[va]).as_torch()
        test_ds = SensorArrayDataset(Xte, self.y[te]).as_torch()
        weight = class_weights(self.y[tr], len(EVENT_CLASSES))

        per_seed: list[float] = []
        eces: list[float] = []
        full_metrics: dict[str, Any] = {}
        n_params: int | None = None
        worst_delta: float | None = None
        t0 = time.perf_counter()
        try:
            for s in spec.seeds:
                set_torch_seed(s)
                model: Any = SensorPatchTST(**model_cfg)
                n_params = sum(p.numel() for p in model.parameters())
                train_ds = SensorArrayDataset(Xtr, self.y[tr]).as_torch()
                model, _ = train_model(
                    model,
                    train_ds,
                    val_ds,
                    epochs=spec.epochs,
                    weight=weight,
                    device=self.device,
                    **train_kwargs,
                )
                proba = predict_proba(model, test_ds, device=self.device)
                per_seed.append(_macro_f1(self.y[te], proba.argmax(1)))
                eces.append(expected_calibration_error(self.y[te], proba))
                full_metrics = classification_metrics(
                    self.y[te], proba.argmax(1), proba, EVENT_CLASSES
                )

            # Robustness on the last seed's model (one pass; keeps the agent loop affordable).
            def predict_raw(Xc: np.ndarray) -> np.ndarray:
                ds = SensorArrayDataset(self.standardizer.transform(Xc), self.y[te]).as_torch()
                return predict_proba(model, ds, device=self.device).argmax(1)

            sweep = missing_channel_sweep(
                predict_raw, self.X[te], self.y[te], macro_f1_fn=_macro_f1
            )
            worst_delta = float(sweep["worst_delta"])
            full_metrics["missing_channel"] = sweep
        except Exception as e:  # noqa: BLE001 - capture, never hide a failed experiment
            return (
                RunResult(label=label, failed=True, error=f"{type(e).__name__}: {e}"),
                {},
            )

        mean, std = mean_std(per_seed)
        metrics_path = None
        if self.out_dir is not None:
            full_metrics["ece"] = float(np.mean(eces))
            full_metrics["n_params"] = n_params
            metrics_path = str(save_metrics(full_metrics, f"{self.out_dir}/metrics_agent_{label}"))

        result = RunResult(
            label=label,
            per_seed_macro_f1=per_seed,
            mean=mean,
            std=std,
            ece=float(np.mean(eces)) if eces else None,
            robustness_worst_delta=worst_delta,
            train_time_s=round(time.perf_counter() - t0, 2),
            n_params=n_params,
            metrics_path=metrics_path,
        )
        return result, full_metrics

    def baseline_spec(self, *, epochs: int, seeds: list[int]) -> ExperimentSpec:
        """A no-op single-knob spec (dropout at its current value) that trains the base config."""
        return ExperimentSpec(
            overrides={"dropout": self.base_cfg.get("model", {}).get("dropout", 0.1)},
            epochs=epochs,
            seeds=seeds,
        )
