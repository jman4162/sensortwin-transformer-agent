"""Minimal, reusable supervised training loop for the deep baselines (roadmap v0.2).

Deliberately a plain function, not a Trainer framework (spec §20: keep dependencies light). It is
enough to train ``SensorCNN``/``SensorLSTM`` reproducibly and will be reused for the transformer in
v0.3. Early stopping monitors **validation macro-F1** (the project's headline metric) and the best
state is restored before returning.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sized
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_f1: float = 0.0


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_amp(dev: torch.device, amp: bool | None) -> bool:
    """AMP only on CUDA. ``None`` auto-enables it on CUDA; an explicit value still can't turn it on
    off-GPU (so CPU/macOS runs stay numerically identical to the pre-v0.7 path)."""
    if dev.type != "cuda":
        return False
    return True if amp is None else amp


def _loader_opts(
    dev: torch.device, num_workers: int | None, pin_memory: bool | None
) -> dict[str, Any]:
    """DataLoader speed knobs, auto-resolved by device. Off CUDA the defaults reproduce the original
    single-process loader exactly (``num_workers=0``), so tests and macOS are unaffected."""
    on_cuda = dev.type == "cuda"
    nw = (2 if on_cuda else 0) if num_workers is None else num_workers
    pm = on_cuda if pin_memory is None else pin_memory
    opts: dict[str, Any] = {"num_workers": nw, "pin_memory": pm}
    if nw > 0:
        opts["persistent_workers"] = True
    return opts


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss (handles class imbalance)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def _build_optimizer(
    model: nn.Module, optimizer: str, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be 'adam' or 'adamw', got {optimizer!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, scheduler: str | None, warmup_epochs: int, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Per-epoch cosine schedule with linear warmup (hand-rolled, no extra dependency)."""
    if scheduler is None:
        return None
    if scheduler != "cosine_warmup":
        raise ValueError(f"scheduler must be 'cosine_warmup' or None, got {scheduler!r}")

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 8,
    weight: torch.Tensor | None = None,
    device: str | None = None,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    scheduler: str | None = None,
    warmup_epochs: int = 0,
    augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
    amp: bool | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[nn.Module, History]:
    """Train ``model`` with early stopping on val macro-F1; restore + return the best state.

    Defaults reproduce the original loop (plain Adam, no smoothing/schedule/augmentation), so
    baseline callers are unaffected. The opt-in args (AdamW + weight decay, label smoothing,
    cosine-warmup schedule, train-time ``augment`` hook) regularize the more overfit-prone
    transformer (v0.3).

    ``amp``/``num_workers``/``pin_memory`` (all ``None`` = auto-by-device, v0.7): on CUDA they turn
    on mixed precision and a pinned multi-worker loader for ~1.5-2x throughput; off CUDA they fall
    back to the original FP32 single-process path, so CPU/macOS runs stay bit-identical. AMP stays
    seed-deterministic on a fixed GPU but is not FP32-bit-identical; pass ``amp=False`` for FP32.
    """
    dev = _resolve_device(device)
    amp_on = _resolve_amp(dev, amp)
    model = model.to(dev)
    lopts = _loader_opts(dev, num_workers, pin_memory)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **lopts)
    val_loader = DataLoader(val_ds, batch_size=batch_size, **lopts)
    criterion = nn.CrossEntropyLoss(
        weight=weight.to(dev) if weight is not None else None, label_smoothing=label_smoothing
    )
    opt = _build_optimizer(model, optimizer, lr, weight_decay)
    sched = _build_scheduler(opt, scheduler, warmup_epochs, epochs)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp_on)

    history = History()
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            if augment is not None:
                x = augment(x)
            opt.zero_grad()
            with torch.amp.autocast(device_type=dev.type, enabled=amp_on):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(x)
        if sched is not None:
            sched.step()
        history.train_loss.append(running / len(cast(Sized, train_loader.dataset)))

        val_loss, val_f1 = _evaluate(model, val_loader, criterion, dev, amp_on)
        history.val_loss.append(val_loss)
        history.val_macro_f1.append(val_f1)

        if val_f1 > history.best_val_macro_f1:
            history.best_val_macro_f1 = val_f1
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    dev: torch.device,
    amp_on: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast(device_type=dev.type, enabled=amp_on):
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * len(x)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return total_loss / len(cast(Sized, loader.dataset)), macro_f1


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    ds: Dataset,
    *,
    batch_size: int = 128,
    device: str | None = None,
    amp: bool | None = None,
) -> np.ndarray:
    """Return softmax probabilities ``[N, num_classes]`` for every item in ``ds``."""
    dev = _resolve_device(device)
    amp_on = _resolve_amp(dev, amp)
    model = model.to(dev).eval()
    loader = DataLoader(ds, batch_size=batch_size, **_loader_opts(dev, None, None))
    out = []
    for x, _ in loader:
        with torch.amp.autocast(device_type=dev.type, enabled=amp_on):
            logits = model(x.to(dev))
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    ds: Dataset,
    *,
    batch_size: int = 128,
    device: str | None = None,
    amp: bool | None = None,
) -> np.ndarray:
    """Return raw logits ``[N, num_classes]`` (pre-softmax) — needed for temperature scaling."""
    dev = _resolve_device(device)
    amp_on = _resolve_amp(dev, amp)
    model = model.to(dev).eval()
    loader = DataLoader(ds, batch_size=batch_size, **_loader_opts(dev, None, None))
    out = []
    for x, _ in loader:
        with torch.amp.autocast(device_type=dev.type, enabled=amp_on):
            logits = model(x.to(dev))
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)
