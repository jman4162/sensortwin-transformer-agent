"""Minimal, reusable supervised training loop for the deep baselines (roadmap v0.2).

Deliberately a plain function, not a Trainer framework (spec §20: keep dependencies light). It is
enough to train ``SensorCNN``/``SensorLSTM`` reproducibly and will be reused for the transformer in
v0.3. Early stopping monitors **validation macro-F1** (the project's headline metric) and the best
state is restored before returning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

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


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss (handles class imbalance)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


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
) -> tuple[nn.Module, History]:
    """Train ``model`` with early stopping on val macro-F1; restore + return the best state."""
    dev = _resolve_device(device)
    model = model.to(dev)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(weight=weight.to(dev) if weight is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = History()
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)
        history.train_loss.append(running / len(train_loader.dataset))  # type: ignore[arg-type]

        val_loss, val_f1 = _evaluate(model, val_loader, criterion, dev)
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
    model: nn.Module, loader: DataLoader, criterion: nn.Module, dev: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(x)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return total_loss / len(loader.dataset), macro_f1  # type: ignore[arg-type]


@torch.no_grad()
def predict_proba(
    model: nn.Module, ds: Dataset, *, batch_size: int = 128, device: str | None = None
) -> np.ndarray:
    """Return softmax probabilities ``[N, num_classes]`` for every item in ``ds``."""
    dev = _resolve_device(device)
    model = model.to(dev).eval()
    loader = DataLoader(ds, batch_size=batch_size)
    out = []
    for x, _ in loader:
        probs = torch.softmax(model(x.to(dev)), dim=1)
        out.append(probs.cpu().numpy())
    return np.concatenate(out).astype(np.float32)
