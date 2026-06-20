"""Self-supervised masked-patch pretraining for SensorPatchTST (roadmap v0.4, spec §11.2).

Randomly mask a fraction of patches and reconstruct them (``loss = MSE`` on masked patches only),
so the encoder learns structure from *unlabeled* signals. We then transfer that encoder into a fresh
classifier and fine-tune on a small labeled subset — the label-efficiency experiment (spec §12.4).

Method (research-backed): BERT-style learnable ``[MASK]`` token (zero-masking is ambiguous because 0
is a valid standardized value); 40% of patches masked **per channel**; a single linear head
``d_model -> patch_len``; MSE in standardized space. No contrastive / momentum / heavy decoder.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sensortwin.models.transformer import SensorPatchTST
from sensortwin.training.loop import (
    History,
    _build_optimizer,
    _build_scheduler,
    _resolve_device,
)

# Parameter-name prefixes that constitute the transferable "encoder" (everything but the head/pool).
_ENCODER_PREFIXES = ("patch_proj", "channel_embedding", "encoder", "mask_token")


def _make_patch_mask(
    batch: int,
    channels: int,
    n_patches: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean mask ``[batch, channels*n_patches]`` with ~``mask_ratio`` patches masked per channel.

    Masking is independent per (sample, channel) so the model cannot trivially copy an unmasked
    channel. Generated on CPU (optionally seeded) for reproducibility; caller moves it to device.
    """
    n_mask = max(1, round(n_patches * mask_ratio))
    scores = torch.rand(batch, channels, n_patches, generator=generator)
    idx = scores.argsort(dim=2)[:, :, :n_mask]  # lowest-scoring patches per (sample, channel)
    mask = torch.zeros(batch, channels, n_patches, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    return mask.reshape(batch, channels * n_patches)


def _apply_mask(tokens: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """Replace masked token rows with the learnable mask token."""
    out = tokens.clone()
    out[mask] = mask_token.reshape(-1)
    return out


@torch.no_grad()
def _eval_recon(
    model: SensorPatchTST, head: nn.Module, loader: DataLoader, mask_ratio: float, dev: torch.device
) -> float:
    model.eval()
    head.eval()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(dev)
        tokens, targets = model.embed(x)
        b, n_tokens, _ = tokens.shape
        mask = _make_patch_mask(b, model.in_channels, n_tokens // model.in_channels, mask_ratio)
        mask = mask.to(dev)
        encoded = model.encode(_apply_mask(tokens, mask, model.mask_token))
        pred = head(encoded)
        total += F.mse_loss(pred[mask], targets[mask]).item() * b
        n += b
    return total / max(n, 1)


def pretrain_model(
    model: SensorPatchTST,
    train_ds: Dataset,
    val_ds: Dataset | None = None,
    *,
    mask_ratio: float = 0.4,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 5,
    weight_decay: float = 0.01,
    optimizer: str = "adamw",
    scheduler: str | None = "cosine_warmup",
    warmup_epochs: int = 5,
    augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: str | None = None,
) -> tuple[nn.Module, nn.Module, History]:
    """Masked-patch reconstruction pretraining. Returns ``(model, recon_head, history)`` with the
    best-by-val-MSE encoder restored. ``model`` must expose ``embed``/``encode``/``mask_token``
    (i.e. ``SensorPatchTST``). Labels in the dataset are ignored."""
    dev = _resolve_device(device)
    model = model.to(dev)
    recon_head = nn.Linear(model.d_model, model.patch_len).to(dev)

    bundle = nn.ModuleList([model, recon_head])
    opt = _build_optimizer(bundle, optimizer, lr, weight_decay)
    sched = _build_scheduler(opt, scheduler, warmup_epochs, epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds is not None else None

    history = History()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        recon_head.train()
        running, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(dev)
            if augment is not None:
                x = augment(x)
            tokens, targets = model.embed(x)
            b, n_tokens, _ = tokens.shape
            mask = _make_patch_mask(b, model.in_channels, n_tokens // model.in_channels, mask_ratio)
            mask = mask.to(dev)
            encoded = model.encode(_apply_mask(tokens, mask, model.mask_token))
            loss = F.mse_loss(recon_head(encoded)[mask], targets[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * b
            n += b
        if sched is not None:
            sched.step()
        train_mse = running / max(n, 1)
        history.train_loss.append(train_mse)

        if val_loader is not None:
            val_mse = _eval_recon(model, recon_head, val_loader, mask_ratio, dev)
        else:
            val_mse = train_mse
        history.val_loss.append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, recon_head, history


def freeze_encoder(model: nn.Module) -> nn.Module:
    """Freeze the transferable encoder (embed + encoder + mask token); leave pool/head trainable.

    Used for the linear-probe arm: the classifier head trains on a frozen pretrained representation.
    """
    for name, p in model.named_parameters():
        if name.startswith(_ENCODER_PREFIXES):
            p.requires_grad_(False)
    return model


def transfer_encoder(pretrained_state: dict, model: nn.Module) -> nn.Module:
    """Copy the pretrained encoder weights into ``model`` (a fresh classifier), leaving its head
    and pooling at their fresh initialization."""
    own = model.state_dict()
    transfer = {
        k: v for k, v in pretrained_state.items() if k in own and k.startswith(_ENCODER_PREFIXES)
    }
    own.update(transfer)
    model.load_state_dict(own)
    return model
