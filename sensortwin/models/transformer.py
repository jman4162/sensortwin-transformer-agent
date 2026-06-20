"""SensorPatchTST: a small patch-transformer classifier (roadmap v0.3, spec §9.2 / §10).

The project's main model. PatchTST-inspired hybrid (spec §9.2 "Recommended v1"): each channel is
patchified independently, a learned channel embedding marks channel identity, and a transformer
encoder runs over *all* channel-patch tokens so attention can mix channels — the inductive bias we
expect to help on cross-channel and compound events where the CNN/feature baselines struggle.

Design choices (see plan + research):
- Linear patch projection + sinusoidal time-patch positional encoding (handles any sequence length,
  no learned position params to overfit). Additive learned channel embedding ``[C, d_model]``.
- No RevIN / instance norm: for classification the per-sample statistics are discriminative, and the
  train-fit ``ChannelStandardizer`` already normalizes. Encoder uses standard pre-norm LayerNorm.
- Attention pooling by default (an interpretability hook: the per-token weights say which
  channel/patch the model relied on). Mean pooling is available for the ablation.

Input ``[B, C, T]`` -> logits ``[B, num_classes]``, matching the CNN/LSTM contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from sensortwin.simulation.events import EVENT_CLASSES, N_CHANNELS


def _sinusoidal_position(n_positions: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard fixed sinusoidal positional encoding ``[n_positions, d_model]``."""
    pos = torch.arange(n_positions, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(n_positions, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class AttentionPool(nn.Module):
    """Single learned-query attention pooling over tokens.

    Returns the pooled ``[B, d_model]`` vector and stores the per-token weights ``[B, n_tokens]`` on
    ``self.last_weights`` for a later interpretability pass (v0.5).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * d_model**-0.5)
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_tokens, d_model]
        scores = (x @ self.query) / math.sqrt(x.shape[-1])  # [B, n_tokens]
        weights = torch.softmax(scores, dim=1)
        self.last_weights = weights.detach()
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # [B, d_model]


class SensorPatchTST(nn.Module):
    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        num_classes: int = len(EVENT_CLASSES),
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        channel_embedding: bool = True,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError(f"pooling must be 'attention' or 'mean', got {pooling!r}")
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.pooling = pooling

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.channel_embedding = (
            nn.Parameter(torch.randn(in_channels, d_model) * d_model**-0.5)
            if channel_embedding
            else None
        )
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: more stable for a small model
        )
        # enable_nested_tensor=False: avoids a benign warning with norm_first and has no effect
        # here since we never pass a padding mask.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.pool = AttentionPool(d_model) if pooling == "attention" else None
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        # Learnable token that replaces masked patch embeddings during v0.4 pretraining; unused in
        # supervised forward, so it adds d_model params and nothing else.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` -> ``[B, C, N_patches, patch_len]`` via a strided sliding window."""
        # unfold over the time dimension; drops a tail shorter than patch_len.
        return x.unfold(dimension=2, size=self.patch_len, step=self.stride)

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, C, T]`` -> (tokens ``[B, C*N, d_model]``, raw patches ``[B, C*N, patch_len]``).

        Patch projection + positional + channel embeddings, before the encoder. Returns the raw
        patches alongside so masked-reconstruction pretraining has its targets in the same token
        order (channel-major, patch-minor).
        """
        B, C, _ = x.shape
        patches = self._patchify(x)  # [B, C, N, patch_len]
        n_patches = patches.shape[2]

        tokens = self.patch_proj(patches)  # [B, C, N, d_model]
        pos = _sinusoidal_position(n_patches, self.d_model, x.device)  # [N, d_model]
        tokens = tokens + pos.view(1, 1, n_patches, self.d_model)
        if self.channel_embedding is not None:
            tokens = tokens + self.channel_embedding.view(1, C, 1, self.d_model)

        tokens = tokens.reshape(B, C * n_patches, self.d_model)
        targets = patches.reshape(B, C * n_patches, self.patch_len)
        return tokens, targets

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, n_tokens, d_model]`` -> encoded ``[B, n_tokens, d_model]`` (dropout + encoder)."""
        return self.encoder(self.dropout(tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        tokens, _ = self.embed(x)
        encoded = self.encode(tokens)  # [B, C*N, d_model]
        pooled = self.pool(encoded) if self.pool is not None else encoded.mean(dim=1)
        return self.head(pooled)
