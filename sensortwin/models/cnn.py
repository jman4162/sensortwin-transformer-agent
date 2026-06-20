"""Compact 1D CNN baseline (roadmap v0.2, spec §9.1).

CNNs are the natural strong baseline for *local* time-series signatures (``current_spike``,
``voltage_sag``): stacked Conv1d blocks build a receptive field over short windows, and global
average pooling makes the classifier shift-invariant with far fewer parameters than a flattened
head. The expected weakness — long-range / cross-channel events — is exactly what motivates the
transformer in v0.3.

Architecture (spec §9.1 pattern): ``Conv1d→BatchNorm→GELU`` blocks → MaxPool → GAP → MLP head.
Input ``[B, C, T]`` → logits ``[B, num_classes]``.
"""

from __future__ import annotations

import torch
from torch import nn

from sensortwin.simulation.events import EVENT_CLASSES, N_CHANNELS


class SensorCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        num_classes: int = len(EVENT_CLASSES),
        widths: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev = in_channels
        for i, w in enumerate(widths):
            blocks += [
                nn.Conv1d(prev, w, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(w),
                nn.GELU(),
            ]
            if i < len(widths) - 1:
                blocks.append(nn.MaxPool1d(2))  # downsample between blocks
            prev = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling over time
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.features(x)
        h = self.pool(h).squeeze(-1)  # [B, width]
        return self.head(h)
