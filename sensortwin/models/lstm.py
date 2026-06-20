"""Bidirectional LSTM baseline (roadmap v0.2, spec §9.1).

The recurrent baseline included "for interpretability and historical comparison" (spec §9.1). A
BiLSTM reads the sequence in both directions and we mean-pool the hidden states over time so every
timestep contributes equally (better than last-hidden for fixed-length classification). On this
data CNNs usually train faster and match or beat it — that contrast is itself part of the lesson.

Input ``[B, C, T]`` (transposed internally to ``[B, T, C]`` for ``nn.LSTM``) → logits
``[B, num_classes]``.
"""

from __future__ import annotations

import torch
from torch import nn

from sensortwin.simulation.events import EVENT_CLASSES, N_CHANNELS


class SensorLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        num_classes: int = len(EVENT_CLASSES),
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes),  # 2x for bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        h, _ = self.lstm(x.transpose(1, 2))  # [B, T, 2*hidden]
        h = h.mean(dim=1)  # mean-pool over time
        return self.head(h)
