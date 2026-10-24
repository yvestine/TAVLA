from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x, (2 * self.dilation, 0)))
        return F.gelu(x + self.dropout(F.gelu(self.norm(y))))


class ForceGapEncoderV2(nn.Module):
    def __init__(self, history_length: int = 32, input_dim: int = 12, latent_dim: int = 32):
        super().__init__()
        self.history_length = history_length
        self.input_dim = input_dim
        self.stem = nn.Conv1d(input_dim, 48, kernel_size=1)
        self.temporal = nn.Sequential(*(CausalBlock(48, d) for d in (1, 2, 4)))
        self.gru = nn.GRU(48, 64, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, latent_dim), nn.Tanh())
        self.quality_head = nn.Linear(latent_dim, 2)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[1:] != (self.history_length, self.input_dim):
            raise ValueError(
                f"Expected [batch,{self.history_length},{self.input_dim}], got {tuple(history.shape)}"
            )
        x = self.temporal(self.stem(history.transpose(1, 2))).transpose(1, 2)
        sequence, _ = self.gru(x)
        latent = self.projection(sequence[:, -1])
        return latent, self.quality_head(latent)
