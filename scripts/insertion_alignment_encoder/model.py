from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = 2 * self.dilation
        y = self.conv(F.pad(x, (pad, 0)))
        y = self.dropout(F.gelu(self.norm(y)))
        return F.gelu(x + y)


class ForceAlignmentEncoder(nn.Module):
    """Map a 32-frame, six-channel force history to a 32-D policy feature."""

    def __init__(self, input_channels: int = 6, latent_dim: int = 32, dropout: float = 0.10) -> None:
        super().__init__()
        self.stem = nn.Conv1d(input_channels, 48, kernel_size=1)
        self.temporal = nn.Sequential(
            CausalBlock(48, dilation=1, dropout=dropout),
            CausalBlock(48, dilation=2, dropout=dropout),
            CausalBlock(48, dilation=4, dropout=dropout),
        )
        self.gru = nn.GRU(input_size=48, hidden_size=64, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, latent_dim), nn.Tanh())
        self.quality_head = nn.Linear(latent_dim, 2)
        self.geometry_head = nn.Linear(latent_dim, 3)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # history: [batch, time, 6]; the right edge is the current frame.
        x = self.stem(history.transpose(1, 2))
        x = self.temporal(x).transpose(1, 2)
        seq, _ = self.gru(x)
        latent = self.projection(seq[:, -1])
        return latent, self.quality_head(latent), self.geometry_head(latent)
