"""Separated structure/strength encoder and frozen-representation motion probes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .motion_features import STRENGTH_INDICES, STRUCTURE_INDICES


class MotionCausalBlock(nn.Module):
    """Normalize channels independently at each time; never mix future frames."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.padding, 0)))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return F.gelu(x + self.dropout(F.gelu(y)))


class MotionForceEncoder(nn.Module):
    def __init__(self, history_length: int = 32):
        super().__init__()
        self.history_length = history_length
        self.structure_stem = nn.Conv1d(len(STRUCTURE_INDICES), 48, 1)
        self.structure_blocks = nn.Sequential(*(MotionCausalBlock(48, dilation) for dilation in (1, 2, 4)))
        self.structure_gru = nn.GRU(48, 64, batch_first=True)
        self.structure_projection = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 24), nn.Tanh())
        self.strength_stem = nn.Conv1d(len(STRENGTH_INDICES), 16, 1)
        self.strength_blocks = nn.Sequential(MotionCausalBlock(16, 1), MotionCausalBlock(16, 2))
        self.strength_gru = nn.GRU(16, 16, batch_first=True)
        self.strength_projection = nn.Sequential(nn.Linear(16, 8), nn.Tanh())
        self.quality_head = nn.Linear(32, 2)
        self.reconstruction_head = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, history_length * 15))

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.history_length, 17):
            raise ValueError(f"Expected [B,{self.history_length},17], got {tuple(history.shape)}")
        structure = history[:, :, STRUCTURE_INDICES].transpose(1, 2)
        strength = history[:, :, STRENGTH_INDICES].transpose(1, 2)
        structure, _ = self.structure_gru(self.structure_blocks(self.structure_stem(structure)).transpose(1, 2))
        strength, _ = self.strength_gru(self.strength_blocks(self.strength_stem(strength)).transpose(1, 2))
        latent = torch.cat((self.structure_projection(structure[:, -1]),
                            self.strength_projection(strength[:, -1])), dim=-1)
        return latent, self.quality_head(latent), self.reconstruction_head(latent).reshape(-1, self.history_length, 15)


class FutureMotionProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 14))

    def forward(self, x):
        return self.layers(x)


def conditional_alignment(latent, labels, domains, reliable, weights, sample_indices):
    """Weighted conditional CORAL on structure, using distinct reliable frames."""
    losses = []
    for phase in (0, 1):
        groups = []
        for domain in (0, 1):
            keep = (labels == phase) & (domains == domain) & reliable
            if len(torch.unique(sample_indices[keep])) < 16:
                break
            z, w = latent[keep, :24], weights[keep]
            w = w / w.sum().clamp_min(1e-8)
            mean = (w[:, None] * z).sum(dim=0)
            centered = z - mean
            covariance = centered.T @ (w[:, None] * centered)
            groups.append((mean, covariance))
        if len(groups) == 2:
            losses.append((groups[0][0] - groups[1][0]).square().mean() +
                          0.25 * (groups[0][1] - groups[1][1]).square().mean())
    return torch.stack(losses).mean() if losses else latent.sum() * 0.0
