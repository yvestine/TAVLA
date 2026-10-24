"""Two force-only branches. Motion heads are trained after the encoders are frozen."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from . import PAST, FUTURE


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(64, 64, 3, dilation=dilation)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        return x + F.gelu(self.norm(y))


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(7, 64)
        self.blocks = nn.Sequential(*(CausalBlock(d) for d in (1, 2, 4, 8, 16)))
        self.out = nn.Linear(64, 16)

    def forward(self, x):
        return self.out(self.blocks(F.gelu(self.stem(x)))[:, -1])


class Encoder(nn.Module):
    def __init__(self, use_fast=True):
        super().__init__()
        self.use_fast = use_fast
        self.trend, self.fast = Branch(), Branch()

    def forward(self, x):
        trend = self.trend(x[:, 0])
        fast = self.fast(x[:, 1]) if self.use_fast else torch.zeros_like(trend)
        return torch.cat((trend, fast), -1)


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.GELU(), nn.Linear(64, outputs))


class Reconstruction(nn.Module):
    def __init__(self, use_fast=True):
        super().__init__()
        self.use_fast = use_fast
        self.slow, self.fast = mlp(16, PAST * 3), mlp(16, PAST * 3)

    def forward(self, z):
        slow = self.slow(z[..., :16]).reshape(*z.shape[:-1], PAST, 3)
        fast = self.fast(z[..., 16:]).reshape_as(slow) if self.use_fast else torch.zeros_like(slow)
        return torch.stack((slow, fast), -3)


class MotionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(15, 32, batch_first=True)
        self.head = mlp(64, FUTURE * 3)

    def forward(self, z, motion, use_force=True):
        _, h = self.gru(motion)
        return self.head(torch.cat((z if use_force else torch.zeros_like(z), h[-1]), -1)).reshape(-1, FUTURE, 3)


class PredictiveForce(nn.Module):
    def __init__(self, stats, use_fast=True):
        super().__init__()
        self.stats, self.use_fast = stats, use_fast
        self.encoders = nn.ModuleList([Encoder(use_fast), Encoder(use_fast)])
        self.reconstructors = nn.ModuleList([Reconstruction(use_fast), Reconstruction(use_fast)])
        self.forecaster = mlp(32, FUTURE * 3)
        self.motion_head = MotionHead()
        self.motion_control = copy.deepcopy(self.motion_head)
        self.register_buffer("target_mean", torch.tensor(stats["target"]["mean"], dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(stats["target"]["scale"], dtype=torch.float32))
        self.stage = None

    def initialize_student(self):
        self.encoders[1].load_state_dict(self.encoders[0].state_dict())
        self.reconstructors[1].load_state_dict(self.reconstructors[0].state_dict())

    def configure(self, stage):
        self.stage = stage
        self.requires_grad_(False)
        if stage in ("teacher", "student"):
            domain = int(stage == "student")
            self.encoders[domain].requires_grad_(True)
            self.reconstructors[domain].requires_grad_(True)
            if not domain:
                self.forecaster.requires_grad_(True)
            if not self.use_fast:
                self.encoders[domain].fast.requires_grad_(False)
                self.reconstructors[domain].fast.requires_grad_(False)
        elif stage in ("motion", "control"):
            (self.motion_head if stage == "motion" else self.motion_control).requires_grad_(True)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def normalize_target(self, value):
        return (value - self.target_mean) / self.target_scale

    def physical_future(self, z):
        return self.forecaster(z).reshape(*z.shape[:-1], FUTURE, 3) * self.target_scale + self.target_mean

    def forward(self, x, domain, motion=None):
        z = self.encoders[domain](x)
        parts = self.reconstructors[domain](z)
        result = dict(z=z, components=parts, reconstruction=parts.sum(-3), future=self.physical_future(z))
        if motion is not None:
            result["joint"] = self.motion_head(z.detach(), motion) * self.target_scale + self.target_mean
            result["control"] = self.motion_control(z.detach(), motion, use_force=False) * self.target_scale + self.target_mean
        return result

    def block(self, batch, domain, with_motion=False):
        x = batch["x"][:, domain]
        b, k = x.shape[:2]
        motion = batch["motion"][:, domain].reshape(b * k, *batch["motion"].shape[-2:]) if with_motion else None
        output = self(x.reshape(b * k, *x.shape[2:]), domain, motion)
        return {name: value.reshape(b, k, *value.shape[1:]) for name, value in output.items()}
