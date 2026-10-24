from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from . import FUTURE, PAST


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(64, 64, 3, dilation=dilation)
        self.norm = nn.LayerNorm(64)
        self.dropout = nn.Dropout(.05)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        return x + self.dropout(F.gelu(self.norm(y)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(21, 64)
        self.blocks = nn.Sequential(*(CausalBlock(d) for d in (1, 2, 4, 8)))
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.out = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 26), nn.LayerNorm(26))

    def forward(self, x):
        y = self.blocks(F.gelu(self.stem(x)))
        y, _ = self.gru(y)
        return self.out(y[:, -1])


class ForceForecast(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.real_encoder = Encoder()
        self.sim_encoder = Encoder()
        self.history_decoder = nn.Sequential(nn.Linear(26, 64), nn.GELU(), nn.Linear(64, PAST * 3))
        self.future_decoder = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, FUTURE * 3))
        self.register_buffer("target_mean", torch.tensor(stats["target"]["mean"], dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(stats["target"]["scale"], dtype=torch.float32))

    def normalize_force(self, force):
        return (force - self.target_mean) / self.target_scale

    def history(self, h):
        return self.history_decoder(h).reshape(*h.shape[:-1], PAST, 3) * self.target_scale + self.target_mean

    def future(self, z):
        return self.future_decoder(z).reshape(*z.shape[:-1], FUTURE, 3) * self.target_scale + self.target_mean

    def adapt(self):
        self.sim_encoder.load_state_dict(copy.deepcopy(self.real_encoder.state_dict()))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("sim_encoder."))

    def train(self, mode=True):
        super().train(mode)
        # Frozen modules stay deterministic; gradients through decoders reach student.
        if not next(self.real_encoder.parameters()).requires_grad:
            self.real_encoder.eval()
            self.history_decoder.eval()
            self.future_decoder.eval()
        return self

    def block(self, batch, domain, motion_only=False):
        x = batch["x"][:, domain]
        b, k = x.shape[:2]
        encoder = (self.real_encoder, self.sim_encoder)[domain]
        h = encoder(x.reshape(b * k, *x.shape[2:])).reshape(b, k, 26)
        past = self.history(h)
        if domain == 0 and not motion_only:
            current, delta = batch["current"], batch["delta"]
        else:
            current = past[..., -1, :]
            delta = torch.cat((torch.zeros_like(current[:, :1]), current[:, 1:] - current[:, :-1]), 1)
            reset = batch["real_reset" if domain == 0 else "sim_reset"]
            delta = torch.where(reset[..., None], torch.zeros_like(delta), delta)
        z = torch.cat((current, delta, h), -1)
        return {"h": h, "z": z, "past": past, "future": self.future(z)}
