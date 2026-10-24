from __future__ import annotations

import torch
from torch import nn

from force_gap_forecast_v1.model import ForceForecast
from . import FUTURE, PAST


class ForceResidual(nn.Module):
    """Observed F/dF history, domain identity, and frozen motion context."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(8, 32, batch_first=True)
        self.residual = nn.Sequential(nn.Linear(58, 64), nn.GELU(), nn.Linear(64, (PAST + FUTURE) * 3))
        self.gate = nn.Linear(58, PAST + FUTURE)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, x, h, domain):
        identity = x.new_zeros((*x.shape[:2], 2))
        identity[..., domain] = 1
        features, _ = self.gru(torch.cat((x[..., :6], identity), -1))
        context = torch.cat((h, features[:, -1]), -1)
        raw = self.residual(context).reshape(-1, PAST + FUTURE, 3)
        gate = torch.sigmoid(self.gate(context))[..., None]
        correction = raw * gate
        return correction[:, :PAST], correction[:, PAST:]


class MotionForceForecast(nn.Module):
    def __init__(self, stats, motion_state=None):
        super().__init__()
        self.stats = stats
        self.motion = ForceForecast(stats)
        if motion_state is not None:
            self.motion.load_state_dict(motion_state)
        self.motion.requires_grad_(False)
        self.motion.eval()
        self.force_residual = ForceResidual()
        self.register_buffer("residual_gain", torch.tensor(1.))

    @property
    def target_scale(self):
        return self.motion.target_scale

    def normalize_force(self, force):
        return self.motion.normalize_force(force)

    def train(self, mode=True):
        super().train(mode)
        self.motion.eval()  # Frozen dropout must stay disabled during training.
        return self

    def encode(self, x, domain):
        motion_x = x.clone()
        motion_x[..., :6] = 0
        with torch.no_grad():
            encoder = (self.motion.real_encoder, self.motion.sim_encoder)[domain]
            h = encoder(motion_x)
            history = self.motion.history(h)
        dp, df = self.force_residual(x, h, domain)
        return h, history, dp * self.target_scale, df * self.target_scale

    def block(self, batch, domain):
        x = batch["x"][:, domain]
        b, k = x.shape[:2]
        h, base_history, dp, df = self.encode(x.reshape(b * k, *x.shape[2:]), domain)
        h = h.reshape(b, k, 26)
        base_history, dp = [v.reshape(b, k, PAST, 3) for v in (base_history, dp)]
        df = df.reshape(b, k, FUTURE, 3)
        base_current = base_history[..., -1, :]
        reset = batch["real_reset" if domain == 0 else "sim_reset"]

        def change(current):
            delta = torch.cat((torch.zeros_like(current[:, :1]), current[:, 1:] - current[:, :-1]), 1)
            return torch.where(reset[..., None], torch.zeros_like(delta), delta)

        with torch.no_grad():
            base_z = torch.cat((base_current, change(base_current), h), -1)
            base_future = self.motion.future(base_z)
        past = base_history + self.residual_gain * dp
        # Real canonical force stays the physical sensor reading, as in v1.
        current = batch["current"] if domain == 0 else past[..., -1, :]
        z = torch.cat((current, change(current), h), -1)
        correction_norm = (dp / self.target_scale).square().mean() + (df / self.target_scale).square().mean()
        return {"h": h, "z": z, "past": past, "future": base_future + self.residual_gain * df,
                "correction_norm": correction_norm}
