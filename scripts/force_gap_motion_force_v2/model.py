from __future__ import annotations

import copy

import torch
from torch import nn

from force_gap_forecast_v1.model import ForceForecast
from . import FUTURE, PAST


class JointAdapter(nn.Module):
    """A causal force/motion history, plus innovation against a motion prior."""

    def __init__(self):
        super().__init__()
        self.expected_force = nn.Sequential(nn.Linear(26, 32), nn.GELU(), nn.Linear(32, 3))
        self.gru = nn.GRU(24, 32, batch_first=True)
        self.context = nn.Sequential(nn.Linear(58, 64), nn.GELU())
        self.latent = nn.Linear(64, 26)
        self.history = nn.Linear(64, PAST * 3)
        self.gate = nn.Linear(64, 26)
        for head in (self.latent, self.history):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, x, h, use_force):
        expected = self.expected_force(h)  # Domain-normalized observed force, never a Real-equivalent force.
        if use_force:
            innovation = x[..., :3] - expected.detach()[:, None, :]
            inputs = torch.cat((x, innovation), -1)
        else:
            clean = x.clone()
            clean[..., :6] = 0
            inputs = torch.cat((clean, torch.zeros_like(clean[..., :3])), -1)
        sequence, _ = self.gru(inputs)
        context = self.context(torch.cat((h, sequence[:, -1]), -1))
        dh = self.latent(context) * torch.sigmoid(self.gate(context))
        dp = self.history(context).reshape(-1, PAST, 3)
        return dh, dp, expected


class JointMotionForceForecast(nn.Module):
    def __init__(self, stats, motion_state=None, use_force=True):
        super().__init__()
        self.stats, self.use_force = stats, bool(use_force)
        self.motion = ForceForecast(stats)
        if motion_state is not None:
            self.motion.load_state_dict(motion_state)
        self.motion.requires_grad_(False)
        self.motion.eval()
        self.adapters = nn.ModuleList([JointAdapter(), JointAdapter()])
        self.future_heads = nn.ModuleList([copy.deepcopy(self.motion.future_decoder) for _ in range(2)])
        self.register_buffer("domain_gains", torch.ones(2))

    @property
    def target_scale(self):
        return self.motion.target_scale

    def train(self, mode=True):
        super().train(mode)
        self.motion.eval()
        return self

    def set_decoder_trainable(self, trainable):
        # Only copies of the future decoder are unfrozen; the motion prior stays immutable.
        self.future_heads.requires_grad_(trainable)

    def encode(self, x, domain):
        clean = x.clone()
        clean[..., :6] = 0
        with torch.no_grad():
            h = (self.motion.real_encoder, self.motion.sim_encoder)[domain](clean)
            history = self.motion.history(h)
        dh, dp, expected = self.adapters[domain](x, h, self.use_force)
        return dict(motion_h=h, base_history=history, dh=dh, dp=dp, expected_force=expected)

    def compose(self, parts, reset, observed_force, domain, gain=None):
        """Parts have shape (batch, time, ...); deltas use consecutive causal endpoints."""
        gain = self.domain_gains[domain] if gain is None else gain
        base_h, base_history = parts["motion_h"], parts["base_history"]
        h = base_h + gain * parts["dh"]
        history = self.motion.history(h) + gain * parts["dp"] * self.target_scale
        base_current = observed_force if domain == 0 else base_history[..., -1, :]
        current = observed_force if domain == 0 else history[..., -1, :]

        def change(values):
            delta = torch.cat((torch.zeros_like(values[:, :1]), values[:, 1:] - values[:, :-1]), 1)
            return torch.where(reset[..., None], torch.zeros_like(delta), delta)

        base_z = torch.cat((base_current, change(base_current), base_h), -1)
        z = torch.cat((current, change(current), h), -1)
        base_future = self.motion.future(base_z)
        candidate = self.future_heads[domain](z).reshape(*z.shape[:-1], FUTURE, 3)
        candidate = candidate * self.target_scale + self.motion.target_mean
        # gain=0 returns exactly the immutable prior. At gain=1 the decoder consumes the joint state.
        future = base_future + gain * (candidate - base_future)
        regularizer = parts["dh"].square().mean() + parts["dp"].square().mean()
        return dict(h=h, z=z, past=history, current=current, future=future,
                    expected_force=parts["expected_force"], regularizer=regularizer)

    def block(self, batch, domain, gain=None):
        x = batch["x"][:, domain]
        b, k = x.shape[:2]
        parts = self.encode(x.reshape(b * k, *x.shape[2:]), domain)
        parts = {key: value.reshape(b, k, *value.shape[1:]) for key, value in parts.items()}
        reset = batch["real_reset" if domain == 0 else "sim_reset"]
        return self.compose(parts, reset, batch["current"], domain, gain)
