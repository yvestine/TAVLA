"""Independent domain encoders and one shared current-plus-future real-force head."""

from __future__ import annotations

from force_predictive_encoder_v3.model import Encoder
from force_predictive_encoder_v3.model import mlp
import torch
from torch import nn

from . import FUTURE


class PredictiveForce(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.stats = stats
        self.encoders = nn.ModuleList([Encoder(), Encoder()])
        self.forecaster = mlp(32, (FUTURE + 1) * 3)
        self.register_buffer("target_mean", torch.tensor(stats["target"]["mean"], dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(stats["target"]["scale"], dtype=torch.float32))
        self.register_buffer("increment_scale", torch.tensor(stats["target"]["increment_scale_n"], dtype=torch.float32))

    def initialize_student(self):
        self.encoders[1].load_state_dict(self.encoders[0].state_dict())

    def configure(self, stage):
        self.requires_grad_(requires_grad=False)
        if stage == "teacher":
            self.encoders[0].requires_grad_(requires_grad=True)
            self.forecaster.requires_grad_(requires_grad=True)
        elif stage == "student":
            self.encoders[1].requires_grad_(requires_grad=True)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def forward(self, x, domain):
        z = self.encoders[domain](x)
        force = self.forecaster(z).reshape(*z.shape[:-1], FUTURE + 1, 3)
        force = force * self.target_scale + self.target_mean
        return {"z": z, "force": force, "current": force[..., 0, :], "future": force[..., 1:, :]}

    def block(self, x, domain):
        x = x[:, domain]
        b, t = x.shape[:2]
        output = self(x.reshape(b * t, *x.shape[2:]), domain)
        return {name: value.reshape(b, t, *value.shape[1:]) for name, value in output.items()}
