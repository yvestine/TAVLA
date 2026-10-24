"""Streaming force-only v4 encoding and current/future REAL-force prediction."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

from force_gap_forecast_v1.common import sha256
from force_predictive_encoder_v3.data import force_window
import numpy as np
import torch

from . import FORMAT
from . import FUTURE
from . import GAP_S
from . import HISTORY
from . import LAYOUT
from .train import load_model


class ForcePredictiveEncoder:
    def __init__(self, manifest, domain="sim", device="cpu", threads=1):
        if domain not in ("real", "sim"):
            raise ValueError("Domain must be real or sim")
        path = Path(manifest).resolve()
        metadata = json.loads(path.read_text())
        if (
            metadata.get("artifact_format") != FORMAT
            or metadata.get("layout") != LAYOUT
            or metadata.get("target_domain") != "real"
            or metadata.get("horizons") != list(range(FUTURE + 1))
        ):
            raise ValueError("Expected a v4 real-force manifest")
        checkpoint = path.parent / metadata["checkpoint"]
        if sha256(checkpoint) != metadata["encoder_sha256"]:
            raise ValueError("Encoder SHA mismatch")
        if device.startswith("cpu"):
            torch.set_num_threads(threads)
        self.model, _ = load_model(checkpoint, device)
        self.model.eval().requires_grad_(requires_grad=False)
        self.domain, self.device = domain, device
        self.reset()

    def reset(self):
        self.force, self.time = deque(maxlen=HISTORY), deque(maxlen=HISTORY)
        self.latents = deque(maxlen=10)
        self.output = None

    @torch.inference_mode()
    def push(self, force_xyz_n, timestamp_s):
        force, timestamp = np.asarray(force_xyz_n, np.float32), float(timestamp_s)
        if force.shape != (3,) or not np.isfinite(force).all() or not np.isfinite(timestamp):
            raise ValueError("Expected finite force XYZ in N and a finite timestamp")
        if self.time and timestamp <= self.time[-1]:
            raise ValueError("Timestamps must increase; reset between episodes")
        if self.time and timestamp - self.time[-1] > GAP_S:
            self.reset()
        self.force.append(force.copy())
        self.time.append(timestamp)
        x = force_window(np.asarray(self.force), np.asarray(self.time), self.model.stats, self.domain)[0]
        self.output = self.model(torch.from_numpy(x[None]).to(self.device), int(self.domain == "sim"))
        if not all(torch.isfinite(v).all() for v in self.output.values()):
            raise FloatingPointError("Non-finite encoder output")
        z = self.output["z"][0].cpu().numpy().copy()
        self.latents.append(z)
        return z.copy()

    def _prediction(self, key):
        if self.output is None:
            raise RuntimeError("Push a force observation first")
        return self.output[key][0].cpu().numpy().copy()

    def predict_current(self):
        return self._prediction("current")

    def predict_future(self):
        return self._prediction("future")

    def effort_history(self):
        if not self.latents:
            raise RuntimeError("Push a force observation first")
        return np.pad(np.stack(self.latents), ((10 - len(self.latents), 0), (0, 0)), mode="edge")
