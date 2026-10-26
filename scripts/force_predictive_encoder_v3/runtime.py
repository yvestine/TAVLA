"""Streaming and batch force-only inference with a separate optional motion decoder."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import numpy as np
import torch

from force_gap_forecast_v1.common import finite_array, sha256
from . import FORMAT, HISTORY, GAP_S, LAYOUT
from .data import force_window, motion_window
from .model import PredictiveForce


class ForcePredictiveEncoder:
    def __init__(self, manifest, domain="sim", device="cpu", threads=1):
        if domain not in ("real", "sim"):
            raise ValueError("domain must be real or sim")
        path = Path(manifest).resolve()
        metadata = json.loads(path.read_text())
        if metadata.get("artifact_format") != FORMAT or metadata.get("layout") != LAYOUT:
            raise ValueError("Not a force-predictive-v3 manifest")
        if domain not in metadata.get("ready_domains", ("real", "sim")):
            raise ValueError(f"The {domain} branch is not trained; complete the Sim student stage first")
        checkpoint = path.parent / metadata["checkpoint"]
        if sha256(checkpoint) != metadata["encoder_sha256"]:
            raise ValueError("Encoder SHA mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("artifact_format") != FORMAT or payload.get("layout") != LAYOUT:
            raise ValueError("Checkpoint format/layout mismatch")
        if str(device).startswith("cpu"):
            torch.set_num_threads(threads)
        self.stats, self.domain, self.device = payload["stats"], domain, device
        self.model = PredictiveForce(self.stats, payload["use_fast"]).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.motion_ready = bool(payload.get("motion_ready", False))
        self.reset()

    def reset(self):
        self.force, self.time, self.latents = deque(maxlen=HISTORY), deque(maxlen=HISTORY), deque(maxlen=10)
        self.output = None

    @torch.inference_mode()
    def push(self, force_xyz_n, timestamp_s):
        force = finite_array(force_xyz_n, (3,), "force_xyz_n")
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp) or (self.time and timestamp <= self.time[-1]):
            raise ValueError("Finite, increasing timestamps required; reset between episodes")
        if self.time and timestamp - self.time[-1] > GAP_S:
            self.reset()
        self.force.append(force.copy())
        self.time.append(timestamp)
        x, _ = force_window(np.asarray(self.force), np.asarray(self.time), self.stats, self.domain)
        self.output = self.model(torch.from_numpy(x[None]).to(self.device), int(self.domain == "sim"))
        if not all(torch.isfinite(v).all() for v in self.output.values()):
            raise FloatingPointError("Non-finite encoder output")
        z = self.output["z"][0].cpu().numpy().copy()
        self.latents.append(z)
        return z.copy()

    def batch(self, force, time):
        force = finite_array(force, (len(time), 3), "force")
        time = finite_array(time, (len(time),), "time", np.float64)
        if not len(time) or np.any(np.diff(time) <= 0):
            raise ValueError("Nonempty increasing timestamps required")
        self.reset()
        return np.stack([self.push(f, t) for f, t in zip(force, time)])

    def effort_history(self):
        if not self.latents:
            raise RuntimeError("Push an observation first")
        return np.pad(np.stack(self.latents), ((10 - len(self.latents), 0), (0, 0)), mode="edge")

    @torch.inference_mode()
    def predict_future(self, q_history=None, dq_history=None):
        if self.output is None:
            raise RuntimeError("Push an observation first")
        if q_history is None and dq_history is None:
            return self.output["future"][0].cpu().numpy().copy()
        if q_history is None or dq_history is None or not self.motion_ready:
            raise ValueError("A trained motion head and both synchronized q/dq histories are required")
        x = motion_window(q_history, dq_history, np.asarray(self.time), self.stats)
        future = self.model.motion_head(self.output["z"], torch.from_numpy(x[None]).to(self.device))
        return (future * self.model.target_scale + self.model.target_mean)[0].cpu().numpy().copy()

    def reconstruct_history(self):
        if self.output is None:
            raise RuntimeError("Push an observation first")
        result = self.output["reconstruction"][0].cpu().numpy()
        return (result * np.asarray(self.stats[self.domain]["scale"][:3], np.float32) +
                np.asarray(self.stats[self.domain]["mean"][:3], np.float32))
