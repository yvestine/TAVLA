"""Joint-state streaming inference using the same composition as training/evaluation."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from force_gap_forecast_v1.common import finite_array, sha256
from . import FORMAT, GAP_S, HISTORY, LAYOUT
from .model import JointMotionForceForecast


class JointMotionForceEncoder:
    def __init__(self, manifest, domain="sim", device="cpu", threads=1):
        if domain not in ("real", "sim"):
            raise ValueError("domain must be real or sim")
        path = Path(manifest).resolve()
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("artifact_format") != FORMAT or self.manifest.get("layout") != LAYOUT:
            raise ValueError("Not a joint-motion-force-v2 manifest")
        checkpoint = path.parent / self.manifest["checkpoint"]
        if sha256(checkpoint) != self.manifest["encoder_sha256"]:
            raise ValueError("Encoder SHA mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("artifact_format") != FORMAT or payload.get("layout") != LAYOUT:
            raise ValueError("Checkpoint format/layout mismatch")
        if str(device).startswith("cpu"):
            torch.set_num_threads(threads)
        self.stats = payload["stats"]
        self.model = JointMotionForceForecast(self.stats, use_force=payload.get("use_force", True)).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device, self.domain = device, domain
        self.mean = np.asarray(self.stats[domain]["mean"], np.float32)
        self.scale = np.asarray(self.stats[domain]["scale"], np.float32)
        self.reset()

    def reset(self):
        self.rows, self.latents = deque(maxlen=HISTORY), deque(maxlen=10)
        self.previous_parts = None
        self.previous_force = self.previous_time = None
        self.last_future = None

    @torch.inference_mode()
    def push(self, force_xyz_n, joint_pos_rad, joint_vel_rad_s, timestamp_s):
        force = finite_array(force_xyz_n, (3,), "force_xyz_n")
        q = finite_array(joint_pos_rad, (7,), "joint_pos_rad")
        dq = finite_array(joint_vel_rad_s, (7,), "joint_vel_rad_s")
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("Finite timestamp required")
        if self.previous_time is not None:
            if timestamp <= self.previous_time:
                raise ValueError("Timestamps must increase; reset between episodes")
            if timestamp - self.previous_time > GAP_S:
                self.reset()
        change = np.zeros(3, np.float32) if self.previous_force is None else force - self.previous_force
        dt = .1 if self.previous_time is None else timestamp - self.previous_time
        row = np.concatenate((force, change, q, dq, np.array([dt], np.float32)))
        self.rows.append((row - self.mean) / self.scale)
        x = np.stack(self.rows)
        if len(x) < HISTORY:
            x = np.concatenate((np.repeat(x[:1], HISTORY - len(x), axis=0), x))
        domain = int(self.domain == "sim")
        parts = self.model.encode(torch.from_numpy(x[None]).to(self.device), domain)
        reset = self.previous_parts is None
        previous = parts if reset else self.previous_parts
        pair = {key: torch.stack((previous[key], parts[key]), 1) for key in parts}
        flags = torch.tensor([[True, reset]], device=self.device)
        previous_force = force if reset else self.previous_force
        observed = torch.from_numpy(np.stack((previous_force, force))[None]).to(self.device)
        output = self.model.compose(pair, flags, observed, domain)
        z = output["z"][0, -1].cpu().numpy().astype(np.float32)
        self.last_future = output["future"][0, -1].cpu().numpy().astype(np.float32)
        if not np.isfinite(z).all() or not np.isfinite(self.last_future).all():
            raise FloatingPointError("Non-finite encoder output")
        self.latents.append(z.copy())
        self.previous_parts = parts
        self.previous_force, self.previous_time = force.copy(), timestamp
        return z

    def batch(self, force, q, dq, time):
        n = len(force)
        force, q, dq, time = [finite_array(value, shape, name, dtype) for value, shape, name, dtype in
                              [(force, (n, 3), "force", np.float32), (q, (n, 7), "q", np.float32),
                               (dq, (n, 7), "dq", np.float32), (time, (n,), "time", np.float64)]]
        if n == 0 or np.any(np.diff(time) <= 0):
            raise ValueError("Nonempty increasing timestamps required")
        self.reset()
        return np.stack([self.push(force[i], q[i], dq[i], time[i]) for i in range(n)])

    def effort_history(self):
        if not self.latents:
            raise RuntimeError("Push a sample first")
        rows = np.stack(self.latents)
        return np.concatenate((np.repeat(rows[:1], 10 - len(rows), axis=0), rows)).astype(np.float32)

    def predict_future(self):
        if self.last_future is None:
            raise RuntimeError("Push a sample first")
        return self.last_future.copy()
