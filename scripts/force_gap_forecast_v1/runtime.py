"""One implementation for streaming and batch inference: no startup tare."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from . import FORMAT, GAP_S, HISTORY, LAYOUT
from .common import finite_array, read_manifest
from .model import ForceForecast


class ForceGapEncoder:
    def __init__(self, manifest, domain, device="cpu", threads=1):
        if domain not in ("real", "sim"):
            raise ValueError("domain must be real or sim")
        self.manifest, checkpoint = read_manifest(manifest)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["artifact_format"] != FORMAT or payload["layout"] != LAYOUT:
            raise ValueError("Checkpoint format/layout mismatch")
        self.stats = payload["stats"]
        if str(device).startswith("cpu"):
            torch.set_num_threads(threads)
        self.model = ForceForecast(self.stats).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device, self.domain = device, domain
        self.encoder = (self.model.real_encoder, self.model.sim_encoder)[domain == "sim"]
        self.mean = np.asarray(self.stats[domain]["mean"], np.float32)
        self.scale = np.asarray(self.stats[domain]["scale"], np.float32)
        self.reset()

    def reset(self):
        self.rows, self.latents = deque(maxlen=HISTORY), deque(maxlen=10)
        self.previous_force = self.previous_canonical = self.previous_time = None

    @torch.inference_mode()
    def push(self, force_xyz_n, joint_pos_rad, joint_vel_rad_s, timestamp_s):
        force = finite_array(force_xyz_n, (3,), "force_xyz_n")
        q = finite_array(joint_pos_rad, (7,), "joint_pos_rad")
        dq = finite_array(joint_vel_rad_s, (7,), "joint_vel_rad_s")
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self.previous_time is not None:
            if timestamp <= self.previous_time:
                raise ValueError("Timestamps must increase; reset on a new episode")
            if timestamp - self.previous_time > GAP_S:
                self.reset()
        delta = np.zeros(3, np.float32) if self.previous_force is None else force - self.previous_force
        dt = .1 if self.previous_time is None else timestamp - self.previous_time
        row = np.concatenate((force, delta, q, dq, np.array([dt], np.float32)))
        self.rows.append((row - self.mean) / self.scale)
        x = np.stack(self.rows)
        if len(x) < HISTORY:
            x = np.concatenate((np.repeat(x[:1], HISTORY - len(x), axis=0), x))
        h = self.encoder(torch.from_numpy(x[None]).to(self.device))
        canonical = force.copy() if self.domain == "real" else self.model.history(h)[0, -1].cpu().numpy()
        change = np.zeros(3, np.float32) if self.previous_canonical is None else canonical - self.previous_canonical
        z = np.concatenate((canonical, change, h[0].cpu().numpy())).astype(np.float32)
        if not np.isfinite(z).all():
            raise FloatingPointError("Encoder produced non-finite features")
        self.latents.append(z.copy())
        self.previous_force, self.previous_canonical, self.previous_time = force.copy(), canonical.copy(), timestamp
        return z

    def batch(self, force, q, dq, time):
        n = len(force)
        force = finite_array(force, (n, 3), "force")
        q = finite_array(q, (n, 7), "q")
        dq = finite_array(dq, (n, 7), "dq")
        time = finite_array(time, (n,), "time", np.float64)
        if n == 0 or np.any(np.diff(time) <= 0):
            raise ValueError("Nonempty increasing timestamps required")
        self.reset()
        return np.stack([self.push(force[i], q[i], dq[i], time[i]) for i in range(n)])

    def effort_history(self):
        if not self.latents:
            raise RuntimeError("Push a sample first")
        rows = np.stack(self.latents)
        return np.concatenate((np.repeat(rows[:1], 10 - len(rows), axis=0), rows)).astype(np.float32)

    @torch.inference_mode()
    def predict_future(self):
        if not self.latents:
            raise RuntimeError("Push a sample first")
        z = torch.from_numpy(self.latents[-1][None]).to(self.device)
        return self.model.future(z)[0].cpu().numpy().astype(np.float32)
