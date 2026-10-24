"""Simulation/hardware adapter with encoder SHA handshake and step tracing."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import LAYOUT
from .common import finite_array
from .runtime import ForceGapEncoder


class TavlaAdapter:
    def __init__(self, manifest, domain, device="cpu"):
        self.force_encoder = ForceGapEncoder(manifest, domain, device)
        self.q = self.dq = self.time = self.force = None

    def reset(self):
        self.force_encoder.reset()
        self.q = self.dq = self.time = self.force = None

    def check_server(self, metadata):
        if (metadata.get("encoder_sha256") != self.force_encoder.manifest["encoder_sha256"]
                or metadata.get("layout") != LAYOUT or metadata.get("effort_shape") != [10, 32]
                or metadata.get("action_shape") != [50, 8] or metadata.get("actions_are_absolute") is not True):
            raise ValueError("Server/adapter SHA, layout or action contract mismatch")

    def update(self, force_xyz_n, actual_q_rad, actual_dq_rad_s, timestamp_s):
        z = self.force_encoder.push(force_xyz_n, actual_q_rad, actual_dq_rad_s, timestamp_s)
        self.force = finite_array(force_xyz_n, (3,), "force").copy()
        self.q = finite_array(actual_q_rad, (7,), "q").copy()
        self.dq = finite_array(actual_dq_rad_s, (7,), "dq").copy()
        self.time = float(timestamp_s)
        return z

    def observation(self, front_rgb, wrist_rgb, gripper_pos, prompt="peg-in-hole"):
        if self.q is None:
            raise RuntimeError("Update synchronized force/q/dq/time first")
        images = {}
        for key, value in (("cam_high", front_rgb), ("cam_left_wrist", wrist_rgb), ("cam_right_wrist", wrist_rgb)):
            value = np.asarray(value)
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) < 4:
                raise ValueError("Cameras must be RGB uint8 HxWx3")
            images[key] = np.ascontiguousarray(value)
        grip = float(gripper_pos)
        if not np.isfinite(grip) or not 0 <= grip <= 1 or not isinstance(prompt, str) or not prompt:
            raise ValueError("Normalized gripper 0..1 and nonempty prompt required")
        return {"images": images, "state": np.r_[self.q, np.float32(grip)].astype(np.float32),
                "effort": self.force_encoder.effort_history(), "prompt": prompt,
                "encoder_sha256": self.force_encoder.manifest["encoder_sha256"]}

    def trace(self, executed_action):
        return {"timestamp_s": self.time, "input_force_n": self.force.copy(), "q_rad": self.q.copy(),
                "dq_rad_s": self.dq.copy(), "z": self.force_encoder.effort_history()[-1].copy(),
                "future_real_force_n": self.force_encoder.predict_future(),
                "executed_action": finite_array(executed_action, (8,), "absolute executed action").copy(),
                "encoder_sha256": self.force_encoder.manifest["encoder_sha256"]}


class EpisodeTrace:
    def __init__(self, path):
        self.path, self.rows = Path(path), []
        if self.path.exists():
            raise FileExistsError("Trace exists; use a new experiment directory")

    def append(self, adapter, executed_action):
        self.rows.append(adapter.trace(executed_action))

    def close(self):
        if not self.rows:
            raise ValueError("No closed-loop trace samples")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **{key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]})
