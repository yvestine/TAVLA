#!/usr/bin/env python3
"""Frozen control-encoder inference for a single trajectory (deployment mirror).

Given the ``encoder.pt`` written by ``export_control_encoder.py`` and one
trajectory's raw signals, this reproduces exactly the encoder input used at
training time -- the 10-D causal force-trend feature (normalized with the saved
force center/scale and clip) optionally concatenated with the normalized causal
pose feature -- windows it by ``history_frames``, and returns the per-frame 32-D
latent.  The same code path serves offline dataset conversion and, sample by
sample, online deployment.

No TAVLA/pi0 weight is read or modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import train_insertion_quality_v3_transformer as v3
from train_insertion_quality_control_encoder import ControlEncoder
from openpi.force_features import build_real_force_trend_features


class ControlEncoderInfer:
    """Load a frozen control encoder and map force(+pose) to latents."""

    def __init__(self, encoder_path: Path, device: str = "cpu"):
        checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=False)
        self.config = checkpoint["config"]
        self.metadata = checkpoint["feature_metadata"]
        self.latent_dim = int(checkpoint["latent_dim"])
        self.history = int(self.config["representation"]["history_frames"])
        self.calibration_frames = int(self.config["representation"]["calibration_frames"])
        rep = self.config["representation"]
        self.axis = np.asarray(rep["insertion_axis"], dtype=np.float64)
        self.axis /= np.linalg.norm(self.axis)
        self.smoothing = int(rep["causal_smoothing_frames"])
        self.trend_lag = int(rep["trend_lag_frames"])
        self.epsilon = float(rep["epsilon"])
        self.mad = float(rep["active_noise_mad_multiplier"])
        self.floor = float(rep["active_noise_floor_newton"])
        self.use_pose = bool(self.metadata.get("pose_feature_names"))
        self.pose_center = np.asarray(self.metadata["pose_center"], dtype=np.float32) if self.metadata.get("pose_center") else None
        self.pose_scale = np.asarray(self.metadata["pose_scale"], dtype=np.float32) if self.metadata.get("pose_scale") else None
        self.force_center = np.asarray(self.metadata["force_center"], dtype=np.float32)
        self.force_scale = np.asarray(self.metadata["force_scale"], dtype=np.float32)
        self.force_clip = float(self.metadata.get("force_clip", 10.0))
        self.device = torch.device(device)
        self.model = ControlEncoder(int(checkpoint["input_dim"]), self.config, use_domain=True).to(self.device)
        # strict=False: a variant trained without the domain head omits it; that
        # never affects encode(), which uses only the shared trunk.
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.eval()

    def features(self, force_xyz: np.ndarray, pose: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        force_xyz = np.asarray(force_xyz, dtype=np.float64)[:, :3]
        free_end = self.calibration_frames - 1
        raw, _ = build_real_force_trend_features(
            force_xyz,
            free_end=free_end,
            insertion_axis=self.axis,
            smoothing_frames=self.smoothing,
            trend_lag_frames=self.trend_lag,
            epsilon=self.epsilon,
            active_noise_mad_multiplier=self.mad,
            active_noise_floor=self.floor,
        )
        force_part = np.clip((raw - self.force_center) / self.force_scale, -self.force_clip, self.force_clip)
        if not self.use_pose:
            return force_part.astype(np.float32)
        pose_part = v3.causal_pose_features(np.asarray(pose, dtype=np.float64), np.asarray(timestamps, dtype=np.float64), free_end, self.axis)
        pose_part = (pose_part - self.pose_center) / self.pose_scale
        return np.column_stack((force_part, pose_part)).astype(np.float32)

    def latents(self, force_xyz: np.ndarray, pose: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Per-frame 32-D latent, left-padded so output length == trajectory length.

        Frame ``t``'s latent uses the causal window ``[t-history+1, t]``; the first
        ``history-1`` frames repeat the earliest computable latent, matching the
        episode-start padding convention used elsewhere.
        """
        feats = self.features(force_xyz, pose, timestamps)
        n = len(feats)
        windows = np.stack([feats[max(0, end - self.history + 1) : end + 1] for end in range(self.history - 1, n)])
        # Left-pad short leading windows (only the very first) to full history.
        fixed = []
        for w in windows:
            if len(w) < self.history:
                w = np.vstack([np.repeat(w[:1], self.history - len(w), axis=0), w])
            fixed.append(w)
        windows = np.asarray(fixed, dtype=np.float32)
        out = []
        with torch.no_grad():
            for s in range(0, len(windows), 512):
                out.append(self.model.encode(torch.from_numpy(windows[s : s + 512]).to(self.device)).cpu().numpy())
        latent = np.concatenate(out) if out else np.zeros((0, self.latent_dim), np.float32)
        pad = np.repeat(latent[:1], self.history - 1, axis=0) if len(latent) else np.zeros((self.history - 1, self.latent_dim), np.float32)
        return np.vstack([pad, latent]).astype(np.float32)
