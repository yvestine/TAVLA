"""Offline and streaming inference for the frozen force-gap v2 encoder."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .features import force_gap_features, normalize
from .model import ForceGapEncoderV2


class ForceGapLatentEncoder:
    """Loads the exported artifact and maps base-frame force [N,3] to [N,32].

    The first 20 force samples are the stationary calibration window. Deployments
    should collect them before asking the policy for an action. Inputs are in N,
    expressed in the robot base frame, at the same 10 Hz rate as training.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("artifact_format") != "tavla_force_gap_encoder_v2":
            raise ValueError(f"Unsupported encoder artifact: {payload.get('artifact_format')!r}")
        self.config = payload["config"]
        self.preprocess = payload["preprocess"]
        self.center = np.asarray(self.preprocess["normalizer_center"], dtype=np.float32)
        self.scale = np.asarray(self.preprocess["normalizer_scale"], dtype=np.float32)
        self.history_length = int(self.config["history_length"])
        self.calibration_frames = int(self.config["baseline_frames"])
        self.latent_dim = int(self.config["latent_dim"])
        self.model = ForceGapEncoderV2(
            history_length=self.history_length,
            input_dim=len(self.preprocess["feature_order"]),
            latent_dim=self.latent_dim,
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()

    def _windows(self, normalized_features: np.ndarray) -> np.ndarray:
        rows = []
        for frame in range(len(normalized_features)):
            lo = max(0, frame - self.history_length + 1)
            window = normalized_features[lo : frame + 1]
            if len(window) < self.history_length:
                window = np.pad(window, ((self.history_length - len(window), 0), (0, 0)), mode="edge")
            rows.append(window)
        return np.asarray(rows, dtype=np.float32)

    def _infer_windows(self, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                x = torch.from_numpy(windows[start : start + batch_size]).to(self.device)
                z, _ = self.model(x)
                outputs.append(z.cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, self.latent_dim), np.float32)

    def latents(self, force_xyz: np.ndarray, timestamps: np.ndarray | None = None) -> np.ndarray:
        force = np.asarray(force_xyz, dtype=np.float32)
        if force.ndim != 2 or force.shape[1] != 3:
            raise ValueError(f"force_xyz must have shape [N,3], got {force.shape}")
        if len(force) < self.calibration_frames:
            raise ValueError(f"need at least {self.calibration_frames} calibration frames, got {len(force)}")
        if timestamps is not None:
            times = np.asarray(timestamps, dtype=np.float64)
            if times.shape != (len(force),) or not np.isfinite(times).all():
                raise ValueError("timestamps must be a finite [N] array matching force_xyz")
            if np.any(np.diff(times) <= 0):
                raise ValueError("timestamps must be strictly increasing")
        features, _ = force_gap_features(
            force,
            baseline_frames=self.calibration_frames,
            insertion_axis=np.asarray(self.preprocess["insertion_axis_base"], dtype=np.float32),
            smoothing_frames=int(self.preprocess["smoothing_frames"]),
            trend_lag_frames=int(self.preprocess["trend_lag_frames"]),
            active_noise_mad_multiplier=float(self.preprocess["active_noise_mad_multiplier"]),
            active_noise_floor_newton=float(self.preprocess["active_noise_floor_newton"]),
        )
        x = normalize(features, self.center, self.scale)
        return self._infer_windows(self._windows(x))


class StreamingForceGapEncoder:
    """Causal adapter; returns no latent until the initial 20-frame calibration ends."""

    def __init__(self, encoder: ForceGapLatentEncoder, output_history: int = 10):
        if output_history < 1:
            raise ValueError("output_history must be positive")
        self.encoder = encoder
        self.output_history = int(output_history)
        self._forces: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._latents: deque[np.ndarray] = deque(maxlen=max(self.output_history, encoder.history_length))

    def push(self, force_xyz: np.ndarray, timestamp: float) -> np.ndarray | None:
        force = np.asarray(force_xyz, dtype=np.float32)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("each force sample must be a finite base-frame [Fx,Fy,Fz] vector in N")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self._timestamps and timestamp <= self._timestamps[-1]):
            raise ValueError("timestamps must be finite and strictly increasing")
        self._forces.append(force)
        self._timestamps.append(timestamp)
        count = len(self._forces)
        if count < self.encoder.calibration_frames:
            return None
        if count == self.encoder.calibration_frames:
            initial = self.encoder.latents(np.asarray(self._forces), np.asarray(self._timestamps))
            self._latents.extend(initial[-self._latents.maxlen :])
            return self._latents[-1].copy()

        force = np.asarray(self._forces, dtype=np.float32)
        features, _ = force_gap_features(
            force,
            baseline_frames=self.encoder.calibration_frames,
            insertion_axis=np.asarray(self.encoder.preprocess["insertion_axis_base"], dtype=np.float32),
            smoothing_frames=int(self.encoder.preprocess["smoothing_frames"]),
            trend_lag_frames=int(self.encoder.preprocess["trend_lag_frames"]),
            active_noise_mad_multiplier=float(self.encoder.preprocess["active_noise_mad_multiplier"]),
            active_noise_floor_newton=float(self.encoder.preprocess["active_noise_floor_newton"]),
        )
        normalized = normalize(features[-self.encoder.history_length :], self.encoder.center, self.encoder.scale)
        if len(normalized) < self.encoder.history_length:
            normalized = np.pad(
                normalized,
                ((self.encoder.history_length - len(normalized), 0), (0, 0)),
                mode="edge",
            )
        window = torch.from_numpy(normalized[None]).to(self.encoder.device)
        with torch.inference_mode():
            latent, _ = self.encoder.model(window)
        result = latent[0].cpu().numpy()
        self._latents.append(result)
        return result.copy()

    def effort_history(self) -> np.ndarray:
        """Return [output_history,32] in chronological order, edge padded at startup."""
        if not self._latents:
            raise RuntimeError("encoder is still collecting its initial calibration frames")
        values = np.asarray(list(self._latents), dtype=np.float32)[-self.output_history :]
        if len(values) < self.output_history:
            values = np.pad(values, ((self.output_history - len(values), 0), (0, 0)), mode="edge")
        return values
