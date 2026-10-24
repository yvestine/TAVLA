"""Batch and streaming motion_v1 inference, independent of the training loader."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .features import normalize
from .motion_features import (FEATURE_NAMES, SOURCE_NAMES, background_from_basis,
                              motion_features, numpy_features, state_basis)
from .motion_model import MotionForceEncoder

ARTIFACT_FORMAT = "tavla_force_gap_encoder_motion_v1"


def _window(features, frame, length):
    values = features[max(0, frame - length + 1):frame + 1]
    return np.pad(values, ((length - len(values), 0), (0, 0)), mode="edge")


class MotionForceLatentEncoder:
    """Move in reviewed free space during startup; no stationary assumption.

    One synchronized F/q/dq/time observation per nominal 100 ms is required.
    The source is a constructor property, never a latent input channel.
    Only completed startup outputs are exposed; pre-startup latents are not
    backfilled into the policy history.
    """

    def __init__(self, checkpoint_path: str | Path, *, source: str, device: str = "cpu"):
        if source not in SOURCE_NAMES:
            raise ValueError(f"source must be one of {SOURCE_NAMES}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("artifact_format") != ARTIFACT_FORMAT:
            raise ValueError("Expected motion_v1 artifact; legacy force-only checkpoints have different semantics")
        self.source, self.device = source, torch.device(device)
        self.config, self.preprocess = payload["config"], payload["preprocess"]
        self.calibration_frames = int(self.config["baseline_frames"])
        self.history_length = int(self.config["history_length"])
        if self.calibration_frames != 20 or self.history_length != 32 or payload.get("version") != 1:
            raise ValueError("Unsupported motion_v1 version/history/calibration")
        self.background = payload["background"]
        for key, shape in (("coefficient", (28, 3)), ("feature_scale", (28,)), ("delta_limit_n", (3,))):
            value = np.asarray(self.background[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid background {key}")
        if np.any(np.asarray(self.background["feature_scale"]) <= 0) or np.any(np.asarray(self.background["delta_limit_n"]) < 0):
            raise ValueError("Invalid background scale/limits")
        if tuple(self.preprocess["feature_order"]) != FEATURE_NAMES:
            raise ValueError("Unexpected feature order")
        self.center = np.asarray(self.preprocess["normalizer_center"], np.float32)
        self.scale = np.asarray(self.preprocess["normalizer_scale"], np.float32)
        if self.center.shape != (17,) or self.scale.shape != (17,) or not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Invalid feature normalizer")
        self.model = MotionForceEncoder(self.history_length)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()
        self.reset()

    def reset(self):
        self._forces, self._joints, self._velocities, self._times = [], [], [], []
        self._latents = deque(maxlen=10)
        self._feature_history = deque(maxlen=self.history_length)
        self._residual_history = deque(maxlen=3)
        self._smooth_history, self._time_history = deque(maxlen=5), deque(maxlen=5)
        self._baseline, self._phi0, self._scales = None, None, None
        self._last_timestamp = None

    @property
    def ready(self):
        return bool(self._latents)

    def features(self, forces, joints, velocities, timestamps):
        return motion_features(forces, joints, velocities, timestamps, self.background,
                               self.source, self.calibration_frames)

    def _infer(self, windows):
        output = []
        with torch.inference_mode():
            for start in range(0, len(windows), 512):
                z, _, _ = self.model(torch.from_numpy(windows[start:start + 512]).to(self.device))
                if not torch.isfinite(z).all():
                    raise FloatingPointError("Non-finite force latent")
                output.append(z.cpu().numpy())
        return np.concatenate(output).astype(np.float32)

    def batch(self, forces, joints, velocities, timestamps):
        """Return [T-19,32]; row zero corresponds to calibration frame 19."""
        signal = self.features(forces, joints, velocities, timestamps)
        features = normalize(signal["features"], self.center, self.scale)
        windows = np.stack([_window(features, frame, self.history_length)
                            for frame in range(self.calibration_frames - 1, len(features))])
        return self._infer(windows)

    def push(self, force_xyz_n, joint_pos_rad, joint_vel_rad_s, timestamp_s):
        force, q, velocity = (np.asarray(value, np.float32) for value in (force_xyz_n, joint_pos_rad, joint_vel_rad_s))
        timestamp = float(timestamp_s)
        if force.shape != (3,) or q.shape != (7,) or velocity.shape != (7,) or not all(np.isfinite(value).all() for value in (force, q, velocity)):
            raise ValueError("Expected finite force[3], joint_pos[7], joint_vel[7]")
        if not np.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            raise ValueError("Timestamp must be finite and strictly increasing; reset between episodes")
        if self._baseline is None:
            self._forces.append(force.copy())
            self._joints.append(q.copy())
            self._velocities.append(velocity.copy())
            self._times.append(timestamp)
            self._last_timestamp = timestamp
            if len(self._forces) < self.calibration_frames:
                return None
            forces, joints, velocities = map(np.asarray, (self._forces, self._joints, self._velocities))
            signal = self.features(forces, joints, velocities, np.asarray(self._times))
            self._baseline = np.median(forces, axis=0)
            self._phi0 = np.median(state_basis(joints, velocities), axis=0)
            self._scales = signal["scales"].copy()
            self._feature_history.extend(signal["features"])
            self._residual_history.extend(signal["residual"][-3:])
            self._smooth_history.extend(signal["vectors"][-5:, 0])
            self._time_history.extend(self._times[-5:])
            self._forces, self._joints, self._velocities, self._times = [], [], [], []
        else:
            phi = state_basis(q[None], velocity[None])[0]
            _, residual = background_from_basis(force, phi, self._baseline, self._phi0, self.background, self.source)
            self._residual_history.append(residual)
            smooth = np.mean(np.stack(self._residual_history), axis=0, dtype=np.float64).astype(np.float32)
            vectors = np.stack((smooth, smooth - self._smooth_history[-1], smooth - self._smooth_history[-5]))
            elapsed = np.asarray([(timestamp - self._time_history[-1]) / .1,
                                  (timestamp - self._time_history[-5]) / .5], np.float32)
            self._feature_history.append(numpy_features(vectors[None], self._scales, elapsed[None])[0])
            self._smooth_history.append(smooth)
            self._time_history.append(timestamp)
            self._last_timestamp = timestamp
        normalized = normalize(np.stack(self._feature_history), self.center, self.scale)
        latent = self._infer(_window(normalized, len(normalized) - 1, self.history_length)[None])[0]
        self._latents.append(latent.copy())
        return latent.copy()

    def effort_history(self):
        if not self.ready:
            raise RuntimeError("Collect 20 synchronized, pre-contact samples; movement is allowed")
        values = np.stack(self._latents).astype(np.float32)
        return np.pad(values, ((10 - len(values), 0), (0, 0)), mode="edge").copy()
