from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import ForceAlignmentEncoder
from .preprocess import force_history_features, initial_baseline, normalize_features


class ForceAlignmentRuntime:
    """Stateful TAVLA-side adapter. Feed 3-D force once per sample."""

    def __init__(self, artifact: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        payload: dict[str, Any] = torch.load(artifact, map_location=self.device, weights_only=True)
        config = payload["config"]
        self.history_length = int(config["history_length"])
        self.calibration_frames = int(config["baseline_frames"])
        self.dt = float(config["sample_period_s"])
        self.scales = np.asarray(payload["preprocess"]["feature_scales"], dtype=np.float32)
        self.geometry_scale_m = float(config["geometry_scale_m"])
        self.geometry_supervision = bool(config.get("geometry_supervision", False))
        self.model = ForceAlignmentEncoder(latent_dim=int(config["latent_dim"])).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self._startup: list[np.ndarray] = []
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)

    def reset(self) -> None:
        self._startup.clear()
        self._history.clear()

    @torch.inference_mode()
    def update(self, force_xyz_n: np.ndarray) -> dict[str, Any] | None:
        force = np.asarray(force_xyz_n, dtype=np.float32).reshape(3)
        if not np.isfinite(force).all():
            raise ValueError("force sample contains NaN or Inf")
        self._startup.append(force.copy())
        self._history.append(force.copy())
        if len(self._startup) < self.calibration_frames or len(self._history) < self.history_length:
            return None

        baseline = initial_baseline(np.asarray(self._startup), self.calibration_frames)
        history = np.asarray(self._history, dtype=np.float32)
        ts = np.arange(len(history), dtype=np.float64) * self.dt
        features = normalize_features(force_history_features(history, ts, baseline), self.scales)
        x = torch.from_numpy(features[None]).to(self.device)
        latent, logits, geometry = self.model(x)
        prob = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return {
            "latent": latent[0].cpu().numpy(),
            "misaligned_probability": float(prob[0]),
            "aligned_probability": float(prob[1]),
            "geometry_error_m": (geometry[0].cpu().numpy() * self.geometry_scale_m) if self.geometry_supervision else None,
        }
