#!/usr/bin/env python3
"""Whole-trajectory inference for the frozen ForceAlignmentEncoder.

This is the dataset-conversion / deployment mirror of the force-only insertion
alignment encoder saved at
``eval_outputs/insertion_alignment_encoder/encoder.pt``.  It reproduces the
encoder's exact training-time input (baseline-subtracted 3-axis force plus its
per-second derivative, 6 channels, normalized with the saved feature scales and
clipped), windows every frame causally with the saved history length, and
returns one 32-D latent per frame.

It deliberately exposes the SAME ``latents(force_xyz, pose, timestamps)`` and
``latent_dim`` interface as ``scripts/control_encoder_infer.py`` so it is a
drop-in replacement wherever that adapter is used.  ``pose`` is accepted for
signature compatibility and ignored -- this encoder is force-only.

No TAVLA/pi0 weight is read or modified, and no existing file is changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import ForceAlignmentEncoder
from .preprocess import force_history_features, initial_baseline, normalize_features


class ForceAlignmentLatentEncoder:
    """Load a frozen ForceAlignmentEncoder and map 3-axis force to latents."""

    def __init__(self, encoder_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        payload: dict[str, Any] = torch.load(encoder_path, map_location=self.device, weights_only=True)
        expected = "tavla_force_alignment_encoder_terminal_reference_v1"
        if payload.get("artifact_format") != expected:
            raise ValueError(
                f"{encoder_path}: expected artifact_format {expected!r}, got "
                f"{payload.get('artifact_format')!r}. Train it with "
                "insertion_alignment_encoder.train_terminal_geometry."
            )
        config = payload["config"]
        self.latent_dim = int(config["latent_dim"])
        self.history = int(config["history_length"])
        self.baseline_frames = int(config["baseline_frames"])
        self.scales = np.asarray(payload["preprocess"]["feature_scales"], dtype=np.float32)
        self.model = ForceAlignmentEncoder(latent_dim=self.latent_dim).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def features(self, force_xyz: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Return the normalized [T, 6] encoder input for a whole trajectory."""
        force = np.asarray(force_xyz, dtype=np.float32)[:, :3]
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        baseline = initial_baseline(force, self.baseline_frames)
        raw = force_history_features(force, timestamps, baseline)
        return normalize_features(raw, self.scales)

    @torch.inference_mode()
    def latents(self, force_xyz: np.ndarray, pose: np.ndarray | None, timestamps: np.ndarray) -> np.ndarray:
        """Per-frame 32-D latent, one row per input frame.

        Frame ``t`` uses the causal window ``[t-history+1, t]``; windows shorter
        than ``history`` (only the leading frames) are left-padded by repeating
        the earliest feature row, matching the edge padding used at training
        time in ``make_windows``.  ``pose`` is ignored (force-only encoder).
        """
        feats = self.features(force_xyz, timestamps)
        n = len(feats)
        if n == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        windows = np.empty((n, self.history, 6), dtype=np.float32)
        for end in range(n):
            lo = max(0, end - self.history + 1)
            chunk = feats[lo : end + 1]
            if len(chunk) < self.history:
                chunk = np.vstack([np.repeat(chunk[:1], self.history - len(chunk), axis=0), chunk])
            windows[end] = chunk
        out = []
        for start in range(0, n, 512):
            batch = torch.from_numpy(windows[start : start + 512]).to(self.device)
            latent, _, _ = self.model(batch)
            out.append(latent.cpu().numpy())
        return np.concatenate(out).astype(np.float32)
