"""Causal serving adapter matching ForceAlignmentLatentEncoder's offline windows."""

from collections import deque

import numpy as np
import torch

from .deploy_infer import ForceAlignmentLatentEncoder
from .preprocess import force_history_features, initial_baseline, normalize_features


class StreamingForceAlignmentEncoder:
    """Per-connection histories sharing one frozen, read-only encoder model.

    The first baseline_frames samples must be collected without task contact.
    Once their fixed baseline is available, backfill the startup latents using
    the same edge-padded windows used by training conversion. Preserve each
    frame's derivative when the window slides; recomputing the derivative at
    every window's left edge would change the training-time features.
    """

    def __init__(self, encoder: ForceAlignmentLatentEncoder, policy_history: int = 10):
        self.encoder = encoder
        self.policy_history = policy_history
        self.reset()

    def reset(self):
        self._startup_force = []
        self._startup_time = []
        self._features = deque(maxlen=self.encoder.history)
        self._latents = deque(maxlen=self.policy_history)
        self._baseline = None
        self._previous_force = None
        self.last_timestamp = None
        self.samples_received = 0

    @property
    def ready(self):
        return self._baseline is not None

    @property
    def samples_needed(self):
        return max(0, self.encoder.baseline_frames - self.samples_received)

    def _window(self):
        features = np.asarray(self._features, dtype=np.float32)
        padding = self.encoder.history - len(features)
        return np.pad(features, ((padding, 0), (0, 0)), mode="edge")

    @torch.inference_mode()
    def _encode(self, windows):
        inputs = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.encoder.device)
        latent, _, _ = self.encoder.model(inputs)
        self._latents.extend(latent.cpu().numpy().astype(np.float32))

    def push(self, force_xyz, timestamp_s):
        force = np.asarray(force_xyz, dtype=np.float32)
        stamp = float(timestamp_s)
        if force.shape != (3,) or not np.isfinite(force).all() or not np.isfinite(stamp):
            raise ValueError("Expected finite XYZ force (3,) and timestamp in seconds")
        if self.last_timestamp is not None and stamp <= self.last_timestamp:
            raise ValueError("Force timestamps must increase; reset between episodes")

        if not self.ready:
            self._startup_force.append(force.copy())
            self._startup_time.append(stamp)
            if len(self._startup_force) == self.encoder.baseline_frames:
                startup = np.asarray(self._startup_force, dtype=np.float32)
                self._baseline = initial_baseline(startup, self.encoder.baseline_frames)
                features = normalize_features(
                    force_history_features(startup, np.asarray(self._startup_time), self._baseline),
                    self.encoder.scales,
                )
                windows = []
                for feature in features:
                    self._features.append(feature)
                    windows.append(self._window())
                self._encode(windows)
                self._startup_force.clear()
                self._startup_time.clear()
        else:
            # The last row uses the actual preceding sample, including when
            # that sample will fall outside the current 32-frame window.
            features = force_history_features(
                np.stack([self._previous_force, force]),
                np.asarray([self.last_timestamp, stamp]),
                self._baseline,
            )[-1:]
            self._features.append(normalize_features(features, self.encoder.scales)[0])
            self._encode([self._window()])

        self._previous_force = force.copy()
        self.last_timestamp = stamp
        self.samples_received += 1

    def effort_history(self):
        if not self.ready:
            raise ValueError(f"Collect {self.samples_needed} more baseline force samples before inference")
        history = np.asarray(self._latents, dtype=np.float32)
        return np.pad(history, ((self.policy_history - len(history), 0), (0, 0)), mode="edge")
