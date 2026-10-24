from __future__ import annotations

import numpy as np


def force_history_features(
    force: np.ndarray,
    timestamps: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Return causal [force residual, first difference per second] channels."""
    force = np.asarray(force, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    residual = force - np.asarray(baseline, dtype=np.float32).reshape(1, 3)
    dt = np.diff(timestamps, prepend=timestamps[:1])
    positive = dt[dt > 1e-4]
    nominal_dt = float(np.median(positive)) if positive.size else 0.1
    dt[0] = nominal_dt
    dt = np.clip(dt, 0.02, 0.5)
    delta = np.diff(residual, axis=0, prepend=residual[:1]) / dt[:, None].astype(np.float32)
    return np.concatenate((residual, delta), axis=1).astype(np.float32)


def normalize_features(features: np.ndarray, scales: np.ndarray) -> np.ndarray:
    scales = np.asarray(scales, dtype=np.float32).reshape(1, 6)
    return np.clip(np.asarray(features, dtype=np.float32) / scales, -10.0, 10.0)


def initial_baseline(force: np.ndarray, frames: int = 20) -> np.ndarray:
    force = np.asarray(force, dtype=np.float32)
    return np.median(force[: min(frames, len(force))], axis=0).astype(np.float32)
