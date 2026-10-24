from __future__ import annotations

import numpy as np

from openpi.force_features import _robust_norm_threshold, causal_mean


FEATURE_NAMES = (
    "force_dir_x",
    "force_dir_y",
    "force_dir_z",
    "trend_dir_x",
    "trend_dir_y",
    "trend_dir_z",
    "axial_signed_fraction",
    "lateral_ratio",
    "log1p_force_norm",
    "log1p_trend_norm",
    "force_valid",
    "trend_valid",
)
FORCE_NORM_INDEX = 8
TREND_NORM_INDEX = 9


def _unit(values: np.ndarray, norm: np.ndarray, valid: np.ndarray, epsilon: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    result[valid] = values[valid] / np.maximum(norm[valid, None], epsilon)
    return result


def force_gap_features(
    force_xyz: np.ndarray,
    *,
    baseline_frames: int = 20,
    insertion_axis: np.ndarray = np.asarray([0.0, 0.0, -1.0]),
    smoothing_frames: int = 3,
    trend_lag_frames: int = 5,
    active_noise_mad_multiplier: float = 6.0,
    active_noise_floor_newton: float = 1e-6,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build causal direction, magnitude and validity features from base-frame force."""
    force = np.asarray(force_xyz, dtype=np.float32)
    if baseline_frames < 1 or force.ndim != 2 or force.shape[1] != 3 or len(force) < baseline_frames:
        raise ValueError(f"Expected force [T,3] with T >= {baseline_frames}, got {force.shape}")
    if smoothing_frames < 1 or trend_lag_frames < 1:
        raise ValueError("smoothing_frames and trend_lag_frames must be positive")
    if not np.isfinite(force).all():
        raise ValueError("force contains NaN or Inf")
    axis = np.array(insertion_axis, dtype=np.float32, copy=True)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= epsilon:
        raise ValueError(f"Invalid insertion axis: {axis}")
    axis /= np.linalg.norm(axis)

    baseline = np.median(force[:baseline_frames], axis=0).astype(np.float32)
    residual = force - baseline
    smoothed = causal_mean(residual, smoothing_frames)
    force_norm = np.linalg.norm(smoothed, axis=1).astype(np.float32)
    _, _, force_gate = _robust_norm_threshold(
        force_norm[:baseline_frames], active_noise_mad_multiplier, active_noise_floor_newton
    )
    force_valid = force_norm > force_gate

    reference = np.repeat(smoothed[:1], len(smoothed), axis=0)
    if trend_lag_frames < len(smoothed):
        reference[trend_lag_frames:] = smoothed[:-trend_lag_frames]
    trend = smoothed - reference
    trend_norm = np.linalg.norm(trend, axis=1).astype(np.float32)
    _, _, trend_gate = _robust_norm_threshold(
        trend_norm[:baseline_frames], active_noise_mad_multiplier, active_noise_floor_newton
    )
    trend_valid = trend_norm > trend_gate

    axial = smoothed @ axis
    lateral = smoothed - axial[:, None] * axis[None]
    lateral_norm = np.linalg.norm(lateral, axis=1)
    axial_fraction = np.divide(axial, force_norm + epsilon)
    lateral_ratio = np.divide(lateral_norm, force_norm + epsilon)
    axial_fraction[~force_valid] = 0.0
    lateral_ratio[~force_valid] = 0.0

    values = np.column_stack(
        (
            _unit(smoothed, force_norm, force_valid, epsilon),
            _unit(trend, trend_norm, trend_valid, epsilon),
            axial_fraction,
            lateral_ratio,
            np.log1p(force_norm),
            np.log1p(trend_norm),
            force_valid.astype(np.float32),
            trend_valid.astype(np.float32),
        )
    ).astype(np.float32)
    if values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise AssertionError(values.shape)
    return values, {
        "baseline_xyz_n": baseline.tolist(),
        "force_noise_gate_n": float(force_gate),
        "trend_noise_gate_n": float(trend_gate),
    }


def fit_normalizer(feature_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(feature_arrays, axis=0).astype(np.float64)
    center = np.median(stacked, axis=0)
    q25, q75 = np.quantile(stacked, [0.25, 0.75], axis=0)
    q05, q95 = np.quantile(stacked, [0.05, 0.95], axis=0)
    scale = np.maximum(q75 - q25, 0.1 * (q95 - q05))
    scale = np.maximum(scale, 1e-3)
    return center.astype(np.float32), scale.astype(np.float32)


def normalize(features: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(features, np.float32) - center) / scale, -10.0, 10.0).astype(np.float32)


def apply_gain_to_normalized(
    normalized: "torch.Tensor", center: "torch.Tensor", scale: "torch.Tensor", gain: "torch.Tensor"
) -> "torch.Tensor":
    """Apply an isotropic force gain to normalized windows without changing directions."""
    import torch

    raw = normalized * scale + center
    result = raw.clone()
    gain_per_window = gain.reshape(-1, 1)
    for index in (FORCE_NORM_INDEX, TREND_NORM_INDEX):
        magnitude = torch.expm1(torch.clamp(raw[..., index], min=0.0))
        result[..., index] = torch.log1p(gain_per_window * magnitude)
    return torch.clamp((result - center) / scale, -10.0, 10.0)
