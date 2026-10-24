"""Causal force features shared by dataset conversion and real-time inference.

The first deployment-oriented force representation is intentionally Real-first:
it removes the per-trajectory free-space bias, smooths only with past samples,
and exposes force direction/trend separately from magnitude.  It does not use
future pose or any insertion label, so the same transform can run on a robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REAL_FORCE_TREND_FEATURE_NAMES = (
    "force_dir_x",
    "force_dir_y",
    "force_dir_z",
    "lateral_force_ratio",
    "axial_force_fraction",
    "trend_dir_x",
    "trend_dir_y",
    "trend_dir_z",
    "force_norm",
    "trend_norm",
)

# Shared free-space calibration window, in frames, counted from the start of a
# trajectory.  Both the offline encoder training/cross-validation and the
# deployment-side ``RealForceTrendBuffer`` estimate the free-space baseline and
# noise gate from this leading window, so the features match at run time.  A
# robot can identify the same window online (the first samples before contact);
# it cannot know a per-trajectory annotated free-space end.
DEFAULT_CALIBRATION_FRAMES = 20


def causal_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Return a causal moving mean; output[t] only uses values[:t+1]."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [time, channels], got {values.shape}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if width == 1:
        return values.copy()
    cumulative = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0)])
    indices = np.arange(values.shape[0])
    starts = np.maximum(0, indices - width + 1)
    counts = (indices - starts + 1).astype(np.float64)[:, None]
    return ((cumulative[indices + 1] - cumulative[starts]) / counts).astype(np.float32)


def _robust_norm_threshold(
    free_norm: np.ndarray,
    mad_multiplier: float,
    floor: float,
) -> tuple[float, float, float]:
    median = float(np.median(free_norm))
    mad = float(np.median(np.abs(free_norm - median)))
    threshold = max(median + mad_multiplier * mad, float(floor))
    return median, mad, threshold


def _direction(values: np.ndarray, norms: np.ndarray, threshold: float, epsilon: float) -> np.ndarray:
    result = values / (norms[:, None] + epsilon)
    result[norms <= threshold] = 0.0
    return result.astype(np.float32)


def build_real_force_trend_features(
    force_xyz: np.ndarray,
    *,
    free_end: int,
    insertion_axis: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float32),
    smoothing_frames: int = 3,
    trend_lag_frames: int = 5,
    epsilon: float = 1e-6,
    active_noise_mad_multiplier: float = 6.0,
    active_noise_floor: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build the 10-D causal Real force representation.

    Features per frame:
      gated force direction (3), lateral ratio (1), axial fraction (1),
      gated k-step trend direction (3), force norm (1), trend norm (1).

    ``free_end`` is used only to estimate the trajectory's Real free-space
    baseline and noise gate.  It is not a label and does not use future force.
    """
    force_xyz = np.asarray(force_xyz, dtype=np.float32)
    if force_xyz.ndim != 2 or force_xyz.shape[1] != 3:
        raise ValueError(f"Expected force_xyz with shape [time, 3], got {force_xyz.shape}")
    if not np.isfinite(force_xyz).all():
        raise ValueError("force_xyz contains NaN or inf")
    if not 0 <= free_end < len(force_xyz):
        raise ValueError(f"free_end={free_end} is outside trajectory length {len(force_xyz)}")
    if smoothing_frames <= 0 or trend_lag_frames <= 0:
        raise ValueError("smoothing_frames and trend_lag_frames must be positive")

    axis = np.asarray(insertion_axis, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.isfinite(axis).all() or axis_norm <= epsilon:
        raise ValueError(f"insertion_axis must be a non-zero finite 3-vector, got {axis}")
    axis = axis / axis_norm

    baseline = np.median(force_xyz[: free_end + 1], axis=0).astype(np.float32)
    residual = force_xyz - baseline
    smoothed = causal_mean(residual, smoothing_frames)
    force_norm = np.linalg.norm(smoothed, axis=1).astype(np.float32)
    _, _, noise_threshold = _robust_norm_threshold(
        force_norm[: free_end + 1], active_noise_mad_multiplier, active_noise_floor
    )

    force_dir = _direction(smoothed, force_norm, noise_threshold, epsilon)
    axial = smoothed @ axis
    lateral_vec = smoothed - axial[:, None] * axis[None, :]
    lateral_norm = np.linalg.norm(lateral_vec, axis=1)
    lateral_ratio = lateral_norm / (force_norm + epsilon)
    axial_fraction = axial / (force_norm + epsilon)
    lateral_ratio[force_norm <= noise_threshold] = 0.0
    axial_fraction[force_norm <= noise_threshold] = 0.0

    trend_reference = np.repeat(smoothed[:1], len(smoothed), axis=0)
    if trend_lag_frames < len(smoothed):
        trend_reference[trend_lag_frames:] = smoothed[:-trend_lag_frames]
    # For the first lag frames, compare with the first available causal sample.
    trend = smoothed - trend_reference
    trend_norm = np.linalg.norm(trend, axis=1).astype(np.float32)
    trend_dir = _direction(trend, trend_norm, noise_threshold, epsilon)

    features = np.column_stack(
        [
            force_dir,
            lateral_ratio,
            axial_fraction,
            trend_dir,
            force_norm,
            trend_norm,
        ]
    ).astype(np.float32)
    if features.shape[1] != len(REAL_FORCE_TREND_FEATURE_NAMES):
        raise AssertionError(features.shape)
    if not np.isfinite(features).all():
        raise ValueError("force feature construction produced NaN or inf")
    metadata: dict[str, float | int] = {
        "free_end": int(free_end),
        "smoothing_frames": int(smoothing_frames),
        "trend_lag_frames": int(trend_lag_frames),
        "baseline_x": float(baseline[0]),
        "baseline_y": float(baseline[1]),
        "baseline_z": float(baseline[2]),
        "free_norm_median": float(np.median(force_norm[: free_end + 1])),
        "free_norm_mad": float(np.median(np.abs(force_norm[: free_end + 1] - np.median(force_norm[: free_end + 1])))),
        "activity_gate_threshold": float(noise_threshold),
    }
    return features, metadata


@dataclass
class RealForceTrendBuffer:
    """Small causal adapter for deployment-side force histories.

    The robot should call ``update(force_xyz)`` once per force sample and pass
    the returned 10-D vectors as the newest ``observation.effort`` history.
    The initial ``baseline_frames`` are treated as free-space calibration.
    """

    baseline_frames: int = DEFAULT_CALIBRATION_FRAMES
    smoothing_frames: int = 3
    trend_lag_frames: int = 5
    epsilon: float = 1e-6
    active_noise_mad_multiplier: float = 6.0
    active_noise_floor: float = 1e-6

    def __post_init__(self) -> None:
        self._raw: list[np.ndarray] = []
        self._features: list[np.ndarray] = []
        self._threshold = float(self.active_noise_floor)
        self._baseline = np.zeros(3, dtype=np.float32)

    def update(self, force_xyz: np.ndarray) -> np.ndarray:
        sample = np.asarray(force_xyz, dtype=np.float32).reshape(-1)
        if sample.shape != (3,) or not np.isfinite(sample).all():
            raise ValueError(f"Expected one finite force sample with shape (3,), got {sample.shape}")
        self._raw.append(sample)
        if len(self._raw) == self.baseline_frames:
            calibration = np.asarray(self._raw, dtype=np.float32)
            self._baseline = np.median(calibration, axis=0)
            residual = calibration - self._baseline
            smoothed = causal_mean(residual, self.smoothing_frames)
            norms = np.linalg.norm(smoothed, axis=1)
            _, _, self._threshold = _robust_norm_threshold(
                norms, self.active_noise_mad_multiplier, self.active_noise_floor
            )
        if len(self._raw) < self.baseline_frames:
            return np.zeros(len(REAL_FORCE_TREND_FEATURE_NAMES), dtype=np.float32)
        features, _ = build_real_force_trend_features(
            np.asarray(self._raw, dtype=np.float32),
            free_end=self.baseline_frames - 1,
            smoothing_frames=self.smoothing_frames,
            trend_lag_frames=self.trend_lag_frames,
            epsilon=self.epsilon,
            active_noise_mad_multiplier=self.active_noise_mad_multiplier,
            active_noise_floor=self.active_noise_floor,
        )
        self._features.append(features[-1])
        return features[-1]

    def history(self, length: int = 10) -> np.ndarray:
        """Return the newest causal feature history with shape ``[length, 10]``.

        Before calibration completes, the history is zero-filled.  Once a
        valid feature exists, short histories are left-padded by their first
        valid feature, matching the usual episode-start padding convention.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if not self._features:
            return np.zeros((length, len(REAL_FORCE_TREND_FEATURE_NAMES)), dtype=np.float32)
        values = np.asarray(self._features, dtype=np.float32)
        if len(values) >= length:
            return values[-length:]
        padding = np.repeat(values[:1], length - len(values), axis=0)
        return np.vstack([padding, values])
