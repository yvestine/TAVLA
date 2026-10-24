#!/usr/bin/env python3
"""Shared data, labeling, and representation utilities for Insertion Quality V1.

This module deliberately does not import any TAVLA training code.  The standalone
pipeline only reads the paired H5 files and writes artifacts below
``eval_outputs/insertion_quality_v1``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


QUALITY_LABELS = {"misaligned_contact": 0, "aligned_progress": 1}
LABEL_NAMES = {0: "misaligned_contact", 1: "aligned_progress", -1: "uncertain"}
FEATURE_NAMES = [
    "force_dir_x",
    "force_dir_y",
    "force_dir_z",
    "lateral_force_ratio",
    "axial_force_fraction",
    "trend_dir_x_k5",
    "trend_dir_y_k5",
    "trend_dir_z_k5",
    "activity_occupancy_5",
    "relative_lateral_strength",
    "relative_axial_strength",
]
MOTION_NAMES = ["future_z_progress", "future_xy_path", "z_stall_ratio", "z_rebound", "future_resume_progress"]
CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.65, "low": 0.35}


@dataclass
class TrajectoryData:
    trajectory: int
    timestamps: np.ndarray
    pose: np.ndarray
    real_force: np.ndarray
    sim_force: np.ndarray


@dataclass
class QuantileMotionRule:
    """Robust one-dimensional projection between class medians.

    The projection uses train-only median/IQR statistics.  The two acceptance
    thresholds are selected from train-score quantiles; samples between them
    remain uncertain.  This is intentionally a conservative weak-label rule,
    not a learned force classifier.
    """

    center: np.ndarray
    scale: np.ndarray
    direction: np.ndarray
    misaligned_max_score: float
    aligned_min_score: float
    train_precision_misaligned: float
    train_precision_aligned: float
    train_coverage: float
    quantile: float
    stall_step_threshold: float

    def score(self, motion: np.ndarray) -> np.ndarray:
        normalized = (motion - self.center) / self.scale
        return normalized @ self.direction

    def predict(self, motion: np.ndarray) -> np.ndarray:
        scores = self.score(motion)
        result = np.full(len(scores), -1, dtype=np.int64)
        result[scores <= self.misaligned_max_score] = 0
        result[scores >= self.aligned_min_score] = 1
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "direction": self.direction.tolist(),
            "misaligned_max_score": self.misaligned_max_score,
            "aligned_min_score": self.aligned_min_score,
            "train_precision_misaligned": self.train_precision_misaligned,
            "train_precision_aligned": self.train_precision_aligned,
            "train_coverage": self.train_coverage,
            "quantile": self.quantile,
            "stall_step_threshold": self.stall_step_threshold,
        }


@dataclass
class TemporalLogisticMotionRule:
    """Train-only motion classifier with causal probability smoothing.

    The classifier never consumes force.  It converts future EE-motion
    descriptors into conservative pseudo quality labels, leaving the interval
    between the two probability thresholds as uncertain.
    """

    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    misaligned_max_probability: float
    aligned_min_probability: float
    probability_smoothing_frames: int
    train_precision_misaligned: float
    train_precision_aligned: float
    train_coverage: float
    quantile: float
    stall_step_threshold: float

    def score(self, motion: np.ndarray) -> np.ndarray:
        normalized = (motion - self.center) / self.scale
        logits = normalized @ self.coefficient + self.intercept
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        width = self.probability_smoothing_frames
        cumulative = np.concatenate(([0.0], np.cumsum(probability)))
        smooth = np.empty_like(probability)
        for frame in range(len(probability)):
            start = max(0, frame - width + 1)
            smooth[frame] = (cumulative[frame + 1] - cumulative[start]) / (frame - start + 1)
        return smooth

    def predict(self, motion: np.ndarray) -> np.ndarray:
        probability = self.score(motion)
        result = np.full(len(probability), -1, dtype=np.int64)
        result[probability <= self.misaligned_max_probability] = 0
        result[probability >= self.aligned_min_probability] = 1
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_type": "temporal_logistic",
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
            "misaligned_max_probability": self.misaligned_max_probability,
            "aligned_min_probability": self.aligned_min_probability,
            "probability_smoothing_frames": self.probability_smoothing_frames,
            "train_precision_misaligned": self.train_precision_misaligned,
            "train_precision_aligned": self.train_precision_aligned,
            "train_coverage": self.train_coverage,
            "quantile": self.quantile,
            "stall_step_threshold": self.stall_step_threshold,
        }


@dataclass
class RepresentationScaler:
    active_threshold: dict[str, float]
    trend_threshold: dict[str, float]
    lateral_cdf: dict[str, np.ndarray]
    axial_cdf: dict[str, np.ndarray]


def read_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_annotations(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            row["trajectory_id"] = int(raw["trajectory"].split("_")[-1])
            row["start_frame"] = int(raw["start_frame"])
            row["end_frame"] = int(raw["end_frame"])
            rows.append(row)
    return rows


def annotation_arrays(
    annotations: list[dict[str, object]], trajectory: int, n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quality = np.full(n_frames, -1, dtype=np.int64)
    terminal = np.zeros(n_frames, dtype=np.float32)
    confidence = np.zeros(n_frames, dtype=np.float32)
    for row in annotations:
        if int(row["trajectory_id"]) != trajectory:
            continue
        start = max(0, int(row["start_frame"]))
        end = min(n_frames - 1, int(row["end_frame"]))
        label = str(row["label"])
        weight = CONFIDENCE_WEIGHTS[str(row["confidence"])]
        if label in QUALITY_LABELS:
            quality[start : end + 1] = QUALITY_LABELS[label]
            confidence[start : end + 1] = weight
        elif label == "terminal":
            terminal[start : end + 1] = 1.0
            confidence[start : end + 1] = weight
    return quality, terminal, confidence


def history_consistent_labels(labels: np.ndarray, history: int) -> np.ndarray:
    """Keep a label only when the complete causal history has one class.

    A window ending immediately after a phase boundary otherwise inherits only
    its last frame's label and mixes free/uncertain/misaligned/aligned evidence.
    Such boundary-crossing windows are intentionally masked.
    """

    if history <= 0:
        raise ValueError("history must be positive")
    result = np.full(len(labels), -1, dtype=np.int64)
    for frame in range(history - 1, len(labels)):
        window = labels[frame - history + 1 : frame + 1]
        if window[-1] >= 0 and np.all(window == window[-1]):
            result[frame] = int(window[-1])
    return result


def load_trajectory(repo: Path, config: dict[str, object], trajectory: int) -> TrajectoryData:
    data_cfg = config["data"]
    real_path = repo / str(data_cfg["real_root"]) / f"traj_{trajectory}" / "data.h5"
    sim_path = repo / str(data_cfg["sim_root"]) / f"traj_{trajectory}" / "data.h5"
    if not real_path.exists():
        raise FileNotFoundError(real_path)
    if not sim_path.exists():
        raise FileNotFoundError(sim_path)
    with h5py.File(real_path, "r") as handle:
        timestamps = np.asarray(handle[str(data_cfg["timestamp_key"])], dtype=np.float64)
        pose = np.asarray(handle[str(data_cfg["pose_key"])], dtype=np.float64)[:, :3]
        real_force = np.asarray(handle[str(data_cfg["real_force_key"])], dtype=np.float64)[:, :3]
    with h5py.File(sim_path, "r") as handle:
        sim_key = str(data_cfg["sim_force_key"])
        if sim_key not in handle:
            raise KeyError(f"{sim_path} does not contain required Sim force field {sim_key!r}; fallback is forbidden")
        sim_timestamps = np.asarray(handle[str(data_cfg["timestamp_key"])], dtype=np.float64)
        sim_pose = np.asarray(handle[str(data_cfg["pose_key"])], dtype=np.float64)[:, :3]
        sim_force = np.asarray(handle[sim_key], dtype=np.float64)[:, :3]
    if not (len(timestamps) == len(pose) == len(real_force) == len(sim_force)):
        raise ValueError(f"traj_{trajectory}: Real/Sim lengths differ")
    if not np.array_equal(timestamps, sim_timestamps):
        raise ValueError(f"traj_{trajectory}: Real/Sim timestamps are not exactly paired")
    if not np.array_equal(pose, sim_pose):
        raise ValueError(f"traj_{trajectory}: Real/Sim EE poses are not exactly paired")
    for name, values in (("real_force", real_force), ("sim_force", sim_force), ("pose", pose)):
        if not np.isfinite(values).all():
            raise ValueError(f"traj_{trajectory}: {name} contains NaN/inf")
    return TrajectoryData(trajectory, timestamps - timestamps[0], pose, real_force, sim_force)


def load_trajectories(repo: Path, config: dict[str, object], ids: list[int]) -> dict[int, TrajectoryData]:
    return {trajectory: load_trajectory(repo, config, trajectory) for trajectory in ids}


def _future_motion_base(pose: np.ndarray, horizon: int, resume_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n_frames = len(pose)
    base = np.zeros((n_frames, 4), dtype=np.float64)
    dz_steps = np.zeros((n_frames, horizon), dtype=np.float64)
    for frame in range(n_frames):
        end = min(frame + horizon, n_frames - 1)
        resume_end = min(frame + resume_horizon, n_frames - 1)
        segment = pose[frame : end + 1]
        steps = np.abs(np.diff(segment[:, 2]))
        dz_steps[frame, : len(steps)] = steps
        base[frame, 0] = pose[frame, 2] - pose[end, 2]
        base[frame, 1] = np.linalg.norm(np.diff(segment[:, :2], axis=0), axis=1).sum()
        base[frame, 2] = np.max(segment[:, 2] - pose[frame, 2])
        base[frame, 3] = pose[frame, 2] - np.min(pose[frame : resume_end + 1, 2])
    return base, dz_steps


def motion_descriptors(
    pose: np.ndarray,
    horizon: int,
    resume_horizon: int,
    stall_step_threshold: float,
) -> np.ndarray:
    base, dz_steps = _future_motion_base(pose, horizon, resume_horizon)
    valid_counts = np.minimum(horizon, np.maximum(0, len(pose) - 1 - np.arange(len(pose))))
    stall = np.ones(len(pose), dtype=np.float64)
    for frame, count in enumerate(valid_counts):
        if count > 0:
            stall[frame] = np.mean(dz_steps[frame, :count] <= stall_step_threshold)
    return np.column_stack((base[:, 0], base[:, 1], stall, base[:, 2], base[:, 3]))


def fit_stall_threshold(
    data: dict[int, TrajectoryData],
    annotations: list[dict[str, object]],
    train_ids: list[int],
    horizon: int,
    resume_horizon: int,
) -> float:
    del resume_horizon
    steps: list[np.ndarray] = []
    for trajectory in train_ids:
        pose = data[trajectory].pose
        quality, _, confidence = annotation_arrays(annotations, trajectory, len(pose))
        frames = np.flatnonzero((quality >= 0) & (confidence >= CONFIDENCE_WEIGHTS["medium"]))
        for frame in frames:
            end = min(frame + horizon, len(pose) - 1)
            if end > frame:
                steps.append(np.abs(np.diff(pose[frame : end + 1, 2])))
    if not steps:
        raise ValueError("Cannot fit stall threshold: no medium/high-confidence quality annotations in training set")
    values = np.concatenate(steps)
    return float(max(np.quantile(values, 0.25), np.finfo(np.float64).eps))


def _precision(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    selected = y_pred == label
    if not np.any(selected):
        return 0.0
    return float(np.mean(y_true[selected] == label))


def fit_quantile_motion_rule(
    data: dict[int, TrajectoryData],
    annotations: list[dict[str, object]],
    train_ids: list[int],
    config: dict[str, object],
) -> QuantileMotionRule:
    weak_cfg = config["weak_labels"]
    horizon = int(weak_cfg["future_horizon_frames"])
    resume_horizon = int(weak_cfg["resume_horizon_frames"])
    stall_threshold = fit_stall_threshold(data, annotations, train_ids, horizon, resume_horizon)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    trajectory_parts: list[np.ndarray] = []
    full_motion: dict[int, np.ndarray] = {}
    full_quality: dict[int, np.ndarray] = {}
    full_confidence: dict[int, np.ndarray] = {}
    for trajectory in train_ids:
        motion = motion_descriptors(data[trajectory].pose, horizon, resume_horizon, stall_threshold)
        quality, _, confidence = annotation_arrays(annotations, trajectory, len(motion))
        full_motion[trajectory] = motion
        full_quality[trajectory] = quality
        full_confidence[trajectory] = confidence
        keep = (quality >= 0) & (confidence >= CONFIDENCE_WEIGHTS["medium"])
        x_parts.append(motion[keep])
        y_parts.append(quality[keep])
        trajectory_parts.append(np.full(np.sum(keep), trajectory))
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    trajectory_ids = np.concatenate(trajectory_parts)
    if len(np.unique(y)) != 2:
        raise ValueError("Both insertion-quality classes are required to fit weak-label thresholds")

    if str(weak_cfg.get("rule_type", "quantile_projection")) == "temporal_logistic":
        from sklearn.linear_model import LogisticRegression

        center = np.median(x, axis=0)
        q25, q75 = np.quantile(x, [0.25, 0.75], axis=0)
        scale = np.maximum(q75 - q25, np.maximum(np.abs(center) * 1e-3, 1e-9))
        # Balance every trajectory and both classes so long search episodes do
        # not dominate the weak rule merely because they contain more frames.
        sample_weight = np.zeros(len(y), dtype=np.float64)
        for trajectory in np.unique(trajectory_ids):
            trajectory_mask = trajectory_ids == trajectory
            for label in (0, 1):
                class_mask = trajectory_mask & (y == label)
                count = int(np.sum(class_mask))
                if count:
                    sample_weight[class_mask] = 1.0 / (len(train_ids) * count)
        model = LogisticRegression(
            C=float(weak_cfg.get("logistic_c", 1.0)),
            max_iter=1000,
            solver="liblinear",
            random_state=int(config["seed"]),
        )
        model.fit((x - center) / scale, y, sample_weight=sample_weight)
        smoothing = int(weak_cfg.get("probability_smoothing_frames", 10))
        candidate_rule = TemporalLogisticMotionRule(
            center=center,
            scale=scale,
            coefficient=model.coef_[0].astype(np.float64),
            intercept=float(model.intercept_[0]),
            misaligned_max_probability=0.0,
            aligned_min_probability=1.0,
            probability_smoothing_frames=smoothing,
            train_precision_misaligned=0.0,
            train_precision_aligned=0.0,
            train_coverage=0.0,
            quantile=float("nan"),
            stall_step_threshold=stall_threshold,
        )
        probability_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        trajectory_label_parts: list[np.ndarray] = []
        for trajectory in train_ids:
            probability = candidate_rule.score(full_motion[trajectory])
            keep = (full_quality[trajectory] >= 0) & (
                full_confidence[trajectory] >= CONFIDENCE_WEIGHTS["medium"]
            )
            probability_parts.append(probability[keep])
            label_parts.append(full_quality[trajectory][keep])
            trajectory_label_parts.append(np.full(int(np.sum(keep)), trajectory, dtype=np.int64))
        probabilities = np.concatenate(probability_parts)
        labels = np.concatenate(label_parts)
        label_trajectories = np.concatenate(trajectory_label_parts)
        fit_precision = float(weak_cfg.get("fit_minimum_precision", weak_cfg["minimum_precision"]))
        candidates = np.unique(probabilities)
        valid_low: list[float] = []
        valid_high: list[float] = []
        for threshold in candidates:
            low_selected = probabilities <= threshold
            high_selected = probabilities >= threshold
            if (
                np.any(low_selected)
                and _precision(labels, np.where(low_selected, 0, -1), 0) >= fit_precision
                and len(np.unique(label_trajectories[low_selected])) >= 2
            ):
                valid_low.append(float(threshold))
            if (
                np.any(high_selected)
                and _precision(labels, np.where(high_selected, 1, -1), 1) >= fit_precision
                and len(np.unique(label_trajectories[high_selected])) >= 2
            ):
                valid_high.append(float(threshold))
        if not valid_low or not valid_high:
            raise ValueError("Cannot find train-only temporal-logistic thresholds at requested precision")
        best_pair: tuple[float, float, float] | None = None
        sorted_high = np.asarray(sorted(valid_high), dtype=np.float64)
        for candidate_low in valid_low:
            higher = sorted_high[sorted_high > candidate_low]
            if not len(higher):
                continue
            candidate_high = float(higher[0])
            coverage = float(
                np.mean((probabilities <= candidate_low) | (probabilities >= candidate_high))
            )
            candidate = (coverage, float(candidate_low), candidate_high)
            if best_pair is None or candidate > best_pair:
                best_pair = candidate
        if best_pair is None:
            raise ValueError("Cannot form a non-overlapping temporal-logistic uncertainty interval")
        _, low, high = best_pair
        prediction = np.full(len(labels), -1, dtype=np.int64)
        prediction[probabilities <= low] = 0
        prediction[probabilities >= high] = 1
        candidate_rule.misaligned_max_probability = low
        candidate_rule.aligned_min_probability = high
        candidate_rule.train_precision_misaligned = _precision(labels, prediction, 0)
        candidate_rule.train_precision_aligned = _precision(labels, prediction, 1)
        candidate_rule.train_coverage = float(np.mean(prediction >= 0))
        return candidate_rule

    center = np.median(x, axis=0)
    q25, q75 = np.quantile(x, [0.25, 0.75], axis=0)
    q05, q95 = np.quantile(x, [0.05, 0.95], axis=0)
    # A zero IQR is common for rebound because most windows do not rebound.
    # Fall back to a fraction of the robust 5--95% span so that this sparse
    # feature cannot numerically dominate the projection.
    scale = np.maximum(q75 - q25, 0.1 * (q95 - q05))
    scale = np.maximum(scale, np.maximum(np.abs(center) * 1e-3, 1e-9))
    class_medians = np.stack([np.median((x[y == label] - center) / scale, axis=0) for label in (0, 1)])
    direction = class_medians[1] - class_medians[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    scores = ((x - center) / scale) @ direction

    minimum_precision = float(weak_cfg["minimum_precision"])
    best: tuple[float, float, float, float, float, float] | None = None
    for quantile in weak_cfg["confidence_grid"]:
        q = float(quantile)
        # Larger q means a narrower, more conservative accepted tail.
        low = float(np.quantile(scores[y == 0], 1.0 - q))
        high = float(np.quantile(scores[y == 1], q))
        if low >= high:
            midpoint = 0.5 * (float(np.median(scores[y == 0])) + float(np.median(scores[y == 1])))
            gap = max(0.02 * float(np.std(scores)), 1e-6)
            low, high = midpoint - gap, midpoint + gap
        prediction = np.full(len(y), -1, dtype=np.int64)
        prediction[scores <= low] = 0
        prediction[scores >= high] = 1
        p0 = _precision(y, prediction, 0)
        p1 = _precision(y, prediction, 1)
        coverage = float(np.mean(prediction >= 0))
        # Require that accepted examples span at least two trajectories per class.
        span0 = len(np.unique(trajectory_ids[prediction == 0]))
        span1 = len(np.unique(trajectory_ids[prediction == 1]))
        valid = p0 >= minimum_precision and p1 >= minimum_precision and span0 >= 2 and span1 >= 2
        candidate = (1.0 if valid else 0.0, coverage, min(p0, p1), low, high, q)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    _, coverage, _, low, high, quantile = best
    prediction = np.full(len(y), -1, dtype=np.int64)
    prediction[scores <= low] = 0
    prediction[scores >= high] = 1
    return QuantileMotionRule(
        center=center,
        scale=scale,
        direction=direction,
        misaligned_max_score=low,
        aligned_min_score=high,
        train_precision_misaligned=_precision(y, prediction, 0),
        train_precision_aligned=_precision(y, prediction, 1),
        train_coverage=coverage,
        quantile=quantile,
        stall_step_threshold=stall_threshold,
    )


def predict_weak_labels(
    rule: QuantileMotionRule,
    trajectory: TrajectoryData,
    config: dict[str, object],
    free_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    weak_cfg = config["weak_labels"]
    motion = motion_descriptors(
        trajectory.pose,
        int(weak_cfg["future_horizon_frames"]),
        int(weak_cfg["resume_horizon_frames"]),
        rule.stall_step_threshold,
    )
    labels = rule.predict(motion)
    labels[: free_end + 1] = -1
    # The terminal tail is never converted into an aligned-progress weak label.
    z = trajectory.pose[:, 2]
    depth_threshold = float(np.quantile(z, float(weak_cfg["terminal_depth_quantile"])))
    progress_threshold = float(
        np.quantile(np.abs(motion[:, 0]), float(weak_cfg["terminal_progress_quantile"]))
    )
    terminal_like = (z <= depth_threshold) & (np.abs(motion[:, 0]) <= progress_threshold)
    labels[terminal_like] = -1
    return labels, motion


def causal_mean(values: np.ndarray, width: int) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    for frame in range(len(values)):
        start = max(0, frame - width + 1)
        result[frame] = (cumulative[frame + 1] - cumulative[start]) / (frame - start + 1)
    return result


def _residual_force(trajectory: TrajectoryData, domain: str, free_end: int) -> np.ndarray:
    if domain == "real":
        baseline = np.median(trajectory.real_force[: free_end + 1], axis=0)
        return trajectory.real_force - baseline
    if domain == "sim":
        return trajectory.sim_force.copy()
    raise ValueError(domain)


def fit_representation_scaler(
    data: dict[int, TrajectoryData],
    train_ids: list[int],
    config: dict[str, object],
) -> RepresentationScaler:
    rep_cfg = config["representation"]
    free_ends = config["free_end_frames"]
    axis = np.asarray(rep_cfg["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    width = int(rep_cfg["causal_smoothing_frames"])
    lag = int(rep_cfg["trend_lag_frames"])
    multiplier = float(rep_cfg["active_noise_mad_multiplier"])
    floor = float(rep_cfg["active_noise_floor_newton"])
    active_threshold: dict[str, float] = {}
    trend_threshold: dict[str, float] = {}
    lateral_cdf: dict[str, np.ndarray] = {}
    axial_cdf: dict[str, np.ndarray] = {}
    for domain in ("real", "sim"):
        free_norms: list[np.ndarray] = []
        free_trends: list[np.ndarray] = []
        lateral_values: list[np.ndarray] = []
        axial_values: list[np.ndarray] = []
        for trajectory_id in train_ids:
            free_end = int(free_ends[str(trajectory_id)])
            residual = _residual_force(data[trajectory_id], domain, free_end)
            smooth = causal_mean(residual, width)
            norm = np.linalg.norm(smooth, axis=1)
            trend = np.zeros_like(smooth)
            trend[lag:] = smooth[lag:] - smooth[:-lag]
            trend_norm = np.linalg.norm(trend, axis=1)
            axial = np.abs(smooth @ axis)
            lateral = np.linalg.norm(smooth - np.outer(smooth @ axis, axis), axis=1)
            free_norms.append(norm[: free_end + 1])
            free_trends.append(trend_norm[: free_end + 1])
            lateral_values.append(lateral)
            axial_values.append(axial)
        free_norm = np.concatenate(free_norms)
        free_trend = np.concatenate(free_trends)
        norm_median = float(np.median(free_norm))
        trend_median = float(np.median(free_trend))
        norm_mad = float(np.median(np.abs(free_norm - norm_median)))
        trend_mad = float(np.median(np.abs(free_trend - trend_median)))
        active_threshold[domain] = max(norm_median + multiplier * norm_mad, floor)
        trend_threshold[domain] = max(trend_median + multiplier * trend_mad, floor)
        lateral_cdf[domain] = np.sort(np.concatenate(lateral_values))
        axial_cdf[domain] = np.sort(np.concatenate(axial_values))
    return RepresentationScaler(active_threshold, trend_threshold, lateral_cdf, axial_cdf)


def empirical_cdf(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    if len(sorted_reference) == 0:
        raise ValueError("Empirical CDF reference is empty")
    return np.searchsorted(sorted_reference, values, side="right").astype(np.float64) / len(sorted_reference)


def transform_representation(
    trajectory: TrajectoryData,
    domain: str,
    free_end: int,
    scaler: RepresentationScaler,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    rep_cfg = config["representation"]
    axis = np.asarray(rep_cfg["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    epsilon = float(rep_cfg["epsilon"])
    lag = int(rep_cfg["trend_lag_frames"])
    occupancy_width = int(rep_cfg["occupancy_frames"])
    residual = _residual_force(trajectory, domain, free_end)
    smooth = causal_mean(residual, int(rep_cfg["causal_smoothing_frames"]))
    norm = np.linalg.norm(smooth, axis=1)
    active = norm > scaler.active_threshold[domain]
    force_direction = np.zeros_like(smooth)
    force_direction[active] = smooth[active] / (norm[active, None] + epsilon)
    axial_signed = smooth @ axis
    axial = np.abs(axial_signed)
    lateral = np.linalg.norm(smooth - np.outer(axial_signed, axis), axis=1)
    lateral_ratio = np.divide(lateral, norm + epsilon)
    axial_fraction = np.divide(axial, norm + epsilon)
    trend = np.zeros_like(smooth)
    trend[lag:] = smooth[lag:] - smooth[:-lag]
    trend_norm = np.linalg.norm(trend, axis=1)
    trend_active = trend_norm > scaler.trend_threshold[domain]
    trend_direction = np.zeros_like(trend)
    trend_direction[trend_active] = trend[trend_active] / (trend_norm[trend_active, None] + epsilon)
    occupancy = causal_mean(active.astype(np.float64)[:, None], occupancy_width)[:, 0]
    features = np.column_stack(
        (
            force_direction,
            lateral_ratio,
            axial_fraction,
            trend_direction,
            occupancy,
            empirical_cdf(lateral, scaler.lateral_cdf[domain]),
            empirical_cdf(axial, scaler.axial_cdf[domain]),
        )
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError(f"Expected 11 features, got {features.shape}")
    if np.min(features[:, [3, 4, 8, 9, 10]]) < -1e-8 or np.max(features[:, [3, 4, 8, 9, 10]]) > 1 + 1e-8:
        raise AssertionError("Bounded scalar representation escaped [0, 1]")
    return features.astype(np.float32), active


def window_view(features: np.ndarray, history: int) -> np.ndarray:
    if len(features) < history:
        return np.empty((0, history, features.shape[1]), dtype=np.float32)
    windows = [features[end - history + 1 : end + 1] for end in range(history - 1, len(features))]
    return np.asarray(windows, dtype=np.float32)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
