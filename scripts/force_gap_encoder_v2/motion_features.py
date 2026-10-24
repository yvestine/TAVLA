"""Moving free-space calibration and continuous force features for motion_v1."""

from __future__ import annotations

import numpy as np
import torch

FEATURE_NAMES = (
    "soft_force_x", "soft_force_y", "soft_force_z",
    "soft_delta1_x", "soft_delta1_y", "soft_delta1_z",
    "soft_delta5_x", "soft_delta5_y", "soft_delta5_z",
    "soft_axial_fraction", "soft_lateral_ratio",
    "log_relative_force", "log_relative_delta1", "log_relative_delta5",
    "log_absolute_force_n", "elapsed1_over_100ms", "elapsed5_over_500ms",
)
STRUCTURE_INDICES = tuple(range(11)) + (15, 16)
STRENGTH_INDICES = (11, 12, 13, 14)
RIDGE_WEIGHTS = (0.001, 0.01, 0.1, 1.0, 10.0)
SOURCE_NAMES = ("real", "sim_contact_isolated")


def state_basis(q: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    q, velocity = np.asarray(q, np.float64), np.asarray(velocity, np.float64)
    if q.ndim != 2 or q.shape[1] != 7 or velocity.shape != q.shape:
        raise ValueError("joint_pos/joint_vel must have matching [T,7] shapes")
    if not np.isfinite(q).all() or not np.isfinite(velocity).all():
        raise ValueError("Non-finite joint state")
    return np.concatenate((np.sin(q), np.cos(q), velocity, np.abs(velocity)), axis=1)


def disabled_background(reason: str) -> dict:
    return {"enabled": False, "reason": reason, "coefficient": np.zeros((28, 3)).tolist(),
            "feature_scale": np.ones(28).tolist(), "delta_limit_n": np.zeros(3).tolist()}


def _free_rows(episode, calibration: int):
    phase, confidence = episode.phase, episode.confidence
    keep = (phase == "free") & (confidence == "high") & (np.arange(len(phase)) >= calibration)
    if not keep.any():
        raise ValueError(f"traj_{episode.trajectory}: no high-confidence free-space fitting frames")
    phi = state_basis(episode.q, episode.velocity)
    x = phi[keep] - np.median(phi[:calibration], axis=0)
    y = episode.force[keep] - np.median(episode.force[:calibration], axis=0)
    return x, y


def fit_background(episodes, ids: list[int], ridge: float, calibration: int = 20) -> dict:
    """Robust, trajectory-balanced ridge fit; zero delta state maps to zero delta force."""
    pieces = [_free_rows(episodes[(trajectory, 0)], calibration) for trajectory in ids]
    x = np.concatenate([piece[0] for piece in pieces]).astype(np.float64)
    y = np.concatenate([piece[1] for piece in pieces]).astype(np.float64)
    base_weight = np.concatenate([np.full(len(a), 1.0 / len(a)) for a, _ in pieces])
    base_weight /= base_weight.mean()
    feature_scale = np.maximum(np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0), 1e-3)
    target_scale = np.maximum(np.quantile(y, 0.75, axis=0) - np.quantile(y, 0.25, axis=0), 0.01)
    xn, yn = x / feature_scale, y / target_scale
    coefficient = np.zeros((28, 3), np.float64)
    for axis in range(3):
        beta = np.zeros(28, np.float64)
        for _ in range(30):
            residual = yn[:, axis] - xn @ beta
            weights = base_weight * np.minimum(1.0, 1.0 / np.maximum(np.abs(residual), 1e-12))
            lhs = xn.T @ (weights[:, None] * xn) / weights.sum() + ridge * np.eye(28)
            rhs = xn.T @ (weights * yn[:, axis]) / weights.sum()
            updated = np.linalg.solve(lhs, rhs)
            if np.max(np.abs(updated - beta)) < 1e-7:
                beta = updated
                break
            beta = updated
        coefficient[:, axis] = beta * target_scale[axis]
    return {"enabled": True, "ridge": ridge, "fit_trajectories": ids,
            "coefficient": coefficient.tolist(), "feature_scale": feature_scale.tolist(),
            "delta_limit_n": np.quantile(np.abs(y), 0.99, axis=0).tolist(),
            "method": "trajectory-balanced Huber IRLS ridge; anchored state delta; 30 iterations; Huber beta=1"}


def background_from_basis(force, phi, baseline, phi0, background, source):
    """Shared batch/stream step with fixed episode anchors."""
    delta = np.zeros_like(force)
    if source == "real" and background["enabled"]:
        x = (phi - phi0) / np.asarray(background["feature_scale"])
        delta = np.clip(x @ np.asarray(background["coefficient"]),
                        -np.asarray(background["delta_limit_n"]), np.asarray(background["delta_limit_n"]))
    prediction = (baseline + delta).astype(np.float32)
    return prediction, (force - prediction).astype(np.float32)


def background_trace(force, q, velocity, background: dict, source: str, calibration: int = 20):
    force = np.asarray(force, np.float32)
    phi = state_basis(q, velocity)
    if source not in SOURCE_NAMES or force.shape != (len(phi), 3) or len(force) < calibration:
        raise ValueError("Invalid source, force shape or calibration length")
    if not np.isfinite(force).all():
        raise ValueError("Non-finite force")
    baseline = np.median(force[:calibration], axis=0)
    phi0 = np.median(phi[:calibration], axis=0)
    return background_from_basis(force, phi, baseline, phi0, background, source)


def background_mse(episodes, ids, background, calibration=20):
    rows = {}
    for trajectory in ids:
        episode = episodes[(trajectory, 0)]
        _, residual = background_trace(episode.force, episode.q, episode.velocity, background, "real", calibration)
        keep = (episode.phase == "free") & (episode.confidence == "high") & (np.arange(len(residual)) >= calibration)
        rows[str(trajectory)] = float(np.mean(residual[keep].astype(np.float64) ** 2))
    return rows


def select_background(episodes, ids: list[int], calibration=20, seed=42):
    """Select ridge with grouped OOF, then enable only for >=20% OOF improvement."""
    if len(ids) < 3:
        raise ValueError("Background grouped validation requires at least three trajectories")
    order = np.random.default_rng(seed).permutation(sorted(ids))
    folds = [list(map(int, group)) for group in np.array_split(order, min(4, len(ids)))]
    constant = background_mse(episodes, ids, disabled_background("constant baseline comparison"), calibration)
    scores = {}
    for ridge in RIDGE_WEIGHTS:
        per_trajectory = {}
        for valid in folds:
            train = sorted(set(ids) - set(valid))
            fitted = fit_background(episodes, train, ridge, calibration)
            per_trajectory.update(background_mse(episodes, valid, fitted, calibration))
        scores[ridge] = per_trajectory
    selected = min(RIDGE_WEIGHTS, key=lambda value: (np.mean(list(scores[value].values())), -value))
    mse0, mse1 = float(np.mean(list(constant.values()))), float(np.mean(list(scores[selected].values())))
    improvement = (mse0 - mse1) / max(mse0, 1e-12)
    report = {"grouped_folds": folds, "selected_ridge": selected, "constant_mse_n2": mse0,
              "learned_oof_mse_n2": mse1, "relative_improvement": improvement,
              "minimum_improvement": 0.20, "enabled": improvement >= 0.20,
              "constant_mse_by_trajectory": constant,
              "oof_mse_by_ridge": {str(k): v for k, v in scores.items()}}
    fitted = fit_background(episodes, ids, selected, calibration) if report["enabled"] else disabled_background("OOF improvement below 20%")
    fitted["selection"] = report
    return fitted


def signal_vectors(residual: np.ndarray, timestamps: np.ndarray, calibration: int = 20):
    # Float64 accumulation avoids prefix-length-dependent rounding and agrees
    # with the streaming mean over the last three float32 residuals.
    residual = np.asarray(residual, np.float32)
    cumulative = np.vstack((np.zeros((1, 3)), np.cumsum(residual, axis=0, dtype=np.float64)))
    indices = np.arange(len(residual))
    starts = np.maximum(indices - 2, 0)
    smooth = ((cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)[:, None]).astype(np.float32)
    vectors = [smooth]
    elapsed = []
    for lag in (1, 5):
        references = np.maximum(np.arange(len(smooth)) - lag, 0)
        vectors.append(smooth - smooth[references])
        elapsed.append((timestamps - timestamps[references]) / (0.1 * lag))
    vectors = np.stack(vectors, axis=1).astype(np.float32)  # [T,force/delta1/delta5,xyz]
    scales = np.maximum(np.median(np.linalg.norm(vectors[:calibration], axis=-1), axis=0), 0.01).astype(np.float32)
    return vectors, scales, np.stack(elapsed, axis=1).astype(np.float32)


def numpy_features(vectors, scales, elapsed, axis=(0.0, 0.0, -1.0)):
    vectors, scales = np.asarray(vectors, np.float32), np.asarray(scales, np.float32)
    norm = np.linalg.norm(vectors, axis=-1)
    soft = vectors / (norm + scales)[..., None]
    insertion_axis = np.asarray(axis, np.float32)
    insertion_axis = insertion_axis / np.linalg.norm(insertion_axis)
    axial = vectors[..., 0, :] @ insertion_axis
    lateral = np.linalg.norm(vectors[..., 0, :] - axial[..., None] * insertion_axis, axis=-1)
    values = np.concatenate((soft.reshape(*soft.shape[:-2], 9),
                             (axial / (norm[..., 0] + scales[..., 0]))[..., None],
                             (lateral / (norm[..., 0] + scales[..., 0]))[..., None],
                             np.log1p(norm / scales), np.log1p(norm[..., :1]), np.asarray(elapsed)), axis=-1)
    return values.astype(np.float32)


def torch_features(vectors, scales, elapsed, axis):
    """Rebuild features from gained physical vectors before normalization/clipping."""
    scales = scales[:, None, :]
    norm = torch.linalg.vector_norm(vectors, dim=-1)
    soft = vectors / (norm + scales).unsqueeze(-1)
    insertion_axis = torch.as_tensor(axis, dtype=vectors.dtype, device=vectors.device)
    insertion_axis = insertion_axis / torch.linalg.vector_norm(insertion_axis)
    axial = vectors[:, :, 0] @ insertion_axis
    lateral = torch.linalg.vector_norm(vectors[:, :, 0] - axial.unsqueeze(-1) * insertion_axis, dim=-1)
    return torch.cat((soft.flatten(-2), (axial / (norm[:, :, 0] + scales[:, :, 0])).unsqueeze(-1),
                      (lateral / (norm[:, :, 0] + scales[:, :, 0])).unsqueeze(-1),
                      torch.log1p(norm / scales), torch.log1p(norm[:, :, :1]), elapsed), dim=-1)


def motion_features(force, q, velocity, timestamps, background, source, calibration=20):
    timestamps = np.asarray(timestamps, np.float64)
    if timestamps.shape != (len(force),) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must be finite, strictly increasing and match force")
    prediction, residual = background_trace(force, q, velocity, background, source, calibration)
    vectors, scales, elapsed = signal_vectors(residual, timestamps, calibration)
    features = numpy_features(vectors, scales, elapsed)
    return {"features": features, "vectors": vectors, "scales": scales, "elapsed": elapsed,
            "background": prediction, "residual": residual,
            "signal_confidence": (np.linalg.norm(vectors[:, 0], axis=1) ** 2 /
                                  (np.linalg.norm(vectors[:, 0], axis=1) ** 2 + scales[0] ** 2)).astype(np.float32)}
