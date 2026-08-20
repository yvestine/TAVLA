"""Compare independent, unpaired sim-to-real wrench alignment methods.

This is an isolated experiment module.  It does not modify the existing
adapter, dataset converter, training configuration, or server.  Because the
simulation and real trajectories are not frame-paired, this script ranks
methods by held-out distribution and temporal-statistics agreement; it does
not report a fake frame-wise MAE.

Methods:
  identity:      no distribution adapter;
  robust_affine: existing median/central-quantile per-channel baseline;
  coral_block:   covariance alignment separately for force and torque;
  coral_full:    six-dimensional covariance alignment;
  quantile:      per-channel empirical CDF/quantile transport;
  copula_block:  target marginal quantiles plus block covariance alignment in
                 Gaussian-rank coordinates;
  polyfit_mlp:   an already-trained unpaired PolyFit-inspired checkpoint,
                 when supplied with --mlp-adapter.

The default split is episode-disjoint and deterministic.  All fit statistics
come from the training episodes; held-out real episodes are only used for
evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import ks_2samp, wasserstein_distance

from openpi.shared.wrench_adapter import load_adapter


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
EPS = 1e-6


def load_episodes(root: Path, pattern: str, h5_key: str | None) -> list[tuple[str, np.ndarray]]:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {root / pattern}")
    result: list[tuple[str, np.ndarray]] = []
    for path in files:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            if h5_key is None:
                raise ValueError(f"--h5-key is required for {path}")
            with h5py.File(path, "r") as h5:
                value = np.asarray(h5[h5_key][:], dtype=np.float64)
        else:
            value = np.asarray(np.genfromtxt(path, delimiter=",", skip_header=1), dtype=np.float64)
            if value.ndim == 1:
                value = value.reshape(1, -1)
        if value.ndim != 2 or value.shape[1] != 6:
            raise ValueError(f"Expected [N,6] in {path}, got {value.shape}")
        if len(value) < 2 or not np.isfinite(value).all():
            raise ValueError(f"Invalid or too-short wrench stream in {path}")
        result.append((str(path), value))
    return result


def split_episodes(items: list[tuple[str, np.ndarray]], fraction: float, seed: int) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    if len(items) < 5:
        raise ValueError(f"Need at least five episodes, got {len(items)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_test = max(1, int(round(len(items) * fraction)))
    test_ids = set(order[:n_test].tolist())
    train = [item for i, item in enumerate(items) if i not in test_ids]
    test = [item for i, item in enumerate(items) if i in test_ids]
    return train, test


def concat(items: list[tuple[str, np.ndarray]]) -> np.ndarray:
    return np.concatenate([value for _, value in items], axis=0)


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = (np.quantile(values, 0.99, axis=0) - np.quantile(values, 0.01, axis=0)) / 2.0
    scale = np.where(scale > 1e-5, scale, np.std(values, axis=0))
    return center, np.maximum(scale, 1e-3)


def _symmetrize(value: np.ndarray) -> np.ndarray:
    return (value + value.T) / 2.0


def _sqrt_and_inv_sqrt(covariance: np.ndarray, shrinkage: float) -> tuple[np.ndarray, np.ndarray]:
    covariance = _symmetrize(covariance)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(covariance + 1e-5 * np.eye(covariance.shape[0]))
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inv_root = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return root, inv_root


def _fit_coral(source: np.ndarray, target: np.ndarray, blocks: tuple[slice, ...], shrinkage: float) -> dict[str, object]:
    source_center, _ = robust_location_scale(source)
    target_center, _ = robust_location_scale(target)
    transforms = []
    for block in blocks:
        source_cov = np.cov((source[:, block] - source_center[block]).T)
        target_cov = np.cov((target[:, block] - target_center[block]).T)
        target_root, _ = _sqrt_and_inv_sqrt(target_cov, shrinkage)
        _, source_inv_root = _sqrt_and_inv_sqrt(source_cov, shrinkage)
        transforms.append(source_inv_root @ target_root)
    return {"source_center": source_center, "target_center": target_center, "blocks": blocks, "transforms": transforms}


def _apply_coral(values: np.ndarray, fit: dict[str, object]) -> np.ndarray:
    output = np.empty_like(values)
    source_center = fit["source_center"]
    target_center = fit["target_center"]
    for block, transform in zip(fit["blocks"], fit["transforms"], strict=True):
        output[:, block] = (values[:, block] - source_center[block]) @ transform + target_center[block]
    return output


def _fit_affine(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    source_center, source_scale = robust_location_scale(source)
    target_center, target_scale = robust_location_scale(target)
    return {
        "source_center": source_center,
        "source_scale": source_scale,
        "target_center": target_center,
        "target_scale": target_scale,
    }


def _apply_affine(values: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    return (values - fit["source_center"]) / fit["source_scale"] * fit["target_scale"] + fit["target_center"]


def _fit_quantile(source: np.ndarray, target: np.ndarray, grid_size: int = 2049) -> dict[str, np.ndarray]:
    q = np.linspace(0.0, 1.0, grid_size)
    return {
        "source_quantiles": np.quantile(source, q, axis=0),
        "target_quantiles": np.quantile(target, q, axis=0),
    }


def _apply_quantile(values: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    output = np.empty_like(values)
    q = np.linspace(0.0, 1.0, fit["source_quantiles"].shape[0])
    for channel in range(6):
        source_grid = fit["source_quantiles"][:, channel]
        target_grid = fit["target_quantiles"][:, channel]
        # np.interp accepts repeated source quantiles and is stable for the
        # small/constant channels present in the wrench data.
        output[:, channel] = np.interp(values[:, channel], source_grid, target_grid)
    return output


def _normal_score(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    q = np.linspace(0.0, 1.0, len(quantiles))
    rank = np.empty_like(values)
    for channel in range(6):
        rank[:, channel] = np.interp(values[:, channel], quantiles[:, channel], q)
    return ndtri(np.clip(rank, 1e-4, 1.0 - 1e-4))


def _fit_copula(source: np.ndarray, target: np.ndarray, shrinkage: float) -> dict[str, object]:
    q = np.linspace(0.0, 1.0, 2049)
    source_quantiles = np.quantile(source, q, axis=0)
    target_quantiles = np.quantile(target, q, axis=0)
    source_z = _normal_score(source, source_quantiles)
    target_z = _normal_score(target, target_quantiles)
    fit = _fit_coral(source_z, target_z, (slice(0, 3), slice(3, 6)), shrinkage)
    fit["source_quantiles"] = source_quantiles
    fit["target_quantiles"] = target_quantiles
    fit["q"] = q
    return fit


def _apply_copula(values: np.ndarray, fit: dict[str, object]) -> np.ndarray:
    source_z = _normal_score(values, fit["source_quantiles"])
    target_z = _apply_coral(source_z, fit)
    output = np.empty_like(values)
    q = fit["q"]
    for channel in range(6):
        # Convert Gaussian score to target CDF, then use target empirical
        # quantiles.  This matches target marginals while retaining adapted
        # dependence between channels.
        target_rank = ndtr(target_z[:, channel])
        output[:, channel] = np.interp(target_rank, q, fit["target_quantiles"][:, channel])
    return output


def _covariance(values: np.ndarray) -> np.ndarray:
    return np.cov(values.T)


def _corr(values: np.ndarray) -> np.ndarray:
    covariance = _covariance(values)
    scale = np.sqrt(np.maximum(np.diag(covariance), EPS))
    return covariance / np.outer(scale, scale)


def _rbf_mmd(left: np.ndarray, right: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(left), len(right), 800)
    left = left[rng.choice(len(left), n, replace=False)]
    right = right[rng.choice(len(right), n, replace=False)]
    d_ll = ((left[:, None, :] - left[None, :, :]) ** 2).mean(axis=-1)
    d_rr = ((right[:, None, :] - right[None, :, :]) ** 2).mean(axis=-1)
    d_lr = ((left[:, None, :] - right[None, :, :]) ** 2).mean(axis=-1)
    result = 0.0
    for bandwidth in (0.5, 1.0, 2.0, 4.0):
        result += float(np.exp(-d_ll / (2.0 * bandwidth**2)).mean())
        result += float(np.exp(-d_rr / (2.0 * bandwidth**2)).mean())
        result -= 2.0 * float(np.exp(-d_lr / (2.0 * bandwidth**2)).mean())
    return result / 4.0


def _sampled_wasserstein(left: np.ndarray, right: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(left), len(right), 2000)
    left = left[rng.choice(len(left), n, replace=False)]
    right = right[rng.choice(len(right), n, replace=False)]
    return float(wasserstein_distance(left, right))


def _episode_norms(items: list[tuple[str, np.ndarray]], adapted: list[np.ndarray], first: int, last: int) -> np.ndarray:
    return np.asarray([np.mean(np.linalg.norm(value[:, first:last], axis=1)) for value in adapted], dtype=np.float64)


def _metrics(
    adapted_items: list[np.ndarray],
    target_items: list[tuple[str, np.ndarray]],
    target_train: np.ndarray,
    seed: int,
) -> dict[str, object]:
    adapted = np.concatenate(adapted_items)
    target = concat(target_items)
    _, target_scale = robust_location_scale(target_train)
    adapted_z = (adapted - np.median(target_train, axis=0)) / target_scale
    target_z = (target - np.median(target_train, axis=0)) / target_scale
    adapted_delta = np.concatenate([np.diff(value, axis=0) for value in adapted_items])
    target_delta = np.concatenate([np.diff(value, axis=0) for _, value in target_items])
    delta_scale = np.maximum(np.std(np.concatenate([adapted_delta, target_delta]), axis=0), 1e-3)
    adapted_delta_z = adapted_delta / delta_scale
    target_delta_z = target_delta / delta_scale
    cov_gap = np.mean(np.abs(_covariance(adapted_z) - _covariance(target_z)))
    corr_gap = np.mean(np.abs(_corr(adapted_z) - _corr(target_z)))
    delta_cov_gap = np.mean(np.abs(_covariance(adapted_delta_z) - _covariance(target_delta_z)))
    per_channel_ks = [ks_2samp(adapted[:, i], target[:, i]).statistic for i in range(6)]
    per_channel_wasserstein = [
        _sampled_wasserstein(adapted_z[:, i], target_z[:, i], seed + i) for i in range(6)
    ]
    force_norm = np.linalg.norm(adapted[:, :3], axis=1)
    target_force_norm = np.linalg.norm(target[:, :3], axis=1)
    torque_norm = np.linalg.norm(adapted[:, 3:], axis=1)
    target_torque_norm = np.linalg.norm(target[:, 3:], axis=1)
    force_scale = max(float(np.quantile(target_force_norm, 0.75) - np.quantile(target_force_norm, 0.25)), 1e-3)
    torque_scale = max(float(np.quantile(target_torque_norm, 0.75) - np.quantile(target_torque_norm, 0.25)), 1e-3)
    target_force_p90 = float(np.quantile(target_force_norm, 0.90))
    target_torque_p90 = float(np.quantile(target_torque_norm, 0.90))
    adapted_episode_force = _episode_norms(target_items, adapted_items, 0, 3)
    target_episode_force = _episode_norms(target_items, [value for _, value in target_items], 0, 3)
    adapted_episode_torque = _episode_norms(target_items, adapted_items, 3, 6)
    target_episode_torque = _episode_norms(target_items, [value for _, value in target_items], 3, 6)
    mean_gap = float(np.mean(np.abs(adapted_z.mean(0) - target_z.mean(0))))
    std_gap = float(np.mean(np.abs(adapted_z.std(0) - target_z.std(0))))
    delta_mean_gap = float(np.mean(np.abs(adapted_delta_z.mean(0) - target_delta_z.mean(0))))
    delta_std_gap = float(np.mean(np.abs(adapted_delta_z.std(0) - target_delta_z.std(0))))
    mmd = _rbf_mmd(adapted_z, target_z, seed)
    ks_mean = float(np.mean(per_channel_ks))
    wasserstein_mean = float(np.mean(per_channel_wasserstein))
    force_norm_median_gap = abs(float(np.median(force_norm) - np.median(target_force_norm))) / force_scale
    force_norm_p95_gap = abs(float(np.quantile(force_norm, 0.95) - np.quantile(target_force_norm, 0.95))) / force_scale
    torque_norm_median_gap = abs(float(np.median(torque_norm) - np.median(target_torque_norm))) / torque_scale
    torque_norm_p95_gap = abs(float(np.quantile(torque_norm, 0.95) - np.quantile(target_torque_norm, 0.95))) / torque_scale
    contact_rate_gap = abs(float(np.mean(force_norm >= target_force_p90) - np.mean(target_force_norm >= target_force_p90)))
    torque_contact_rate_gap = abs(float(np.mean(torque_norm >= target_torque_p90) - np.mean(target_torque_norm >= target_torque_p90)))
    episode_force_wasserstein = _sampled_wasserstein(adapted_episode_force, target_episode_force, seed + 100)
    episode_torque_wasserstein = _sampled_wasserstein(adapted_episode_torque, target_episode_torque, seed + 101)
    alignment_score = (
        mean_gap + std_gap + cov_gap + corr_gap + mmd + ks_mean + wasserstein_mean
        + delta_mean_gap + delta_std_gap + delta_cov_gap
        + 0.5 * (force_norm_median_gap + force_norm_p95_gap + torque_norm_median_gap + torque_norm_p95_gap)
        + contact_rate_gap + torque_contact_rate_gap
    )
    return {
        "frames": int(len(adapted)),
        "alignment_score_lower_is_better": float(alignment_score),
        "mean_gap_z": mean_gap,
        "std_gap_z": std_gap,
        "covariance_gap_z": float(cov_gap),
        "correlation_gap": corr_gap,
        "joint_rbf_mmd": float(mmd),
        "per_channel_ks_mean": ks_mean,
        "per_channel_ks": {name: float(value) for name, value in zip(CHANNELS, per_channel_ks, strict=True)},
        "per_channel_wasserstein_z_mean": wasserstein_mean,
        "per_channel_wasserstein_z": {name: float(value) for name, value in zip(CHANNELS, per_channel_wasserstein, strict=True)},
        "delta_mean_gap_z": delta_mean_gap,
        "delta_std_gap_z": delta_std_gap,
        "delta_covariance_gap_z": float(delta_cov_gap),
        "force_norm_median": float(np.median(force_norm)),
        "target_force_norm_median": float(np.median(target_force_norm)),
        "force_norm_p95": float(np.quantile(force_norm, 0.95)),
        "target_force_norm_p95": float(np.quantile(target_force_norm, 0.95)),
        "torque_norm_median": float(np.median(torque_norm)),
        "target_torque_norm_median": float(np.median(target_torque_norm)),
        "torque_norm_p95": float(np.quantile(torque_norm, 0.95)),
        "target_torque_norm_p95": float(np.quantile(target_torque_norm, 0.95)),
        "force_norm_median_gap_iqr": float(force_norm_median_gap),
        "force_norm_p95_gap_iqr": float(force_norm_p95_gap),
        "torque_norm_median_gap_iqr": float(torque_norm_median_gap),
        "torque_norm_p95_gap_iqr": float(torque_norm_p95_gap),
        "contact_rate_gap_at_target_p90": contact_rate_gap,
        "torque_rate_gap_at_target_p90": torque_contact_rate_gap,
        "episode_force_mean_wasserstein": episode_force_wasserstein,
        "episode_torque_mean_wasserstein": episode_torque_wasserstein,
    }


def _method_fits(source_train: np.ndarray, target_train: np.ndarray) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    affine = _fit_affine(source_train, target_train)
    block = _fit_coral(source_train, target_train, (slice(0, 3), slice(3, 6)), shrinkage=0.10)
    full = _fit_coral(source_train, target_train, (slice(0, 6),), shrinkage=0.10)
    quantile = _fit_quantile(source_train, target_train)
    copula = _fit_copula(source_train, target_train, shrinkage=0.10)
    affine_transform = lambda value: _apply_affine(value, affine)
    copula_transform = lambda value: _apply_copula(value, copula)

    def blended(value: np.ndarray, amount: float) -> np.ndarray:
        return (1.0 - amount) * affine_transform(value) + amount * copula_transform(value)

    return {
        "identity": lambda value: value.copy(),
        "robust_affine": affine_transform,
        "coral_block": lambda value: _apply_coral(value, block),
        "coral_full": lambda value: _apply_coral(value, full),
        "quantile": lambda value: _apply_quantile(value, quantile),
        "copula_block": copula_transform,
        "copula_affine_blend_25": lambda value: blended(value, 0.25),
        "copula_affine_blend_50": lambda value: blended(value, 0.50),
        "copula_affine_blend_75": lambda value: blended(value, 0.75),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data-sim-wrench-final-hdf5"))
    parser.add_argument("--source-pattern", default="episode_*/data.h5")
    parser.add_argument("--source-h5-key", default="decision/obs/state/wrench_final")
    parser.add_argument("--target-dir", type=Path, default=Path("data"))
    parser.add_argument("--target-pattern", default="traj_*/data.h5")
    parser.add_argument("--target-h5-key", default="obs/state/ee_wrench_base")
    parser.add_argument("--mlp-adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/wrench_alignment/method_comparison"))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    source = load_episodes(args.source_dir, args.source_pattern, args.source_h5_key)
    target = load_episodes(args.target_dir, args.target_pattern, args.target_h5_key)
    source_train_items, source_test_items = split_episodes(source, args.test_fraction, args.seed)
    target_train_items, target_test_items = split_episodes(target, args.test_fraction, args.seed + 1)
    source_train = concat(source_train_items)
    target_train = concat(target_train_items)
    source_test = concat(source_test_items)
    target_test = concat(target_test_items)
    methods = _method_fits(source_train, target_train)
    if args.mlp_adapter is not None:
        mlp = load_adapter(args.mlp_adapter)
        methods["polyfit_mlp_existing"] = mlp.transform_numpy

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": {
            "source_train_episodes": len(source_train_items),
            "source_test_episodes": len(source_test_items),
            "target_train_episodes": len(target_train_items),
            "target_test_episodes": len(target_test_items),
            "source_train_frames": len(source_train),
            "source_test_frames": len(source_test),
            "target_train_frames": len(target_train),
            "target_test_frames": len(target_test),
            "seed": args.seed,
            "note": "Unpaired episode-disjoint evaluation; no frame-wise MAE is valid.",
        },
        "methods": {},
    }
    for name, transform in methods.items():
        adapted_items = [transform(value) for _, value in source_test_items]
        if not all(np.isfinite(value).all() for value in adapted_items):
            raise FloatingPointError(f"Method {name} produced NaN/Inf")
        metrics = _metrics(adapted_items, target_test_items, target_train, args.seed)
        report["methods"][name] = metrics
        np.save(args.output_dir / f"{name}_heldout.npy", np.concatenate(adapted_items))
        print(f"{name}: score={metrics['alignment_score_lower_is_better']:.6f} mmd={metrics['joint_rbf_mmd']:.6f} ks={metrics['per_channel_ks_mean']:.6f}")

    ranking = sorted(
        ((name, value["alignment_score_lower_is_better"]) for name, value in report["methods"].items()),
        key=lambda item: item[1],
    )
    report["ranking_lower_is_better"] = [{"method": name, "score": float(score)} for name, score in ranking]
    report["best_method"] = ranking[0][0]
    report["best_method_selection"] = "lowest composite held-out distribution score; inspect physical constraints before deployment"
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"best_method": report["best_method"], "ranking": report["ranking_lower_is_better"]}, indent=2))


if __name__ == "__main__":
    main()
