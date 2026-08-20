"""Fit and audit a robust affine wrench baseline from unpaired datasets.

This command is intentionally separate from ``train_wrench_adapter.py``.  It
can use the current 50 simulated and 40 real trajectories, but it only aligns
per-channel marginal location/scale.  It does *not* learn contact-state
correspondence and must not be described as a PolyFit DLA result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openpi.shared.wrench_adapter import build_robust_affine_adapter
from openpi.shared.wrench_adapter import save_adapter


def _load_file(path: Path, h5_key: str | None) -> np.ndarray:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        if h5_key is None:
            raise ValueError(f"--h5-key is required for {path}")
        with h5py.File(path, "r") as h5:
            value = np.asarray(h5[h5_key][:], dtype=np.float32)
    else:
        value = np.asarray(np.genfromtxt(path, delimiter=",", skip_header=1), dtype=np.float32)
        if value.ndim == 1:
            value = value.reshape(1, -1)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError(f"Expected [N, 6] wrench values in {path}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"NaN/Inf in {path}")
    return value


def _load_episodes(root: Path, pattern: str, h5_key: str | None) -> list[tuple[Path, np.ndarray]]:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {root / pattern}")
    return [(path, _load_file(path, h5_key)) for path in files]


def _split(values: list[tuple[Path, np.ndarray]], test_fraction: float, seed: int) -> tuple[list[tuple[Path, np.ndarray]], list[tuple[Path, np.ndarray]]]:
    if len(values) < 5:
        raise ValueError("Need at least five episodes for a held-out distribution audit")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(values))
    n_test = max(1, int(round(len(values) * test_fraction)))
    test_indices = set(order[:n_test].tolist())
    train = [value for index, value in enumerate(values) if index not in test_indices]
    test = [value for index, value in enumerate(values) if index in test_indices]
    return train, test


def _summary(values: np.ndarray) -> dict[str, object]:
    return {
        "frames": int(len(values)),
        "mean": values.mean(axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "p95_abs": np.quantile(np.abs(values), 0.95, axis=0).tolist(),
        "force_norm_median": float(np.median(np.linalg.norm(values[:, :3], axis=1))),
        "force_norm_p95": float(np.quantile(np.linalg.norm(values[:, :3], axis=1), 0.95)),
        "torque_norm_median": float(np.median(np.linalg.norm(values[:, 3:], axis=1))),
        "torque_norm_p95": float(np.quantile(np.linalg.norm(values[:, 3:], axis=1), 0.95)),
    }


def _concat(items: list[tuple[Path, np.ndarray]]) -> np.ndarray:
    return np.concatenate([value for _, value in items], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-pattern", required=True, help="For example episode_*/wrench_final.csv")
    parser.add_argument("--source-h5-key")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--target-pattern", required=True, help="For example traj_*/data.h5")
    parser.add_argument("--target-h5-key")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    source_items = _load_episodes(args.source_dir, args.source_pattern, args.source_h5_key)
    target_items = _load_episodes(args.target_dir, args.target_pattern, args.target_h5_key)
    source_train, source_test = _split(source_items, args.test_fraction, args.seed)
    target_train, target_test = _split(target_items, args.test_fraction, args.seed + 1)
    source_train_values = _concat(source_train)
    target_train_values = _concat(target_train)
    adapter = build_robust_affine_adapter(source_train_values, target_train_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_adapter(
        adapter,
        args.output,
        train_frames=int(len(source_train_values)),
        train_episodes=int(len(source_train)),
        val_frames=0,
        val_episodes=0,
    )

    source_test_values = _concat(source_test)
    target_test_values = _concat(target_test)
    adapted_source_test = adapter.transform_numpy(source_test_values)
    report = {
        "source": {"root": str(args.source_dir), "pattern": args.source_pattern, "train": _summary(source_train_values), "held_out": _summary(source_test_values)},
        "target": {"root": str(args.target_dir), "pattern": args.target_pattern, "train": _summary(target_train_values), "held_out": _summary(target_test_values)},
        "adapted_source_held_out": _summary(adapted_source_test),
        "held_out_marginal_gap_before": np.abs(source_test_values.mean(axis=0) - target_test_values.mean(axis=0)).tolist(),
        "held_out_marginal_gap_after": np.abs(adapted_source_test.mean(axis=0) - target_test_values.mean(axis=0)).tolist(),
        "warning": "Unpaired affine alignment is a baseline only; it does not establish frame-level sim-real correspondence and is not PolyFit DLA.",
    }
    args.output.with_name("unpaired_affine_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
