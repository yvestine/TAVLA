"""Train a PolyFit-inspired wrench adapter without frame-level pairing.

This is deliberately different from supervised PolyFit DLA.  The source and
target streams may come from different trajectories.  The adapter is trained
with distribution-level objectives:

* robust mean/std and covariance matching in the target wrench domain;
* RBF-MMD matching for the joint force/torque distribution;
* matching of consecutive-frame difference statistics from independent
  source/target episodes;
* a bounded residual penalty around the robust affine baseline.

The model still has separate force and torque encoders followed by a fused
feature block, so it keeps PolyFit's useful inductive bias without pretending
that unrelated frames are supervised pairs.  Inputs remain only six wrench
components, making the adapter task-agnostic.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from openpi.shared.wrench_adapter import PolyFitWrenchDLA
from openpi.shared.wrench_adapter import WrenchAdapterConfig
from openpi.shared.wrench_adapter import robust_location_scale
from openpi.shared.wrench_adapter import save_adapter


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def _read_csv(path: Path) -> np.ndarray:
    value = np.asarray(np.genfromtxt(path, delimiter=",", skip_header=1), dtype=np.float32)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    return value


def _load_episodes(root: Path, pattern: str, h5_key: str | None) -> list[tuple[str, np.ndarray]]:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {root / pattern}")
    result = []
    for path in files:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            if h5_key is None:
                raise ValueError(f"--h5-key is required for {path}")
            with h5py.File(path, "r") as h5:
                value = np.asarray(h5[h5_key][:], dtype=np.float32)
        else:
            value = _read_csv(path)
        if value.ndim != 2 or value.shape[1] != 6:
            raise ValueError(f"Expected [N, 6] in {path}, got {value.shape}")
        if len(value) < 2:
            raise ValueError(f"Episode {path} must contain at least two frames")
        if not np.isfinite(value).all():
            raise ValueError(f"NaN/Inf in {path}")
        result.append((str(path), value))
    return result


def _split(items: list[tuple[str, np.ndarray]], fraction: float, seed: int) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    if len(items) < 5:
        raise ValueError(f"Need at least five episodes for a held-out split, got {len(items)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_test = max(1, int(round(len(items) * fraction)))
    test = [items[index] for index in order[:n_test]]
    train = [items[index] for index in order[n_test:]]
    return train, test


class EpisodeSampler:
    def __init__(self, items: list[tuple[str, np.ndarray]], device: torch.device, seed: int) -> None:
        self.values = [torch.from_numpy(value).to(device) for _, value in items]
        self.rng = np.random.default_rng(seed)

    def batch(self, size: int) -> torch.Tensor:
        episodes = self.rng.integers(0, len(self.values), size=size)
        rows = [self.values[index][int(self.rng.integers(0, len(self.values[index])))] for index in episodes]
        return torch.stack(rows)

    def adjacent_batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        episodes = self.rng.integers(0, len(self.values), size=size)
        left = []
        right = []
        for index in episodes:
            episode = self.values[index]
            start = int(self.rng.integers(0, len(episode) - 1))
            left.append(episode[start])
            right.append(episode[start + 1])
        return torch.stack(left), torch.stack(right)


def _standardize(value: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (value - center) / scale


def _moment_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean = (prediction.mean(0) - target.mean(0)).square().mean()
    std = (prediction.std(0, unbiased=False) - target.std(0, unbiased=False)).square().mean()
    return mean + std


def _covariance(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(0, keepdim=True)
    return centered.T @ centered / max(value.shape[0] - 1, 1)


def _coral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (_covariance(prediction) - _covariance(target)).square().mean()


def _mmd_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RBF MMD with fixed bandwidths in robust-standardized coordinates."""

    distance_pp = torch.cdist(prediction, prediction).square() / prediction.shape[-1]
    distance_qq = torch.cdist(target, target).square() / target.shape[-1]
    distance_pq = torch.cdist(prediction, target).square() / prediction.shape[-1]
    result = prediction.new_zeros(())
    for bandwidth in (0.5, 1.0, 2.0, 4.0):
        result = result + (
            torch.exp(-distance_pp / (2 * bandwidth**2)).mean()
            + torch.exp(-distance_qq / (2 * bandwidth**2)).mean()
            - 2 * torch.exp(-distance_pq / (2 * bandwidth**2)).mean()
        )
    return result / 4.0


def _score(prediction: np.ndarray, target: np.ndarray, target_center: np.ndarray, target_scale: np.ndarray) -> dict[str, object]:
    prediction_z = (prediction - target_center) / target_scale
    target_z = (target - target_center) / target_scale
    mean_gap = np.abs(prediction_z.mean(0) - target_z.mean(0)).mean()
    std_gap = np.abs(prediction_z.std(0) - target_z.std(0)).mean()
    covariance_gap = np.abs(np.cov(prediction_z.T) - np.cov(target_z.T)).mean()
    if prediction.shape == target.shape:
        error = np.abs(prediction - target)
        mean_mae: float | None = float(error.mean())
        per_dimension_mae: dict[str, float] | None = {
            name: float(value) for name, value in zip(CHANNELS, error.mean(0), strict=True)
        }
    else:
        mean_mae = None
        per_dimension_mae = None
    return {
        "frames": int(len(prediction)),
        "alignment_score": float(mean_gap + std_gap + covariance_gap),
        "mean_gap_z": float(mean_gap),
        "std_gap_z": float(std_gap),
        "covariance_gap_z": float(covariance_gap),
        "mean_mae": mean_mae,
        "per_dimension_mae": per_dimension_mae,
        "force_norm_median": float(np.median(np.linalg.norm(prediction[:, :3], axis=1))),
        "target_force_norm_median": float(np.median(np.linalg.norm(target[:, :3], axis=1))),
        "torque_norm_median": float(np.median(np.linalg.norm(prediction[:, 3:], axis=1))),
        "target_torque_norm_median": float(np.median(np.linalg.norm(target[:, 3:], axis=1))),
    }


def _concat(items: list[tuple[str, np.ndarray]]) -> np.ndarray:
    return np.concatenate([value for _, value in items], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-pattern", required=True)
    parser.add_argument("--source-h5-key")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--target-pattern", required=True)
    parser.add_argument("--target-h5-key")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction", choices=("sim_to_real",), default="sim_to_real")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--residual-limit", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--moment-weight", type=float, default=1.0)
    parser.add_argument("--coral-weight", type=float, default=0.20)
    parser.add_argument("--mmd-weight", type=float, default=0.50)
    parser.add_argument("--delta-weight", type=float, default=0.30)
    parser.add_argument("--baseline-weight", type=float, default=0.05)
    parser.add_argument("--residual-penalty", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    source_items = _load_episodes(args.source_dir, args.source_pattern, args.source_h5_key)
    target_items = _load_episodes(args.target_dir, args.target_pattern, args.target_h5_key)
    source_fit, source_test = _split(source_items, args.test_fraction, args.seed)
    target_fit, target_test = _split(target_items, args.test_fraction, args.seed + 1)
    source_train, source_val = _split(source_fit, args.test_fraction, args.seed + 2)
    target_train, target_val = _split(target_fit, args.test_fraction, args.seed + 3)
    source_train_values = _concat(source_train)
    target_train_values = _concat(target_train)
    source_center, source_scale = robust_location_scale(source_train_values)
    target_center, target_scale = robust_location_scale(target_train_values)
    config = WrenchAdapterConfig(
        direction=args.direction,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        residual_limit=args.residual_limit,
    )
    model = PolyFitWrenchDLA(source_center, source_scale, target_center, target_scale, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    source_sampler = EpisodeSampler(source_train, device, args.seed)
    target_sampler = EpisodeSampler(target_train, device, args.seed + 1)
    iterator = itertools.count(1)
    source_center_t = torch.from_numpy(source_center).to(device)
    source_scale_t = torch.from_numpy(source_scale).to(device)
    target_center_t = torch.from_numpy(target_center).to(device)
    target_scale_t = torch.from_numpy(target_scale).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_score = float("inf")
    best_step = 0
    stale_evals = 0
    history: list[dict[str, float]] = []
    for _ in range(args.steps):
        step = next(iterator)
        model.train()
        source_batch = source_sampler.batch(args.batch_size)
        target_batch = target_sampler.batch(args.batch_size)
        prediction = model(source_batch)
        prediction_z = _standardize(prediction, target_center_t, target_scale_t)
        target_z = _standardize(target_batch, target_center_t, target_scale_t)
        loss_moment = _moment_loss(prediction_z, target_z)
        loss_coral = _coral_loss(prediction_z, target_z)
        loss_mmd = _mmd_loss(torch.clamp(prediction_z, -8, 8), torch.clamp(target_z, -8, 8))

        source_left, source_right = source_sampler.adjacent_batch(args.batch_size)
        target_left, target_right = target_sampler.adjacent_batch(args.batch_size)
        prediction_delta_z = (model(source_right) - model(source_left)) / target_scale_t
        target_delta_z = (target_right - target_left) / target_scale_t
        loss_delta = _moment_loss(prediction_delta_z, target_delta_z)

        baseline = (source_batch - source_center_t) / source_scale_t * target_scale_t + target_center_t
        loss_baseline = ((prediction - baseline) / target_scale_t).square().mean()
        loss_residual = model.normalized_residual(source_batch).square().mean()
        loss = (
            args.moment_weight * loss_moment
            + args.coral_weight * loss_coral
            + args.mmd_weight * loss_mmd
            + args.delta_weight * loss_delta
            + args.baseline_weight * loss_baseline
            + args.residual_penalty * loss_residual
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step % args.eval_every != 0 and step != args.steps:
            continue
        model.eval()
        with torch.no_grad():
            source_val_values_t = torch.from_numpy(_concat(source_val)).to(device)
            prediction_val = model(source_val_values_t).cpu().numpy()
        target_val_values = _concat(target_val)
        metrics = _score(prediction_val, target_val_values, target_center, target_scale)
        score = float(metrics["alignment_score"])
        history.append({"step": float(step), "train_loss": float(loss.item()), "validation_alignment_score": score})
        print(f"step={step} loss={loss.item():.6f} validation_alignment_score={score:.6f}")
        if score < best_score:
            best_score = score
            best_step = step
            stale_evals = 0
            save_adapter(
                model,
                args.output_dir / "best.pt",
                train_frames=int(len(source_train_values)),
                train_episodes=int(len(source_train)),
                val_frames=int(len(_concat(target_val))),
                val_episodes=int(len(target_val)),
            )
        else:
            stale_evals += 1
            if stale_evals >= args.patience:
                print(f"early stop at step={step}; best_step={best_step}")
                break

    save_adapter(
        model,
        args.output_dir / "last.pt",
        train_frames=int(len(source_train_values)),
        train_episodes=int(len(source_train)),
        val_frames=int(len(_concat(target_val))),
        val_episodes=int(len(target_val)),
    )
    payload = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    source_test_values = _concat(source_test)
    target_test_values = _concat(target_test)
    source_train_prediction = model(torch.from_numpy(source_train_values).to(device)).detach().cpu().numpy()
    source_test_prediction = model(torch.from_numpy(source_test_values).to(device)).detach().cpu().numpy()
    report = {
        "mode": "polyfit_inspired_unpaired_distribution_adaptation",
        "direction": args.direction,
        "source_train_episodes": [name for name, _ in source_train],
        "source_validation_episodes": [name for name, _ in source_val],
        "source_test_episodes": [name for name, _ in source_test],
        "target_train_episodes": [name for name, _ in target_train],
        "target_validation_episodes": [name for name, _ in target_val],
        "target_test_episodes": [name for name, _ in target_test],
        "best_step": best_step,
        "best_alignment_score": best_score,
        "train_source_after": _score(source_train_prediction, target_train_values, target_center, target_scale),
        "validation_source_after": _score(
            model(torch.from_numpy(_concat(source_val)).to(device)).detach().cpu().numpy(),
            _concat(target_val),
            target_center,
            target_scale,
        ),
        "held_out_source_after": _score(source_test_prediction, target_test_values, target_center, target_scale),
        "held_out_source_before": _score(
            (source_test_values - source_center) / source_scale * target_scale + target_center,
            target_test_values,
            target_center,
            target_scale,
        ),
        "history": history,
        "warning": "This uses unpaired distribution adaptation inspired by PolyFit; it is not supervised paired DLA.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"best_step": best_step, "held_out_before": report["held_out_source_before"], "held_out_after": report["held_out_source_after"]}, indent=2))


if __name__ == "__main__":
    main()
