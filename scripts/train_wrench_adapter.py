"""Train a PolyFit-style wrench DLA on an explicitly paired dataset.

The input HDF5 must contain aligned rows:

    real_wrench [N, 6]   (or the key passed with --source-key)
    sim_wrench  [N, 6]   (or the key passed with --target-key)
    episode_index [N]    (required; used for episode-level splitting)

``--direction sim_to_real`` is recommended for TAVLA: adapt simulation
wrench into the real-domain convention before sim policy training.  Use
``real_to_sim`` only when reproducing the original PolyFit DLA deployment
direction.  This script never treats unrelated real and simulation episodes as
paired examples.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from openpi.shared.wrench_adapter import PolyFitWrenchDLA
from openpi.shared.wrench_adapter import WrenchAdapterConfig
from openpi.shared.wrench_adapter import robust_location_scale
from openpi.shared.wrench_adapter import save_adapter


def _read_dataset(h5: h5py.File, key: str) -> np.ndarray:
    if key not in h5:
        raise KeyError(f"Missing HDF5 dataset {key!r}; available top-level keys: {list(h5.keys())}")
    value = np.asarray(h5[key][:], dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError(f"{key} must have shape [N, 6], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains NaN/Inf")
    return value


def _read_paired(path: Path, source_key: str, target_key: str, episode_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        source = _read_dataset(h5, source_key)
        target = _read_dataset(h5, target_key)
        if episode_key not in h5:
            raise KeyError(
                f"Missing {episode_key!r}. Paired data must include episode IDs so validation cannot leak frames "
                "from the same trajectory into training."
            )
        episodes = np.asarray(h5[episode_key][:]).reshape(-1)
    if source.shape != target.shape or source.shape[0] != len(episodes):
        raise ValueError(f"Paired shape mismatch: source={source.shape}, target={target.shape}, episodes={episodes.shape}")
    return source, target, episodes


def _split_episode_indices(episodes: np.ndarray, val_fraction: float, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(episodes)
    if len(unique) < 5:
        raise ValueError(f"Need at least five distinct episodes for train/val/test splitting, got {len(unique)}")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    n_val = max(1, int(round(len(unique) * val_fraction)))
    if n_test + n_val >= len(unique):
        raise ValueError("val_fraction + test_fraction leaves no training episodes")
    return shuffled[: len(unique) - n_val - n_test], shuffled[len(unique) - n_val - n_test : len(unique) - n_test], shuffled[len(unique) - n_test :]


def _mask_for(episodes: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.isin(episodes, selected)


def _metrics(model: PolyFitWrenchDLA, source: np.ndarray, target: np.ndarray, device: str) -> dict[str, object]:
    prediction = model.transform_numpy(source, device=device)
    error = np.abs(prediction - target)
    return {
        "frames": int(len(source)),
        "mean_mae": float(error.mean()),
        "median_mae": float(np.median(error)),
        "p95_frame_mae": float(np.quantile(error.mean(axis=1), 0.95)),
        "per_dimension_mae": {name: float(value) for name, value in zip(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"), error.mean(axis=0), strict=True)},
        "max_abs_error": float(error.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-key", help="Input wrench dataset key; defaults from --direction")
    parser.add_argument("--target-key", help="Paired target wrench dataset key; defaults from --direction")
    parser.add_argument("--episode-key", default="episode_index")
    parser.add_argument("--direction", choices=("sim_to_real", "real_to_sim"), default="sim_to_real")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--residual-limit", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="Number of evals without improvement before stopping")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--residual-penalty", type=float, default=0.01)
    parser.add_argument("--input-noise", type=float, default=0.01, help="Gaussian noise in source robust-scale units")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto; the adapter is small and CPU is sufficient")
    args = parser.parse_args()

    if args.source_key is None:
        args.source_key = "sim_wrench" if args.direction == "sim_to_real" else "real_wrench"
    if args.target_key is None:
        args.target_key = "real_wrench" if args.direction == "sim_to_real" else "sim_wrench"

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    source, target, episode_ids = _read_paired(args.paired_hdf5, args.source_key, args.target_key, args.episode_key)
    train_eps, val_eps, test_eps = _split_episode_indices(episode_ids, args.val_fraction, args.test_fraction, args.seed)
    train_mask = _mask_for(episode_ids, train_eps)
    val_mask = _mask_for(episode_ids, val_eps)
    test_mask = _mask_for(episode_ids, test_eps)

    source_center, source_scale = robust_location_scale(source[train_mask])
    target_center, target_scale = robust_location_scale(target[train_mask])
    config = WrenchAdapterConfig(
        direction=args.direction,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        residual_limit=args.residual_limit,
    )
    model = PolyFitWrenchDLA(source_center, source_scale, target_center, target_scale, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(source[train_mask]), torch.from_numpy(target[train_mask])),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    iterator = itertools.cycle(loader)
    source_noise_scale = torch.from_numpy(source_scale).to(device)
    best_value = float("inf")
    best_step = 0
    stale_evals = 0
    history: list[dict[str, float]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        model.train()
        x, y = next(iterator)
        x = x.to(device)
        y = y.to(device)
        if args.input_noise > 0:
            x = x + torch.randn_like(x) * source_noise_scale * args.input_noise
        prediction = model(x)
        residual = model.normalized_residual(x)
        loss_data = torch.abs(prediction - y).mean()
        loss_regularizer = residual.square().mean()
        loss = loss_data + args.residual_penalty * loss_regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step % args.eval_every != 0 and step != args.steps:
            continue
        model.eval()
        val_metrics = _metrics(model, source[val_mask], target[val_mask], str(device))
        value = float(val_metrics["mean_mae"])
        history.append({"step": float(step), "train_loss": float(loss.item()), "val_mae": value})
        print(f"step={step} train_loss={loss.item():.6f} val_mae={value:.6f}")
        if value < best_value:
            best_value = value
            best_step = step
            stale_evals = 0
            save_adapter(
                model,
                args.output_dir / "best.pt",
                train_frames=int(train_mask.sum()),
                train_episodes=int(len(train_eps)),
                val_frames=int(val_mask.sum()),
                val_episodes=int(len(val_eps)),
            )
        else:
            stale_evals += 1
            if stale_evals >= args.patience:
                print(f"early stop at step={step}; best_step={best_step}")
                break

    save_adapter(
        model,
        args.output_dir / "last.pt",
        train_frames=int(train_mask.sum()),
        train_episodes=int(len(train_eps)),
        val_frames=int(val_mask.sum()),
        val_episodes=int(len(val_eps)),
    )
    model.load_state_dict(torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)["state_dict"])
    model.eval()
    report = {
        "paired_hdf5": str(args.paired_hdf5),
        "direction": args.direction,
        "seed": args.seed,
        "episodes": {
            "train": train_eps.tolist(),
            "val": val_eps.tolist(),
            "test": test_eps.tolist(),
        },
        "best_step": best_step,
        "best_val_mae": best_value,
        "validation": _metrics(model, source[val_mask], target[val_mask], str(device)),
        "held_out_test": _metrics(model, source[test_mask], target[test_mask], str(device)),
        "history": history,
        "warning": "DLA requires genuinely paired sim-real rows; unrelated episodes must not be used as pairs.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"best_step": best_step, "validation": report["validation"], "held_out_test": report["held_out_test"]}, indent=2))


if __name__ == "__main__":
    main()
