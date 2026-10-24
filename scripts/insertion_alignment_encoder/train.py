#!/usr/bin/env python3
"""Train/evaluate the force-only insertion-alignment encoder.

When exact per-frame peg geometry is not mounted, the script can use the
previously reconstructed per-trajectory 3 mm cutoff frames. That reconstruction
uses the Sim EE pose, hole position, and a shared fitted tool offset; its error is
recorded in the artifact and geometry regression is disabled in that mode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from insertion_alignment_encoder.model import ForceAlignmentEncoder
from insertion_alignment_encoder.preprocess import (
    force_history_features,
    initial_baseline,
    normalize_features,
)


LABEL_IDS = {"misaligned_contact": 0, "aligned_progress": 1}
LABEL_NAMES = ["misaligned_contact", "aligned_progress"]
# First Sim frame satisfying planar XY <3 mm and |Z|<3 mm, reconstructed from
# the previously inspected ee_pose_sim.csv and replay_metadata hole positions.
# Shared fitted EE-local peg offset: [2.576, 32.243, -0.898] mm; endpoint
# residual p95=2.345 mm, max=2.697 mm. These cutoffs are approximate.
RECONSTRUCTED_CUTOFFS = {
    0: 251, 1: 172, 2: 170, 3: 153, 4: 163, 5: 159, 6: 164, 7: 240, 8: 171, 9: 354,
    10: 307, 11: 173, 12: 249, 13: 279, 14: 141, 15: 121, 16: 219, 17: 255, 18: 127, 19: 137,
    20: 223, 21: 186, 22: 158, 23: 160, 24: 163, 25: 186, 26: 96, 27: 144, 28: 155, 29: 161,
    30: 96, 31: 97, 32: 150, 33: 121, 34: 208, 35: 111, 36: 92, 37: 91, 38: 111, 39: 74,
}


@dataclass
class Episode:
    trajectory: int
    domain: int  # 0=Real, 1=Sim
    force: np.ndarray
    time: np.ndarray
    raw_features: np.ndarray
    geometry: np.ndarray
    cutoff: int


@dataclass
class PairWindows:
    x: np.ndarray  # [N, 2, history, 6], Real then Sim
    y: np.ndarray
    weight: np.ndarray
    geometry: np.ndarray
    trajectory: np.ndarray
    frame: np.ndarray


def _geometry_file(root: Path, trajectory: int) -> Path:
    candidates = [
        root / f"traj_{trajectory}" / "peg_hole_error.csv",
        root / f"traj_{trajectory}.csv",
        root / f"traj_{trajectory}" / "geometry.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"missing explicit peg-tip geometry CSV for traj_{trajectory}; tried "
        + ", ".join(str(p) for p in candidates)
        + ". Expected frame,dx_m,dy_m,dz_m (or dx_mm,dy_mm,dz_mm)."
    )


def read_geometry(root: Path, trajectory: int, expected_length: int) -> np.ndarray:
    combined = root if root.is_file() else root / "peg_hole_error.csv"
    if not combined.is_file():
        combined = _geometry_file(root, trajectory)
    path = combined
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = set(reader.fieldnames or [])
        if {"dx_m", "dy_m", "dz_m"}.issubset(names):
            columns = ("dx_m", "dy_m", "dz_m")
            scale = 1.0
        elif {"dx_mm", "dy_mm", "dz_mm"}.issubset(names):
            columns = ("dx_mm", "dy_mm", "dz_mm")
            scale = 0.001
        else:
            raise ValueError(f"{path}: need dx_m,dy_m,dz_m or dx_mm,dy_mm,dz_mm columns")
        rows = list(reader)
    if "trajectory" in names:
        rows = [row for row in rows if int(str(row["trajectory"]).removeprefix("traj_")) == trajectory]
    if len(rows) != expected_length:
        raise ValueError(f"{path}: {len(rows)} geometry rows != {expected_length} force frames")
    if "frame" in (names or set()):
        frames = np.asarray([int(row["frame"]) for row in rows])
        if not np.array_equal(frames, np.arange(expected_length)):
            raise ValueError(f"{path}: frame column must be exactly 0..{expected_length - 1}")
    values = np.asarray([[float(row[col]) for col in columns] for row in rows], dtype=np.float32) * scale
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: geometry contains NaN or Inf")
    return values


def read_labels(path: Path) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            trajectory = int(row["trajectory"].removeprefix("traj_"))
            result.setdefault(trajectory, []).append(
                {
                    "start": int(row["start_frame"]),
                    "end": int(row["end_frame"]),
                    "label": row["label"],
                    "confidence": row["confidence"],
                }
            )
    return result


def load_episodes(args: argparse.Namespace) -> tuple[dict[tuple[int, int], Episode], dict[int, int], dict[str, Any]]:
    labels = read_labels(args.labels)
    episodes: dict[tuple[int, int], Episode] = {}
    cutoffs: dict[int, int] = {}
    trajectories = sorted(labels)
    geometry_meta: dict[str, Any] = (
        {"mode": "exact_sim_geometry", "supervision": True}
        if args.geometry_root
        else {
            "mode": "reconstructed_cutoffs",
            "supervision": False,
            "tool_offset_local_mm": [2.576, 32.243, -0.898],
            "endpoint_residual_p95_mm": 2.345,
            "endpoint_residual_max_mm": 2.697,
            "note": "cutoffs were derived earlier from ee_pose_sim.csv and per-trajectory hole positions; the raw geometry export is not currently mounted",
        }
    )
    for trajectory in trajectories:
        geom: np.ndarray | None = None
        for domain, root in enumerate((args.real_root, args.sim_root)):
            path = root / f"traj_{trajectory}" / "data.h5"
            with h5py.File(path, "r") as h5:
                force = np.asarray(h5[args.force_key], dtype=np.float32)
                time = np.asarray(h5["timestamps"], dtype=np.float64)
            if force.ndim != 2 or force.shape[1] != 3 or len(time) != len(force):
                raise ValueError(f"{path}: expected force (T,3) and matching timestamps")
            if geom is None:
                if args.geometry_root:
                    geom = read_geometry(args.geometry_root, trajectory, len(force))
                    radial_xy = np.linalg.norm(geom[:, :2], axis=1)
                    hit = np.flatnonzero((radial_xy < args.xy_threshold_m) & (np.abs(geom[:, 2]) < args.z_threshold_m))
                    cutoffs[trajectory] = int(hit[0]) if hit.size else len(force)
                else:
                    if trajectory not in RECONSTRUCTED_CUTOFFS:
                        raise ValueError(f"no reconstructed cutoff for traj_{trajectory}; provide --geometry-root")
                    cutoffs[trajectory] = min(RECONSTRUCTED_CUTOFFS[trajectory], len(force))
                    geom = np.zeros((len(force), 3), dtype=np.float32)
            elif len(force) != len(geom):
                raise ValueError(f"traj_{trajectory}: Real/Sim/geometry frame counts differ")
            if not np.isfinite(force).all() or not np.isfinite(time).all():
                raise ValueError(f"{path}: force or timestamps contain NaN or Inf")
            base = initial_baseline(force, args.baseline_frames)
            features = force_history_features(force, time, base)
            episodes[(trajectory, domain)] = Episode(trajectory, domain, force, time, features, geom, cutoffs[trajectory])

    for trajectory, rows in labels.items():
        n = len(episodes[(trajectory, 0)].force)
        cursor = 0
        for row in rows:
            if row["start"] != cursor or row["end"] < row["start"]:
                raise ValueError(f"traj_{trajectory}: non-contiguous or invalid label intervals near frame {cursor}")
            cursor = row["end"] + 1
        if cursor != n:
            raise ValueError(f"traj_{trajectory}: labels cover {cursor} frames, force data has {n}")
        if len(episodes[(trajectory, 1)].force) != n:
            raise ValueError(f"traj_{trajectory}: Real and Sim frame counts differ")
    return episodes, cutoffs, geometry_meta


def make_windows(
    episodes: dict[tuple[int, int], Episode],
    labels: dict[int, list[dict[str, Any]]],
    trajectories: list[int],
    scales: np.ndarray,
    history: int,
    guard: int,
) -> PairWindows:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    w_rows: list[float] = []
    g_rows: list[np.ndarray] = []
    t_rows: list[int] = []
    f_rows: list[int] = []
    for trajectory in trajectories:
        real = episodes[(trajectory, 0)]
        sim = episodes[(trajectory, 1)]
        real_x = normalize_features(real.raw_features, scales)
        sim_x = normalize_features(sim.raw_features, scales)
        for interval in labels[trajectory]:
            label = interval["label"]
            if label not in LABEL_IDS:
                continue
            start = interval["start"] + guard
            end = min(interval["end"] - guard, real.cutoff - 1)
            if end < start:
                continue
            confidence = 1.0 if interval["confidence"] == "high" else 0.65
            for frame in range(start, end + 1):
                lo = max(0, frame - history + 1)
                xr = real_x[lo : frame + 1]
                xs = sim_x[lo : frame + 1]
                if len(xr) < history:
                    xr = np.pad(xr, ((history - len(xr), 0), (0, 0)), mode="edge")
                    xs = np.pad(xs, ((history - len(xs), 0), (0, 0)), mode="edge")
                x_rows.append(np.stack((xr, xs), axis=0))
                y_rows.append(LABEL_IDS[label])
                w_rows.append(confidence)
                g_rows.append(real.geometry[frame])
                t_rows.append(trajectory)
                f_rows.append(frame)
    if not x_rows:
        raise ValueError("no labeled force windows remain after geometry cutoff and boundary guard")
    return PairWindows(
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int64),
        np.asarray(w_rows, dtype=np.float32),
        np.asarray(g_rows, dtype=np.float32),
        np.asarray(t_rows, dtype=np.int32),
        np.asarray(f_rows, dtype=np.int32),
    )


def fit_feature_scales(episodes: dict[tuple[int, int], Episode], train_ids: list[int]) -> np.ndarray:
    signal = np.concatenate(
        [episodes[(t, d)].raw_features[: episodes[(t, d)].cutoff] for t in train_ids for d in (0, 1)],
        axis=0,
    )
    if len(signal) == 0:
        raise ValueError("no pre-cutoff Sim/Real force frames are available to fit normalization")
    scales = np.quantile(np.abs(signal), 0.95, axis=0)
    floors = np.asarray([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)
    return np.maximum(scales, floors).astype(np.float32)


def grouped_folds(windows: PairWindows, trajectories: list[int], n_folds: int, seed: int) -> list[list[int]]:
    per_traj = {
        t: np.bincount(windows.y[windows.trajectory == t], minlength=2).astype(np.float64)
        for t in trajectories
    }
    target = np.bincount(windows.y, minlength=2).astype(np.float64) / n_folds
    rng = np.random.default_rng(seed)
    best: list[list[int]] | None = None
    best_score = float("inf")
    ids = np.asarray(trajectories, dtype=np.int32)
    for _ in range(2500):
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        folds = [list(map(int, part)) for part in np.array_split(shuffled, n_folds)]
        counts = np.asarray([sum((per_traj[t] for t in fold), start=np.zeros(2)) for fold in folds])
        missing = np.sum(counts == 0)
        score = float(np.square((counts - target) / np.maximum(target, 1.0)).sum() + 1000 * missing)
        if score < best_score:
            best, best_score = folds, score
    assert best is not None
    if any(not fold for fold in best):
        raise ValueError("a grouped CV fold is empty")
    return best


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (y.astype(np.int64), pred.astype(np.int64)), 1)
    precision, recall, f1 = [], [], []
    for cls in range(2):
        tp = int(cm[cls, cls])
        fp = int(cm[:, cls].sum() - tp)
        fn = int(cm[cls, :].sum() - tp)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(p + r, 1e-12))
    return {
        "count": int(cm.sum()),
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            LABEL_NAMES[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(cm[i].sum())}
            for i in range(2)
        },
        "confusion_matrix": cm.tolist(),
    }


def transition_latency(
    records: list[dict[str, np.ndarray]],
    labels: dict[int, list[dict[str, Any]]],
    guard: int,
    persistence: int = 3,
) -> dict[str, Any]:
    """Measure how soon the OOF classifier settles on aligned after a reviewed transition."""
    by_trajectory: dict[int, dict[int, int]] = {}
    for record in records:
        for trajectory, frame, pred in zip(record["trajectory"], record["frame"], record["pred"]):
            by_trajectory.setdefault(int(trajectory), {})[int(frame)] = int(pred)
    delays: list[int] = []
    misses = 0
    transitions = 0
    for trajectory, intervals in labels.items():
        predictions = by_trajectory.get(trajectory, {})
        seen_misaligned = False
        for interval in intervals:
            if interval["label"] == "misaligned_contact":
                seen_misaligned = True
            if interval["label"] != "aligned_progress" or not seen_misaligned:
                continue
            transitions += 1
            first = interval["start"] + guard
            last = interval["end"] - guard
            detected = None
            for frame in range(first, last - persistence + 2):
                if all(predictions.get(frame + offset) == 1 for offset in range(persistence)):
                    detected = frame
                    break
            if detected is None:
                misses += 1
            else:
                delays.append(detected - interval["start"])
    return {
        "transition_count": transitions,
        "detected_count": len(delays),
        "miss_count": misses,
        "miss_rate": float(misses / transitions) if transitions else None,
        "median_delay_frames": float(np.median(delays)) if delays else None,
        "p90_delay_frames": float(np.quantile(delays, 0.9)) if delays else None,
        "median_delay_seconds": float(np.median(delays) / 10.0) if delays else None,
    }


def predict(model: ForceAlignmentEncoder, x: torch.Tensor, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            z, logits, _ = model(x[start : start + batch_size])
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            latents.append(z.cpu().numpy())
    return np.concatenate(probs), np.concatenate(latents)


def train_one(
    train: PairWindows,
    valid: PairWindows | None,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[ForceAlignmentEncoder, int, dict[str, np.ndarray] | None]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = ForceAlignmentEncoder(latent_dim=config["latent_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    x_train = torch.from_numpy(train.x).to(device)
    y_train = torch.from_numpy(train.y).to(device)
    w_train = torch.from_numpy(train.weight).to(device)
    g_train = torch.from_numpy(train.geometry / config["geometry_scale_m"]).to(device)
    counts = np.bincount(train.y, minlength=2).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"training fold has only one class: {counts.tolist()}")
    sampling_weights = np.asarray([1.0 / counts[y] for y in train.y], dtype=np.float64) * train.weight
    sampling_weights /= sampling_weights.sum()
    batches_per_epoch = max(1, int(np.ceil(len(train.y) / config["batch_size"])))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = config["epochs"]
    best_score = -1.0
    best_valid: dict[str, np.ndarray] | None = None
    stale = 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        sampled = np_rng.choice(len(train.y), size=batches_per_epoch * config["batch_size"], replace=True, p=sampling_weights)
        for start in range(0, len(sampled), config["batch_size"]):
            idx = torch.as_tensor(sampled[start : start + config["batch_size"]], device=device)
            x = x_train[idx].reshape(-1, config["history_length"], 6)
            y = y_train[idx].repeat_interleave(2)
            weights = w_train[idx].repeat_interleave(2)
            geom = g_train[idx].repeat_interleave(2, dim=0)
            # Small independent gain jitter keeps the representation stable across sensor scales.
            gains = torch.empty((len(idx), 2, 1, 1), device=device).uniform_(0.85, 1.15)
            x = (x_train[idx] * gains).reshape(-1, config["history_length"], 6)
            z, logits, geom_hat = model(x)
            quality_loss = (F.cross_entropy(logits, y, reduction="none") * weights).mean()
            z_pair = z.reshape(-1, 2, config["latent_dim"])
            consistency_loss = (1.0 - F.cosine_similarity(z_pair[:, 0], z_pair[:, 1], dim=-1)).mean()
            geometry_loss = F.smooth_l1_loss(geom_hat, geom, beta=0.2)
            loss = quality_loss + 0.15 * consistency_loss + config["geometry_loss_weight"] * geometry_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        if valid is None:
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            continue
        valid_x = torch.from_numpy(valid.x).to(device).reshape(-1, config["history_length"], 6)
        valid_prob, valid_z = predict(model, valid_x)
        pred = valid_prob.reshape(-1, 2, 2).argmax(axis=-1)
        metrics = classification_metrics(np.repeat(valid.y[:, None], 2, axis=1).reshape(-1), pred.reshape(-1))
        real_metrics = classification_metrics(valid.y, pred[:, 0])
        score = real_metrics["macro_f1"]
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_valid = {
                "probability": valid_prob.reshape(-1, 2, 2),
                "latent": valid_z.reshape(-1, 2, config["latent_dim"]),
                "metrics_all": metrics,
            }
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_epoch, best_valid


def prepare_for_fold(
    episodes: dict[tuple[int, int], Episode],
    labels: dict[int, list[dict[str, Any]]],
    train_ids: list[int],
    all_ids: list[int],
    args: argparse.Namespace,
) -> tuple[PairWindows, PairWindows, np.ndarray]:
    scales = fit_feature_scales(episodes, train_ids)
    all_windows = make_windows(episodes, labels, all_ids, scales, args.history_length, args.boundary_guard)
    train_mask = np.isin(all_windows.trajectory, train_ids)
    def subset(mask: np.ndarray) -> PairWindows:
        return PairWindows(*(getattr(all_windows, field)[mask] for field in ("x", "y", "weight", "geometry", "trajectory", "frame")))
    return subset(train_mask), subset(~train_mask), scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--geometry-root", type=Path, default=None, help="optional exact per-frame peg-tip/hole geometry CSV root")
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--baseline-frames", type=int, default=20)
    parser.add_argument("--boundary-guard", type=int, default=5)
    parser.add_argument("--xy-threshold-m", type=float, default=0.003)
    parser.add_argument("--z-threshold-m", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
        if visible not in ("0", "0,", "0\n"):
            raise RuntimeError("GPU training must use only physical GPU 0; launch with CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected environment")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    episodes, cutoffs, geometry_meta = load_episodes(args)
    labels = read_labels(args.labels)
    ids = sorted(labels)
    # A preliminary all-data window table is used only to stratify trajectory groups.
    all_scales = fit_feature_scales(episodes, ids)
    all_windows = make_windows(episodes, labels, ids, all_scales, args.history_length, args.boundary_guard)
    folds = grouped_folds(all_windows, ids, args.folds, args.seed)
    config: dict[str, Any] = {
        "history_length": args.history_length,
        "baseline_frames": args.baseline_frames,
        "sample_period_s": 0.1,
        "latent_dim": 32,
        "geometry_scale_m": 0.01,
        "geometry_supervision": bool(args.geometry_root),
        "geometry_loss_weight": 0.10 if args.geometry_root else 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "boundary_guard": args.boundary_guard,
        "xy_threshold_m": args.xy_threshold_m,
        "z_threshold_m": args.z_threshold_m,
        "classes": LABEL_NAMES,
        "input_force_key": args.force_key,
    }

    oof: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    fold_reports: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    for fold_index, valid_ids in enumerate(folds):
        train_ids = sorted(set(ids) - set(valid_ids))
        train_windows, valid_windows, _ = prepare_for_fold(episodes, labels, train_ids, ids, args)
        model, best_epoch, valid_out = train_one(train_windows, valid_windows, config, device, args.seed + fold_index)
        if valid_out is None:
            raise RuntimeError("cross-validation produced no validation predictions")
        best_epochs.append(best_epoch)
        pred = valid_out["probability"].argmax(axis=-1)
        per_domain = {}
        for domain in (0, 1):
            metric = classification_metrics(valid_windows.y, pred[:, domain])
            per_domain["real" if domain == 0 else "sim"] = metric
            oof[domain].append(
                {
                    "y": valid_windows.y.copy(),
                    "pred": pred[:, domain].copy(),
                    "trajectory": valid_windows.trajectory.copy(),
                    "frame": valid_windows.frame.copy(),
                }
            )
        fold_reports.append({"fold": fold_index, "held_out_trajectories": valid_ids, "best_epoch": best_epoch, "metrics": per_domain})
        print(f"fold {fold_index + 1}/{args.folds}: epoch={best_epoch} real_macro_f1={per_domain['real']['macro_f1']:.3f} sim_macro_f1={per_domain['sim']['macro_f1']:.3f}", flush=True)

    aggregate: dict[str, Any] = {}
    for domain in (0, 1):
        chunks = oof[domain]
        y = np.concatenate([r["y"] for r in chunks])
        pred = np.concatenate([r["pred"] for r in chunks])
        domain_name = "real" if domain == 0 else "sim"
        aggregate[domain_name] = classification_metrics(y, pred)
        latency_records = []
        for chunk in chunks:
            latency_records.append({key: chunk[key] for key in ("trajectory", "frame", "pred")})
        aggregate[domain_name]["transition_latency"] = transition_latency(
            latency_records, labels, args.boundary_guard
        )

    final_scales = fit_feature_scales(episodes, ids)
    final_windows = make_windows(episodes, labels, ids, final_scales, args.history_length, args.boundary_guard)
    final_epochs = max(1, int(np.median(best_epochs)))
    final_config = dict(config, epochs=final_epochs, patience=final_epochs + 1)
    final_model, _, _ = train_one(final_windows, None, final_config, device, args.seed + 1000)
    final_model.to("cpu")

    payload = {
        "artifact_format": "tavla_force_alignment_encoder_v1",
        "config": config,
        "preprocess": {
            "feature_scales": final_scales.tolist(),
            "feature_order": ["force_x", "force_y", "force_z", "dforce_x_per_s", "dforce_y_per_s", "dforce_z_per_s"],
            "baseline": "coordinate-wise median of the first 20 force samples",
            "geometry_units": "meters; peg-tip position minus hole center in simulator XYZ" if args.geometry_root else "not supervised; reconstructed cutoff only",
        },
        "model_state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
        "metrics": {"grouped_oof": aggregate, "folds": fold_reports, "deploy_training_epochs": final_epochs},
        "data_summary": {
            "trajectories": ids,
            "labelled_windows_after_cutoff_and_guard": int(len(all_windows.y)),
            "class_counts": {LABEL_NAMES[i]: int((all_windows.y == i).sum()) for i in range(2)},
            "cutoff_frame_by_trajectory": {str(t): int(cutoffs[t]) for t in sorted(cutoffs)},
            "grouped_fold_assignment": [list(map(int, fold)) for fold in folds],
            "geometry_source": geometry_meta,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"OOF Real: macro-F1={aggregate['real']['macro_f1']:.3f}, per-class F1={aggregate['real']['per_class']}")
    print(f"OOF Sim:  macro-F1={aggregate['sim']['macro_f1']:.3f}, per-class F1={aggregate['sim']['per_class']}")
    print(f"Saved one artifact (overwrites on rerun): {args.output}")


if __name__ == "__main__":
    main()
