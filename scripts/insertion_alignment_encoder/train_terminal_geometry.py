#!/usr/bin/env python3
"""Train a force-only alignment encoder with geometry recovered from terminal EE poses.

The simulator constructed each hole from the final end-effector pose.  In this
recording, Real and Sim carry the same EE pose sequence, so the terminal peg-tip
pose is a useful per-trajectory hole reference.  This is a *terminal reference*,
not an independently measured hole center.  The tool offset is only needed to
account for rotation; its contribution to within-trajectory XY error is tiny.

Only force history reaches the encoder.  Pose is used offline to make training
targets.  All non-free frames, including final insertion and terminal holding,
receive geometry supervision; the first 3 mm hit is a reporting marker only.
The eight held-out trajectories never select an epoch, scale, or parameter.

Run from TAVLA with physical GPU 0 visible:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts .venv/bin/python \
      -m insertion_alignment_encoder.train_terminal_geometry

The new artifact is saved at
eval_outputs/insertion_alignment_encoder_full_insertion/encoder.pt.
Existing output files are never overwritten.  Retraining TAVLA on this
encoder's newly generated latents is required before deploying it to a policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
import uuid

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F

from .model import ForceAlignmentEncoder
from .preprocess import force_history_features, initial_baseline, normalize_features
from .train import (
    Episode,
    LABEL_IDS,
    PairWindows,
    RECONSTRUCTED_CUTOFFS,
    classification_metrics,
    grouped_folds,
    read_labels,
)


HELD_OUT_TRAJECTORIES = [0, 2, 21, 22, 23, 33, 34, 39]
TRAINING_RECIPE = "full_insertion_v4"
GEOMETRY_PHASES = {"uncertain", "misaligned_contact", "aligned_progress", "terminal"}


@dataclass
class InsertionWindows(PairWindows):
    # y=-1 means geometry/alignment only, with no two-class quality target.
    phase: np.ndarray


def _load_force_and_pose(path: Path, force_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        force = np.asarray(h5[force_key], dtype=np.float32)
        time = np.asarray(h5["timestamps"], dtype=np.float64)
        pose = np.asarray(h5["obs/state/ee_pose"], dtype=np.float64)
    if (force.ndim != 2 or force.shape[1] != 3 or len(force) == 0
            or pose.shape != (len(force), 7) or time.shape != (len(force),)):
        raise ValueError(f"{path}: expected force (T,3), ee_pose (T,7), and timestamps (T,)")
    if not all(np.isfinite(array).all() for array in (force, time, pose)):
        raise ValueError(f"{path}: force, time, or pose contains NaN/Inf")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    return force, time, pose


def terminal_reference_geometry(pose: np.ndarray, tool_offset_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return signed peg-tip minus terminal-tip XYZ and the terminal-tip position.

    The recorded quaternion is wxyz.  Both tip and hole reference receive the
    same local tool offset, which nearly cancels because orientation is stable.
    """
    q_wxyz = pose[:, 3:7]
    q_xyzw = np.concatenate((q_wxyz[:, 1:], q_wxyz[:, :1]), axis=1)
    offset_m = np.asarray(tool_offset_mm, dtype=np.float64) * 0.001
    tip = pose[:, :3] + Rotation.from_quat(q_xyzw).apply(np.broadcast_to(offset_m, (len(pose), 3)))
    hole_reference = tip[-1].copy()
    return (tip - hole_reference).astype(np.float32), hole_reference


def _first_hit(geometry: np.ndarray, xy_threshold_m: float, z_threshold_m: float) -> int:
    radius = np.linalg.norm(geometry[:, :2], axis=1)
    hit = np.flatnonzero((radius < xy_threshold_m) & (np.abs(geometry[:, 2]) < z_threshold_m))
    return int(hit[0]) if hit.size else len(geometry)


def load_terminal_episodes(args: argparse.Namespace, labels: dict[int, list[dict[str, Any]]]):
    episodes: dict[tuple[int, int], Episode] = {}
    cutoffs: dict[int, int] = {}
    hole_references: dict[int, list[float]] = {}
    max_pose_disagreement_mm = 0.0
    for trajectory in sorted(labels):
        real_path = args.real_root / f"traj_{trajectory}" / "data.h5"
        sim_path = args.sim_root / f"traj_{trajectory}" / "data.h5"
        real_force, real_time, real_pose = _load_force_and_pose(real_path, args.force_key)
        sim_force, sim_time, sim_pose = _load_force_and_pose(sim_path, args.force_key)
        if len(real_force) != len(sim_force):
            raise ValueError(f"traj_{trajectory}: Real/Sim frame counts differ")
        if not np.allclose(real_time, sim_time, rtol=0, atol=1e-8):
            raise ValueError(f"traj_{trajectory}: Real/Sim timestamps are not paired")
        if len(real_force) < args.baseline_frames:
            raise ValueError(f"traj_{trajectory}: fewer samples than the baseline window")
        disagreement_mm = float(np.max(np.abs(real_pose[:, :3] - sim_pose[:, :3])) * 1000)
        max_pose_disagreement_mm = max(max_pose_disagreement_mm, disagreement_mm)
        if disagreement_mm > 0.1:
            raise ValueError(f"traj_{trajectory}: paired Real/Sim EE positions differ by {disagreement_mm:.3f} mm")
        geometry, hole_reference = terminal_reference_geometry(sim_pose, args.tool_offset_mm)
        cutoff = _first_hit(geometry, args.xy_threshold_m, args.z_threshold_m)
        cutoffs[trajectory] = cutoff
        hole_references[trajectory] = hole_reference.tolist()
        for domain, force, time in ((0, real_force, real_time), (1, sim_force, sim_time)):
            baseline = initial_baseline(force, args.baseline_frames)
            features = force_history_features(force, time, baseline)
            episodes[(trajectory, domain)] = Episode(
                trajectory, domain, force, time, features, geometry, cutoff
            )
        cursor = 0
        for interval in labels[trajectory]:
            if interval["start"] != cursor or interval["end"] < interval["start"]:
                raise ValueError(f"traj_{trajectory}: invalid label interval near frame {cursor}")
            if interval["label"] not in GEOMETRY_PHASES | {"free"}:
                raise ValueError(f"traj_{trajectory}: unknown phase {interval['label']!r}")
            cursor = interval["end"] + 1
        if cursor != len(sim_force):
            raise ValueError(f"traj_{trajectory}: phase labels do not cover the pose sequence")
    return episodes, cutoffs, hole_references, max_pose_disagreement_mm


def _take(windows: InsertionWindows, keep: np.ndarray) -> InsertionWindows:
    return InsertionWindows(*(getattr(windows, field.name)[keep] for field in fields(windows)))


def _subset(windows: InsertionWindows, ids: list[int]) -> InsertionWindows:
    return _take(windows, np.isin(windows.trajectory, ids))


def _classification_windows(windows: InsertionWindows) -> InsertionWindows:
    return _take(windows, windows.y >= 0)


def make_full_insertion_windows(episodes, labels, ids, scales, history, guard) -> InsertionWindows:
    """Keep every non-free frame; guard only the reviewed quality labels.

    Uncertain/terminal phases and guarded boundary frames have a valid pose
    target, but are not relabeled as aligned_progress or misaligned_contact.
    The first-hit marker is deliberately not consulted here.
    """
    rows: list[tuple[Any, ...]] = []
    for trajectory in ids:
        real, sim = episodes[(trajectory, 0)], episodes[(trajectory, 1)]
        features = [normalize_features(episode.raw_features, scales) for episode in (real, sim)]
        selected_frames: set[int] = set()
        for interval in labels[trajectory]:
            phase = interval["label"]
            if phase == "free":
                continue
            confidence = 1.0 if interval["confidence"] == "high" else 0.65
            class_start, class_end = interval["start"] + guard, interval["end"] - guard
            for frame in range(interval["start"], interval["end"] + 1):
                chunks = []
                lo = max(0, frame - history + 1)
                for feature in features:
                    chunk = feature[lo:frame + 1]
                    if len(chunk) < history:
                        chunk = np.pad(chunk, ((history - len(chunk), 0), (0, 0)), mode="edge")
                    chunks.append(chunk)
                quality = LABEL_IDS.get(phase, -1) if class_start <= frame <= class_end else -1
                rows.append((np.stack(chunks), quality, confidence, real.geometry[frame],
                             trajectory, frame, phase))
                selected_frames.add(frame)
        near = ((np.linalg.norm(real.geometry[:, :2], axis=1) < 0.003)
                & (np.abs(real.geometry[:, 2]) < 0.003))
        missing = set(map(int, np.flatnonzero(near))) - selected_frames
        if missing:
            raise ValueError(f"traj_{trajectory}: {len(missing)} near-terminal frames were excluded; "
                             "review the free-phase labels before training")
    if not rows:
        raise ValueError("no non-free insertion windows are available")
    return InsertionWindows(
        np.asarray([row[0] for row in rows], dtype=np.float32),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        np.asarray([row[2] for row in rows], dtype=np.float32),
        np.asarray([row[3] for row in rows], dtype=np.float32),
        np.asarray([row[4] for row in rows], dtype=np.int32),
        np.asarray([row[5] for row in rows], dtype=np.int32),
        np.asarray([row[6] for row in rows], dtype="U32"),
    )


def _window_table(episodes, labels, fit_ids, ids, args):
    # Include the end of TRAINING trajectories when fitting a shared scale.
    # Validation/test trajectories never contribute their force statistics.
    signal = np.concatenate([episodes[(t, d)].raw_features for t in fit_ids for d in (0, 1)])
    floors = np.asarray([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)
    scales = np.maximum(np.quantile(np.abs(signal), 0.95, axis=0), floors).astype(np.float32)
    return make_full_insertion_windows(episodes, labels, ids, scales,
                                       args.history_length, args.boundary_guard), scales


def _depth_masks(windows: InsertionWindows) -> dict[str, np.ndarray]:
    centered = np.linalg.norm(windows.geometry[:, :2], axis=1) < 0.003
    depth = np.abs(windows.geometry[:, 2])
    return {
        "0_3mm": centered & (depth < 0.003),
        "3_5mm": centered & (depth >= 0.003) & (depth < 0.005),
        "5_8mm": centered & (depth >= 0.005) & (depth < 0.008),
        "8_10mm": centered & (depth >= 0.008) & (depth < 0.010),
        "other": ~centered | (depth >= 0.010),
    }


def window_coverage(windows: InsertionWindows) -> dict[str, Any]:
    depth_masks = _depth_masks(windows)
    return {
        "pair_windows": len(windows.y),
        "classification_pair_windows": int((windows.y >= 0).sum()),
        "geometry_only_pair_windows": int((windows.y < 0).sum()),
        "by_phase": {str(phase): int((windows.phase == phase).sum()) for phase in np.unique(windows.phase)},
        "by_depth": {name: int(mask.sum()) for name, mask in depth_masks.items()},
        "per_trajectory": {
            f"traj_{trajectory}": {
                "pair_windows": int((windows.trajectory == trajectory).sum()),
                "near_0_3mm": int(((windows.trajectory == trajectory) & depth_masks["0_3mm"]).sum()),
            }
            for trajectory in np.unique(windows.trajectory)
        },
    }


def _classification_probabilities(windows: InsertionWindows) -> np.ndarray:
    """Balance classes and temper trajectory length; use confidence in CE only.

    Each class has half the probability mass. Within a class, a trajectory
    receives mass proportional to sqrt(window_count), rather than its full
    window count. This reduces domination by long, overlapping segments while
    avoiding giving a one-frame segment the same mass as a long trajectory.
    """
    class_prob = np.zeros(len(windows.y), dtype=np.float64)
    for cls in range(2):
        mask = windows.y == cls
        if not mask.any():
            raise ValueError(f"training group is missing quality class {cls}")
        trajectories, counts = np.unique(windows.trajectory[mask], return_counts=True)
        masses = np.sqrt(counts.astype(np.float64))
        for trajectory, count, mass in zip(trajectories, counts, masses, strict=True):
            class_prob[mask & (windows.trajectory == trajectory)] = mass / masses.sum() / count / 2
    return class_prob / class_prob.sum()


def _geometry_probabilities(windows: InsertionWindows) -> np.ndarray:
    """Balance depth, phase and trajectory without changing quality batches."""
    geometry_prob = np.zeros(len(windows.y), dtype=np.float64)
    masks = [mask for mask in _depth_masks(windows).values() if mask.any()]
    for depth_mask in masks:
        phases = np.unique(windows.phase[depth_mask])
        for phase in phases:
            phase_mask = depth_mask & (windows.phase == phase)
            trajectories = np.unique(windows.trajectory[phase_mask])
            for trajectory in trajectories:
                mask = phase_mask & (windows.trajectory == trajectory)
                # Pose targets remain valid even when phase-label confidence
                # is medium. Do not reuse that confidence for geometry.
                geometry_prob[mask] = 1.0 / mask.sum() / len(masks) / len(phases) / len(trajectories)
    return geometry_prob / geometry_prob.sum()


def geometry_metrics(target: np.ndarray, estimate: np.ndarray, min_radius_m: float = 0.003) -> dict[str, float | int | None]:
    if len(target) == 0:
        return {"count": 0, "xy_mae_mm": None, "xy_radial_error_mae_mm": None,
                "z_mae_mm": None, "direction_count": 0, "direction_cosine": None,
                "direction_within_90deg": None, "target_z_mean_mm": None,
                "predicted_z_mean_mm": None, "predicted_z_std_mm": None}
    error = estimate - target
    radius = np.linalg.norm(target[:, :2], axis=1)
    valid = radius >= min_radius_m
    t = target[valid, :2]
    p = estimate[valid, :2]
    denominator = np.linalg.norm(t, axis=1) * np.linalg.norm(p, axis=1)
    cosine = np.divide(np.sum(t * p, axis=1), denominator, out=np.zeros(len(t)), where=denominator > 1e-8)
    return {
        "count": len(target),
        "xy_mae_mm": float(np.mean(np.abs(error[:, :2])) * 1000),
        "xy_radial_error_mae_mm": float(np.mean(np.abs(np.linalg.norm(estimate[:, :2], axis=1) - radius)) * 1000),
        "z_mae_mm": float(np.mean(np.abs(error[:, 2])) * 1000),
        "target_z_mean_mm": float(np.mean(target[:, 2]) * 1000),
        "predicted_z_mean_mm": float(np.mean(estimate[:, 2]) * 1000),
        "predicted_z_std_mm": float(np.std(estimate[:, 2]) * 1000),
        "direction_count": int(valid.sum()),
        "direction_cosine": float(np.mean(cosine)) if len(cosine) else None,
        "direction_within_90deg": float(np.mean(cosine > 0)) if len(cosine) else None,
    }


def _selection_depth_error(by_phase_depth: dict[str, Any]) -> float | None:
    """Weight the final 3 mm separately from the more forgiving 3--10 mm."""
    final_errors, approach_errors = [], []
    for bands in by_phase_depth.values():
        for name, metrics in bands.items():
            if not metrics["count"] or name == "other":
                continue
            (final_errors if name == "0_3mm" else approach_errors).append(metrics["z_mae_mm"])
    groups = [float(np.mean(errors)) for errors in (final_errors, approach_errors) if errors]
    return float(np.mean(groups)) if groups else None


def _domain_report(windows: InsertionWindows, prediction: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    classified = windows.y >= 0
    depth_masks = _depth_masks(windows)
    by_depth = {
        name: geometry_metrics(windows.geometry[mask], estimate[mask])
        for name, mask in depth_masks.items()
    }
    by_phase_depth = {
        str(phase): {
            name: geometry_metrics(windows.geometry[mask & (windows.phase == phase)],
                                   estimate[mask & (windows.phase == phase)])
            for name, mask in depth_masks.items()
        }
        for phase in np.unique(windows.phase)
    }
    # Equal weight per nonempty depth/phase group. Long terminal holds should
    # not hide poor predictions on frames that are still actively inserting.
    near_errors = [metrics["z_mae_mm"] for bands in by_phase_depth.values()
                   for name, metrics in bands.items() if name != "other" and metrics["count"]]
    return {
        "class": classification_metrics(windows.y[classified], prediction[classified]),
        "by_trajectory": {
            f"traj_{trajectory}": {
                "class": classification_metrics(
                    windows.y[classified & (windows.trajectory == trajectory)],
                    prediction[classified & (windows.trajectory == trajectory)],
                ),
                "geometry": geometry_metrics(
                    windows.geometry[windows.trajectory == trajectory],
                    estimate[windows.trajectory == trajectory],
                ),
            }
            for trajectory in np.unique(windows.trajectory)
        },
        "geometry": geometry_metrics(windows.geometry, estimate),
        "geometry_by_depth": by_depth,
        "geometry_by_phase_and_depth": by_phase_depth,
        "near_terminal_z_mae_mm": float(np.mean(near_errors)) if near_errors else None,
        "selection_depth_mae_mm": _selection_depth_error(by_phase_depth),
    }


def _evaluate(model: ForceAlignmentEncoder, windows: InsertionWindows, device: torch.device, geometry_scale_m: float):
    model.eval()
    x = torch.from_numpy(windows.x.reshape(-1, windows.x.shape[-2], 6)).to(device)
    probs: list[np.ndarray] = []
    geom: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            _, logits, prediction = model(x[start:start + 512])
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            geom.append(prediction.cpu().numpy() * geometry_scale_m)
    probability = np.concatenate(probs).reshape(-1, 2, 2)
    estimate = np.concatenate(geom).reshape(-1, 2, 3)
    reports = {}
    for domain, name in ((0, "real"), (1, "sim")):
        reports[name] = _domain_report(windows, probability[:, domain].argmax(axis=-1), estimate[:, domain])
    return reports, probability, estimate


def _score(reports: dict[str, Any], depth_scale_mm: float) -> float:
    depth_scores = []
    class_scores = []
    for domain in ("real", "sim"):
        report = reports[domain]
        depth_error = report["selection_depth_mae_mm"]
        if depth_error is None:
            depth_error = report["geometry"]["z_mae_mm"]
        depth_scores.append(float(np.exp(-depth_error / depth_scale_mm)))
        class_scores.append(report["class"]["macro_f1"])
    return float(0.60 * min(class_scores) + 0.40 * np.mean(depth_scores))


def _selection_failures(reports: dict[str, Any], epoch: int, args: argparse.Namespace) -> list[str]:
    failures = []
    if epoch < args.min_selection_epoch:
        failures.append(f"epoch<{args.min_selection_epoch}")
    for domain in ("real", "sim"):
        report = reports[domain]
        if report["class"]["macro_f1"] < args.min_macro_f1:
            failures.append(f"{domain}.macro_f1<{args.min_macro_f1}")
        for label, metrics in report["class"]["per_class"].items():
            if metrics["support"] == 0 or metrics["recall"] < args.min_class_recall:
                failures.append(f"{domain}.{label}.recall<{args.min_class_recall}")
        if report["selection_depth_mae_mm"] is None:
            failures.append(f"{domain}.missing_near_depth_windows")
    return failures


def _learning_rate(epoch: int, args: argparse.Namespace) -> float:
    # The final fit uses the SAME prefix of the CV learning-rate schedule.
    if args.warmup_epochs and epoch <= args.warmup_epochs:
        return args.learning_rate * epoch / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
    return float(args.min_learning_rate + 0.5 * (args.learning_rate - args.min_learning_rate)
                 * (1.0 + np.cos(np.pi * min(progress, 1.0))))


def train_one(train: InsertionWindows, valid: InsertionWindows | None,
              args: argparse.Namespace, device: torch.device, seed: int, epochs: int,
              run_name: str, history_path: Path, checkpoint_context: dict[str, Any] | None = None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ForceAlignmentEncoder(latent_dim=32, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    x = torch.from_numpy(train.x).to(device)
    y = torch.from_numpy(train.y).to(device)
    weight = torch.from_numpy(train.weight).to(device)
    target = torch.from_numpy(train.geometry / args.geometry_scale_m).to(device)
    class_probability = _classification_probabilities(train)
    geometry_probability = _geometry_probabilities(train)
    # A larger geometry table must not silently multiply the number of draws
    # from the much smaller, heavily overlapping classification table.
    steps = int(np.ceil(np.count_nonzero(train.y >= 0) / args.batch_size))
    checkpoint_dir = history_path.parent / f"{run_name}_checkpoints" if valid is not None else None
    if checkpoint_dir is not None:
        if checkpoint_context is None:
            raise ValueError("CV checkpoint context must include fold IDs and normalization scales")
        checkpoint_dir.mkdir(exist_ok=False)
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    print(f"TRAIN {run_name}: epochs={epochs}, steps_per_epoch={steps}, "
          f"class_pairs_per_step={args.batch_size}, geometry_pairs_per_step={args.batch_size}", flush=True)
    # Keep all CV epochs so selection can compare the same epoch across folds.
    # Save every row immediately, including if training later fails.
    with history_path.open("x", encoding="utf-8") as trace:
        for epoch in range(1, epochs + 1):
            model.train()
            learning_rate = _learning_rate(epoch, args)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            class_samples = rng.choice(len(train.y), size=steps * args.batch_size,
                                       replace=True, p=class_probability)
            geometry_samples = rng.choice(len(train.y), size=steps * args.batch_size,
                                          replace=True, p=geometry_probability)
            sums = {key: 0.0 for key in ("total", "class", "geometry", "consistency", "direction",
                                         "gradient_norm", "class_batch_accuracy")}
            for start in range(0, len(class_samples), args.batch_size):
                class_index = torch.as_tensor(class_samples[start:start + args.batch_size], device=device)
                geometry_index = torch.as_tensor(geometry_samples[start:start + args.batch_size], device=device)
                gain = torch.empty((len(class_index), 2, 1, 1), device=device).uniform_(0.85, 1.15)
                class_inputs = (x[class_index] * gain).reshape(-1, args.history_length, 6)
                gain = torch.empty((len(geometry_index), 2, 1, 1), device=device).uniform_(0.85, 1.15)
                geometry_inputs = (x[geometry_index] * gain).reshape(-1, args.history_length, 6)
                class_latent, logits, _ = model(class_inputs)
                geometry_latent, _, geom_hat = model(geometry_inputs)
                class_target = y[class_index].repeat_interleave(2)
                class_weight = weight[class_index].repeat_interleave(2)
                geom_target = target[geometry_index].repeat_interleave(2, dim=0)
                # Only the class-balanced batch reaches this loss. Geometry
                # oversampling cannot alter the quality-label distribution.
                class_loss = (F.cross_entropy(logits, class_target, reduction="none",
                                              label_smoothing=args.label_smoothing) * class_weight).mean()
                pair_latent = torch.cat((class_latent.reshape(-1, 2, 32),
                                         geometry_latent.reshape(-1, 2, 32)), dim=0)
                consistency = (1 - F.cosine_similarity(pair_latent[:, 0], pair_latent[:, 1], dim=-1)).mean()
                geom_loss = F.smooth_l1_loss(geom_hat, geom_target, beta=0.2)
                radius = torch.linalg.vector_norm(geom_target[:, :2], dim=-1)
                direction_mask = radius >= args.xy_threshold_m / args.geometry_scale_m
                direction_loss = (
                    (1 - F.cosine_similarity(geom_hat[direction_mask, :2],
                                            geom_target[direction_mask, :2], dim=-1)).mean()
                    if direction_mask.any() else torch.zeros((), device=device)
                )
                loss = (args.class_loss_weight * class_loss + args.consistency_loss_weight * consistency
                        + args.geometry_loss_weight * geom_loss + args.direction_loss_weight * direction_loss)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{run_name}: non-finite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(f"{run_name}: non-finite gradient at epoch {epoch}")
                optimizer.step()
                for key, value in (("total", loss), ("class", class_loss), ("geometry", geom_loss),
                                   ("consistency", consistency), ("direction", direction_loss),
                                   ("gradient_norm", gradient_norm)):
                    sums[key] += float(value.detach().item())
                sums["class_batch_accuracy"] += float((logits.argmax(dim=-1) == class_target).float().mean().item())
            reports = _evaluate(model, valid, device, args.geometry_scale_m)[0] if valid is not None else None
            failures = _selection_failures(reports, epoch, args) if reports is not None else []
            row = {
                "epoch": epoch, "optimizer_steps": epoch * steps, "steps_per_epoch": steps,
                "learning_rate": learning_rate, "elapsed_s": time.monotonic() - started,
                "train": {key: value / steps for key, value in sums.items()},
                "validation": reports,
                "selection_score": _score(reports, args.selection_depth_scale_mm) if reports is not None else None,
                "selection_failures": failures,
            }
            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
                _save_new_checkpoint({
                    # A CV snapshot deliberately has a different format from
                    # deployable encoders and cannot be loaded as one.
                    "artifact_format": "tavla_force_alignment_encoder_cv_snapshot_v1",
                    "training_recipe": TRAINING_RECIPE,
                    "run_name": run_name, "epoch": epoch, "seed": seed,
                    "optimizer_steps": epoch * steps,
                    "context": checkpoint_context,
                    "validation": reports,
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }, checkpoint_path)
                row["checkpoint"] = str(checkpoint_path)
            history.append(row)
            trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            trace.flush()
            if epoch == 1 or epoch % args.log_every == 0 or epoch == epochs:
                message = (f"{run_name} epoch={epoch}/{epochs} steps={epoch * steps} lr={learning_rate:.6g} "
                           f"loss={row['train']['total']:.4f} class={row['train']['class']:.4f} "
                           f"geometry={row['train']['geometry']:.4f} consistency={row['train']['consistency']:.4f}")
                if reports is not None:
                    message += (f" real_macro_f1={reports['real']['class']['macro_f1']:.3f} "
                                f"sim_macro_f1={reports['sim']['class']['macro_f1']:.3f} "
                                f"real_depth_mae_mm={reports['real']['selection_depth_mae_mm']:.3f} "
                                f"sim_depth_mae_mm={reports['sim']['selection_depth_mae_mm']:.3f} "
                                f"eligible={not failures}")
                print(message, flush=True)
    return model, history


def _merge_geometry_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    nonempty = [report for report in reports if report["count"]]
    if not nonempty:
        return geometry_metrics(np.empty((0, 3)), np.empty((0, 3)))
    count = sum(report["count"] for report in nonempty)
    result: dict[str, Any] = {"count": count}
    for key in ("xy_mae_mm", "xy_radial_error_mae_mm", "z_mae_mm", "target_z_mean_mm", "predicted_z_mean_mm"):
        result[key] = float(sum(report["count"] * report[key] for report in nonempty) / count)
    second_moment = sum(report["count"] * (report["predicted_z_std_mm"] ** 2
                                           + report["predicted_z_mean_mm"] ** 2) for report in nonempty) / count
    result["predicted_z_std_mm"] = float(np.sqrt(max(0, second_moment - result["predicted_z_mean_mm"] ** 2)))
    directional = [report for report in nonempty if report["direction_count"]]
    direction_count = sum(report["direction_count"] for report in directional)
    result["direction_count"] = direction_count
    for key in ("direction_cosine", "direction_within_90deg"):
        result[key] = (float(sum(report["direction_count"] * report[key] for report in directional) / direction_count)
                       if direction_count else None)
    return result


def _aggregate_validation(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool out-of-fold predictions at the SAME epoch, without held-out data."""
    result = {}
    for domain in ("real", "sim"):
        domain_reports = [report[domain] for report in reports]
        confusion = np.sum([report["class"]["confusion_matrix"] for report in domain_reports], axis=0)
        truth = np.repeat([0, 0, 1, 1], confusion.reshape(-1))
        prediction = np.repeat([0, 1, 0, 1], confusion.reshape(-1))
        phases = sorted({phase for report in domain_reports for phase in report["geometry_by_phase_and_depth"]})
        by_phase_depth = {
            phase: {
                band: _merge_geometry_metrics([
                    report["geometry_by_phase_and_depth"][phase][band]
                    for report in domain_reports if phase in report["geometry_by_phase_and_depth"]
                ]) for band in ("0_3mm", "3_5mm", "5_8mm", "8_10mm", "other")
            } for phase in phases
        }
        result[domain] = {
            "class": classification_metrics(truth, prediction),
            "geometry": _merge_geometry_metrics([report["geometry"] for report in domain_reports]),
            "geometry_by_phase_and_depth": by_phase_depth,
            "selection_depth_mae_mm": _selection_depth_error(by_phase_depth),
            "by_trajectory": {
                trajectory: metrics
                for report in domain_reports for trajectory, metrics in report["by_trajectory"].items()
            },
        }
    return result


def select_common_epoch(histories: list[list[dict[str, Any]]], args: argparse.Namespace,
                        summary_path: Path) -> tuple[int | None, list[dict[str, Any]]]:
    summaries = []
    for index in range(args.epochs):
        epoch = index + 1
        reports = _aggregate_validation([history[index]["validation"] for history in histories])
        failures = _selection_failures(reports, epoch, args)
        summaries.append({"epoch": epoch, "score": _score(reports, args.selection_depth_scale_mm),
                          "eligible": not failures, "failures": failures, "metrics": reports})
    candidates = [row for row in summaries if row["eligible"]]
    selected = max(candidates, key=lambda row: row["score"]) if candidates else None
    best_unqualified = max(summaries, key=lambda row: row["score"])
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump({"status": "passed_cv" if selected else "rejected_cv",
                   "selected_epoch": selected["epoch"] if selected else None,
                   "best_diagnostic_epoch": best_unqualified["epoch"], "epochs": summaries},
                  handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    if selected is None:
        print(f"QUALITY REJECTED: no common CV epoch passes all requirements. "
              f"Best diagnostic epoch={best_unqualified['epoch']}, "
              f"failures={best_unqualified['failures']}", flush=True)
        for domain in ("real", "sim"):
            report = best_unqualified["metrics"][domain]
            recalls = {name: round(metrics["recall"], 4)
                       for name, metrics in report["class"]["per_class"].items()}
            print(f"BEST DIAGNOSTIC {domain}: macro_f1={report['class']['macro_f1']:.4f}, "
                  f"recalls={recalls}, depth_mae_mm={report['selection_depth_mae_mm']}", flush=True)
        return None, summaries
    print(f"Selected common CV epoch={selected['epoch']}, score={selected['score']:.4f}, "
          f"real_macro_f1={selected['metrics']['real']['class']['macro_f1']:.3f}, "
          f"sim_macro_f1={selected['metrics']['sim']['class']['macro_f1']:.3f}", flush=True)
    return selected["epoch"], summaries


def _save_new_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Atomically create a checkpoint, refusing to replace any existing file."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _linear_features(windows: InsertionWindows) -> np.ndarray:
    force = windows.x[..., :3]
    return np.concatenate((
        force[:, :, -1],
        force[:, :, -5:].mean(axis=2),
        force[:, :, -5:].mean(axis=2) - force[:, :, -15:-10].mean(axis=2),
        force.std(axis=2),
    ), axis=-1)


def linear_baselines(train: InsertionWindows, test: InsertionWindows) -> dict[str, Any]:
    x_train = _linear_features(train).reshape(-1, 12)
    x_test = _linear_features(test)
    y_geom = np.repeat(train.geometry[:, None, :], 2, axis=1).reshape(-1, 3)
    design = np.concatenate((x_train.astype(np.float64), np.ones((len(x_train), 1))), axis=1)
    ridge = design.T @ design + np.eye(design.shape[1]) * 10.0
    direction_coef = np.linalg.solve(ridge, design.T @ (y_geom * 1000))
    classified = np.repeat(train.y >= 0, 2)
    class_design = design[classified]
    y_class = np.repeat(train.y, 2)[classified]
    class_target = np.eye(2, dtype=np.float64)[y_class]
    class_weight = 1.0 / np.bincount(y_class, minlength=2)[y_class]
    weighted = class_design * class_weight[:, None]
    class_coef = np.linalg.solve(
        class_design.T @ weighted + np.eye(design.shape[1]) * 10.0,
        weighted.T @ class_target,
    )
    result = {}
    for domain, name in ((0, "real"), (1, "sim")):
        features = np.concatenate((x_test[:, domain], np.ones((len(test.y), 1))), axis=1)
        estimate = (features @ direction_coef).astype(np.float32) * 0.001
        class_pred = (features @ class_coef).argmax(axis=1)
        result[name] = _domain_report(test, class_pred, estimate)
    return result


def _training_config(args: argparse.Namespace, source_snapshot: bytes, labels_sha256: str) -> dict[str, Any]:
    return {
        "history_length": args.history_length,
        "baseline_frames": args.baseline_frames,
        "sample_period_s": 0.1,
        "latent_dim": 32,
        "geometry_scale_m": args.geometry_scale_m,
        "geometry_supervision": True,
        "geometry_target": "peg-tip pose minus terminal peg-tip pose; terminal hole reference, not independent metrology",
        "xy_threshold_m": args.xy_threshold_m,
        "z_threshold_m": args.z_threshold_m,
        "tool_offset_local_mm": args.tool_offset_mm.tolist(),
        "classes": ["misaligned_contact", "aligned_progress"],
        "input_force_key": args.force_key,
        "training_recipe": TRAINING_RECIPE,
        "training_first_hit_cutoff_enabled": False,
        "geometry_phases": sorted(GEOMETRY_PHASES),
        "quality_boundary_guard": args.boundary_guard,
        "sampling": "independent class-balanced/trajectory-tempered and depth/phase/trajectory-balanced batches",
        "classification_trajectory_mass": "sqrt(reviewed_window_count), within each equally weighted class",
        "classification_confidence_usage": "cross-entropy loss only; not sampling probability",
        "pair_batch_size_per_task": args.batch_size,
        "steps_per_epoch_basis": "ceil(reviewed_classification_pair_windows/batch_size)",
        "gain_augmentation_range": [0.85, 1.15],
        "selection_depth_scale_mm": args.selection_depth_scale_mm,
        "selection_score": "0.60*min_domain_macro_f1+0.40*mean_domain_exp(-selection_depth_mae_mm/depth_scale_mm)",
        "min_selection_epoch": args.min_selection_epoch,
        "min_macro_f1": args.min_macro_f1,
        "min_class_recall": args.min_class_recall,
        "cv_epoch_budget": args.epochs,
        "early_stopping_enabled": False,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_epochs": args.warmup_epochs,
        "learning_rate_schedule": "epoch warmup then cosine; final fit uses the CV schedule prefix",
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "training_source_sha256": hashlib.sha256(source_snapshot).hexdigest(),
        "labels_sha256": labels_sha256,
        "real_root": str(args.real_root.resolve()),
        "sim_root": str(args.sim_root.resolve()),
        "loss_weights": {"class": args.class_loss_weight, "consistency": args.consistency_loss_weight,
                         "geometry": args.geometry_loss_weight, "direction": args.direction_loss_weight},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--output", type=Path,
                        default=Path("eval_outputs/insertion_alignment_encoder_full_insertion/encoder.pt"))
    parser.add_argument("--tool-offset-mm", nargs=3, type=float, default=[2.576, 32.243, -0.898])
    parser.add_argument("--xy-threshold-m", type=float, default=0.003)
    parser.add_argument("--z-threshold-m", type=float, default=0.003)
    parser.add_argument("--geometry-scale-m", type=float, default=0.01)
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--baseline-frames", type=int, default=20)
    parser.add_argument("--boundary-guard", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100, help="full epoch budget for each CV fold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0", help="logical device within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--held-out-trajectories", nargs="+", type=int, default=HELD_OUT_TRAJECTORIES)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--min-selection-epoch", type=int, default=1)
    parser.add_argument("--min-macro-f1", type=float, default=0.70)
    parser.add_argument("--min-class-recall", type=float, default=0.60)
    parser.add_argument("--selection-depth-scale-mm", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--min-learning-rate", type=float, default=0.00003)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.20, help="temporal dropout during training; inactive at inference")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.15)
    parser.add_argument("--geometry-loss-weight", type=float, default=0.12)
    parser.add_argument("--direction-loss-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--geometry-only", "--prepare-only", dest="prepare_only", action="store_true",
                        help="print first-hit markers and full-insertion coverage, without training or writing")
    args = parser.parse_args()
    args.tool_offset_mm = np.asarray(args.tool_offset_mm, dtype=np.float64)
    if (args.history_length < 15 or args.baseline_frames < 1 or args.boundary_guard < 0
            or args.batch_size < 1 or args.epochs < 1 or args.cv_folds < 2 or args.log_every < 1):
        parser.error("history-length must be >=15 for the linear baseline; other sizes must be positive (guard >=0)")
    if (args.geometry_scale_m <= 0 or args.xy_threshold_m <= 0 or args.z_threshold_m <= 0
            or args.selection_depth_scale_mm <= 0):
        parser.error("geometry and selection scales/thresholds must be positive")
    if not 1 <= args.min_selection_epoch <= args.epochs:
        parser.error("min-selection-epoch must be within 1..epochs")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("warmup-epochs must be >=0 and less than epochs")
    if not 0 <= args.min_macro_f1 <= 1 or not 0 <= args.min_class_recall <= 1:
        parser.error("classification quality requirements must be within [0,1]")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        parser.error("learning rates must satisfy 0 < min-learning-rate <= learning-rate")
    if not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing < 1 or args.weight_decay < 0:
        parser.error("dropout/label-smoothing must be within [0,1); weight-decay must be nonnegative")
    if (args.class_loss_weight <= 0 or args.geometry_loss_weight <= 0
            or args.consistency_loss_weight < 0 or args.direction_loss_weight < 0):
        parser.error("class/geometry weights must be positive; other loss weights must be nonnegative")
    report_path = args.output.with_suffix(".report.json")
    if not args.prepare_only and (args.output.exists() or report_path.exists()):
        raise FileExistsError(f"output already exists; choose a new --output: {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    # Capture the source used by this run before any long GPU work starts.
    source_snapshot = Path(__file__).read_bytes()
    supporting_sources = {
        name: Path(__file__).with_name(name).read_bytes()
        for name in ("model.py", "preprocess.py", "train.py")
    }
    labels_sha256 = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    config = _training_config(args, source_snapshot, labels_sha256)
    config["supporting_source_sha256"] = {
        name: hashlib.sha256(source).hexdigest() for name, source in supporting_sources.items()
    }
    labels = read_labels(args.labels)
    ids = sorted(labels)
    test_ids = sorted(set(args.held_out_trajectories))
    if not test_ids or not set(test_ids).issubset(ids):
        parser.error("held-out-trajectories must name existing trajectories in the label CSV")
    train_ids = sorted(set(ids) - set(test_ids))
    if len(train_ids) < args.cv_folds:
        parser.error("not enough training trajectories for the requested CV folds")
    episodes, cutoffs, hole_references, pose_disagreement = load_terminal_episodes(args, labels)
    print(f"Recovered terminal-referenced first-hit markers for {len(ids)} trajectories; "
          f"max paired pose difference={pose_disagreement:.6f} mm", flush=True)
    print("First-hit markers (not training cutoffs):",
          {f"traj_{key}": value for key, value in cutoffs.items()}, flush=True)
    print(f"Locked held-out trajectories: {test_ids}", flush=True)
    table, scales = _window_table(episodes, labels, train_ids, ids, args)
    train_windows, test_windows = _subset(table, train_ids), _subset(table, test_ids)
    coverage = {"train": window_coverage(train_windows), "held_out": window_coverage(test_windows)}
    for split, counts in coverage.items():
        print(f"COVERAGE {split}: paired geometry windows={counts['pair_windows']}, "
              f"classification windows={counts['classification_pair_windows']}, "
              f"phases={counts['by_phase']}, depth bands={counts['by_depth']}", flush=True)
        if counts["by_depth"]["0_3mm"] == 0:
            raise ValueError(f"{split}: no supervised windows in the last 3 mm")
    if args.prepare_only:
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the GPU host and select a GPU with CUDA_VISIBLE_DEVICES")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output.parent / f"{args.output.stem}_training_history"
    trace_dir.mkdir(exist_ok=False)
    with (trace_dir / "train_terminal_geometry.py").open("xb") as handle:
        handle.write(source_snapshot)
    for name, source in supporting_sources.items():
        with (trace_dir / name).open("xb") as handle:
            handle.write(source)
    inner_folds = grouped_folds(_classification_windows(train_windows), train_ids, args.cv_folds, args.seed + 1)
    # Save the actual settings and split before fitting. Failed runs need the
    # same provenance as successful runs, without consuming held-out metrics.
    prepared_hash = hashlib.sha256()
    for (trajectory, domain), episode in sorted(episodes.items()):
        prepared_hash.update(f"traj={trajectory},domain={domain}".encode())
        for array in (episode.force, episode.time, episode.geometry):
            prepared_hash.update(f"{array.shape}:{array.dtype}".encode())
            prepared_hash.update(array.tobytes())
    config["prepared_data_sha256"] = prepared_hash.hexdigest()
    with (trace_dir / "run_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump({"config": config, "train_trajectories": train_ids, "held_out_trajectories": test_ids,
                   "cv_validation_trajectories": inner_folds, "window_coverage": coverage,
                   "final_fit_feature_scales": scales.tolist()},
                  handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    fold_histories: list[list[dict[str, Any]]] = []
    print(f"Full CV budget: {args.cv_folds} folds x {args.epochs} epochs; "
          f"selection starts at epoch {args.min_selection_epoch}, "
          f"min_macro_f1={args.min_macro_f1}, min_each_class_recall={args.min_class_recall}", flush=True)
    print(f"Regularization: weight_decay={args.weight_decay}, dropout={args.dropout}, "
          f"label_smoothing={args.label_smoothing}; steps/epoch follow classification window count", flush=True)
    for index, valid_ids in enumerate(inner_folds):
        fit_ids = sorted(set(train_ids) - set(valid_ids))
        table, fold_scales = _window_table(episodes, labels, fit_ids, train_ids, args)
        checkpoint_context = {
            "config": config, "fit_trajectories": fit_ids, "valid_trajectories": valid_ids,
            "feature_scales": fold_scales.tolist(),
        }
        model, history = train_one(_subset(table, fit_ids), _subset(table, valid_ids),
                                   args, device, args.seed + index, args.epochs,
                                   f"cv_fold_{index + 1}", trace_dir / f"cv_fold_{index + 1}.jsonl",
                                   checkpoint_context=checkpoint_context)
        fold_histories.append(history)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    final_epochs, cv_summaries = select_common_epoch(fold_histories, args, trace_dir / "cv_epoch_summary.json")
    if final_epochs is None:
        best = max(cv_summaries, key=lambda row: row["score"])
        rejected_report = {
            "status": "rejected_cv", "config": config,
            "reason": "No common CV epoch meets all quality requirements; final fit was not performed.",
            "metrics": {
                "window_coverage": coverage,
                "train_trajectories": train_ids, "held_out_trajectories": test_ids,
                "held_out_test": None, "final_training_epochs": None,
                "total_cv_optimizer_steps": sum(history[-1]["optimizer_steps"] for history in fold_histories),
                "training_history_directory": str(trace_dir),
                "best_diagnostic_epoch": best["epoch"],
                "best_diagnostic_epoch_metrics": best["metrics"],
                "best_diagnostic_epoch_failures": best["failures"],
                "best_diagnostic_fold_checkpoints": [history[best["epoch"] - 1]["checkpoint"]
                                                    for history in fold_histories],
            },
        }
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(rejected_report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Saved rejected-run report: {report_path}. CV snapshots retained in {trace_dir}. "
              "No deployable encoder.pt was created. Exit code 2 indicates failed quality requirements.", flush=True)
        return 2
    inner_reports = [
        {"valid_trajectories": valid_ids, "selected_epoch": final_epochs,
         "epochs_ran": len(history), "optimizer_steps": history[-1]["optimizer_steps"],
         "metrics": history[final_epochs - 1]["validation"]}
        for valid_ids, history in zip(inner_folds, fold_histories, strict=True)
    ]
    model, final_history = train_one(train_windows, None, args, device, args.seed + 1000, final_epochs,
                                     "final_fit", trace_dir / "final_fit.jsonl")
    test_reports, _, _ = _evaluate(model, test_windows, device, args.geometry_scale_m)
    baseline_reports = linear_baselines(train_windows, test_windows)
    held_out_failures = _selection_failures(test_reports, final_epochs, args)
    for domain in ("real", "sim"):
        classification = test_reports[domain]["class"]
        print(f"HELD-OUT {domain}: accuracy={classification['accuracy']:.3f}, "
              f"macro_f1={classification['macro_f1']:.3f}, "
              f"misaligned_recall={classification['per_class']['misaligned_contact']['recall']:.3f}, "
              f"aligned_recall={classification['per_class']['aligned_progress']['recall']:.3f}, "
              f"selection_depth_mae_mm={test_reports[domain]['selection_depth_mae_mm']:.3f}, "
              f"near_z_mae_mm={test_reports[domain]['near_terminal_z_mae_mm']:.3f}", flush=True)
    if held_out_failures:
        print(f"HELD-OUT quality requirements were not met: {held_out_failures}. "
              "This checkpoint is a diagnostic result; held-out data did not select its epoch.", flush=True)
    model.to("cpu")
    payload = {
        "artifact_format": "tavla_force_alignment_encoder_terminal_reference_v1",
        "config": {**config, "model_parameter_count": sum(parameter.numel() for parameter in model.parameters())},
        "preprocess": {
            "feature_scales": scales.tolist(),
            "feature_order": ["force_x", "force_y", "force_z", "dforce_x_per_s", "dforce_y_per_s", "dforce_z_per_s"],
            "baseline": f"coordinate-wise median of the first {args.baseline_frames} force samples",
        },
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "geometry": {
            "hole_reference_world_m": {f"traj_{key}": value for key, value in hole_references.items()},
            "first_hit_cutoff_frames": {f"traj_{key}": value for key, value in cutoffs.items()},
            "previous_reconstructed_cutoff_frames": {f"traj_{key}": RECONSTRUCTED_CUTOFFS[key]
                                                       for key in ids if key in RECONSTRUCTED_CUTOFFS},
            "max_real_sim_pose_disagreement_mm": pose_disagreement,
            "first_hit_marker_usage": "diagnostic only; no frames are cut from training at this marker",
        },
        "metrics": {
            "held_out_test": test_reports,
            "linear_force_baseline": baseline_reports,
            "inner_folds": inner_reports,
            "train_trajectories": train_ids,
            "held_out_trajectories": test_ids,
            "train_pair_windows": len(train_windows.y),
            "test_pair_windows": len(test_windows.y),
            "final_training_epochs": final_epochs,
            "window_coverage": coverage,
            "cv_selected_epoch_metrics": cv_summaries[final_epochs - 1]["metrics"],
            "final_optimizer_steps": final_history[-1]["optimizer_steps"],
            "total_cv_optimizer_steps": sum(history[-1]["optimizer_steps"] for history in fold_histories),
            "training_history_directory": str(trace_dir),
            "held_out_quality_requirements": {"met": not held_out_failures, "failures": held_out_failures},
        },
    }
    _save_new_checkpoint(payload, args.output)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump({"status": "passed_cv" if not held_out_failures else "diagnostic_held_out_failed",
                   "artifact_format": payload["artifact_format"], "config": payload["config"],
                   "geometry": payload["geometry"], "metrics": payload["metrics"]},
                  handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved new encoder: {args.output}", flush=True)
    print(f"Saved depth/phase metrics and coverage: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
