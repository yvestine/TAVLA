#!/usr/bin/env python3
"""Train/evaluate a paired force-only encoder while preserving insertion phase."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from force_gap_encoder_v2.features import apply_gain_to_normalized, fit_normalizer, force_gap_features, normalize
from force_gap_encoder_v2.model import ForceGapEncoderV2


LABEL_IDS = {"misaligned_contact": 0, "aligned_progress": 1}
LABEL_NAMES = ["misaligned_contact", "aligned_progress"]
DEFAULT_HOLDOUT = (0, 2, 21, 22, 23, 33, 34, 39)


@dataclass
class Episode:
    trajectory: int
    domain: int  # 0=Real, 1=Sim
    raw_features: np.ndarray
    cutoff: int


@dataclass
class PairWindows:
    x: np.ndarray  # [N,2,history,12], Real then Sim
    y: np.ndarray
    weight: np.ndarray
    trajectory: np.ndarray
    frame: np.ndarray


def read_labels(path: Path) -> dict[int, list[dict[str, Any]]]:
    labels: dict[int, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            trajectory = int(row["trajectory"].removeprefix("traj_"))
            labels.setdefault(trajectory, []).append(
                {
                    "start": int(row["start_frame"]),
                    "end": int(row["end_frame"]),
                    "label": row["label"],
                    "confidence": row["confidence"].strip().lower(),
                }
            )
    return labels


def load_cutoffs(path: Path, trajectories: list[int]) -> tuple[dict[int, int], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("artifact_format") != "tavla_force_alignment_encoder_terminal_reference_v1":
        raise ValueError(f"Unexpected cutoff artifact format in {path}")
    raw = payload.get("geometry", {}).get("first_hit_cutoff_frames", {})
    cutoffs = {int(str(key).removeprefix("traj_")): int(value) for key, value in raw.items()}
    missing = sorted(set(trajectories) - set(cutoffs))
    if missing:
        raise ValueError(f"cutoff checkpoint is missing trajectories: {missing}")
    return cutoffs, payload.get("geometry", {})


def load_episodes(args: argparse.Namespace, labels: dict[int, list[dict[str, Any]]]):
    cutoffs, geometry_meta = load_cutoffs(args.cutoff_checkpoint, sorted(labels))
    episodes: dict[tuple[int, int], Episode] = {}
    sampling_diagnostics: dict[str, dict[str, float]] = {}
    for trajectory in sorted(labels):
        lengths = []
        timestamp_sequences = []
        period_summaries = []
        for domain, root in enumerate((args.real_root, args.sim_root)):
            path = root / f"traj_{trajectory}" / "data.h5"
            with h5py.File(path, "r") as h5:
                force = np.asarray(h5[args.force_key], dtype=np.float32)
                timestamps = np.asarray(h5["timestamps"], dtype=np.float64)
            if force.ndim != 2 or force.shape[1] < 3:
                raise ValueError(f"{path}: expected force [T,>=3], got {force.shape}")
            force = force[:, :3]
            if len(force) != len(timestamps) or len(force) < args.baseline_frames:
                raise ValueError(f"{path}: force/timestamp length mismatch or episode shorter than calibration")
            if not np.isfinite(force).all() or not np.isfinite(timestamps).all():
                raise ValueError(f"{path}: non-finite force or timestamp")
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"{path}: timestamps must be strictly increasing")
            periods = np.diff(timestamps)
            median_period = float(np.median(periods))
            if not 0.08 <= median_period <= 0.12:
                raise ValueError(
                    f"{path}: median frame period {median_period:.6f}s is outside the expected 10 Hz range"
                )
            features, _ = force_gap_features(
                force,
                baseline_frames=args.baseline_frames,
                insertion_axis=np.asarray(args.insertion_axis, dtype=np.float32),
                smoothing_frames=args.smoothing_frames,
                trend_lag_frames=args.trend_lag_frames,
                active_noise_mad_multiplier=args.active_noise_mad_multiplier,
                active_noise_floor_newton=args.active_noise_floor_newton,
            )
            lengths.append(len(force))
            timestamp_sequences.append(timestamps)
            period_summaries.append(
                {
                    "median_period_s": median_period,
                    "p01_period_s": float(np.quantile(periods, 0.01)),
                    "p99_period_s": float(np.quantile(periods, 0.99)),
                }
            )
            episodes[(trajectory, domain)] = Episode(
                trajectory=trajectory,
                domain=domain,
                raw_features=features,
                cutoff=min(cutoffs[trajectory], len(force)),
            )
        if lengths[0] != lengths[1]:
            raise ValueError(f"traj_{trajectory}: Real/Sim lengths differ: {lengths}")
        timestamp_difference = float(np.max(np.abs(timestamp_sequences[0] - timestamp_sequences[1])))
        if timestamp_difference > 1e-6:
            raise ValueError(
                f"traj_{trajectory}: Real/Sim timestamps differ by up to {timestamp_difference:.9g}s; "
                "paired force windows would not be frame-aligned"
            )
        sampling_diagnostics[str(trajectory)] = {
            "real_median_period_s": period_summaries[0]["median_period_s"],
            "real_p01_period_s": period_summaries[0]["p01_period_s"],
            "real_p99_period_s": period_summaries[0]["p99_period_s"],
            "sim_median_period_s": period_summaries[1]["median_period_s"],
            "sim_p01_period_s": period_summaries[1]["p01_period_s"],
            "sim_p99_period_s": period_summaries[1]["p99_period_s"],
            "max_real_sim_timestamp_difference_s": timestamp_difference,
        }
        cursor = 0
        for row in labels[trajectory]:
            if row["start"] != cursor or row["end"] < row["start"]:
                raise ValueError(f"traj_{trajectory}: label intervals must be contiguous near frame {cursor}")
            cursor = row["end"] + 1
        if cursor != lengths[0]:
            raise ValueError(f"traj_{trajectory}: labels cover {cursor} frames, expected {lengths[0]}")
    return episodes, cutoffs, geometry_meta, sampling_diagnostics


def build_windows(
    episodes: dict[tuple[int, int], Episode],
    labels: dict[int, list[dict[str, Any]]],
    trajectories: list[int],
    center: np.ndarray,
    scale: np.ndarray,
    history: int,
    guard: int,
) -> PairWindows:
    rows, targets, weights, trajectory_ids, frames = [], [], [], [], []
    for trajectory in trajectories:
        real, sim = episodes[(trajectory, 0)], episodes[(trajectory, 1)]
        xr = normalize(real.raw_features, center, scale)
        xs = normalize(sim.raw_features, center, scale)
        for interval in labels[trajectory]:
            label = interval["label"]
            if label not in LABEL_IDS or interval["confidence"] == "low":
                continue
            start = interval["start"] + guard
            end = min(interval["end"] - guard, real.cutoff - 1)
            confidence_weight = 1.0 if interval["confidence"] == "high" else 0.65
            for frame in range(start, end + 1, 1):
                lo = max(0, frame - history + 1)
                wr, ws = xr[lo : frame + 1], xs[lo : frame + 1]
                if len(wr) < history:
                    pad = history - len(wr)
                    wr = np.pad(wr, ((pad, 0), (0, 0)), mode="edge")
                    ws = np.pad(ws, ((pad, 0), (0, 0)), mode="edge")
                rows.append(np.stack((wr, ws)))
                targets.append(LABEL_IDS[label])
                weights.append(confidence_weight)
                trajectory_ids.append(trajectory)
                frames.append(frame)
    if not rows:
        raise ValueError("No labeled windows remain after confidence, boundary, and cutoff filters")
    return PairWindows(
        np.asarray(rows, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        np.asarray(trajectory_ids, dtype=np.int32),
        np.asarray(frames, dtype=np.int32),
    )


def fit_fold_normalizer(episodes, train_ids):
    arrays = [
        episodes[(trajectory, domain)].raw_features[: episodes[(trajectory, domain)].cutoff]
        for trajectory in train_ids
        for domain in (0, 1)
    ]
    return fit_normalizer(arrays)


def split_folds(trajectories: list[int], holdout: set[int], folds: int, seed: int) -> list[list[int]]:
    available = np.asarray(sorted(set(trajectories) - holdout), dtype=np.int32)
    if len(available) < folds:
        raise ValueError(f"need at least {folds} development trajectories, got {len(available)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(available)
    return [list(map(int, group)) for group in np.array_split(available, folds)]


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (y.astype(np.int64), prediction.astype(np.int64)), 1)
    f1s = []
    per_class = {}
    for cls, name in enumerate(LABEL_NAMES):
        tp = int(matrix[cls, cls])
        precision = tp / max(int(matrix[:, cls].sum()), 1)
        recall = tp / max(int(matrix[cls].sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[cls].sum())}
    return {
        "count": int(matrix.sum()),
        "accuracy": float(np.trace(matrix) / max(int(matrix.sum()), 1)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def select_shared_threshold(y: np.ndarray, real_probability: np.ndarray, sim_probability: np.ndarray):
    candidates = np.linspace(0.20, 0.80, 121)
    scores = []
    for threshold in candidates:
        real_result = metrics(y, (real_probability >= threshold).astype(np.int64))
        sim_result = metrics(y, (sim_probability >= threshold).astype(np.int64))
        scores.append((min(real_result["macro_f1"], sim_result["macro_f1"]), -abs(threshold - 0.5), threshold))
    _, _, selected = max(scores)
    return float(selected)


def paired_cosine(values: np.ndarray) -> float:
    paired = np.asarray(values, dtype=np.float32).reshape(len(values), 2, -1)
    dot = np.sum(paired[:, 0] * paired[:, 1], axis=-1)
    denom = np.linalg.norm(paired[:, 0], axis=-1) * np.linalg.norm(paired[:, 1], axis=-1)
    return float(np.mean(dot / np.maximum(denom, 1e-8)))


def predict(model: ForceGapEncoderV2, x: np.ndarray, device: torch.device):
    model.eval()
    probability, latent = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            inputs = torch.from_numpy(x[start : start + 512]).to(device)
            z, logits = model(inputs)
            probability.append(torch.softmax(logits, dim=-1).cpu().numpy())
            latent.append(z.cpu().numpy())
    return np.concatenate(probability), np.concatenate(latent)


def train_model(windows, config, device, seed, validation=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ForceGapEncoderV2(config["history_length"], len(config["feature_order"]), config["latent_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    x = torch.from_numpy(windows.x).to(device)
    y = torch.from_numpy(windows.y).to(device)
    weights = torch.from_numpy(windows.weight).to(device)
    class_weighting = config.get("class_weighting", "none")
    if class_weighting == "inverse":
        class_counts = np.bincount(windows.y.astype(np.int64), minlength=len(LABEL_NAMES)).astype(np.float32)
        class_weights = len(windows.y) / np.maximum(len(LABEL_NAMES) * class_counts, 1.0)
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    elif class_weighting == "none":
        class_weights = torch.ones(len(LABEL_NAMES), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported class weighting mode: {class_weighting}")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_score, stale = config["epochs"], -float("inf"), 0
    best_validation = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = rng.permutation(len(windows.y))
        for start in range(0, len(order), config["batch_size"]):
            index = torch.as_tensor(order[start : start + config["batch_size"]], device=device)
            pair = x[index]
            flat = pair.reshape(-1, config["history_length"], len(config["feature_order"]))
            gain_min, gain_max = config.get("augmentation_gain_range", (0.70, 1.30))
            gain_sampling = config.get("augmentation_gain_sampling", "uniform")
            if gain_sampling == "log_uniform":
                gain = torch.empty((len(flat), 1), device=device).uniform_(np.log(gain_min), np.log(gain_max)).exp()
            elif gain_sampling == "uniform":
                gain = torch.empty((len(flat), 1), device=device).uniform_(gain_min, gain_max)
            else:
                raise ValueError(f"Unsupported gain sampling mode: {gain_sampling}")
            flat = apply_gain_to_normalized(
                flat,
                torch.as_tensor(config["normalizer_center"], device=device),
                torch.as_tensor(config["normalizer_scale"], device=device),
                gain,
            )
            latent, logits = model(flat)
            labels = y[index].repeat_interleave(2)
            sample_weights = weights[index].repeat_interleave(2) * class_weights[labels]
            quality_loss = (F.cross_entropy(logits, labels, reduction="none") * sample_weights).mean()
            paired = latent.reshape(-1, 2, config["latent_dim"])
            alignment_loss = (1.0 - F.cosine_similarity(paired[:, 0], paired[:, 1], dim=-1)).mean()
            loss = quality_loss + config["alignment_weight"] * alignment_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        if validation is None:
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            continue
        probability, latent = predict(model, validation.x.reshape(-1, config["history_length"], len(config["feature_order"])), device)
        prediction = probability.argmax(axis=-1).reshape(-1, 2)
        real_score = metrics(validation.y, prediction[:, 0])["macro_f1"]
        sim_score = metrics(validation.y, prediction[:, 1])["macro_f1"]
        score = min(real_score, sim_score)
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            paired = latent.reshape(-1, 2, config["latent_dim"])
            best_validation = {
                "real_prediction": prediction[:, 0],
                "sim_prediction": prediction[:, 1],
                "real_probability": probability.reshape(-1, 2, 2)[:, 0, 1],
                "sim_probability": probability.reshape(-1, 2, 2)[:, 1, 1],
                "raw_feature_pair_cosine": paired_cosine(validation.x),
                "latent_pair_cosine": float(
                    np.mean(
                        np.sum(paired[:, 0] * paired[:, 1], axis=-1)
                        / np.maximum(np.linalg.norm(paired[:, 0], axis=-1) * np.linalg.norm(paired[:, 1], axis=-1), 1e-8)
                    )
                ),
            }
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_epoch, best_validation


def main():
    recipe_parser = argparse.ArgumentParser(add_help=False)
    recipe_parser.add_argument("--recipe", choices=("legacy_v2", "motion_v1"), default="legacy_v2")
    recipe, remaining = recipe_parser.parse_known_args()
    if recipe.recipe == "motion_v1":
        from force_gap_encoder_v2.motion_train import main as motion_main

        sys.argv = [sys.argv[0], *remaining]
        motion_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("legacy_v2", "motion_v1"), default="legacy_v2")
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--cutoff-checkpoint", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/encoder.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--baseline-frames", type=int, default=20)
    parser.add_argument("--boundary-guard", type=int, default=5)
    parser.add_argument("--smoothing-frames", type=int, default=3)
    parser.add_argument("--trend-lag-frames", type=int, default=5)
    parser.add_argument("--active-noise-mad-multiplier", type=float, default=6.0)
    parser.add_argument("--active-noise-floor-newton", type=float, default=1e-6)
    parser.add_argument("--insertion-axis", type=float, nargs=3, default=(0.0, 0.0, -1.0))
    parser.add_argument("--alignment-weights", type=float, nargs="+", default=(0.0, 0.01, 0.05))
    parser.add_argument("--class-weighting", choices=("none", "inverse"), default="none")
    parser.add_argument("--augmentation-gain-range", type=float, nargs=2, default=(0.70, 1.30), metavar=("MIN", "MAX"))
    parser.add_argument("--augmentation-gain-sampling", choices=("uniform", "log_uniform"), default="uniform")
    parser.add_argument("--holdout-trajectories", type=int, nargs="+", default=DEFAULT_HOLDOUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.folds < 2 or args.history_length < 1 or args.baseline_frames < 1:
        raise ValueError("folds, history length, and baseline frames must be positive (folds >= 2)")
    if not args.alignment_weights or any(weight < 0 for weight in args.alignment_weights):
        raise ValueError("alignment weights must be a non-empty list of nonnegative values")
    if args.augmentation_gain_range[0] <= 0 or args.augmentation_gain_range[1] < args.augmentation_gain_range[0]:
        raise ValueError("augmentation gain range must satisfy 0 < min <= max")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a valid device or use --device cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    labels = read_labels(args.labels)
    episodes, cutoffs, geometry_meta, sampling_diagnostics = load_episodes(args, labels)
    ids = sorted(labels)
    holdout = set(args.holdout_trajectories) & set(ids)
    folds = split_folds(ids, holdout, args.folds, args.seed)
    dev_ids = sorted(set(ids) - holdout)
    if not dev_ids:
        raise ValueError("no development trajectories remain after holdout selection")

    config_base = {
        "history_length": args.history_length,
        "baseline_frames": args.baseline_frames,
        "latent_dim": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "class_weighting": args.class_weighting,
        "augmentation_gain_range": list(args.augmentation_gain_range),
        "augmentation_gain_sampling": args.augmentation_gain_sampling,
        "feature_order": [
            "force_dir_x", "force_dir_y", "force_dir_z", "trend_dir_x", "trend_dir_y", "trend_dir_z",
            "axial_signed_fraction", "lateral_ratio", "log1p_force_norm", "log1p_trend_norm", "force_valid", "trend_valid",
        ],
    }
    results: dict[float, dict[str, Any]] = {}
    for alignment_weight in args.alignment_weights:
        fold_reports, oof_real_y, oof_sim_y, oof_real_pred, oof_sim_pred = [], [], [], [], []
        oof_real_probability, oof_sim_probability = [], []
        fold_epochs, raw_cosine_values, cosine_values = [], [], []
        for fold_index, validation_ids in enumerate(folds):
            train_ids = sorted(set(dev_ids) - set(validation_ids))
            center, scale = fit_fold_normalizer(episodes, train_ids)
            train_all = build_windows(episodes, labels, train_ids, center, scale, args.history_length, args.boundary_guard)
            valid = build_windows(episodes, labels, validation_ids, center, scale, args.history_length, args.boundary_guard)
            config = dict(
                config_base,
                alignment_weight=float(alignment_weight),
                normalizer_center=center.tolist(),
                normalizer_scale=scale.tolist(),
            )
            model, best_epoch, valid_result = train_model(
                train_all, config, torch.device(args.device), args.seed + fold_index, valid
            )
            if valid_result is None:
                raise RuntimeError("fold training did not produce validation predictions")
            fold_epochs.append(best_epoch)
            raw_cosine_values.append(valid_result["raw_feature_pair_cosine"])
            cosine_values.append(valid_result["latent_pair_cosine"])
            oof_real_y.append(valid.y)
            oof_sim_y.append(valid.y)
            oof_real_pred.append(valid_result["real_prediction"])
            oof_sim_pred.append(valid_result["sim_prediction"])
            oof_real_probability.append(valid_result["real_probability"])
            oof_sim_probability.append(valid_result["sim_probability"])
            report = {
                "fold": fold_index,
                "held_out_trajectories": validation_ids,
                "best_epoch": best_epoch,
                "real": metrics(valid.y, valid_result["real_prediction"]),
                "sim": metrics(valid.y, valid_result["sim_prediction"]),
                "raw_feature_pair_cosine": valid_result["raw_feature_pair_cosine"],
                "latent_pair_cosine": valid_result["latent_pair_cosine"],
            }
            fold_reports.append(report)

        real_metric = metrics(np.concatenate(oof_real_y), np.concatenate(oof_real_pred))
        sim_metric = metrics(np.concatenate(oof_sim_y), np.concatenate(oof_sim_pred))
        oof_y = np.concatenate(oof_real_y)
        real_prob = np.concatenate(oof_real_probability)
        sim_prob = np.concatenate(oof_sim_probability)
        threshold = select_shared_threshold(oof_y, real_prob, sim_prob)
        results[float(alignment_weight)] = {
            "folds": fold_reports,
            "real": real_metric,
            "sim": sim_metric,
            "balanced_macro_f1": min(real_metric["macro_f1"], sim_metric["macro_f1"]),
            "raw_feature_pair_cosine": float(np.mean(raw_cosine_values)),
            "latent_pair_cosine": float(np.mean(cosine_values)),
            "fold_epochs": fold_epochs,
            "oof_threshold": threshold,
            "real_at_oof_threshold": metrics(oof_y, (real_prob >= threshold).astype(np.int64)),
            "sim_at_oof_threshold": metrics(oof_y, (sim_prob >= threshold).astype(np.int64)),
        }
        print(
            f"lambda={alignment_weight:g} OOF real_f1={real_metric['macro_f1']:.3f} "
            f"sim_f1={sim_metric['macro_f1']:.3f} "
            f"raw_pair_cos={np.mean(raw_cosine_values):.3f} "
            f"latent_pair_cos={np.mean(cosine_values):.3f}",
            flush=True,
        )

    best_task_score = max(result["balanced_macro_f1"] for result in results.values())
    eligible = [
        weight for weight, result in results.items()
        if result["balanced_macro_f1"] >= best_task_score - 0.02
    ]
    selected_weight = max(eligible, key=lambda weight: (results[weight]["latent_pair_cosine"], -weight))

    final_center, final_scale = fit_fold_normalizer(episodes, dev_ids)
    final_train = build_windows(episodes, labels, dev_ids, final_center, final_scale, args.history_length, args.boundary_guard)
    selected = results[selected_weight]
    final_epochs = max(1, int(np.median(selected["fold_epochs"])))
    final_config = dict(
        config_base,
        epochs=final_epochs,
        patience=final_epochs + 1,
        alignment_weight=float(selected_weight),
        normalizer_center=final_center.tolist(),
        normalizer_scale=final_scale.tolist(),
    )
    final_model, _, _ = train_model(final_train, final_config, torch.device(args.device), args.seed + 10_000)

    holdout_report = None
    if holdout:
        test = build_windows(episodes, labels, sorted(holdout), final_center, final_scale, args.history_length, args.boundary_guard)
        probability, latent = predict(
            final_model,
            test.x.reshape(-1, args.history_length, len(config_base["feature_order"])),
            torch.device(args.device),
        )
        prediction = probability.argmax(axis=-1).reshape(-1, 2)
        probability = probability.reshape(-1, 2, 2)
        paired = latent.reshape(-1, 2, config_base["latent_dim"])
        selected_threshold = float(selected["oof_threshold"])
        holdout_report = {
            "trajectories": sorted(holdout),
            "real": metrics(test.y, prediction[:, 0]),
            "sim": metrics(test.y, prediction[:, 1]),
            "diagnostic_threshold_selected_on_grouped_oof": selected_threshold,
            "real_at_oof_threshold": metrics(test.y, (probability[:, 0, 1] >= selected_threshold).astype(np.int64)),
            "sim_at_oof_threshold": metrics(test.y, (probability[:, 1, 1] >= selected_threshold).astype(np.int64)),
            "raw_feature_pair_cosine": paired_cosine(test.x),
            "latent_pair_cosine": float(
                np.mean(
                    np.sum(paired[:, 0] * paired[:, 1], axis=-1)
                    / np.maximum(np.linalg.norm(paired[:, 0], axis=-1) * np.linalg.norm(paired[:, 1], axis=-1), 1e-8)
                )
            ),
        }

    preprocess = {
        "feature_order": config_base["feature_order"],
        "normalizer_center": final_center.tolist(),
        "normalizer_scale": final_scale.tolist(),
        "insertion_axis_base": list(args.insertion_axis),
        "baseline": "coordinate-wise median of first 20 force samples",
        "smoothing_frames": args.smoothing_frames,
        "trend_lag_frames": args.trend_lag_frames,
        "active_noise_mad_multiplier": args.active_noise_mad_multiplier,
        "active_noise_floor_newton": args.active_noise_floor_newton,
        "force_units": "newtons",
        "force_frame": "robot base frame",
        "sample_rate_hz": 10.0,
    }
    payload = {
        "artifact_format": "tavla_force_gap_encoder_v2",
        "config": final_config,
        "preprocess": preprocess,
        "model_state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
        "selection": {
            "alignment_weight_by_cv": {str(weight): result for weight, result in results.items()},
            "selected_alignment_weight": selected_weight,
            "rule": "maximize the lower of Real/Sim grouped-OOF macro-F1; among candidates within 0.02, choose the highest paired latent cosine",
        },
        "holdout": holdout_report,
        "data_summary": {
            "development_trajectories": dev_ids,
            "held_out_trajectories": sorted(holdout),
            "cutoff_frame_by_trajectory": {str(key): int(value) for key, value in cutoffs.items()},
            "cutoff_source": geometry_meta,
            "sampling_diagnostics": sampling_diagnostics,
            "timebase_note": "windows use HDF5 frame order; timestamps confirm paired Real/Sim alignment and nominal median 10 Hz cadence",
            "labeled_window_count_development": int(len(final_train.y)),
            "class_counts_development": {LABEL_NAMES[i]: int((final_train.y == i).sum()) for i in range(2)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps({key: value for key, value in payload.items() if key != "model_state_dict"}, indent=2), encoding="utf-8")
    print(f"selected lambda={selected_weight:g}; encoder saved to {args.output}; report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
