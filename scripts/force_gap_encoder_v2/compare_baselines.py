#!/usr/bin/env python3
"""Compare raw force features and the saved old/new encoders on fixed groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from force_gap_encoder_v2.model import ForceGapEncoderV2
from force_gap_encoder_v2.train import (
    DEFAULT_HOLDOUT,
    build_windows,
    load_episodes,
    metrics,
    paired_cosine,
    read_labels,
)
from insertion_alignment_encoder.model import ForceAlignmentEncoder
from insertion_alignment_encoder.preprocess import force_history_features, initial_baseline, normalize_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--cutoff-checkpoint", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--old-checkpoint", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--new-checkpoint", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/encoder.pt"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_old_features(
    args: argparse.Namespace,
    trajectories: list[int],
    scales: np.ndarray,
    baseline_frames: int,
) -> dict[tuple[int, int], np.ndarray]:
    result: dict[tuple[int, int], np.ndarray] = {}
    for trajectory in trajectories:
        for domain, root in enumerate((args.real_root, args.sim_root)):
            path = root / f"traj_{trajectory}" / "data.h5"
            with h5py.File(path, "r") as episode:
                force = np.asarray(episode[args.force_key], dtype=np.float32)[:, :3]
                timestamps = np.asarray(episode["timestamps"], dtype=np.float64)
            baseline = initial_baseline(force, baseline_frames)
            features = force_history_features(force, timestamps, baseline)
            result[(trajectory, domain)] = normalize_features(features, scales)
    return result


def gather_windows(windows, features_by_episode: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    history = windows.x.shape[-2]
    rows = np.empty((len(windows.y), 2, history, next(iter(features_by_episode.values())).shape[-1]), dtype=np.float32)
    for i, (trajectory, frame) in enumerate(zip(windows.trajectory, windows.frame, strict=True)):
        for domain in (0, 1):
            values = features_by_episode[(int(trajectory), domain)]
            start = max(0, int(frame) - history + 1)
            sample = values[start : int(frame) + 1]
            if len(sample) < history:
                sample = np.pad(sample, ((history - len(sample), 0), (0, 0)), mode="edge")
            rows[i, domain] = sample
    return rows


def infer(model: nn.Module, values: np.ndarray, device: torch.device, old: bool = False):
    model.eval()
    latents, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(values), 384):
            batch = torch.from_numpy(values[start : start + 384]).to(device)
            output = model(batch)
            latent, logits = output[0], output[1]
            latents.append(latent.cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(latents), np.concatenate(probabilities)


def binary_summary(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (labels.astype(np.int64), predictions.astype(np.int64)), 1)
    class_f1 = []
    for cls in (0, 1):
        tp = int(matrix[cls, cls])
        precision = tp / max(int(matrix[:, cls].sum()), 1)
        recall = tp / max(int(matrix[cls].sum()), 1)
        class_f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "count": int(matrix.sum()),
        "accuracy": float(np.trace(matrix) / max(int(matrix.sum()), 1)),
        "macro_f1": float(np.mean(class_f1)),
        "class_f1": {"class_0": class_f1[0], "class_1": class_f1[1]},
        "confusion_matrix": matrix.tolist(),
    }


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dot = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return dot / np.maximum(denominator, 1e-8)


def build_similarity_controls(windows, minimum_frame_gap: int = 5):
    rng = np.random.default_rng(42)
    same_trajectory, different_trajectory = [], []
    for i, (trajectory, frame, label) in enumerate(
        zip(windows.trajectory, windows.frame, windows.y, strict=True)
    ):
        same_mask = (
            (windows.trajectory == trajectory)
            & (windows.y == label)
            & (np.abs(windows.frame - frame) >= minimum_frame_gap)
        )
        same_candidates = np.flatnonzero(same_mask)
        if same_candidates.size:
            j = int(rng.choice(same_candidates))
            same_trajectory.extend(((i, j), (j, i)))

        other_mask = (windows.trajectory != trajectory) & (windows.y == label)
        other_candidates = np.flatnonzero(other_mask)
        if other_candidates.size:
            j = int(rng.choice(other_candidates))
            different_trajectory.extend(((i, j), (j, i)))
    return {
        "paired_same_frame": [(i, i) for i in range(len(windows.y))],
        "same_trajectory_same_phase_gap_ge_5_frames": same_trajectory,
        "different_trajectory_same_phase": different_trajectory,
    }


def similarity_category_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    def summarize(selection: np.ndarray) -> dict[str, Any]:
        selected = values[selection]
        if not len(selected):
            return {"count": 0}
        q25, median, q75 = np.quantile(selected, [0.25, 0.5, 0.75])
        return {
            "count": int(len(selected)),
            "mean": float(np.mean(selected)),
            "median": float(median),
            "q25": float(q25),
            "q75": float(q75),
        }

    report = {"all_phases": summarize(np.ones(len(values), dtype=bool)), "by_phase": {}}
    for label_id, label_name in enumerate(("misaligned_contact", "aligned_progress")):
        report["by_phase"][label_name] = summarize(labels == label_id)
    return report


def matched_vs_control_similarity(representation: np.ndarray, windows, controls) -> dict[str, Any]:
    paired = representation.reshape(len(windows.y), 2, -1)
    result = {}
    for category, index_pairs in controls.items():
        indices = np.asarray(index_pairs, dtype=np.int64).reshape(-1, 2)
        if len(indices):
            values = cosine_rows(paired[indices[:, 0], 0], paired[indices[:, 1], 1])
            labels = windows.y[indices[:, 0]]
        else:
            values, labels = np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64)
        result[category] = similarity_category_summary(values, labels)
    return result


def cosine_sample_summary(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"count": 0}
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "minimum": float(np.min(values)),
        "below_zero_count": int(np.sum(values < 0.0)),
        "below_0_5_count": int(np.sum(values < 0.5)),
    }


def force_magnitude_gap_report(windows, episodes, augmentation_gain_range: tuple[float, float]) -> dict[str, Any]:
    real_norm, sim_norm, real_valid, sim_valid = [], [], [], []
    for trajectory, frame in zip(windows.trajectory, windows.frame, strict=True):
        real_feature = episodes[(int(trajectory), 0)].raw_features[int(frame)]
        sim_feature = episodes[(int(trajectory), 1)].raw_features[int(frame)]
        real_norm.append(float(np.expm1(real_feature[8])))
        sim_norm.append(float(np.expm1(sim_feature[8])))
        real_valid.append(bool(real_feature[10] > 0.5))
        sim_valid.append(bool(sim_feature[10] > 0.5))
    real_norm = np.asarray(real_norm)
    sim_norm = np.asarray(sim_norm)
    real_valid = np.asarray(real_valid)
    sim_valid = np.asarray(sim_valid)
    ratio_mask = (real_norm > 1e-6) & (sim_norm > 1e-6)
    ratio = real_norm[ratio_mask] / sim_norm[ratio_mask]

    gain_min, gain_max = augmentation_gain_range
    low_gain_reach, high_gain_reach = gain_min / gain_max, gain_max / gain_min

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        both = mask & ratio_mask
        ratio_values = real_norm[both] / sim_norm[both]
        if len(ratio_values):
            q10, q25, median, q75, q90 = np.quantile(ratio_values, [0.1, 0.25, 0.5, 0.75, 0.9])
        else:
            q10 = q25 = median = q75 = q90 = None
        selected_real, selected_sim = real_valid[mask], sim_valid[mask]
        return {
            "window_count": int(mask.sum()),
            "both_force_valid_count": int(np.sum(selected_real & selected_sim)),
            "real_only_force_valid_count": int(np.sum(selected_real & ~selected_sim)),
            "sim_only_force_valid_count": int(np.sum(~selected_real & selected_sim)),
            "neither_force_valid_count": int(np.sum(~selected_real & ~selected_sim)),
            "force_valid_mismatch_rate": float(np.mean(selected_real != selected_sim)) if len(selected_real) else None,
            "real_to_sim_smoothed_force_norm_ratio_both_valid": {
                "count": int(len(ratio_values)),
                "q10": None if q10 is None else float(q10),
                "q25": None if q25 is None else float(q25),
                "median": None if median is None else float(median),
                "q75": None if q75 is None else float(q75),
                "q90": None if q90 is None else float(q90),
                "outside_independent_augmentation_gain_reach_fraction": (
                    float(np.mean((ratio_values < low_gain_reach) | (ratio_values > high_gain_reach)))
                    if len(ratio_values) else None
                ),
            },
        }

    result = {"all_phases": summarize(np.ones(len(real_norm), dtype=bool)), "by_phase": {}}
    for label_id, label_name in enumerate(("misaligned_contact", "aligned_progress")):
        result["by_phase"][label_name] = summarize(windows.y == label_id)
    return result


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    balance_classes: bool,
    seed: int,
) -> dict[str, Any]:
    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    center = train_x.mean(axis=0, keepdims=True)
    scale = np.maximum(train_x.std(axis=0, keepdims=True), 1e-3)
    train_x = np.clip((train_x - center) / scale, -10.0, 10.0)
    test_x = np.clip((test_x - center) / scale, -10.0, 10.0)

    weights = np.asarray(train_weight, dtype=np.float32).copy()
    if balance_classes:
        counts = np.bincount(train_y.astype(np.int64), minlength=2).astype(np.float32)
        class_weight = len(train_y) / np.maximum(2.0 * counts, 1.0)
        weights *= class_weight[train_y.astype(np.int64)]
    weights *= len(weights) / max(float(weights.sum()), 1e-8)

    torch.manual_seed(seed)
    x_t = torch.from_numpy(train_x)
    y_t = torch.from_numpy(train_y.astype(np.int64))
    w_t = torch.from_numpy(weights)
    probe = nn.Linear(train_x.shape[1], 2)
    optimizer = torch.optim.LBFGS(probe.parameters(), lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x_t)
        loss = (F.cross_entropy(logits, y_t, reduction="none") * w_t).mean()
        loss = loss + 0.5 * probe.weight.square().sum() / max(len(train_y), 1)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        prediction = probe(torch.from_numpy(test_x)).argmax(dim=-1).numpy()
    return binary_summary(test_y, prediction)


def domain_probe_report(train_pair: np.ndarray, hold_pair: np.ndarray, seed: int) -> dict[str, Any]:
    if train_pair.ndim == 4:
        train_x = train_pair.reshape(-1, train_pair.shape[-2] * train_pair.shape[-1])
        test_x = hold_pair.reshape(-1, hold_pair.shape[-2] * hold_pair.shape[-1])
    else:
        train_x = train_pair.reshape(-1, train_pair.shape[-1])
        test_x = hold_pair.reshape(-1, hold_pair.shape[-1])
    train_y = np.tile(np.asarray([0, 1], dtype=np.int64), len(train_pair))
    test_y = np.tile(np.asarray([0, 1], dtype=np.int64), len(hold_pair))
    return fit_linear_probe(
        train_x,
        train_y,
        np.ones(len(train_y), dtype=np.float32),
        test_x,
        test_y,
        balance_classes=True,
        seed=seed,
    )


def cross_domain_task_report(
    train_pair: np.ndarray,
    hold_pair: np.ndarray,
    train_windows,
    hold_windows,
    seed: int,
) -> dict[str, Any]:
    results = {}
    for source, target, name in ((0, 1, "real_to_sim"), (1, 0, "sim_to_real")):
        results[name] = fit_linear_probe(
            train_pair[:, source].reshape(len(train_windows.y), -1),
            train_windows.y,
            train_windows.weight,
            hold_pair[:, target].reshape(len(hold_windows.y), -1),
            hold_windows.y,
            balance_classes=True,
            seed=seed + source,
        )
    return results


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment; use --device cpu")
    torch.set_num_threads(2)
    device = torch.device(args.device)

    new_payload = torch.load(args.new_checkpoint, map_location="cpu", weights_only=False)
    old_payload = torch.load(args.old_checkpoint, map_location="cpu", weights_only=False)
    if new_payload.get("artifact_format") != "tavla_force_gap_encoder_v2":
        raise ValueError(f"Unexpected new checkpoint format: {args.new_checkpoint}")
    if old_payload.get("artifact_format") != "tavla_force_alignment_encoder_terminal_reference_v1":
        raise ValueError(f"Unexpected old checkpoint format: {args.old_checkpoint}")

    args.baseline_frames = int(new_payload["config"]["baseline_frames"])
    args.insertion_axis = new_payload["preprocess"]["insertion_axis_base"]
    args.smoothing_frames = int(new_payload["preprocess"]["smoothing_frames"])
    args.trend_lag_frames = int(new_payload["preprocess"]["trend_lag_frames"])
    args.active_noise_mad_multiplier = float(new_payload["preprocess"]["active_noise_mad_multiplier"])
    args.active_noise_floor_newton = float(new_payload["preprocess"]["active_noise_floor_newton"])
    labels = read_labels(args.labels)
    episodes, _, _, _ = load_episodes(args, labels)
    preprocess = new_payload["preprocess"]
    center = np.asarray(preprocess["normalizer_center"], dtype=np.float32)
    scale = np.asarray(preprocess["normalizer_scale"], dtype=np.float32)
    history = int(new_payload["config"]["history_length"])
    all_ids = sorted(labels)
    holdout_ids = sorted(set(new_payload["data_summary"]["held_out_trajectories"]) & set(all_ids))
    development_ids = sorted(set(all_ids) - set(holdout_ids))
    if holdout_ids != sorted(DEFAULT_HOLDOUT):
        raise ValueError(f"Unexpected holdout split: {holdout_ids}")

    dev = build_windows(episodes, labels, development_ids, center, scale, history, guard=5)
    hold = build_windows(episodes, labels, holdout_ids, center, scale, history, guard=5)
    old_features = read_old_features(
        args,
        all_ids,
        np.asarray(old_payload["preprocess"]["feature_scales"], dtype=np.float32),
        int(old_payload["config"]["baseline_frames"]),
    )
    old_dev_x = gather_windows(dev, old_features)
    old_hold_x = gather_windows(hold, old_features)

    old_model = ForceAlignmentEncoder(latent_dim=int(old_payload["config"]["latent_dim"])).to(device)
    old_model.load_state_dict(old_payload["model_state_dict"])
    new_model = ForceGapEncoderV2(history, len(preprocess["feature_order"]), int(new_payload["config"]["latent_dim"])).to(device)
    new_model.load_state_dict(new_payload["model_state_dict"])

    old_dev_z, _ = infer(old_model, old_dev_x.reshape(-1, history, old_dev_x.shape[-1]), device, old=True)
    old_hold_z_flat, old_hold_prob_flat = infer(
        old_model, old_hold_x.reshape(-1, history, old_hold_x.shape[-1]), device, old=True
    )
    new_dev_z, _ = infer(new_model, dev.x.reshape(-1, history, dev.x.shape[-1]), device)
    new_hold_z_flat, new_hold_prob_flat = infer(new_model, hold.x.reshape(-1, history, hold.x.shape[-1]), device)
    old_dev_z = old_dev_z.reshape(len(dev.y), 2, -1)
    old_hold_z = old_hold_z_flat.reshape(len(hold.y), 2, -1)
    new_dev_z = new_dev_z.reshape(len(dev.y), 2, -1)
    new_hold_z = new_hold_z_flat.reshape(len(hold.y), 2, -1)
    old_hold_prob = old_hold_prob_flat.reshape(len(hold.y), 2, 2)
    new_hold_prob = new_hold_prob_flat.reshape(len(hold.y), 2, 2)

    representations = {
        "raw_12d_history": (dev.x, hold.x),
        "old_encoder_latent": (old_dev_z, old_hold_z),
        "force_gap_v2_latent": (new_dev_z, new_hold_z),
    }
    raw_train, raw_hold = representations["raw_12d_history"]
    raw_domain = domain_probe_report(raw_train, raw_hold, seed=42)
    domain_reports = {"raw_12d_history": raw_domain}
    task_reports = {}
    for index, (name, (train_pair, hold_pair)) in enumerate(representations.items()):
        if index:
            domain_reports[name] = domain_probe_report(train_pair, hold_pair, seed=42 + index)
        task_reports[name] = cross_domain_task_report(train_pair, hold_pair, dev, hold, seed=100 + index * 10)

    controls = build_similarity_controls(hold, minimum_frame_gap=5)
    similarity_inputs = {
        "raw_12d_history": hold.x,
        "old_preprocessed_input": old_hold_x,
        "old_encoder_latent": old_hold_z,
        "force_gap_v2_latent": new_hold_z,
    }
    matched_vs_control = {
        name: matched_vs_control_similarity(values, hold, controls)
        for name, values in similarity_inputs.items()
    }
    raw_pair_values = cosine_rows(hold.x[:, 0].reshape(len(hold.y), -1), hold.x[:, 1].reshape(len(hold.y), -1))
    old_input_pair_values = cosine_rows(
        old_hold_x[:, 0].reshape(len(hold.y), -1), old_hold_x[:, 1].reshape(len(hold.y), -1)
    )
    old_latent_pair_values = cosine_rows(old_hold_z[:, 0], old_hold_z[:, 1])
    new_latent_pair_values = cosine_rows(new_hold_z[:, 0], new_hold_z[:, 1])
    similarity_by_trajectory = {}
    for trajectory in holdout_ids:
        mask = hold.trajectory == trajectory
        similarity_by_trajectory[str(trajectory)] = {
            "count": int(mask.sum()),
            "real_sim_pair_cosine": {
                "raw_12d_history": cosine_sample_summary(raw_pair_values[mask]),
                "old_preprocessed_input": cosine_sample_summary(old_input_pair_values[mask]),
                "old_encoder_latent": cosine_sample_summary(old_latent_pair_values[mask]),
                "force_gap_v2_latent": cosine_sample_summary(new_latent_pair_values[mask]),
            },
        }
    worst_indices = np.argsort(new_latent_pair_values)[:12]
    worst_pairs = [
        {
            "trajectory": int(hold.trajectory[i]),
            "frame": int(hold.frame[i]),
            "phase": ("misaligned_contact", "aligned_progress")[int(hold.y[i])],
            "confidence_weight": float(hold.weight[i]),
            "raw_12d_pair_cosine": float(raw_pair_values[i]),
            "old_encoder_pair_cosine": float(old_latent_pair_values[i]),
            "force_gap_v2_pair_cosine": float(new_latent_pair_values[i]),
            "real_smoothed_force_norm_n": float(np.expm1(episodes[(int(hold.trajectory[i]), 0)].raw_features[int(hold.frame[i]), 8])),
            "sim_smoothed_force_norm_n": float(np.expm1(episodes[(int(hold.trajectory[i]), 1)].raw_features[int(hold.frame[i]), 8])),
            "real_force_valid": bool(episodes[(int(hold.trajectory[i]), 0)].raw_features[int(hold.frame[i]), 10] > 0.5),
            "sim_force_valid": bool(episodes[(int(hold.trajectory[i]), 1)].raw_features[int(hold.frame[i]), 10] > 0.5),
        }
        for i in worst_indices
    ]

    old_head = {
        "real": metrics(hold.y, old_hold_prob[:, 0].argmax(axis=-1)),
        "sim": metrics(hold.y, old_hold_prob[:, 1].argmax(axis=-1)),
    }
    new_head = {
        "real": metrics(hold.y, new_hold_prob[:, 0].argmax(axis=-1)),
        "sim": metrics(hold.y, new_hold_prob[:, 1].argmax(axis=-1)),
    }
    augmentation_gain_range = tuple(
        float(value) for value in new_payload["config"].get("augmentation_gain_range", (0.7, 1.3))
    )
    relative_gain_reach = [
        augmentation_gain_range[0] / augmentation_gain_range[1],
        augmentation_gain_range[1] / augmentation_gain_range[0],
    ]
    augmentation_gain_sampling = new_payload["config"].get("augmentation_gain_sampling", "uniform")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "holdout_trajectories": holdout_ids,
        "development_trajectory_count": len(development_ids),
        "labeled_window_count": int(len(hold.y)),
        "checkpoint_sha256": {
            "old": hashlib.sha256(args.old_checkpoint.read_bytes()).hexdigest(),
            "new": hashlib.sha256(args.new_checkpoint.read_bytes()).hexdigest(),
        },
        "pretrained_phase_heads": {"old_encoder": old_head, "force_gap_v2": new_head},
        "paired_cosine": {
            "old_preprocessed_input": paired_cosine(old_hold_x),
            "old_encoder_latent": paired_cosine(old_hold_z),
            "new_preprocessed_input": paired_cosine(hold.x),
            "force_gap_v2_latent": paired_cosine(new_hold_z),
        },
        "heldout_domain_probe": {
            "chance_accuracy": 0.5,
            "interpretation": "lower Real-versus-Sim accuracy means less domain information remains linearly recoverable",
            "representations": domain_reports,
        },
        "heldout_cross_domain_phase_probe": {
            "training_scope": "development trajectories only; source domain only",
            "representations": task_reports,
        },
        "heldout_matched_vs_control_similarity": {
            "purpose": "distinguish same-frame force alignment from similarity caused by shared trajectory/phase labels",
            "control_sampling": "for each labeled window, randomly choose one same-phase control; includes both Real_i-to-Sim_j and Real_j-to-Sim_i comparisons",
            "minimum_frame_gap": 5,
            "representations": matched_vs_control,
        },
        "heldout_pair_cosine_failure_cases": {
            "force_gap_v2_overall": cosine_sample_summary(new_latent_pair_values),
            "force_gap_v2_by_phase": {
                name: cosine_sample_summary(new_latent_pair_values[hold.y == label_id])
                for label_id, name in enumerate(("misaligned_contact", "aligned_progress"))
            },
            "by_trajectory": similarity_by_trajectory,
            "lowest_12_force_gap_v2_pairs": worst_pairs,
        },
        "paired_force_magnitude_gap": {
            "augmentation_gain_range": list(augmentation_gain_range),
            "augmentation_gain_sampling": augmentation_gain_sampling,
            "relative_gain_augmentation_reach": relative_gain_reach,
            "note": "Pair magnitude ratio is Real/Sim at the same labeled frame using pre-normalizer 3-frame-smoothed force norms; reach interval follows independent gains from the checkpoint config on each domain.",
            "development": force_magnitude_gap_report(dev, episodes, augmentation_gain_range),
            "holdout": force_magnitude_gap_report(hold, episodes, augmentation_gain_range),
        },
        "probe_protocol": {
            "classifier": "L2-regularized linear softmax probe, fit on development trajectories only",
            "feature_standardization": "mean/std fit on the probe training subset only; std floor 1e-3",
            "phase_class_balance": "inverse-frequency weights from the probe training subset",
            "raw_representation": "flattened normalized 32-frame x 12-feature v2 input window",
            "old_representation": "32-D latent from the frozen terminal-reference encoder",
            "new_representation": "32-D latent from the frozen force-gap v2 encoder",
            "domain_probe": "paired Real and Sim windows from the same labeled frames; trajectory-group holdout",
        },
    }

    report_path = args.new_checkpoint.with_suffix(".report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["baseline_comparison"] = summary
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_path = args.new_checkpoint.parent / "train.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nBASELINE COMPARISON (CPU)\n")
        log.write(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Updated existing report: {report_path}")
    print(f"Appended to existing log: {log_path}")


if __name__ == "__main__":
    main()
