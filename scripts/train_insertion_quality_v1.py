#!/usr/bin/env python3
"""Train and evaluate the standalone Insertion Quality Encoder V1.

The script enforces the weak-label gate, performs trajectory-level LOTO, and
does not import or modify TAVLA policy training code.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from insertion_quality_v1_lib import CONFIDENCE_WEIGHTS
from insertion_quality_v1_lib import FEATURE_NAMES
from insertion_quality_v1_lib import MOTION_NAMES
from insertion_quality_v1_lib import RepresentationScaler
from insertion_quality_v1_lib import annotation_arrays
from insertion_quality_v1_lib import fit_quantile_motion_rule
from insertion_quality_v1_lib import fit_representation_scaler
from insertion_quality_v1_lib import history_consistent_labels
from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import motion_descriptors
from insertion_quality_v1_lib import predict_weak_labels
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config
from insertion_quality_v1_lib import save_json
from insertion_quality_v1_lib import transform_representation
from insertion_quality_v1_lib import window_view
from insertion_quality_v1_lib import write_csv


@dataclass
class FoldArrays:
    x: np.ndarray
    human_y: np.ndarray
    human_weight: np.ndarray
    weak_y: np.ndarray
    terminal_y: np.ndarray
    terminal_mask: np.ndarray
    motion_y: np.ndarray
    motion_mask: np.ndarray
    pair_x: np.ndarray
    pair_mask: np.ndarray
    trajectory: np.ndarray
    frame: np.ndarray
    domain: np.ndarray


class InsertionQualityEncoder(nn.Module):
    def __init__(self, input_dim: int, channels: int, gru_hidden: int, latent_dim: int):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(channels, gru_hidden, batch_first=True)
        self.to_latent = nn.Sequential(nn.Linear(gru_hidden, latent_dim), nn.Tanh())
        self.quality_head = nn.Linear(latent_dim, 2)
        self.terminal_head = nn.Linear(latent_dim, 1)
        self.motion_head = nn.Linear(latent_dim, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        _, hidden = self.gru(encoded)
        latent = self.to_latent(hidden[-1])
        return self.quality_head(latent), self.terminal_head(latent).squeeze(-1), self.motion_head(latent), latent


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def binary_metrics(y_true: np.ndarray, probability_aligned: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (probability_aligned >= 0.5).astype(np.int64)
    result: dict[str, float] = {"accuracy": float(np.mean(y_true == y_pred))}
    f1_values: list[float] = []
    for label, name in ((0, "misaligned"), (1, "aligned")):
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
        result[f"{name}_precision"] = precision
        result[f"{name}_recall"] = recall
        result[f"{name}_f1"] = f1
        if np.isfinite(f1):
            f1_values.append(f1)
    result["macro_f1"] = float(np.mean(f1_values)) if f1_values else float("nan")
    positives = probability_aligned[y_true == 1]
    negatives = probability_aligned[y_true == 0]
    if len(positives) and len(negatives):
        comparisons = (positives[:, None] > negatives[None, :]).mean()
        ties = (positives[:, None] == negatives[None, :]).mean()
        result["auroc"] = float(comparisons + 0.5 * ties)
    else:
        result["auroc"] = float("nan")
    return result


def f1_score_binary(y_true: np.ndarray, probability: np.ndarray) -> float:
    y_pred = probability >= 0.5
    truth = y_true >= 0.5
    tp = float(np.sum(y_pred & truth))
    fp = float(np.sum(y_pred & ~truth))
    fn = float(np.sum(~y_pred & truth))
    return safe_divide(2 * tp, 2 * tp + fp + fn)


def representation_scaler_metadata(scaler: RepresentationScaler) -> dict[str, object]:
    return {
        "active_threshold": scaler.active_threshold,
        "trend_threshold": scaler.trend_threshold,
        "cdf_sample_counts": {
            domain: {
                "lateral": len(scaler.lateral_cdf[domain]),
                "axial": len(scaler.axial_cdf[domain]),
            }
            for domain in ("real", "sim")
        },
    }


def fit_motion_normalization(
    data: dict[int, object],
    train_ids: list[int],
    config: dict[str, object],
    rule: object,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    free_ends = {int(key): int(value) for key, value in config["free_end_frames"].items()}
    weak_cfg = config["weak_labels"]
    for trajectory_id in train_ids:
        motion = motion_descriptors(
            data[trajectory_id].pose,
            int(weak_cfg["future_horizon_frames"]),
            int(weak_cfg["resume_horizon_frames"]),
            rule.stall_step_threshold,
        )[:, :3]
        values.append(motion[free_ends[trajectory_id] + 1 :])
    stacked = np.concatenate(values)
    center = np.median(stacked, axis=0)
    q25, q75 = np.quantile(stacked, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, np.maximum(np.abs(center) * 1e-3, 1e-8))
    return center.astype(np.float32), scale.astype(np.float32)


def build_fold_arrays(
    data: dict[int, object],
    ids: list[int],
    manual_ids: set[int],
    annotations: list[dict[str, object]],
    config: dict[str, object],
    rule: object,
    scaler: RepresentationScaler,
    motion_center: np.ndarray,
    motion_scale: np.ndarray,
) -> tuple[FoldArrays, dict[tuple[int, str], np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    history = int(config["representation"]["history_frames"])
    tolerance = int(config["model"]["pair_frame_tolerance"])
    free_ends = {int(key): int(value) for key, value in config["free_end_frames"].items()}
    feature_by_key: dict[tuple[int, str], np.ndarray] = {}
    active_by_key: dict[tuple[int, str], np.ndarray] = {}
    weak_by_trajectory: dict[int, np.ndarray] = {}
    motion_by_trajectory: dict[int, np.ndarray] = {}
    human_by_trajectory: dict[int, np.ndarray] = {}
    terminal_by_trajectory: dict[int, np.ndarray] = {}
    confidence_by_trajectory: dict[int, np.ndarray] = {}

    for trajectory_id in ids:
        free_end = free_ends[trajectory_id]
        weak, motion = predict_weak_labels(rule, data[trajectory_id], config, free_end)
        weak_by_trajectory[trajectory_id] = weak
        motion_by_trajectory[trajectory_id] = motion
        human, terminal, confidence = annotation_arrays(annotations, trajectory_id, len(weak))
        human_by_trajectory[trajectory_id] = human
        terminal_by_trajectory[trajectory_id] = terminal
        confidence_by_trajectory[trajectory_id] = confidence
        for domain in ("real", "sim"):
            feature, active = transform_representation(data[trajectory_id], domain, free_end, scaler, config)
            feature_by_key[(trajectory_id, domain)] = feature
            active_by_key[(trajectory_id, domain)] = active

    fields: dict[str, list[object]] = {
        name: []
        for name in (
            "x",
            "human_y",
            "human_weight",
            "weak_y",
            "terminal_y",
            "terminal_mask",
            "motion_y",
            "motion_mask",
            "pair_x",
            "pair_mask",
            "trajectory",
            "frame",
            "domain",
        )
    }
    for trajectory_id in ids:
        n_frames = len(data[trajectory_id].pose)
        free_end = free_ends[trajectory_id]
        weak = weak_by_trajectory[trajectory_id]
        motion = motion_by_trajectory[trajectory_id]
        human = human_by_trajectory[trajectory_id]
        terminal = terminal_by_trajectory[trajectory_id]
        confidence = confidence_by_trajectory[trajectory_id]
        human_window = history_consistent_labels(human, history)
        weak_window = history_consistent_labels(weak, history)
        confidence_window = np.zeros_like(confidence)
        for frame in range(history - 1, n_frames):
            confidence_window[frame] = np.min(confidence[frame - history + 1 : frame + 1])
        for domain_id, domain in enumerate(("real", "sim")):
            other_domain = "sim" if domain == "real" else "real"
            windows = window_view(feature_by_key[(trajectory_id, domain)], history)
            other_windows = window_view(feature_by_key[(trajectory_id, other_domain)], history)
            active = active_by_key[(trajectory_id, domain)]
            other_active = active_by_key[(trajectory_id, other_domain)]
            for offset, frame in enumerate(range(history - 1, n_frames)):
                fields["x"].append(windows[offset])
                is_human_real = trajectory_id in manual_ids and domain == "real" and human_window[frame] >= 0
                fields["human_y"].append(int(human_window[frame]) if is_human_real else -1)
                fields["human_weight"].append(float(confidence_window[frame]) if is_human_real else 0.0)
                fields["weak_y"].append(int(weak_window[frame]))
                fields["terminal_y"].append(float(terminal[frame]))
                fields["terminal_mask"].append(float(trajectory_id in manual_ids))
                fields["motion_y"].append((motion[frame, :3] - motion_center) / motion_scale)
                fields["motion_mask"].append(float(frame > free_end and terminal[frame] < 0.5))
                fields["trajectory"].append(trajectory_id)
                fields["frame"].append(frame)
                fields["domain"].append(domain_id)

                paired_frame = -1
                if active[frame] and weak_window[frame] >= 0:
                    candidates = [
                        candidate
                        for candidate in range(max(history - 1, frame - tolerance), min(n_frames, frame + tolerance + 1))
                        if other_active[candidate] and weak_window[candidate] == weak_window[frame]
                    ]
                    if candidates:
                        paired_frame = min(candidates, key=lambda candidate: abs(candidate - frame))
                if paired_frame >= 0:
                    fields["pair_x"].append(other_windows[paired_frame - history + 1])
                    fields["pair_mask"].append(1.0)
                else:
                    fields["pair_x"].append(np.zeros((history, len(FEATURE_NAMES)), dtype=np.float32))
                    fields["pair_mask"].append(0.0)

    arrays = FoldArrays(
        x=np.asarray(fields["x"], dtype=np.float32),
        human_y=np.asarray(fields["human_y"], dtype=np.int64),
        human_weight=np.asarray(fields["human_weight"], dtype=np.float32),
        weak_y=np.asarray(fields["weak_y"], dtype=np.int64),
        terminal_y=np.asarray(fields["terminal_y"], dtype=np.float32),
        terminal_mask=np.asarray(fields["terminal_mask"], dtype=np.float32),
        motion_y=np.asarray(fields["motion_y"], dtype=np.float32),
        motion_mask=np.asarray(fields["motion_mask"], dtype=np.float32),
        pair_x=np.asarray(fields["pair_x"], dtype=np.float32),
        pair_mask=np.asarray(fields["pair_mask"], dtype=np.float32),
        trajectory=np.asarray(fields["trajectory"], dtype=np.int64),
        frame=np.asarray(fields["frame"], dtype=np.int64),
        domain=np.asarray(fields["domain"], dtype=np.int64),
    )
    return arrays, feature_by_key, weak_by_trajectory, motion_by_trajectory


def select_arrays(arrays: FoldArrays, mask: np.ndarray) -> FoldArrays:
    return FoldArrays(**{field: getattr(arrays, field)[mask] for field in arrays.__dataclass_fields__})


def to_dataset(arrays: FoldArrays) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.x),
        torch.from_numpy(arrays.human_y),
        torch.from_numpy(arrays.human_weight),
        torch.from_numpy(arrays.weak_y),
        torch.from_numpy(arrays.terminal_y),
        torch.from_numpy(arrays.terminal_mask),
        torch.from_numpy(arrays.motion_y),
        torch.from_numpy(arrays.motion_mask),
        torch.from_numpy(arrays.pair_x),
        torch.from_numpy(arrays.pair_mask),
    )


def train_encoder(
    train: FoldArrays,
    config: dict[str, object],
    device: torch.device,
    motion_weight_override: float | None = None,
    epochs_override: int | None = None,
) -> tuple[InsertionQualityEncoder, list[dict[str, float]]]:
    model_cfg = config["model"]
    model = InsertionQualityEncoder(
        int(model_cfg["input_dim"]),
        int(model_cfg["conv_channels"]),
        int(model_cfg["gru_hidden"]),
        int(model_cfg["latent_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_cfg["learning_rate"]), weight_decay=float(model_cfg["weight_decay"])
    )
    weights = dict(model_cfg["loss_weights"])
    if motion_weight_override is not None:
        weights["motion_anchor"] = motion_weight_override
    human_labels = train.human_y[train.human_y >= 0]
    counts = np.bincount(human_labels, minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / np.maximum(2 * counts, 1), dtype=torch.float32, device=device)
    positive_terminal = max(float(np.sum(train.terminal_y * train.terminal_mask)), 1.0)
    negative_terminal = max(float(np.sum((1 - train.terminal_y) * train.terminal_mask)), 1.0)
    terminal_pos_weight = torch.tensor(negative_terminal / positive_terminal, device=device)
    loader = DataLoader(to_dataset(train), batch_size=int(model_cfg["batch_size"]), shuffle=True, drop_last=False)
    epochs = epochs_override or int(model_cfg["epochs"])
    patience = int(model_cfg["patience"])
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        sums = {name: 0.0 for name in ("total", "human", "weak", "motion", "terminal", "pair")}
        batches = 0
        for batch in loader:
            x, human_y, human_weight, weak_y, terminal_y, terminal_mask, motion_y, motion_mask, pair_x, pair_mask = [
                value.to(device) for value in batch
            ]
            quality_logits, terminal_logits, motion_prediction, _ = model(x)
            human_keep = human_y >= 0
            if torch.any(human_keep):
                human_loss = (
                    F.cross_entropy(quality_logits[human_keep], human_y[human_keep], weight=class_weights, reduction="none")
                    * human_weight[human_keep]
                ).sum() / human_weight[human_keep].sum().clamp_min(1.0)
            else:
                human_loss = quality_logits.sum() * 0.0
            weak_keep = weak_y >= 0
            weak_loss = (
                F.cross_entropy(quality_logits[weak_keep], weak_y[weak_keep])
                if torch.any(weak_keep)
                else quality_logits.sum() * 0.0
            )
            motion_keep = motion_mask > 0.5
            motion_loss = (
                F.smooth_l1_loss(motion_prediction[motion_keep], motion_y[motion_keep])
                if torch.any(motion_keep)
                else motion_prediction.sum() * 0.0
            )
            terminal_keep = terminal_mask > 0.5
            terminal_loss = (
                F.binary_cross_entropy_with_logits(
                    terminal_logits[terminal_keep], terminal_y[terminal_keep], pos_weight=terminal_pos_weight
                )
                if torch.any(terminal_keep)
                else terminal_logits.sum() * 0.0
            )
            pair_keep = pair_mask > 0.5
            if torch.any(pair_keep):
                pair_logits, pair_terminal, _, _ = model(pair_x[pair_keep])
                pair_loss = F.mse_loss(
                    torch.softmax(quality_logits[pair_keep], dim=-1), torch.softmax(pair_logits, dim=-1)
                ) + 0.25 * F.mse_loss(torch.sigmoid(terminal_logits[pair_keep]), torch.sigmoid(pair_terminal))
            else:
                pair_loss = quality_logits.sum() * 0.0
            loss = (
                float(weights["human_quality"]) * human_loss
                + float(weights["weak_quality"]) * weak_loss
                + float(weights["motion_anchor"]) * motion_loss
                + float(weights["terminal"]) * terminal_loss
                + float(weights["paired_prediction_consistency"]) * pair_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in (
                ("total", loss),
                ("human", human_loss),
                ("weak", weak_loss),
                ("motion", motion_loss),
                ("terminal", terminal_loss),
                ("pair", pair_loss),
            ):
                sums[name] += float(value.detach().cpu())
            batches += 1
        row = {"epoch": float(epoch), **{name: value / max(batches, 1) for name, value in sums.items()}}
        history.append(row)
        if row["total"] < best_loss - 1e-4:
            best_loss = row["total"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:03d}/{epochs} total={row['total']:.4f} human={row['human']:.4f} "
                f"weak={row['weak']:.4f} motion={row['motion']:.4f} pair={row['pair']:.4f}",
                flush=True,
            )
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history


def predict_encoder(
    model: InsertionQualityEncoder, x: np.ndarray, device: torch.device, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    quality_parts: list[np.ndarray] = []
    terminal_parts: list[np.ndarray] = []
    motion_parts: list[np.ndarray] = []
    latent_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            quality, terminal, motion, latent = model(torch.from_numpy(x[start : start + batch_size]).to(device))
            quality_parts.append(torch.softmax(quality, dim=-1).cpu().numpy())
            terminal_parts.append(torch.sigmoid(terminal).cpu().numpy())
            motion_parts.append(motion.cpu().numpy())
            latent_parts.append(latent.cpu().numpy())
    return (
        np.concatenate(quality_parts),
        np.concatenate(terminal_parts),
        np.concatenate(motion_parts),
        np.concatenate(latent_parts),
    )


def handcrafted_features(x: np.ndarray) -> np.ndarray:
    lateral = x[:, :, 3]
    axial = x[:, :, 4]
    occupancy = x[:, :, 8]
    direction_change = np.linalg.norm(np.diff(x[:, :, :3], axis=1), axis=2)
    impacts = np.sum(np.diff(occupancy, axis=1) > 0.25, axis=1)
    return np.column_stack(
        (
            lateral.mean(axis=1),
            lateral.std(axis=1),
            axial.mean(axis=1),
            axial.std(axis=1),
            direction_change.mean(axis=1),
            impacts,
            occupancy.mean(axis=1),
            x[:, :, 9].max(axis=1),
            x[:, :, 10].max(axis=1),
        )
    ).astype(np.float32)


def ablation_features(x: np.ndarray, name: str) -> np.ndarray:
    if name == "logistic_handcrafted":
        if len(x) == 0:
            return np.empty((0, 9), dtype=np.float32)
        return handcrafted_features(x)
    if name == "force_direction_only":
        return x[:, :, [0, 1, 2]].reshape(len(x), x.shape[1] * 3)
    if name == "magnitude_trend_only":
        return x[:, :, [3, 4, 5, 6, 7, 8, 9, 10]].reshape(len(x), x.shape[1] * 8)
    if name == "full_11d_logistic":
        return x.reshape(len(x), x.shape[1] * x.shape[2])
    raise ValueError(name)


def train_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample_weight: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    center = np.median(train_x, axis=0)
    q25, q75 = np.quantile(train_x, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    train_scaled = np.clip((train_x - center) / scale, -20, 20).astype(np.float32)
    test_scaled = np.clip((test_x - center) / scale, -20, 20).astype(np.float32)
    model = LinearProbe(train_scaled.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-3)
    x_tensor = torch.from_numpy(train_scaled)
    y_tensor = torch.from_numpy(train_y.astype(np.int64))
    w_tensor = torch.from_numpy(sample_weight.astype(np.float32))
    for _ in range(250):
        logits = model(x_tensor)
        loss = (F.cross_entropy(logits, y_tensor, reduction="none") * w_tensor).sum() / w_tensor.sum().clamp_min(1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(test_scaled)), dim=-1).numpy()


def nearest_boundary_errors(
    frames: np.ndarray, predictions: np.ndarray, annotations: list[dict[str, object]], trajectory_id: int
) -> list[int]:
    errors: list[int] = []
    if not len(frames):
        return errors
    ordered = sorted((int(frame), int(prediction)) for frame, prediction in zip(frames, predictions))
    for row in annotations:
        if int(row["trajectory_id"]) != trajectory_id or str(row["label"]) not in ("misaligned_contact", "aligned_progress"):
            continue
        boundary = int(row["start_frame"])
        target = 0 if str(row["label"]) == "misaligned_contact" else 1
        candidates = [
            frame
            for index, (frame, prediction) in enumerate(ordered)
            if prediction == target and (index == 0 or ordered[index - 1][1] != target)
        ]
        if candidates:
            errors.append(min(abs(frame - boundary) for frame in candidates))
    return errors


def fit_domain_probe(train_latent: np.ndarray, train_domain: np.ndarray, test_latent: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    keep: list[int] = []
    count = min(np.sum(train_domain == 0), np.sum(train_domain == 1))
    for label in (0, 1):
        candidates = np.flatnonzero(train_domain == label)
        keep.extend(rng.choice(candidates, int(count), replace=False).tolist())
    keep_array = np.asarray(keep)
    return train_linear_probe(
        train_latent[keep_array], train_domain[keep_array], np.ones(len(keep_array), dtype=np.float32), test_latent, seed
    )


def save_fold_scaler(
    fold_dir: Path,
    scaler: RepresentationScaler,
    motion_center: np.ndarray,
    motion_scale: np.ndarray,
    rule: object,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        fold_dir / "preprocess.json",
        {
            **representation_scaler_metadata(scaler),
            "motion_center": motion_center.tolist(),
            "motion_scale": motion_scale.tolist(),
            "weak_label_rule": rule.as_dict(),
            "feature_names": FEATURE_NAMES,
        },
    )
    np.savez_compressed(
        fold_dir / "empirical_cdf_train_only.npz",
        real_lateral=scaler.lateral_cdf["real"],
        real_axial=scaler.axial_cdf["real"],
        sim_lateral=scaler.lateral_cdf["sim"],
        sim_axial=scaler.axial_cdf["sim"],
    )


def run(
    repo: Path,
    config_path: Path,
    annotations_path: Path,
    output_dir: Path,
    device_name: str,
    epochs_override: int | None,
    max_folds: int | None,
) -> None:
    config = read_config(config_path)
    annotations = read_annotations(annotations_path)
    gate_path = output_dir / "weak_label_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Run scripts/audit_insertion_quality_labels_v1.py first; missing {gate_path}")
    with gate_path.open(encoding="utf-8") as handle:
        gate = json.load(handle)
    if not gate.get("gate_passed", False):
        raise RuntimeError("Weak-label gate failed. Encoder training is intentionally blocked.")

    reviewed_ids = [
        int(value) for value in config.get("reviewed_trajectories", config["manual_trajectories"])
    ]
    evaluation_ids = [int(value) for value in config.get("evaluation_trajectories", reviewed_ids)]
    all_ids = [int(value) for value in config["all_trajectories"]]
    data = load_trajectories(repo, config, all_ids)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable in this process")
    device = torch.device(device_name)
    set_seed(int(config["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = evaluation_ids[: max_folds or len(evaluation_ids)]
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    all_boundary_errors: list[int] = []
    all_paired_probability_differences: list[float] = []
    started = time.time()

    for fold_index, held_out in enumerate(folds, start=1):
        print(f"\n[fold {fold_index}/{len(folds)}] held_out=traj_{held_out} device={device}", flush=True)
        fold_dir = output_dir / "folds" / f"traj_{held_out}"
        train_manual_ids = [trajectory for trajectory in reviewed_ids if trajectory != held_out]
        train_ids = [trajectory for trajectory in all_ids if trajectory != held_out]
        rule = fit_quantile_motion_rule(data, annotations, train_manual_ids, config)
        scaler = fit_representation_scaler(data, train_ids, config)
        motion_center, motion_scale = fit_motion_normalization(data, train_ids, config, rule)
        save_fold_scaler(fold_dir, scaler, motion_center, motion_scale, rule)
        arrays, _, _, _ = build_fold_arrays(
            data, all_ids, set(reviewed_ids), annotations, config, rule, scaler, motion_center, motion_scale
        )
        train = select_arrays(arrays, arrays.trajectory != held_out)
        test = select_arrays(arrays, arrays.trajectory == held_out)
        manifest_rows = [
            {
                "trajectory": f"traj_{int(arrays.trajectory[index])}",
                "end_frame": int(arrays.frame[index]),
                "domain": "real" if int(arrays.domain[index]) == 0 else "sim",
                "split": "test" if int(arrays.trajectory[index]) == held_out else "train",
                "human_quality_label": int(arrays.human_y[index]),
                "human_confidence_weight": float(arrays.human_weight[index]),
                "weak_quality_label": int(arrays.weak_y[index]),
                "terminal_label": float(arrays.terminal_y[index]),
                "terminal_mask": float(arrays.terminal_mask[index]),
                "motion_mask": float(arrays.motion_mask[index]),
                "paired_consistency_mask": float(arrays.pair_mask[index]),
            }
            for index in range(len(arrays.x))
        ]
        write_csv(fold_dir / "window_manifest.csv", manifest_rows)
        checkpoint_path = fold_dir / "encoder.pt"
        if checkpoint_path.exists() and epochs_override is None:
            print(f"  resuming: loading {checkpoint_path}", flush=True)
            model_cfg = config["model"]
            model = InsertionQualityEncoder(
                int(model_cfg["input_dim"]),
                int(model_cfg["conv_channels"]),
                int(model_cfg["gru_hidden"]),
                int(model_cfg["latent_dim"]),
            ).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
        else:
            model, history = train_encoder(train, config, device, epochs_override=epochs_override)
            write_csv(fold_dir / "training_history.csv", history)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "held_out_trajectory": held_out,
                    "feature_names": FEATURE_NAMES,
                },
                checkpoint_path,
            )

        train_quality, train_terminal, _, train_latent = predict_encoder(model, train.x, device)
        test_quality, test_terminal, _, test_latent = predict_encoder(model, test.x, device)
        eval_mask = (test.domain == 0) & (test.human_y >= 0) & (test.human_weight >= CONFIDENCE_WEIGHTS["medium"])
        all_confidence_eval = (test.domain == 0) & (test.human_y >= 0)
        terminal_eval = (test.domain == 0) & (test.terminal_mask > 0.5)
        metrics = binary_metrics(test.human_y[eval_mask], test_quality[eval_mask, 1]) if np.any(eval_mask) else {
            name: float("nan")
            for name in (
                "accuracy",
                "misaligned_precision",
                "misaligned_recall",
                "misaligned_f1",
                "aligned_precision",
                "aligned_recall",
                "aligned_f1",
                "macro_f1",
                "auroc",
            )
        }
        all_metrics = (
            binary_metrics(test.human_y[all_confidence_eval], test_quality[all_confidence_eval, 1])
            if np.any(all_confidence_eval)
            else {"macro_f1": float("nan")}
        )
        terminal_f1 = f1_score_binary(test.terminal_y[terminal_eval], test_terminal[terminal_eval])

        baseline_metrics: dict[str, dict[str, float]] = {}
        baseline_probability_by_name: dict[str, np.ndarray] = {}
        train_target = np.where(train.human_y >= 0, train.human_y, train.weak_y)
        train_weight = np.where(train.human_y >= 0, train.human_weight, 0.3).astype(np.float32)
        train_keep = train_target >= 0
        for baseline_name in (
            "logistic_handcrafted",
            "force_direction_only",
            "magnitude_trend_only",
            "full_11d_logistic",
        ):
            if np.any(eval_mask):
                probability = train_linear_probe(
                    ablation_features(train.x[train_keep], baseline_name),
                    train_target[train_keep],
                    train_weight[train_keep],
                    ablation_features(test.x[eval_mask], baseline_name),
                    int(config["seed"]) + held_out,
                )
                baseline_metrics[baseline_name] = binary_metrics(test.human_y[eval_mask], probability[:, 1])
                full_probability = np.full(len(test.x), np.nan, dtype=np.float64)
                full_probability[eval_mask] = probability[:, 1]
                baseline_probability_by_name[baseline_name] = full_probability
            else:
                baseline_metrics[baseline_name] = {"macro_f1": float("nan")}
                baseline_probability_by_name[baseline_name] = np.full(len(test.x), np.nan, dtype=np.float64)

        no_motion_path = fold_dir / "encoder_no_motion_anchor.pt"
        if no_motion_path.exists() and epochs_override is None:
            model_cfg = config["model"]
            no_motion_model = InsertionQualityEncoder(
                int(model_cfg["input_dim"]),
                int(model_cfg["conv_channels"]),
                int(model_cfg["gru_hidden"]),
                int(model_cfg["latent_dim"]),
            ).to(device)
            no_motion_model.load_state_dict(torch.load(no_motion_path, map_location=device)["state_dict"])
        else:
            no_motion_model, _ = train_encoder(
                train, config, device, motion_weight_override=0.0, epochs_override=epochs_override
            )
            torch.save(
                {
                    "state_dict": no_motion_model.state_dict(),
                    "config": config,
                    "held_out_trajectory": held_out,
                    "motion_anchor_weight": 0.0,
                    "feature_names": FEATURE_NAMES,
                },
                no_motion_path,
            )
        no_motion_quality, _, _, _ = predict_encoder(no_motion_model, test.x, device)
        no_motion_metrics = (
            binary_metrics(test.human_y[eval_mask], no_motion_quality[eval_mask, 1])
            if np.any(eval_mask)
            else {"macro_f1": float("nan")}
        )

        domain_probability = fit_domain_probe(
            train_latent, train.domain, test_latent, int(config["seed"]) + held_out
        )[:, 1]
        domain_accuracy = float(np.mean((domain_probability >= 0.5) == test.domain))
        paired_probability_differences: list[float] = []
        paired_real = (test.domain == 0) & (test.human_y >= 0) & (test.pair_mask > 0.5)
        if np.any(paired_real):
            paired_quality, _, _, _ = predict_encoder(model, test.pair_x[paired_real], device)
            paired_probability_differences = np.abs(
                test_quality[paired_real, 0] - paired_quality[:, 0]
            ).astype(float).tolist()
        pair_median = float(np.median(paired_probability_differences)) if paired_probability_differences else float("nan")
        all_paired_probability_differences.extend(paired_probability_differences)
        predictions = np.argmax(test_quality, axis=1)
        real_mask = test.domain == 0
        boundary_errors = nearest_boundary_errors(
            test.frame[real_mask], predictions[real_mask], annotations, held_out
        )
        all_boundary_errors.extend(boundary_errors)

        row: dict[str, object] = {
            "held_out_trajectory": f"traj_{held_out}",
            "quality_frames_evaluated": int(np.sum(eval_mask)),
            "quality_classes_present": int(len(np.unique(test.human_y[eval_mask]))) if np.any(eval_mask) else 0,
            **{f"encoder_{key}": value for key, value in metrics.items()},
            "encoder_all_confidence_macro_f1": all_metrics.get("macro_f1", float("nan")),
            "terminal_f1": terminal_f1,
            "boundary_median_error_frames": float(np.median(boundary_errors)) if boundary_errors else float("nan"),
            "paired_probability_difference_median": pair_median,
            "paired_windows_evaluated": len(paired_probability_differences),
            "latent_domain_accuracy": domain_accuracy,
            "no_motion_macro_f1": no_motion_metrics.get("macro_f1", float("nan")),
        }
        for baseline_name, baseline in baseline_metrics.items():
            row[f"{baseline_name}_macro_f1"] = baseline.get("macro_f1", float("nan"))
        fold_rows.append(row)

        timestamps = data[held_out].timestamps
        for index in np.flatnonzero(real_mask):
            p_aligned = float(test_quality[index, 1])
            entropy = -sum(
                probability * math.log(max(probability, 1e-12)) for probability in test_quality[index]
            ) / math.log(2.0)
            prediction_rows.append(
                {
                    "trajectory": f"traj_{held_out}",
                    "frame": int(test.frame[index]),
                    "time_s": float(timestamps[test.frame[index]]),
                    "human_quality_label": int(test.human_y[index]),
                    "human_confidence_weight": float(test.human_weight[index]),
                    "human_terminal_label": float(test.terminal_y[index]),
                    "p_misaligned": float(test_quality[index, 0]),
                    "p_aligned_progress": p_aligned,
                    "p_terminal": float(test_terminal[index]),
                    "no_motion_p_aligned": float(no_motion_quality[index, 1]),
                    "confidence": float(1.0 - entropy),
                    **{
                        f"{name}_p_aligned": float(probability[index])
                        for name, probability in baseline_probability_by_name.items()
                    },
                }
            )
        print(
            f"  result macro_f1={metrics['macro_f1']:.3f} misaligned_f1={metrics['misaligned_f1']:.3f} "
            f"domain_acc={domain_accuracy:.3f} pair_diff={pair_median:.3f}",
            flush=True,
        )

    write_csv(output_dir / "loto_metrics.csv", fold_rows)
    write_csv(output_dir / "loto_predictions.csv", prediction_rows)

    def finite_mean(key: str) -> float:
        values = np.asarray([float(row[key]) for row in fold_rows], dtype=np.float64)
        return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")

    pooled_rows = [
        row
        for row in prediction_rows
        if int(row["human_quality_label"]) >= 0
        and float(row["human_confidence_weight"]) >= CONFIDENCE_WEIGHTS["medium"] - 1e-6
    ]
    pooled_true = np.asarray([int(row["human_quality_label"]) for row in pooled_rows], dtype=np.int64)
    pooled_encoder = binary_metrics(
        pooled_true, np.asarray([float(row["p_aligned_progress"]) for row in pooled_rows])
    )
    pooled_baselines = {
        name: binary_metrics(
            pooled_true, np.asarray([float(row[f"{name}_p_aligned"]) for row in pooled_rows])
        )
        for name in (
            "logistic_handcrafted",
            "force_direction_only",
            "magnitude_trend_only",
            "full_11d_logistic",
        )
    }
    pooled_no_motion = binary_metrics(
        pooled_true, np.asarray([float(row["no_motion_p_aligned"]) for row in pooled_rows])
    )
    terminal_true = np.asarray([float(row["human_terminal_label"]) for row in prediction_rows])
    terminal_probability = np.asarray([float(row["p_terminal"]) for row in prediction_rows])
    terminal_f1_pooled = f1_score_binary(terminal_true, terminal_probability)
    quality_folds_evaluable = sum(int(row["quality_frames_evaluated"]) > 0 for row in fold_rows)
    quality_folds_with_both_classes = sum(int(row["quality_classes_present"]) == 2 for row in fold_rows)
    aggregate = {
        "folds_completed": len(fold_rows),
        "quality_folds_evaluable": quality_folds_evaluable,
        "quality_folds_with_both_classes": quality_folds_with_both_classes,
        "quality_frames_pooled": len(pooled_rows),
        "encoder_macro_f1_pooled": pooled_encoder["macro_f1"],
        "encoder_misaligned_f1_pooled": pooled_encoder["misaligned_f1"],
        "encoder_aligned_f1_pooled": pooled_encoder["aligned_f1"],
        "encoder_accuracy_pooled": pooled_encoder["accuracy"],
        "encoder_auroc_pooled": pooled_encoder["auroc"],
        "encoder_macro_f1_fold_mean_descriptive": finite_mean("encoder_macro_f1"),
        "terminal_f1_pooled": terminal_f1_pooled,
        "terminal_f1_fold_mean_descriptive": finite_mean("terminal_f1"),
        "latent_domain_accuracy_mean": finite_mean("latent_domain_accuracy"),
        "paired_folds_evaluable": sum(int(row["paired_windows_evaluated"]) > 0 for row in fold_rows),
        "paired_windows_evaluated": len(all_paired_probability_differences),
        "paired_probability_difference_median": float(np.median(all_paired_probability_differences))
        if all_paired_probability_differences
        else float("nan"),
        "boundary_median_error_frames": float(np.median(all_boundary_errors)) if all_boundary_errors else float("nan"),
        "logistic_handcrafted_macro_f1_pooled": pooled_baselines["logistic_handcrafted"]["macro_f1"],
        "force_direction_only_macro_f1_pooled": pooled_baselines["force_direction_only"]["macro_f1"],
        "magnitude_trend_only_macro_f1_pooled": pooled_baselines["magnitude_trend_only"]["macro_f1"],
        "full_11d_logistic_macro_f1_pooled": pooled_baselines["full_11d_logistic"]["macro_f1"],
        "no_motion_macro_f1_pooled": pooled_no_motion["macro_f1"],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    aggregate["encoder_gain_over_logistic"] = (
        aggregate["encoder_macro_f1_pooled"] - aggregate["logistic_handcrafted_macro_f1_pooled"]
    )
    aggregate["success_criteria_met"] = bool(
        len(fold_rows) == len(folds)
        and quality_folds_evaluable == len(folds)
        and aggregate["encoder_macro_f1_pooled"] >= 0.70
        and aggregate["encoder_misaligned_f1_pooled"] >= 0.65
        and aggregate["encoder_gain_over_logistic"] >= 0.05
        and aggregate["boundary_median_error_frames"] <= 5
        and aggregate["paired_probability_difference_median"] <= 0.15
    )
    save_json(output_dir / "aggregate_metrics.json", aggregate)

    # Confusion matrix and trajectory probability timelines.
    pred_array = prediction_rows
    evaluable = [row for row in pred_array if int(row["human_quality_label"]) >= 0]
    confusion = np.zeros((2, 2), dtype=np.int64)
    for row in evaluable:
        true = int(row["human_quality_label"])
        predicted = int(float(row["p_aligned_progress"]) >= 0.5)
        confusion[true, predicted] += 1
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion, cmap="Blues")
    for y in range(2):
        for x in range(2):
            axis.text(x, y, str(confusion[y, x]), ha="center", va="center")
    axis.set_xticks([0, 1], ["misaligned", "aligned"])
    axis.set_yticks([0, 1], ["misaligned", "aligned"])
    axis.set_xlabel("predicted")
    axis.set_ylabel("human label")
    axis.set_title("LOTO insertion-quality confusion matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(folds), 1, figsize=(15, 2.1 * len(folds)), sharex=False)
    if len(folds) == 1:
        axes = [axes]
    for axis, trajectory_id in zip(axes, folds):
        rows = [row for row in pred_array if row["trajectory"] == f"traj_{trajectory_id}"]
        frame = np.asarray([int(row["frame"]) for row in rows])
        axis.plot(frame, [row["p_misaligned"] for row in rows], label="p_misaligned", color="#d62728")
        axis.plot(frame, [row["p_aligned_progress"] for row in rows], label="p_aligned", color="#2ca02c")
        axis.plot(frame, [row["p_terminal"] for row in rows], label="p_terminal", color="#9467bd", alpha=0.8)
        for annotation in annotations:
            if int(annotation["trajectory_id"]) != trajectory_id:
                continue
            label = str(annotation["label"])
            if label == "misaligned_contact":
                axis.axvspan(int(annotation["start_frame"]), int(annotation["end_frame"]), color="#d62728", alpha=0.08)
            elif label == "aligned_progress":
                axis.axvspan(int(annotation["start_frame"]), int(annotation["end_frame"]), color="#2ca02c", alpha=0.08)
            elif label == "terminal":
                axis.axvspan(int(annotation["start_frame"]), int(annotation["end_frame"]), color="#9467bd", alpha=0.08)
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(f"traj_{trajectory_id}", loc="left", fontsize=9)
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.suptitle("LOTO structured outputs; shaded regions are human intervals")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_dir / "quality_state_timelines.png", dpi=180)
    plt.close(fig)

    report = [
        "# Standalone Insertion Quality Encoder V1",
        "",
        "## Outcome",
        "",
        f"**Acceptance gate: {'PASS' if aggregate['success_criteria_met'] else 'FAIL'}**. TAVLA integration is therefore "
        + ("allowed by the predefined criteria." if aggregate["success_criteria_met"] else "not allowed yet."),
        "",
        f"- Encoder pooled Macro-F1: `{aggregate['encoder_macro_f1_pooled']:.3f}`",
        f"- Misaligned pooled F1: `{aggregate['encoder_misaligned_f1_pooled']:.3f}`",
        f"- Aligned-progress pooled F1: `{aggregate['encoder_aligned_f1_pooled']:.3f}`",
        f"- Pooled AUROC: `{aggregate['encoder_auroc_pooled']:.3f}`",
        f"- Quality-evaluable folds: `{aggregate['quality_folds_evaluable']}/{len(folds)}` (both classes present in `{aggregate['quality_folds_with_both_classes']}/{len(folds)}`)",
        f"- Terminal pooled F1: `{aggregate['terminal_f1_pooled']:.3f}`",
        f"- Handcrafted Logistic pooled Macro-F1: `{aggregate['logistic_handcrafted_macro_f1_pooled']:.3f}`",
        f"- Encoder gain over Logistic: `{aggregate['encoder_gain_over_logistic']:+.3f}`",
        f"- Boundary median error: `{aggregate['boundary_median_error_frames']:.2f}` frames",
        f"- Paired Real/Sim probability median difference: `{aggregate['paired_probability_difference_median']:.3f}`",
        f"- Active paired windows: `{aggregate['paired_windows_evaluated']}` across `{aggregate['paired_folds_evaluable']}/{len(folds)}` folds",
        f"- Latent domain classifier accuracy (diagnostic): `{aggregate['latent_domain_accuracy_mean']:.3f}`",
        "",
        "## Interpretation",
        "",
        ("The encoder meets the predefined standalone acceptance criteria. It may now be frozen for a controlled TAVLA ablation."
         if aggregate["success_criteria_met"]
         else "The encoder does not yet meet all predefined standalone acceptance criteria; inspect the metrics and ablations below before any TAVLA integration."),
        "",
        "Quality probabilities outside high-confidence misaligned/aligned intervals are diagnostic only: `free` and `uncertain` were intentionally masked. The current entropy-derived confidence is not calibrated and must not be interpreted as physical certainty. Visual inspection shows overconfident aligned predictions in these masked phases, which is another reason not to expose this 4D output to TAVLA.",
        "The phase-output calibration table is generated by `plot_insertion_quality_v1_results.py` as `phase_output_summary.csv`.",
        "",
        f"This run uses `{len(reviewed_ids)}` video-reviewed trajectories for Real strong supervision and evaluates on `{len(folds)}` held-out trajectories selected because both quality classes contain complete `{config['representation']['history_frames']}`-frame windows.",
        "",
        "## Ablations",
        "",
        f"- Force direction only pooled Macro-F1: `{aggregate['force_direction_only_macro_f1_pooled']:.3f}`",
        f"- Magnitude/trend only pooled Macro-F1: `{aggregate['magnitude_trend_only_macro_f1_pooled']:.3f}`",
        f"- Full 11D Logistic pooled Macro-F1: `{aggregate['full_11d_logistic_macro_f1_pooled']:.3f}`",
        f"- Encoder without motion-anchor pooled Macro-F1: `{aggregate['no_motion_macro_f1_pooled']:.3f}`",
        "",
        "## Leakage controls",
        "",
        "Each fold holds out one complete paired trajectory. Weak-label thresholds, activity gates, empirical CDFs and motion normalization are fit only on train trajectories. The held-out Real and Sim replay are never split across train/test. Pair consistency is applied to output probabilities only; latent equality is not enforced.",
        f"Every fold stores `preprocess.json`, train-only empirical CDF arrays, and `window_manifest.csv` with label sources/masks under `{output_dir.relative_to(repo)}/folds/`.",
        "",
        "## Structured output",
        "",
        "The deployment-facing output is `[p_misaligned, p_aligned_progress, p_terminal, confidence]`. Confidence is `1 - normalized binary entropy`. The 16D latent remains an audit value and is not a default TAVLA observation.",
        "Because the acceptance gate failed, no deployment/frozen checkpoint is produced. Fold checkpoints are retained for audit only, and TAVLA training logic remains unchanged.",
    ]
    report_path = repo / str(config.get("report_filename", "insertion_quality_encoder_v1_report.md"))
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nAggregate metrics:")
    print(json.dumps(aggregate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_v1.json"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/insertion_quality_labels_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/insertion_quality_v1"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for smoke tests")
    parser.add_argument("--max-folds", type=int, default=None, help="Limit folds for smoke tests")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    annotations = args.annotations if args.annotations.is_absolute() else repo / args.annotations
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, config, annotations, output_dir, args.device, args.epochs, args.max_folds)


if __name__ == "__main__":
    main()
