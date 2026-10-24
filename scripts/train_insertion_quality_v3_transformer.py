#!/usr/bin/env python3
"""Standalone temporal Transformer ablation for insertion quality.

This script is intentionally separate from TAVLA policy training.  It compares
force-only history, force+motion history, and masked-pretraining+GRL.  Sim
torque is excluded because the current Sim field is a compatibility zero.
"""

from __future__ import annotations

import argparse
import copy
import json
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
from torch.utils.data import DataLoader, TensorDataset

from openpi.force_features import REAL_FORCE_TREND_FEATURE_NAMES
from openpi.force_features import build_real_force_trend_features

from insertion_quality_v1_lib import CONFIDENCE_WEIGHTS
from insertion_quality_v1_lib import FEATURE_NAMES
from insertion_quality_v1_lib import annotation_arrays
from insertion_quality_v1_lib import fit_quantile_motion_rule
from insertion_quality_v1_lib import fit_representation_scaler
from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import predict_weak_labels
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config
from insertion_quality_v1_lib import transform_representation
from insertion_quality_v1_lib import write_csv


POSE_FEATURE_NAMES = [
    "ee_velocity_x",
    "ee_velocity_y",
    "ee_velocity_z",
    "ee_delta_x",
    "ee_delta_y",
    "ee_delta_z",
    "insertion_depth",
]


@dataclass
class WindowArrays:
    x: np.ndarray
    human_y: np.ndarray
    human_weight: np.ndarray
    weak_y: np.ndarray
    terminal_y: np.ndarray
    terminal_mask: np.ndarray
    trajectory: np.ndarray
    frame: np.ndarray
    domain: np.ndarray


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metrics(y_true: np.ndarray, probability_aligned: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    prediction = (probability_aligned >= 0.5).astype(np.int64)
    result = {"accuracy": float(np.mean(y_true == prediction))}
    f1_values: list[float] = []
    for label, name in ((0, "misaligned"), (1, "aligned")):
        tp = float(np.sum((y_true == label) & (prediction == label)))
        fp = float(np.sum((y_true != label) & (prediction == label)))
        fn = float(np.sum((y_true == label) & (prediction != label)))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
        result[f"{name}_precision"] = precision
        result[f"{name}_recall"] = recall
        result[f"{name}_f1"] = f1
        if np.isfinite(f1):
            f1_values.append(f1)
    result["macro_f1"] = float(np.mean(f1_values)) if f1_values else float("nan")
    positive = probability_aligned[y_true == 1]
    negative = probability_aligned[y_true == 0]
    if len(positive) and len(negative):
        result["auroc"] = float(
            (positive[:, None] > negative[None, :]).mean()
            + 0.5 * (positive[:, None] == negative[None, :]).mean()
        )
    else:
        result["auroc"] = float("nan")
    return result


def causal_pose_features(pose: np.ndarray, timestamps: np.ndarray, free_end: int, axis: np.ndarray) -> np.ndarray:
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[0] = np.median(np.diff(timestamps)) if len(timestamps) > 1 else 0.1
    dt = np.maximum(dt, 1e-4)
    velocity = np.vstack((np.zeros((1, 3)), np.diff(pose, axis=0))) / dt[:, None]
    reference = pose[min(free_end, len(pose) - 1)]
    delta = pose - reference
    depth = delta @ axis
    return np.column_stack((velocity, delta, depth))


def fit_pose_scaler(data: dict[int, object], train_ids: list[int], config: dict[str, object], pose_reference: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(config["representation"]["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    values = [
        causal_pose_features(data[trajectory].pose, data[trajectory].timestamps, pose_reference[trajectory], axis)
        for trajectory in train_ids
    ]
    stacked = np.concatenate(values, axis=0)
    center = np.median(stacked, axis=0)
    q25, q75 = np.quantile(stacked, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, np.maximum(np.abs(center) * 1e-3, 1e-8))
    return center.astype(np.float32), scale.astype(np.float32)


def build_features(
    data: dict[int, object],
    ids: list[int],
    train_ids: list[int],
    config: dict[str, object],
    scaler: object,
    use_pose: bool | str,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, object]]:
    need_pose = bool(use_pose) or not config.get("quality_only", False)
    calibration_frames = config["representation"].get("calibration_frames")
    # When calibration_frames is set, both force and pose use the same fixed
    # leading window as their trajectory reference, matching deployment.  Older
    # configs without it keep the per-trajectory annotated free-space end.
    pose_reference = {
        trajectory: (int(calibration_frames) - 1 if calibration_frames is not None else free_ends_value)
        for trajectory, free_ends_value in {int(k): int(v) for k, v in config["free_end_frames"].items()}.items()
    }
    pose_center, pose_scale = fit_pose_scaler(data, train_ids, config, pose_reference) if need_pose else (None, None)
    axis = np.asarray(config["representation"]["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    free_ends = {int(key): int(value) for key, value in config["free_end_frames"].items()}
    features: dict[tuple[int, str], np.ndarray] = {}
    domains = tuple(config.get("input_domains", ["real", "sim"]))
    force_mode = config.get("force_feature_mode", "v3_bounded")
    raw_force_features: dict[tuple[int, str], np.ndarray] = {}
    if force_mode == "policy_force_trend_v1":
        rep = config["representation"]
        # Calibrate the force baseline/noise gate from a fixed leading window so
        # offline features match the deployment RealForceTrendBuffer, which sees
        # only the first ``calibration_frames`` samples before contact.  When
        # ``calibration_frames`` is absent (historical configs) fall back to the
        # per-trajectory annotated free-space end.
        calibration_frames = rep.get("calibration_frames")
        for trajectory in ids:
            force_free_end = (
                int(calibration_frames) - 1 if calibration_frames is not None else free_ends[trajectory]
            )
            for domain in domains:
                force = data[trajectory].real_force if domain == "real" else data[trajectory].sim_force
                values, _ = build_real_force_trend_features(
                    force,
                    free_end=force_free_end,
                    insertion_axis=axis,
                    smoothing_frames=int(rep["causal_smoothing_frames"]),
                    trend_lag_frames=int(rep["trend_lag_frames"]),
                    epsilon=float(rep["epsilon"]),
                    active_noise_mad_multiplier=float(rep["active_noise_mad_multiplier"]),
                    active_noise_floor=float(rep["active_noise_floor_newton"]),
                )
                raw_force_features[(trajectory, domain)] = values
        train_values = np.concatenate(
            [raw_force_features[(trajectory, domain)] for trajectory in train_ids for domain in domains]
        )
        force_center = np.median(train_values, axis=0)
        q25, q75 = np.quantile(train_values, [0.25, 0.75], axis=0)
        q05, q95 = np.quantile(train_values, [0.05, 0.95], axis=0)
        force_scale = np.maximum(q75 - q25, 0.1 * (q95 - q05))
        force_scale = np.maximum(force_scale, 1e-3)
    elif force_mode != "v3_bounded":
        raise ValueError(f"Unknown force_feature_mode: {force_mode}")
    for trajectory in ids:
        pose_part = None
        if need_pose:
            pose_part = causal_pose_features(
                data[trajectory].pose, data[trajectory].timestamps, pose_reference[trajectory], axis
            )
            pose_part = (pose_part - pose_center) / pose_scale
        for domain in domains:
            if force_mode == "policy_force_trend_v1":
                force_part = np.clip((raw_force_features[(trajectory, domain)] - force_center) / force_scale, -10, 10)
            else:
                force_part, _ = transform_representation(data[trajectory], domain, free_ends[trajectory], scaler, config)
            if use_pose == "pose_only":
                features[(trajectory, domain)] = pose_part
            elif use_pose:
                features[(trajectory, domain)] = np.column_stack((force_part, pose_part))
            else:
                features[(trajectory, domain)] = force_part
    metadata = {
        "pose_center": pose_center.tolist() if pose_center is not None else None,
        "pose_scale": pose_scale.tolist() if pose_scale is not None else None,
        "force_feature_names": [] if use_pose == "pose_only" else (list(REAL_FORCE_TREND_FEATURE_NAMES) if force_mode == "policy_force_trend_v1" else FEATURE_NAMES),
        "pose_feature_names": POSE_FEATURE_NAMES if use_pose else [],
        "torque_included": False,
        "force_feature_mode": force_mode,
    }
    if force_mode == "policy_force_trend_v1":
        metadata.update(force_center=force_center.tolist(), force_scale=force_scale.tolist(), force_clip=10)
        metadata["force_calibration_frames"] = (
            int(config["representation"]["calibration_frames"])
            if config["representation"].get("calibration_frames") is not None
            else None
        )
    return features, metadata


def build_windows(
    data: dict[int, object],
    ids: list[int],
    reviewed_ids: set[int],
    annotations: list[dict[str, object]],
    config: dict[str, object],
    rule: object,
    features: dict[tuple[int, str], np.ndarray],
) -> WindowArrays:
    history = int(config["representation"]["history_frames"])
    free_ends = {int(key): int(value) for key, value in config["free_end_frames"].items()}
    fields: dict[str, list[object]] = {name: [] for name in (
        "x", "human_y", "human_weight", "weak_y", "terminal_y", "terminal_mask",
        "trajectory", "frame", "domain"
    )}
    for trajectory in ids:
        n_frames = len(data[trajectory].pose)
        weak = predict_weak_labels(rule, data[trajectory], config, free_ends[trajectory])[0] if config.get("use_weak_labels", True) else np.full(n_frames, -1, dtype=np.int64)
        human, terminal, confidence = annotation_arrays(annotations, trajectory, n_frames)
        for domain_id, domain in enumerate(config.get("input_domains", ["real", "sim"])):
            sequence = features[(trajectory, domain)]
            for frame in range(history - 1, n_frames):
                fields["x"].append(sequence[frame - history + 1 : frame + 1])
                human_valid = trajectory in reviewed_ids and domain in config.get("supervised_domains", ["real"]) and human[frame] >= 0
                if human_valid and config.get("require_consistent_history", False):
                    human_valid = bool(np.all(human[frame - history + 1 : frame + 1] == human[frame]))
                fields["human_y"].append(int(human[frame]) if human_valid else -1)
                fields["human_weight"].append(float(confidence[frame]) if human_valid else 0.0)
                fields["weak_y"].append(int(weak[frame]))
                fields["terminal_y"].append(float(terminal[frame]))
                fields["terminal_mask"].append(float(trajectory in reviewed_ids and domain in config.get("supervised_domains", ["real"])))
                fields["trajectory"].append(trajectory)
                fields["frame"].append(frame)
                fields["domain"].append(domain_id)
    return WindowArrays(
        x=np.asarray(fields["x"], dtype=np.float32),
        human_y=np.asarray(fields["human_y"], dtype=np.int64),
        human_weight=np.asarray(fields["human_weight"], dtype=np.float32),
        weak_y=np.asarray(fields["weak_y"], dtype=np.int64),
        terminal_y=np.asarray(fields["terminal_y"], dtype=np.float32),
        terminal_mask=np.asarray(fields["terminal_mask"], dtype=np.float32),
        trajectory=np.asarray(fields["trajectory"], dtype=np.int64),
        frame=np.asarray(fields["frame"], dtype=np.int64),
        domain=np.asarray(fields["domain"], dtype=np.int64),
    )


def select(arrays: WindowArrays, mask: np.ndarray) -> WindowArrays:
    return WindowArrays(**{field: getattr(arrays, field)[mask] for field in arrays.__dataclass_fields__})


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, config: dict[str, object], use_domain: bool):
        super().__init__()
        model_cfg = config["model"]
        d_model = int(model_cfg["d_model"])
        self.input_dim = input_dim
        self.force_dim = len(REAL_FORCE_TREND_FEATURE_NAMES) if config.get("force_feature_mode") == "policy_force_trend_v1" else len(FEATURE_NAMES)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, int(config["representation"]["history_frames"]), d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(model_cfg["nhead"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(model_cfg["num_layers"]))
        latent_dim = int(model_cfg["latent_dim"])
        self.to_latent = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim), nn.Tanh())
        self.quality_head = nn.Linear(latent_dim, 2)
        quality_only = bool(config.get("quality_only", False))
        if quality_only and use_domain:
            raise ValueError("quality_only cannot use a domain head")
        self.terminal_head = None if quality_only else nn.Linear(latent_dim, 1)
        self.domain_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 2)) if use_domain else None
        self.reconstruction_head = None if quality_only else nn.Linear(d_model, self.force_dim)

    def encode_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.input_projection(x) + self.position[:, : x.shape[1]]
        sequence = self.temporal(sequence)
        latent = self.to_latent(sequence[:, -1])
        return sequence, latent

    def forward(self, x: torch.Tensor, domain_lambda: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        sequence, latent = self.encode_sequence(x)
        domain = None
        if self.domain_head is not None:
            domain = self.domain_head(GradientReversal.apply(latent, domain_lambda))
        terminal = self.terminal_head(latent).squeeze(-1) if self.terminal_head is not None else None
        return self.quality_head(latent), terminal, latent, sequence, domain


def dataset(arrays: WindowArrays) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.x), torch.from_numpy(arrays.human_y), torch.from_numpy(arrays.human_weight),
        torch.from_numpy(arrays.weak_y), torch.from_numpy(arrays.terminal_y), torch.from_numpy(arrays.terminal_mask),
        torch.from_numpy(arrays.domain),
    )


def masked_batch(x: torch.Tensor, mask_ratio: float, span: int, force_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    masked = x.clone()
    mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
    count = max(1, int(x.shape[1] * mask_ratio))
    for batch in range(x.shape[0]):
        start = int(torch.randint(0, max(1, x.shape[1] - span + 1), (1,), device=x.device))
        mask[batch, start : start + min(span, count)] = True
    force_values = masked[:, :, :force_dim]
    force_values[mask] = 0.0
    return masked, mask


def predict(model: TemporalTransformer, x: np.ndarray, device: torch.device, domain_lambda: float) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    q_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            q, t, _, _, _ = model(torch.from_numpy(x[start : start + 512]).to(device), domain_lambda)
            q_parts.append(torch.softmax(q, dim=-1).cpu().numpy())
            if t is not None:
                t_parts.append(torch.sigmoid(t).cpu().numpy())
    return np.concatenate(q_parts), np.concatenate(t_parts) if t_parts else None


def pretrain_masked(model: TemporalTransformer, train: WindowArrays, config: dict[str, object], device: torch.device) -> list[dict[str, float]]:
    if model.reconstruction_head is None:
        raise ValueError("Masked pretraining requires a reconstruction head")
    model.train()
    cfg = config["model"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loader = DataLoader(dataset(train), batch_size=int(cfg["batch_size"]), shuffle=True)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(cfg["pretrain_epochs"]) + 1):
        total = 0.0
        batches = 0
        for batch in loader:
            x = batch[0].to(device)
            masked, mask = masked_batch(x, float(cfg["mask_ratio"]), int(cfg["mask_span"]), model.force_dim)
            _, _, _, sequence, _ = model(masked, 0.0)
            reconstruction = model.reconstruction_head(sequence)
            if torch.any(mask):
                target = x[:, :, : model.force_dim]
                loss = F.mse_loss(reconstruction[mask], target[mask])
            else:
                loss = reconstruction.sum() * 0.0
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        row = {"epoch": float(epoch), "masked_reconstruction": total / max(batches, 1)}
        history.append(row)
        print(f"    pretrain epoch {epoch:02d}/{cfg['pretrain_epochs']} recon={row['masked_reconstruction']:.4f}", flush=True)
    return history


def train_model(train: WindowArrays, input_dim: int, config: dict[str, object], variant: dict[str, object], device: torch.device) -> tuple[TemporalTransformer, list[dict[str, float]]]:
    model = TemporalTransformer(input_dim, config, use_domain=float(variant["domain_lambda"]) > 0).to(device)
    cfg = config["model"]
    history: list[dict[str, float]] = []
    if bool(variant["use_pretraining"]):
        pretrain_history = pretrain_masked(model, train, config, device)
        for row in pretrain_history:
            history.append({
                "epoch": row["epoch"],
                "pretrain_reconstruction": row["masked_reconstruction"],
                "total": float("nan"),
                "human": float("nan"),
                "weak": float("nan"),
                "terminal": float("nan"),
                "domain": float("nan"),
                "consistency": float("nan"),
            })
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loader = DataLoader(dataset(train), batch_size=int(cfg["batch_size"]), shuffle=True)
    human_labels = train.human_y[train.human_y >= 0]
    counts = np.bincount(human_labels, minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / np.maximum(2 * counts, 1), dtype=torch.float32, device=device)
    weights = cfg["loss_weights"]
    terminal_pos_weight = None
    if float(weights.get("terminal", 0.0)) > 0:
        positive_terminal = max(float(np.sum(train.terminal_y * train.terminal_mask)), 1.0)
        negative_terminal = max(float(np.sum((1 - train.terminal_y) * train.terminal_mask)), 1.0)
        terminal_pos_weight = torch.tensor(negative_terminal / positive_terminal, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        sums = {name: 0.0 for name in ("total", "human", "weak", "terminal", "domain", "consistency")}
        batches = 0
        for batch in loader:
            x, human_y, human_weight, weak_y, terminal_y, terminal_mask, domain_y = [value.to(device) for value in batch]
            quality_logits, terminal_logits, latent, _, domain_logits = model(x, float(variant["domain_lambda"]))
            human_keep = human_y >= 0
            human_loss = (
                (F.cross_entropy(quality_logits[human_keep], human_y[human_keep], weight=class_weights, reduction="none") * human_weight[human_keep]).sum()
                / human_weight[human_keep].sum().clamp_min(1.0)
                if torch.any(human_keep) else quality_logits.sum() * 0.0
            )
            weak_loss = quality_logits.sum() * 0.0
            if float(weights.get("weak_quality", 0.0)) > 0:
                weak_keep = (weak_y >= 0) & ~human_keep
                if torch.any(weak_keep):
                    weak_loss = F.cross_entropy(quality_logits[weak_keep], weak_y[weak_keep])
            terminal_loss = quality_logits.sum() * 0.0
            if terminal_logits is not None and terminal_pos_weight is not None:
                terminal_keep = terminal_mask > 0.5
                if torch.any(terminal_keep):
                    terminal_loss = F.binary_cross_entropy_with_logits(
                        terminal_logits[terminal_keep], terminal_y[terminal_keep], pos_weight=terminal_pos_weight
                    )
            domain_loss = F.cross_entropy(domain_logits, domain_y) if domain_logits is not None else quality_logits.sum() * 0.0
            noise = torch.randn_like(x) * 0.01
            noisy_quality, _, _, _, _ = model(x + noise, float(variant["domain_lambda"]))
            consistency_loss = F.mse_loss(torch.softmax(quality_logits.detach(), dim=-1), torch.softmax(noisy_quality, dim=-1))
            loss = (
                float(weights["human_quality"]) * human_loss
                + float(weights.get("weak_quality", 0.0)) * weak_loss
                + float(weights.get("terminal", 0.0)) * terminal_loss
                + float(variant["domain_lambda"]) * float(weights.get("domain", 0.0)) * domain_loss
                + float(weights.get("temporal_consistency", 0.0)) * consistency_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in (("total", loss), ("human", human_loss), ("weak", weak_loss), ("terminal", terminal_loss), ("domain", domain_loss), ("consistency", consistency_loss)):
                sums[name] += float(value.detach().cpu())
            batches += 1
        row = {
            "epoch": float(epoch),
            "pretrain_reconstruction": float("nan"),
            **{name: value / max(batches, 1) for name, value in sums.items()},
        }
        history.append(row)
        if row["total"] < best_loss - 1e-4:
            best_loss = row["total"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == int(cfg["epochs"]):
            print(f"    epoch {epoch:02d}/{cfg['epochs']} total={row['total']:.4f} human={row['human']:.4f} weak={row['weak']:.4f} domain={row['domain']:.4f}", flush=True)
        if stale >= int(cfg["patience"]):
            break
    model.load_state_dict(best_state)
    return model, history


def run(
    repo: Path,
    config_path: Path,
    annotations_path: Path,
    output_dir: Path,
    device_name: str,
    max_folds: int | None,
    variant_names: list[str] | None = None,
    epochs_override: int | None = None,
) -> None:
    config = read_config(config_path)
    if "force_config" in config:
        force_config_path = Path(config["force_config"])
        if not force_config_path.is_absolute():
            force_config_path = repo / force_config_path
        force_config = read_config(force_config_path)
        config["free_end_frames"] = force_config["free_end_frames"]
        config["representation"] = {**force_config["representation"], **config.get("representation", {})}
    if epochs_override is not None:
        if epochs_override < 1:
            raise ValueError("--epochs must be positive")
        config["model"]["epochs"] = epochs_override
    if variant_names is not None:
        config["variants"] = {name: config["variants"][name] for name in variant_names}
    annotations = read_annotations(annotations_path)
    if config.get("require_weak_label_gate", True):
        gate = read_json(repo / "eval_outputs/insertion_quality_v2_final/weak_label_gate.json")
        if not gate.get("gate_passed", False):
            raise RuntimeError("The V2 weak-label gate must pass before V3 training")
    all_ids = [int(v) for v in config.get("all_trajectories", sorted(int(key) for key in config["free_end_frames"]))]
    reviewed = [int(v) for v in config.get("reviewed_trajectories", all_ids)]
    folds = [int(v) for v in config["evaluation_trajectories"]]
    if max_folds is not None:
        folds = folds[:max_folds]
    domains = list(config.get("input_domains", ["real", "sim"]))
    if not domains or any(domain not in ("real", "sim") for domain in domains) or len(set(domains)) != len(domains):
        raise ValueError(f"Invalid input_domains: {domains}")
    evaluation_domain = config.get("evaluation_domain", "real")
    if evaluation_domain not in domains:
        raise ValueError(f"evaluation_domain {evaluation_domain!r} is not in input_domains")
    if not set(config.get("supervised_domains", ["real"])) <= set(domains):
        raise ValueError("supervised_domains must be a subset of input_domains")
    if config.get("quality_only", False):
        if config.get("force_feature_mode") != "policy_force_trend_v1":
            raise ValueError("quality_only requires policy_force_trend_v1")
        if config.get("use_weak_labels", True):
            raise ValueError("quality_only requires use_weak_labels=false")
        if any(float(config["model"]["loss_weights"].get(key, 0.0)) != 0.0 for key in ("weak_quality", "terminal", "domain")):
            raise ValueError("quality_only cannot enable weak, terminal, or domain losses")
        if any(variant["use_pretraining"] or float(variant["domain_lambda"]) for variant in config["variants"].values()):
            raise ValueError("quality_only cannot enable pretraining or domain adaptation")
    data = load_trajectories(repo, config, all_ids)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_name} is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_config.json", config)
    started = time.time()
    all_metrics: dict[str, list[dict[str, object]]] = {name: [] for name in config["variants"]}
    all_predictions: dict[str, list[dict[str, object]]] = {name: [] for name in config["variants"]}

    for fold_index, held_out in enumerate(folds, start=1):
        print(f"\n[fold {fold_index}/{len(folds)}] held_out=traj_{held_out} device={device}", flush=True)
        train_ids = [trajectory for trajectory in all_ids if trajectory != held_out]
        train_reviewed = [trajectory for trajectory in reviewed if trajectory != held_out]
        rule = fit_quantile_motion_rule(data, annotations, train_reviewed, config) if config.get("use_weak_labels", True) else None
        scaler = fit_representation_scaler(data, train_ids, config) if config.get("force_feature_mode", "v3_bounded") == "v3_bounded" else None
        fold_dir = output_dir / "folds" / f"traj_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(fold_dir / "preprocess.json", {"weak_label_rule": rule.as_dict() if rule is not None else None, "variants": config["variants"]})

        feature_cache: dict[str, tuple[dict[tuple[int, str], np.ndarray], dict[str, object], WindowArrays]] = {}
        for variant_name, variant in config["variants"].items():
            print(f"  [{variant_name}]", flush=True)
            use_pose = variant["use_pose"]
            feature_key = str(use_pose)
            if feature_key not in feature_cache:
                features, feature_metadata = build_features(data, all_ids, train_ids, config, scaler, use_pose)
                arrays = build_windows(data, all_ids, set(reviewed), annotations, config, rule, features)
                feature_cache[feature_key] = (features, feature_metadata, arrays)
            _, feature_metadata, arrays = feature_cache[feature_key]
            train = select(arrays, arrays.trajectory != held_out)
            if config.get("train_labeled_only", False):
                train = select(train, train.human_y >= 0)
            test = select(arrays, arrays.trajectory == held_out)
            input_dim = train.x.shape[-1]
            set_seed(int(config["seed"]) + held_out)
            model, history = train_model(train, input_dim, config, variant, device)
            variant_dir = fold_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            write_csv(variant_dir / "training_history.csv", history)
            torch.save({"state_dict": model.state_dict(), "variant": variant_name, "feature_metadata": feature_metadata}, variant_dir / "model.pt")
            test_q, test_terminal = predict(model, test.x, device, float(variant["domain_lambda"]))
            eval_mask = (test.domain == domains.index(evaluation_domain)) & (test.human_y >= 0) & (test.human_weight >= CONFIDENCE_WEIGHTS["medium"])
            metrics = binary_metrics(test.human_y[eval_mask], test_q[eval_mask, 1]) if np.any(eval_mask) else {"macro_f1": float("nan")}
            row = {"held_out_trajectory": f"traj_{held_out}", "quality_windows": int(np.sum(eval_mask)), **{f"quality_{k}": v for k, v in metrics.items()}}
            row["domain_head_accuracy"] = float("nan")
            if model.domain_head is not None:
                with torch.no_grad():
                    domain_logits = []
                    for start in range(0, len(test.x), 512):
                        _, _, _, _, domain = model(torch.from_numpy(test.x[start : start + 512]).to(device), 0.0)
                        domain_logits.append(torch.argmax(domain, dim=-1).cpu().numpy())
                    domain_pred = np.concatenate(domain_logits)
                row["domain_head_accuracy"] = float(np.mean(domain_pred == test.domain))
            all_metrics[variant_name].append(row)
            domain_mask = test.domain == domains.index(evaluation_domain)
            for index in np.flatnonzero(domain_mask):
                prediction = {
                    "trajectory": f"traj_{held_out}", "frame": int(test.frame[index]),
                    "p_misaligned": float(test_q[index, 0]), "p_aligned": float(test_q[index, 1]),
                    "human_label": int(test.human_y[index]),
                }
                if test_terminal is not None:
                    prediction["p_terminal"] = float(test_terminal[index])
                all_predictions[variant_name].append(prediction)
            print(f"    result macro_f1={metrics.get('macro_f1', float('nan')):.3f} aligned_f1={metrics.get('aligned_f1', float('nan')):.3f}", flush=True)

    aggregate: dict[str, object] = {"folds_completed": len(folds), "elapsed_minutes": (time.time() - started) / 60.0, "variants": {}}
    for variant_name, rows in all_metrics.items():
        predictions = all_predictions[variant_name]
        valid = [row for row in predictions if row["human_label"] >= 0]
        y = np.asarray([row["human_label"] for row in valid], dtype=np.int64)
        p = np.asarray([row["p_aligned"] for row in valid], dtype=np.float64)
        metrics = binary_metrics(y, p)
        fold_values = np.asarray([float(row.get("quality_macro_f1", float("nan"))) for row in rows])
        domain_values = [float(row.get("domain_head_accuracy", float("nan"))) for row in rows]
        domain_values = [value for value in domain_values if np.isfinite(value)]
        aggregate["variants"][variant_name] = {
            **{f"pooled_{key}": value for key, value in metrics.items()},
            "fold_macro_f1_mean": float(np.nanmean(fold_values)),
            "domain_head_accuracy_mean": float(np.mean(domain_values)) if domain_values else None,
            "folds": len(rows),
        }
        write_csv(output_dir / f"{variant_name}_fold_metrics.csv", rows)
        write_csv(output_dir / f"{variant_name}_predictions.csv", predictions)
    save_json(output_dir / "aggregate_metrics.json", aggregate)

    selected = list(config["variants"].keys())
    figure, axes = plt.subplots(len(selected), 1, figsize=(16, 3.0 * len(selected)), sharex=False)
    if len(selected) == 1:
        axes = [axes]
    for axis, variant_name in zip(axes, selected):
        rows = [row for row in all_predictions[variant_name] if row["trajectory"] in {"traj_0", "traj_6", "traj_25", "traj_32"}]
        for trajectory in (0, 6, 25, 32):
            part = sorted([row for row in rows if row["trajectory"] == f"traj_{trajectory}"], key=lambda row: row["frame"])
            if not part:
                continue
            frame = [row["frame"] for row in part]
            axis.plot(frame, [row["p_misaligned"] for row in part], label=f"traj_{trajectory} p_mis")
            axis.plot(frame, [row["p_aligned"] for row in part], linestyle="--", label=f"traj_{trajectory} p_aligned")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(variant_name, loc="left")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=7)
    axes[-1].set_xlabel("frame")
    figure.suptitle(f"Transformer insertion-quality predictions on held-out {evaluation_domain} trajectories")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_dir / "v3_variant_timelines.png", dpi=180)
    plt.close(figure)

    report = [
        f"# {config.get('report_title', 'Insertion Quality V3 Transformer Audit')}",
        "",
        "## Setup",
        "",
        f"- History window: `{config['representation']['history_frames']}` frames.",
        f"- Force input: `{config.get('force_feature_mode', 'v3_bounded')}` derived from Fx/Fy/Fz; torque excluded.",
        f"- Input domains: `{domains}`; supervised domains: `{config.get('supervised_domains', ['real'])}`; evaluation domain: `{evaluation_domain}`.",
        f"- Sim source: `{config['data']['sim_root']}/{config['data']['sim_force_key']}`.",
        f"- History must remain within one reviewed phase: `{config.get('require_consistent_history', False)}`.",
        "- Labels: reviewed `misaligned_contact` and `aligned_progress`; uncertain/free/terminal endpoints are masked for quality classification.",
        "- No TAVLA policy code was modified.",
        "",
        "## Results",
        "",
    ]
    for variant_name, values in aggregate["variants"].items():
        report.extend([
            f"### {variant_name}",
            "",
            f"- Pooled Macro-F1: `{values['pooled_macro_f1']:.3f}`",
            f"- Misaligned F1: `{values['pooled_misaligned_f1']:.3f}`",
            f"- Aligned F1: `{values['pooled_aligned_f1']:.3f}`",
            f"- Accuracy: `{values['pooled_accuracy']:.3f}`",
            f"- AUROC: `{values['pooled_auroc']:.3f}`",
            f"- Fold Macro-F1 mean: `{values['fold_macro_f1_mean']:.3f}`",
            f"- Domain-head accuracy: `{values['domain_head_accuracy_mean']:.3f}`" if values["domain_head_accuracy_mean"] is not None else "- Domain-head accuracy: n/a",
            "",
        ])
    report.extend(["## Interpretation", ""])
    if config.get("force_feature_mode") == "policy_force_trend_v1":
        report.extend([
            "This is held-out Sim classification using phase labels transferred by paired trajectory/frame from the reviewed Real/older-Sim set. It does not validate the new Sim contact state independently.",
            "The encoder is standalone; policy integration and Real-domain performance require separate evaluation.",
        ])
    else:
        report.extend([
            "The primary comparison is whether adding motion improves aligned-progress F1 over force-only Transformer, and whether C lowers domain-head accuracy without losing quality F1. The domain-head result is diagnostic; it is not sufficient by itself to justify integration.",
            "",
            "The current V2 pooled baseline was Macro-F1 0.690, aligned F1 0.550, and latent domain accuracy 0.892. This V3 report must be compared against those values and against the handcrafted Logistic baseline 0.736.",
        ])
    report_text = "\n".join(report) + "\n"
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")
    if max_folds is None and epochs_override is None:
        (repo / config.get("report_filename", "insertion_quality_encoder_v3_transformer_report.md")).write_text(report_text, encoding="utf-8")
    print("\nAggregate metrics:")
    print(json.dumps(aggregate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_v3_transformer.json"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/insertion_quality_v3_transformer"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for a smoke run")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    annotations = args.annotations if args.annotations.is_absolute() else repo / args.annotations
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, config, annotations, output_dir, args.device, args.max_folds, args.variant, args.epochs)


if __name__ == "__main__":
    main()
