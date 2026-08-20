"""General sim-to-real wrench adaptation utilities.

The module implements the data-level adaptation (DLA) part of PolyFit for a
six-dimensional wrench.  It deliberately does not contain task-specific
features: the default input is only ``[Fx, Fy, Fz, Tx, Ty, Tz]``.  A separate
robust affine adapter is provided for unpaired data; that adapter is a useful
diagnostic baseline, but it is not a replacement for PolyFit's paired-data
supervision.

The adapter is expressed as a direction, because both use cases are useful:

* ``sim_to_real``: canonicalize simulated wrench before training a real-domain
  policy (the recommended direction for the current TAVLA checkpoint).
* ``real_to_sim``: reproduce PolyFit's original DLA deployment convention.

All statistics are stored in the checkpoint and are computed from the
training split only by ``scripts/train_wrench_adapter.py``.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


WRENCH_DIM = 6
FORCE_DIM = 3
TORQUE_DIM = 3
WRENCH_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def _as_float32_stats(value: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (WRENCH_DIM,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def robust_location_scale(values: np.ndarray, *, quantile_low: float = 0.01, quantile_high: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """Return robust per-channel center and scale.

    The center is the median and the scale is half the central quantile range.
    A standard-deviation fallback prevents a constant channel from producing a
    numerical singularity.  The function is intentionally channel-wise: a
    global scale would let large force values dominate the torque channels.
    """

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != WRENCH_DIM:
        raise ValueError(f"Expected [N, 6] wrench values, got {values.shape}")
    if len(values) == 0:
        raise ValueError("Cannot compute wrench statistics from an empty array")
    if not np.isfinite(values).all():
        raise ValueError("Wrench values contain NaN/Inf")
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("Invalid robust quantile range")

    center = np.median(values, axis=0)
    low = np.quantile(values, quantile_low, axis=0)
    high = np.quantile(values, quantile_high, axis=0)
    scale = (high - low) / 2.0
    std = np.std(values, axis=0)
    scale = np.where(scale > 1e-5, scale, std)
    scale = np.maximum(scale, 1e-3)
    return center.astype(np.float32), scale.astype(np.float32)


def _mlp(in_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Sequential:
    if depth < 1:
        raise ValueError("MLP depth must be at least one")
    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(depth):
        layers.extend((nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = hidden_dim
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class WrenchAdapterConfig:
    """Architecture and preprocessing metadata saved with an adapter."""

    kind: str = "polyfit_dla"
    direction: str = "sim_to_real"
    input_dim: int = WRENCH_DIM
    hidden_dim: int = 64
    depth: int = 3
    dropout: float = 0.10
    residual_limit: float = 3.0
    version: int = 1

    def validate(self) -> None:
        if self.kind not in {"polyfit_dla", "robust_affine"}:
            raise ValueError(f"Unsupported adapter kind: {self.kind}")
        if self.direction not in {"sim_to_real", "real_to_sim"}:
            raise ValueError(f"Unsupported adapter direction: {self.direction}")
        if self.input_dim != WRENCH_DIM:
            raise ValueError(f"Only six-dimensional wrench input is supported, got {self.input_dim}")


class PolyFitWrenchDLA(nn.Module):
    """A bounded residual MLP following PolyFit's DLA structure.

    Force and torque receive separate encoders, followed by a fused feature
    block and separate force/torque heads.  The output is a residual in robust
    standardized coordinates.  Zero-initialized heads make the initial model
    the robust affine baseline, while ``tanh`` bounds the learned correction.
    """

    def __init__(
        self,
        source_center: np.ndarray | list[float],
        source_scale: np.ndarray | list[float],
        target_center: np.ndarray | list[float],
        target_scale: np.ndarray | list[float],
        config: WrenchAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or WrenchAdapterConfig()
        self.config.validate()
        self.register_buffer("source_center", torch.from_numpy(_as_float32_stats(source_center, "source_center")))
        self.register_buffer("source_scale", torch.from_numpy(_as_float32_stats(source_scale, "source_scale")))
        self.register_buffer("target_center", torch.from_numpy(_as_float32_stats(target_center, "target_center")))
        self.register_buffer("target_scale", torch.from_numpy(_as_float32_stats(target_scale, "target_scale")))

        hidden = self.config.hidden_dim
        self.force_encoder = _mlp(FORCE_DIM, hidden, self.config.depth, self.config.dropout)
        self.torque_encoder = _mlp(TORQUE_DIM, hidden, self.config.depth, self.config.dropout)
        self.fusion = _mlp(2 * hidden, hidden, 2, self.config.dropout)
        self.force_head = nn.Linear(hidden, FORCE_DIM)
        self.torque_head = nn.Linear(hidden, TORQUE_DIM)
        nn.init.zeros_(self.force_head.weight)
        nn.init.zeros_(self.force_head.bias)
        nn.init.zeros_(self.torque_head.weight)
        nn.init.zeros_(self.torque_head.bias)

    def _standardize_source(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.source_center) / self.source_scale

    def _destandardize_target(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.target_scale + self.target_center

    def normalized_residual(self, wrench: torch.Tensor) -> torch.Tensor:
        value = self._standardize_source(wrench)
        force_feature = self.force_encoder(value[..., :FORCE_DIM])
        torque_feature = self.torque_encoder(value[..., FORCE_DIM:])
        fused = self.fusion(torch.cat((force_feature, torque_feature), dim=-1))
        residual = torch.cat((self.force_head(fused), self.torque_head(fused)), dim=-1)
        return torch.tanh(residual) * self.config.residual_limit

    def forward(self, wrench: torch.Tensor) -> torch.Tensor:
        if wrench.shape[-1] != WRENCH_DIM:
            raise ValueError(f"Expected wrench last dimension 6, got {wrench.shape}")
        source_z = self._standardize_source(wrench)
        target_z = source_z + self.normalized_residual(wrench)
        return self._destandardize_target(target_z)

    @torch.no_grad()
    def transform_numpy(self, values: np.ndarray, *, device: str | torch.device = "cpu") -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != WRENCH_DIM:
            raise ValueError(f"Expected wrench last dimension 6, got {array.shape}")
        was_training = self.training
        self.eval()
        output = self.to(device)(torch.from_numpy(array).to(device)).cpu().numpy()
        if was_training:
            self.train()
        return output.astype(np.float32, copy=False)


class RobustAffineWrenchAdapter:
    """Distribution-only baseline for unpaired sim and real wrench streams."""

    def __init__(
        self,
        source_center: np.ndarray | list[float],
        source_scale: np.ndarray | list[float],
        target_center: np.ndarray | list[float],
        target_scale: np.ndarray | list[float],
        *,
        direction: str = "sim_to_real",
    ) -> None:
        self.config = WrenchAdapterConfig(kind="robust_affine", direction=direction)
        self.config.validate()
        self.source_center = _as_float32_stats(source_center, "source_center")
        self.source_scale = np.maximum(_as_float32_stats(source_scale, "source_scale"), 1e-3)
        self.target_center = _as_float32_stats(target_center, "target_center")
        self.target_scale = np.maximum(_as_float32_stats(target_scale, "target_scale"), 1e-3)

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != WRENCH_DIM:
            raise ValueError(f"Expected wrench last dimension 6, got {array.shape}")
        output = (array - self.source_center) / self.source_scale * self.target_scale + self.target_center
        return output.astype(np.float32, copy=False)


def build_robust_affine_adapter(source: np.ndarray, target: np.ndarray, *, direction: str = "sim_to_real") -> RobustAffineWrenchAdapter:
    source_center, source_scale = robust_location_scale(source)
    target_center, target_scale = robust_location_scale(target)
    return RobustAffineWrenchAdapter(source_center, source_scale, target_center, target_scale, direction=direction)


def adapter_metadata(
    adapter: PolyFitWrenchDLA | RobustAffineWrenchAdapter,
    *,
    train_frames: int | None = None,
    train_episodes: int | None = None,
    val_frames: int | None = None,
    val_episodes: int | None = None,
) -> dict[str, Any]:
    def as_list(value: np.ndarray | torch.Tensor) -> list[float]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value).tolist()

    metadata: dict[str, Any] = {
        "format": "tavla_wrench_adapter",
        "wrench_names": list(WRENCH_NAMES),
        "source_center": as_list(adapter.source_center),
        "source_scale": as_list(adapter.source_scale),
        "target_center": as_list(adapter.target_center),
        "target_scale": as_list(adapter.target_scale),
        "config": asdict(adapter.config),
    }
    if train_frames is not None:
        metadata["train_frames"] = train_frames
    if train_episodes is not None:
        metadata["train_episodes"] = train_episodes
    if val_frames is not None:
        metadata["val_frames"] = val_frames
    if val_episodes is not None:
        metadata["val_episodes"] = val_episodes
    return metadata


def save_adapter(adapter: PolyFitWrenchDLA | RobustAffineWrenchAdapter, path: str | Path, **counts: int) -> None:
    """Save an adapter as a portable ``.pt`` checkpoint plus JSON metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = adapter_metadata(adapter, **counts)
    if isinstance(adapter, PolyFitWrenchDLA):
        payload = {
            "metadata": metadata,
            "state_dict": adapter.state_dict(),
        }
        torch.save(payload, path)
    else:
        payload = {"metadata": metadata}
        torch.save(payload, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_adapter(path: str | Path, *, device: str | torch.device = "cpu") -> PolyFitWrenchDLA | RobustAffineWrenchAdapter:
    """Load either a PolyFit DLA or a robust affine adapter."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = payload["metadata"]
    config = WrenchAdapterConfig(**metadata["config"])
    config.validate()
    if config.kind == "robust_affine":
        return RobustAffineWrenchAdapter(
            metadata["source_center"],
            metadata["source_scale"],
            metadata["target_center"],
            metadata["target_scale"],
            direction=config.direction,
        )
    adapter = PolyFitWrenchDLA(
        metadata["source_center"],
        metadata["source_scale"],
        metadata["target_center"],
        metadata["target_scale"],
        config=config,
    )
    adapter.load_state_dict(payload["state_dict"])
    return adapter.to(device)
