#!/usr/bin/env python3
"""Serve force-alignment cotraining with raw wrench encoding on the server CPU."""

import argparse
import copy
import dataclasses
import hashlib
import logging
from pathlib import Path
import sys

import numpy as np


CONFIG_NAME = "pi0_lora_user_single_arm_force_align_latent_cotrain_50_50"
REPO_IDS = [
    "local/tavla_single_arm_force_align_latent",
    "local/tavla_single_arm_force_align_latent_sim",
]
SAMPLE_WEIGHTS = [0.5, 0.5]
ENCODER_SHA256 = "db793ebd4f3158b8a8add895cf04368b2289df7fad450f1d2a39b09bb46acd80"
DEFAULT_CHECKPOINT = (
    "checkpoints/pi0_lora_user_single_arm_force_align_latent_cotrain_50_50/"
    "force_align_latent_cotrain_50_50_v1/29999"
)


def validate_observation(obs):
    state = np.asarray(obs.get("state"))
    if state.shape != (8,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("state must be finite float32 (8,): seven joint angles and normalized gripper")
    images = obs.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("images must be a mapping")
    names = ["cam_high", "cam_left_wrist"]
    if "cam_right_wrist" in images:
        names.append("cam_right_wrist")
    for name in names:
        value = np.asarray(images.get(name))
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) < 4:
            raise ValueError(f"images/{name} must be RGB uint8 (H,W,3)")
    if "prompt" in obs and (not isinstance(obs["prompt"], str) or not obs["prompt"]):
        raise ValueError("prompt must be a nonempty string")


class ValidatedPolicy:
    def __init__(self, policy):
        self._policy = policy

    def reset(self):
        self._policy.reset()

    def infer(self, obs):
        if not isinstance(obs, dict):
            raise ValueError("Request must be a mapping")
        validate_observation(obs)
        effort = np.asarray(obs.get("effort"))
        if effort.shape != (10, 32) or effort.dtype != np.float32 or not np.isfinite(effort).all():
            raise ValueError("effort must be finite float32 (10,32) latent history")
        result = self._policy.infer(obs)
        actions = np.asarray(result.get("actions"))
        if actions.shape != (50, 8) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite absolute actions with shape (50,8); got {actions.shape}")
        return result


class RawWrenchPolicy:
    """Accept real base-frame wrench and maintain this connection's encoder state."""

    def __init__(self, policy, encoder):
        from insertion_alignment_encoder.streaming import StreamingForceAlignmentEncoder

        self._policy = policy
        self._stream = StreamingForceAlignmentEncoder(encoder)
        self._explicit_timestamps = None

    def reset(self):
        self._stream.reset()
        self._explicit_timestamps = None
        self._policy.reset()

    def infer(self, obs):
        if not isinstance(obs, dict):
            raise ValueError("Request must be a mapping")
        wrench = np.asarray(obs.get("effort"), dtype=np.float32)
        if wrench.shape == (6,):
            wrench = wrench[None]
        if wrench.ndim != 2 or wrench.shape[1] != 6 or not len(wrench) or not np.isfinite(wrench).all():
            raise ValueError("effort must be finite raw [Fx,Fy,Fz,Tx,Ty,Tz], shape (6,) or (N,6)")
        if obs.get("force_source", "ee_wrench_base") != "ee_wrench_base":
            raise ValueError("Use real ee_wrench_base, with the training coordinate frame, sign, and units")
        update_only = obs.get("update_only", False)
        if not isinstance(update_only, bool):
            raise ValueError("update_only must be a boolean")
        if not update_only:
            validate_observation(obs)

        scalar_time, vector_time = "timestamp_s" in obs, "timestamps_s" in obs
        if scalar_time and vector_time:
            raise ValueError("Supply only one of timestamp_s or timestamps_s")
        explicit = scalar_time or vector_time
        if self._explicit_timestamps is not None and explicit != self._explicit_timestamps:
            raise ValueError("Keep the same timestamp mode within an episode; reset before switching")
        if scalar_time:
            stamp = np.asarray(obs["timestamp_s"], dtype=np.float64)
            if stamp.ndim != 0:
                raise ValueError("timestamp_s must be a scalar; use timestamps_s for batches")
            times = stamp.reshape(1)
        elif vector_time:
            times = np.asarray(obs["timestamps_s"], dtype=np.float64)
        else:
            # Legacy clients sending just wrench use the training sample period.
            # Network arrival time is not a substitute for sensor sample time.
            period = self._stream.encoder.sample_period_s
            start = 0.0 if self._stream.last_timestamp is None else self._stream.last_timestamp + period
            times = start + np.arange(len(wrench), dtype=np.float64) * period
        if times.shape != (len(wrench),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("One finite, strictly increasing timestamp in seconds per NEW wrench sample is required")
        if self._stream.last_timestamp is not None and times[0] <= self._stream.last_timestamp:
            raise ValueError("Send only NEW chronological samples; reset between episodes")

        self._explicit_timestamps = explicit
        for sample, stamp in zip(wrench, times, strict=True):
            # This checkpoint's frozen encoder was trained on XYZ only.
            self._stream.push(sample[:3], stamp)
        status = {
            "ready": self._stream.ready,
            "updated": len(wrench),
            "samples_received": self._stream.samples_received,
            "samples_needed": self._stream.samples_needed,
            "timestamp_s": float(times[-1]),
        }
        if update_only or not self._stream.ready:
            return status
        inputs = {key: obs[key] for key in ("images", "state", "prompt") if key in obs}
        inputs["effort"] = self._stream.effort_history()
        return {**self._policy.infer(inputs), **status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tavla-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--encoder", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--input-mode", choices=("raw", "latent"), default="raw")
    args = parser.parse_args()

    root = args.tavla_root.resolve()
    checkpoint = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    encoder = args.encoder if args.encoder.is_absolute() else root / args.encoder
    if hashlib.sha256(encoder.read_bytes()).hexdigest() != ENCODER_SHA256:
        raise ValueError("Encoder differs from the one used to build the force-alignment datasets")
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets/norm_stats.json").is_file():
        raise FileNotFoundError(f"Missing cotraining params or saved normalization statistics: {checkpoint}")
    datasets = (checkpoint / "assets/datasets.txt").read_text().splitlines()
    if datasets != REPO_IDS:
        raise ValueError(f"Expected real + sim datasets in the 50:50 checkpoint; found {datasets}")

    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src"), str(root / "scripts")]
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    cfg = config.get_config(CONFIG_NAME)
    if list(cfg.data.repo_id) != REPO_IDS or list(cfg.data.dataset_sampling_weights) != SAMPLE_WEIGHTS:
        raise ValueError("Repository cotraining config is not the expected 50:50 force-alignment setup")
    # Policy normalization is loaded from checkpoint/assets by create_trained_policy.
    # Avoid a second lookup of per-dataset training-only assets during serving.
    data = dataclasses.replace(
        cfg.data,
        assets=config.AssetsConfig(assets_dir=str(checkpoint / "assets"), asset_id="."),
    )
    cfg = dataclasses.replace(cfg, data=data)
    policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt="peg-in-hole")
    frozen_encoder = None
    if args.input_mode == "raw":
        import torch
        from insertion_alignment_encoder.deploy_infer import ForceAlignmentLatentEncoder

        torch.set_num_threads(1)
        frozen_encoder = ForceAlignmentLatentEncoder(encoder, device="cpu")
        frozen_encoder.model.requires_grad_(False)
    metadata = {
        **policy.metadata,
        "policy_variant": "force_align_latent_cotrain_50_50",
        "training_config": CONFIG_NAME,
        "checkpoint": str(checkpoint),
        "training_datasets": REPO_IDS,
        "dataset_sampling_weights": SAMPLE_WEIGHTS,
        "encoder_artifact_format": "tavla_force_alignment_encoder_terminal_reference_v1",
        "encoder_sha256": ENCODER_SHA256,
        "encoder_history_frames": 32,
        "baseline_frames": 20,
        "policy_history_frames": 10,
        "input_mode": args.input_mode,
        "encoder_location": "server_cpu" if frozen_encoder is not None else "client",
        "effort_shape": [1, 6] if frozen_encoder is not None else [10, 32],
        "internal_effort_shape": [10, 32],
        "effort_shapes": [[6], [1, 6], ["N", 6]] if frozen_encoder is not None else [[10, 32]],
        "force_channels_used": ["Fx", "Fy", "Fz"],
        "force_source": "ee_wrench_base",
        "timestamp_required": False,
        "timestamp_default": "consecutive_samples_at_10hz",
        "history_isolated_per_connection": True,
        "update_only_supported": frozen_encoder is not None,
        "warmup_samples": 20 if frozen_encoder is not None else 0,
        "state_shape": [8],
        "action_shape": [50, 8],
        "actions_are_absolute": True,
        "sample_period_s": 0.1,
        "default_prompt": "peg-in-hole",
    }
    logging.info("Serving 50:50 force-alignment cotrain policy at %s:%s", args.host, args.port)
    logging.info("Checkpoint: %s", checkpoint)
    logging.info("Encoder SHA256: %s", ENCODER_SHA256)
    logging.info("Input mode: %s; encoder location: %s", args.input_mode, metadata["encoder_location"])

    class SessionServer(WebsocketPolicyServer):
        async def _handler(self, websocket):
            # Sharing compiled model arrays is cheap; each policy copy keeps its
            # own RNG, while RawWrenchPolicy owns its baseline and both histories.
            session_policy = ValidatedPolicy(copy.copy(policy))
            if frozen_encoder is not None:
                session_policy = RawWrenchPolicy(session_policy, frozen_encoder)
            await WebsocketPolicyServer(session_policy, metadata=metadata)._handler(websocket)

    server = SessionServer(policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
