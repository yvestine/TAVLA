#!/usr/bin/env python3
"""Serve the completed pure-real force-alignment-latent policy checkpoint.

Uses normalization statistics stored WITH the policy checkpoint and publishes
the required encoder identity/schema in the websocket handshake metadata.
Run in TAVLA's .venv. The simulator computes its 32-D latent history locally.
"""

import argparse
import dataclasses
import hashlib
import logging
from pathlib import Path
import sys

import numpy as np


CONFIG_NAME = "pi0_lora_user_single_arm_control_latent"
REPO_ID = "local/tavla_single_arm_force_align_latent"
ENCODER_SHA256 = "db793ebd4f3158b8a8add895cf04368b2289df7fad450f1d2a39b09bb46acd80"
DEFAULT_CHECKPOINT = "checkpoints/pi0_lora_user_single_arm_control_latent/force_align_latent_gpu0123_fsdp/29999"


class ValidatedPolicy:
    def __init__(self, policy):
        self._policy = policy

    def reset(self):
        self._policy.reset()

    def infer(self, obs):
        if not isinstance(obs, dict):
            raise ValueError("Request must be a mapping")
        for name, shape in (("state", (8,)), ("effort", (10, 32))):
            value = np.asarray(obs.get(name))
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite float32 {shape}; effort is unnormalized latent history")
        images = obs.get("images", {})
        for name in ("cam_high", "cam_left_wrist"):
            value = np.asarray(images.get(name))
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) < 4:
                raise ValueError(f"images/{name} must be RGB uint8 (H,W,3)")
        if "cam_right_wrist" in images:
            value = np.asarray(images["cam_right_wrist"])
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) < 4:
                raise ValueError("images/cam_right_wrist must be RGB uint8 (H,W,3)")
        if "prompt" in obs and (not isinstance(obs["prompt"], str) or not obs["prompt"]):
            raise ValueError("prompt must be a nonempty string")
        return self._policy.infer(obs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tavla-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--encoder", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    root = args.tavla_root.resolve()
    checkpoint = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    encoder = args.encoder if args.encoder.is_absolute() else root / args.encoder
    if hashlib.sha256(encoder.read_bytes()).hexdigest() != ENCODER_SHA256:
        raise ValueError("Encoder changed since this policy's dataset was built; use this bundle's assets/encoder.pt")
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets/norm_stats.json").is_file():
        raise FileNotFoundError(f"Missing policy params or checkpoint normalization statistics: {checkpoint}")
    datasets = (checkpoint / "assets/datasets.txt").read_text().splitlines()
    if datasets != [REPO_ID]:
        raise ValueError(f"This launcher serves the pure-real policy only; found datasets {datasets}")
    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src")]
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    cfg = config.get_config(CONFIG_NAME)
    # Redirect config initialization to the checkpoint's exact stats too, so
    # no old ControlEncoder dataset/assets are consulted at server startup.
    data = dataclasses.replace(
        cfg.data,
        repo_id=REPO_ID,
        assets=config.AssetsConfig(assets_dir=str(checkpoint / "assets"), asset_id="."),
    )
    cfg = dataclasses.replace(cfg, data=data)
    policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt="peg-in-hole")
    metadata = {
        **policy.metadata,
        "policy_variant": "force_align_latent_real_only",
        "training_config": CONFIG_NAME,
        "checkpoint": str(checkpoint),
        "training_dataset": REPO_ID,
        "encoder_artifact_format": "tavla_force_alignment_encoder_terminal_reference_v1",
        "encoder_sha256": ENCODER_SHA256,
        "encoder_history_frames": 32,
        "baseline_frames": 20,
        "policy_history_frames": 10,
        "effort_shape": [10, 32],
        "state_shape": [8],
        "action_shape": [50, 8],
        "actions_are_absolute": True,
        "sample_period_s": 0.1,
        "default_prompt": "peg-in-hole",
        "state_units": "seven joints in rad; gripper_pos normalized with maximum total width 0.08 m",
    }
    logging.info("Serving pure-real force-alignment-latent policy on %s:%s", args.host, args.port)
    logging.info("Encoder SHA256: %s", ENCODER_SHA256)
    server = WebsocketPolicyServer(ValidatedPolicy(policy), host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
