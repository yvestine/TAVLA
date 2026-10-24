"""Serve a new forecast-feature policy on port 8006 with strict artifact contracts."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import socket

import numpy as np

from . import FORMAT, LAYOUT
from .common import read_manifest, sha256
from .prepare_policy import CONFIG_NAME


class ValidatedPolicy:
    def __init__(self, policy, encoder_sha256):
        self.policy, self.encoder_sha256 = policy, encoder_sha256

    def reset(self):
        self.policy.reset()

    def infer(self, obs):
        if not isinstance(obs, dict) or obs.get("encoder_sha256") != self.encoder_sha256:
            raise ValueError("Client encoder SHA does not match trained policy")
        for key, shape in (("state", (8,)), ("effort", (10, 32))):
            value = np.asarray(obs.get(key))
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite float32 {shape}")
        camera_keys = ["cam_high", "cam_left_wrist"]
        if "cam_right_wrist" in obs.get("images", {}):
            camera_keys.append("cam_right_wrist")
        for key in camera_keys:
            value = np.asarray(obs.get("images", {}).get(key))
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) < 4:
                raise ValueError(f"{key} must be RGB uint8 HxWx3")
        if not isinstance(obs.get("prompt"), str) or not obs["prompt"]:
            raise ValueError("Nonempty prompt required")
        request = {key: value for key, value in obs.items() if key != "encoder_sha256"}
        result = self.policy.infer(request)
        actions = np.asarray(result.get("actions"))
        if actions.shape != (50, 8) or not np.isfinite(actions).all():
            raise ValueError("Expected finite absolute joint/gripper action chunk (50,8)")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    with socket.socket() as probe:
        probe.bind((args.host, args.port))
    manifest = json.loads(args.policy_manifest.read_text())
    if manifest.get("artifact_format") != FORMAT or manifest.get("status") != "policy_train_ready" or manifest.get("layout") != LAYOUT:
        raise ValueError("Unaccepted policy manifest")
    accepted, _ = read_manifest(manifest["encoder_manifest"], require_ready=True)
    if accepted["encoder_sha256"] != manifest["encoder_sha256"]:
        raise ValueError("Encoder / policy manifest mismatch")
    checkpoint = args.checkpoint_dir.resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError("Policy params not found")
    saved_contract = json.loads((checkpoint / "assets/force_gap_policy_manifest.json").read_text())
    if saved_contract != manifest:
        raise ValueError("Checkpoint policy contract differs from requested manifest")
    norm = checkpoint / "assets/norm_stats.json"
    # Config may pad state/actions to 32 before checkpoint saving; compare against
    # the validated training statistics after the same padding, below.
    os.environ["FORCE_GAP_FORECAST_POLICY_MANIFEST"] = str(args.policy_manifest.resolve())
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from openpi.training import config
    from openpi.shared import normalize
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    cfg = config.get_config(CONFIG_NAME)
    trained_norm = cfg.data.create(cfg.assets_dirs, cfg.model).norm_stats
    saved_norm = normalize.load(checkpoint / "assets")
    if set(trained_norm) != set(saved_norm) or any(
            not np.array_equal(trained_norm[key].mean, saved_norm[key].mean)
            or not np.array_equal(trained_norm[key].std, saved_norm[key].std) for key in trained_norm):
        raise ValueError("Saved policy normalization differs from accepted shared statistics")
    policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt="peg-in-hole", norm_stats=saved_norm)
    metadata = {**policy.metadata, "policy_variant": "force_gap_forecast_cotrain_50_50", "encoder_sha256": manifest["encoder_sha256"],
                "encoder_artifact_format": FORMAT, "layout": LAYOUT, "encoder_history_frames": 32,
                "future_force_frames": 5, "baseline_frames": 0, "policy_history_frames": 10,
                "state_shape": [8], "effort_shape": [10, 32], "action_shape": [50, 8],
                "actions_are_absolute": True, "sample_period_s": .1, "execute_action_index": 1,
                "encoder_required_inputs": ["force_xyz_n", "actual_q_rad", "actual_dq_rad_s", "timestamp_s"]}
    logging.info("Serving %s on %s:%s", checkpoint, args.host, args.port)
    WebsocketPolicyServer(ValidatedPolicy(policy, manifest["encoder_sha256"]), host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
