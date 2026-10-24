#!/usr/bin/env python3
"""Serve force-predictive v3 TAVLA with raw wrench encoding on the server."""

import argparse
import copy
import dataclasses
import json
import logging
from pathlib import Path

from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
import numpy as np

CONFIG_NAME = "pi0_lora_user_single_arm_force_predictive_v3_cotrain_50_50"
DEFAULT_CHECKPOINT = f"checkpoints/{CONFIG_NAME}/force_predictive_v3_cotrain_20261022_120747_3643501_thpfix/29999"
FORCE_SOURCES = ("ee_wrench_base", "panda_link7_wrench_calibrated")


class RawForcePolicy:
    """One connection's causal force and latent histories; the TAVLA model is shared."""

    def __init__(self, policy, encoder, force_source="ee_wrench_base"):
        if force_source not in FORCE_SOURCES:
            raise ValueError(f"force_source must be one of {FORCE_SOURCES}")
        self._policy = policy
        self._encoder = copy.deepcopy(encoder)
        self._encoder.reset()
        self._default_source = force_source
        self._source = None

    def reset(self):
        self._encoder.reset()
        self._source = None
        self._policy.reset()

    def infer(self, obs):
        if not isinstance(obs, dict):
            raise ValueError("Request must be a mapping")
        wrench = np.asarray(obs.get("effort"), dtype=np.float32)
        if wrench.shape == (6,):
            wrench = wrench[None]
        if wrench.ndim != 2 or wrench.shape[1] != 6 or not len(wrench) or not np.isfinite(wrench).all():
            raise ValueError("effort must be finite raw [Fx,Fy,Fz,Tx,Ty,Tz], shape (6,) or (N,6)")
        if ("timestamp_s" in obs) == ("timestamps_s" in obs):
            raise ValueError("Supply exactly one of timestamp_s or timestamps_s, in simulation seconds")
        if "timestamp_s" in obs:
            stamp = np.asarray(obs["timestamp_s"], dtype=np.float64)
            if stamp.ndim != 0:
                raise ValueError("timestamp_s must be a scalar; use timestamps_s for batches")
            times = stamp.reshape(1)
        else:
            times = np.asarray(obs["timestamps_s"], dtype=np.float64)
        if times.shape != (len(wrench),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("One finite, strictly increasing simulation timestamp per wrench is required")
        if self._encoder.time and times[0] <= self._encoder.time[-1]:
            raise ValueError("Send only NEW force samples in chronological order; reset between episodes")
        source = obs.get("force_source", self._source or self._default_source)
        if source not in FORCE_SOURCES:
            raise ValueError(f"force_source must be one of {FORCE_SOURCES}")
        if source == "panda_link7_wrench_calibrated" and self._encoder.domain != "sim":
            raise ValueError("calibrated link7 conversion is defined only for the sim branch")
        if self._source is not None and source != self._source:
            raise ValueError("force_source cannot change within an episode; reset first")
        update_only = obs.get("update_only", False)
        if not isinstance(update_only, bool):
            raise ValueError("update_only must be a boolean")
        if not update_only:
            state = np.asarray(obs.get("state"))
            if state.shape != (8,) or state.dtype != np.float32 or not np.isfinite(state).all():
                raise ValueError("state must be finite float32 (8,): seven joints and normalized gripper")
            images = obs.get("images", {})
            if not isinstance(images, dict):
                raise ValueError("images must be a mapping")
            for name in ("cam_high", "cam_left_wrist", *(["cam_right_wrist"] if "cam_right_wrist" in images else [])):
                value = np.asarray(images.get(name))
                if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) < 4:
                    raise ValueError(f"images/{name} must be RGB uint8 (H,W,3)")
            if "prompt" in obs and (not isinstance(obs["prompt"], str) or not obs["prompt"]):
                raise ValueError("prompt must be a nonempty string")
        # Conversion matches sim-data-aligned/build_aligned_sim.py: sign=-1, scale=1.
        # Already aligned ee_wrench_base uses +1. Torque never enters this encoder.
        sign = -1.0 if source == "panda_link7_wrench_calibrated" else 1.0
        self._source = source
        for sample, stamp in zip(wrench, times, strict=True):
            self._encoder.push(sign * sample[:3], stamp)
        if update_only:
            return {"updated": len(wrench), "timestamp_s": float(times[-1])}
        inputs = {key: obs[key] for key in ("images", "state", "prompt") if key in obs}
        inputs["effort"] = self._encoder.effort_history().astype(np.float32)
        result = self._policy.infer(inputs)
        actions = np.asarray(result.get("actions"))
        if actions.shape != (50, 8) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite absolute actions (50,8), got {actions.shape}")
        return {**result, "predicted_force": self._encoder.predict_future(), "timestamp_s": float(times[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--domain", choices=("sim", "real"), default="sim")
    parser.add_argument("--force-source", choices=FORCE_SOURCES, default="ee_wrench_base")
    args = parser.parse_args()
    if args.domain == "real" and args.force_source != "ee_wrench_base":
        parser.error("real domain requires ee_wrench_base")
    checkpoint = args.checkpoint_dir.resolve()
    assets = checkpoint / "assets"
    manifest_path = assets / "force_predictive_v3_encoder_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads((assets / "force_predictive_v3_training_inputs.json").read_text())
    repos = (assets / "datasets.txt").read_text().splitlines()
    if (
        contract["encoder_sha256"] != manifest["encoder_sha256"]
        or not contract["encoder_frozen"]
        or contract["effort_shape"] != [10, 32]
        or contract["repo_ids"] != repos
        or contract["sampling_weights"] != [0.5, 0.5]
    ):
        raise ValueError("Checkpoint encoder, datasets, or training input contract mismatch")
    if not (checkpoint / "params").is_dir() or not (assets / "norm_stats.json").is_file():
        raise FileNotFoundError(f"Missing params or saved normalization statistics in {checkpoint}")
    # The small frozen PyTorch encoder runs on CPU; JAX TAVLA uses the selected GPU.
    encoder = ForcePredictiveEncoder(manifest_path, args.domain, device="cpu", threads=1)
    encoder.model.requires_grad_(False)

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    cfg = config.get_config(CONFIG_NAME)
    if list(cfg.data.repo_id) != repos or list(cfg.data.dataset_sampling_weights) != contract["sampling_weights"]:
        raise ValueError("Training configuration does not match checkpoint datasets/weights")
    cfg = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, assets=config.AssetsConfig(assets_dir=str(assets), asset_id="."))
    )
    policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt="peg-in-hole")
    metadata = {
        **policy.metadata,
        "policy_variant": "force_predictive_v3_cotrain_50_50_server_encoder",
        "training_config": CONFIG_NAME,
        "checkpoint": str(checkpoint),
        "training_datasets": repos,
        "dataset_sampling_weights": contract["sampling_weights"],
        "encoder_sha256": manifest["encoder_sha256"],
        "encoder_domain": args.domain,
        "encoder_location": "server_cpu",
        "encoder_history_frames": 32,
        "policy_history_frames": 10,
        "baseline_frames": 0,
        "effort_shapes": [[6], [1, 6], ["N", 6]],
        "internal_effort_shape": [10, 32],
        "force_channels_used": ["Fx", "Fy", "Fz"],
        "force_units": "N",
        "force_frame": "robot_base",
        "default_force_source": args.force_source,
        "force_sources": list(FORCE_SOURCES) if args.domain == "sim" else ["ee_wrench_base"],
        "calibrated_link7_to_aligned_force": "-wrench[:3]; scale=1",
        "timestamp_required": True,
        "sample_period_s": 0.1,
        "reset_gap_s": 0.5,
        "history_isolated_per_connection": True,
        "update_only_supported": True,
        "state_shape": [8],
        "action_shape": [50, 8],
        "actions_are_absolute": True,
        "predicted_force_shape": [5, 3],
        "predicted_force_domain": "real",
        "predicted_force_units": "N",
        "predicted_force_head": "force_only",
        "default_prompt": "peg-in-hole",
    }

    class SessionServer(WebsocketPolicyServer):
        async def _handler(self, websocket):
            session_policy = RawForcePolicy(policy, encoder, args.force_source)
            # Reuse the existing wire protocol with a separate wrapper per connection.
            await WebsocketPolicyServer(session_policy, metadata=metadata)._handler(websocket)

    logging.info(
        "Serving %s on %s:%s, encoder=%s, domain=%s",
        checkpoint,
        args.host,
        args.port,
        manifest["encoder_sha256"],
        args.domain,
    )
    SessionServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
