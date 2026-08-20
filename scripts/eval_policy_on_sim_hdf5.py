#!/usr/bin/env python3
"""Compare a trained TAVLA policy with 10 Hz simulation action labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from openpi.policies import policy_config
from openpi.training import config as train_config


ACTION_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]


def _decode_jpeg(value: np.ndarray) -> np.ndarray:
    image_bgr = cv2.imdecode(np.asarray(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode HDF5 JPEG image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _evaluate_episode(policy, path: Path, prompt: str, max_frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    labels = []
    with h5py.File(path, "r") as h5:
        state = np.concatenate(
            [h5["decision/obs/state/joint_pos"][:], h5["decision/obs/state/gripper_pos"][:]], axis=-1
        ).astype(np.float32)
        effort = h5["decision/obs/state/wrench_model"][:].astype(np.float32)
        labels_array = h5["decision/action/ppo_joint_targets"][:].astype(np.float32)
        front = h5["decision/images/front_jpeg"][:]
        wrist = h5["decision/images/wrist_jpeg"][:]

        count = state.shape[0] if max_frames is None else min(state.shape[0], max_frames)
        for index in range(count):
            result = policy.infer(
                {
                    "images": {
                        "cam_high": _decode_jpeg(front[index]),
                        "cam_left_wrist": _decode_jpeg(wrist[index]),
                    },
                    "state": state[index],
                    "effort": effort[index][None, :],
                    "prompt": prompt,
                }
            )
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            predictions.append(action_chunk[0, :8])
            labels.append(labels_array[index])

    return np.asarray(predictions), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data-sim-hdf5"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/pi0_lora_user_single_arm_ee_wrench_sim_overfit/sim_overfit20/999"),
    )
    parser.add_argument("--config-name", default="pi0_lora_user_single_arm_ee_wrench_sim_overfit")
    parser.add_argument("--prompt", default="peg-in-hole")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-frames-per-episode", type=int)
    parser.add_argument("--num-sample-steps", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("eval_outputs/sim_overfit20_action_error.json"))
    args = parser.parse_args()

    cfg = train_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint_dir,
        sample_kwargs={"num_steps": args.num_sample_steps},
        default_prompt=args.prompt,
    )

    files = sorted(args.source_dir.glob("episode_*/data.h5"))[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes under {args.source_dir}")

    all_predictions = []
    all_labels = []
    for path in files:
        predictions, labels = _evaluate_episode(policy, path, args.prompt, args.max_frames_per_episode)
        all_predictions.append(predictions)
        all_labels.append(labels)
        print(f"{path.parent.name}: {len(labels)} frames", flush=True)

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    absolute_error = np.abs(predictions - labels)
    result = {
        "checkpoint": str(args.checkpoint_dir),
        "episodes": len(files),
        "frames": int(labels.shape[0]),
        "prompt": args.prompt,
        "mean_mae": float(absolute_error.mean()),
        "per_dimension_mae": {
            name: float(value) for name, value in zip(ACTION_NAMES, absolute_error.mean(axis=0), strict=True)
        },
        "max_frame_mae": float(absolute_error.mean(axis=1).max()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
