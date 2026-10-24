#!/usr/bin/env python3
"""Convert paired raw HDF5 episodes to LeRobot with frozen force-gap v2 effort."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import tqdm
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.utils import serialize_dict, write_json

from convert_user_hdf5_to_lerobot import _create_dataset, _read_video, _trajectory_id
from force_gap_encoder_v2.deploy_infer import ForceGapLatentEncoder


def load_episode(path: Path, force_key: str, encoder: ForceGapLatentEncoder):
    with h5py.File(path, "r") as episode:
        state = np.concatenate(
            [episode["obs/state/joint_pos"][:], episode["obs/state/gripper_pos"][:]], axis=-1
        ).astype(np.float32)
        action = np.concatenate(
            [episode["action/actual/arm"][:], episode["action/actual/gripper"][:]], axis=-1
        ).astype(np.float32)
        force = np.asarray(episode[force_key], dtype=np.float32)
        timestamps = np.asarray(episode["timestamps"], dtype=np.float64)
    if force.ndim != 2 or force.shape[1] < 3:
        raise ValueError(f"{path}: {force_key} must have shape [N,>=3], got {force.shape}")
    if len(state) != len(action) or len(force) != len(state):
        raise ValueError(f"{path}: state/action/force lengths differ")
    effort = encoder.latents(force[:, :3], timestamps).astype(np.float32)
    if len(effort) != len(state):
        raise ValueError(f"{path}: encoder returned {len(effort)} latents for {len(state)} frames")
    front = _read_video(path.parent / "front_camera.mp4", len(state))
    wrist = _read_video(path.parent / "wrist_camera.mp4", len(state))
    return state, action, effort, front, wrist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--encoder-path", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/encoder.pt"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--task", default="peg-in-hole")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.raw_dir.glob("traj_*/data.h5"), key=_trajectory_id)
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No traj_*/data.h5 files found under {args.raw_dir}")

    encoder = ForceGapLatentEncoder(args.encoder_path, device="cpu")
    dataset = _create_dataset(
        args.repo_id,
        fps=args.fps,
        effort_dim=encoder.latent_dim,
        effort_names=tuple(f"latent_{index}" for index in range(encoder.latent_dim)),
        overwrite=args.overwrite,
        use_videos=not args.no_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )
    for path in tqdm.tqdm(files, desc=f"Converting {args.repo_id}"):
        state, action, effort, front, wrist = load_episode(path, args.force_key, encoder)
        for index in range(len(state)):
            dataset.add_frame(
                {
                    "observation.state": state[index],
                    "observation.effort": effort[index],
                    "action": action[index],
                    "observation.images.cam_high": front[index],
                    "observation.images.cam_left_wrist": wrist[index],
                    "observation.images.cam_right_wrist": wrist[index],
                }
            )
        dataset.save_episode(task=args.task)
    dataset.consolidate(run_compute_stats=False)
    dataset.meta.stats = compute_stats(dataset, batch_size=8, num_workers=0)
    write_json(serialize_dict(dataset.meta.stats), dataset.root / "meta/stats.json")
    print(f"Wrote {LEROBOT_HOME / args.repo_id} with {encoder.latent_dim}-D force-gap latent")


if __name__ == "__main__":
    main()
