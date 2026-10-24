#!/usr/bin/env python3
"""Build a 32-D-effort LeRobot dataset from a frozen force encoder.

This is a standalone sibling of ``scripts/convert_user_hdf5_to_lerobot.py``.  It
reuses that converter's dataset-building blocks unchanged (``_read_video``,
``_create_dataset``, ``_trajectory_id``) and only swaps in the force-only
encoder while preserving each domain's state, actions and images.

For each ``traj_*/data.h5`` it assembles observation.state / action / images
exactly like the ``--action-mode actual`` path of the stock converter, then sets
observation.effort to the per-frame 32-D latent produced by
``ForceAlignmentLatentEncoder`` (force-only; pose is not used).

Run from TAVLA with scripts/ and src/ on PYTHONPATH, e.g.:

    PYTHONPATH=scripts:src .venv/bin/python \
      scripts/insertion_alignment_encoder/convert_force_alignment_latent.py \
      --raw-dir data \
      --repo-id local/tavla_single_arm_force_align_latent \
      --encoder-path eval_outputs/insertion_alignment_encoder/encoder.pt \
      --no-videos --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import tqdm
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.utils import serialize_dict, write_json

# Reused, unmodified, from the stock converter.
from convert_user_hdf5_to_lerobot import _create_dataset, _read_video, _trajectory_id

from insertion_alignment_encoder.deploy_infer import ForceAlignmentLatentEncoder


def _load_episode(path: Path, effort_key: str, encoder, encoder_kind="force_alignment", domain="real"):
    with h5py.File(path, "r") as ep:
        state = np.concatenate(
            [ep["obs/state/joint_pos"][:], ep["obs/state/gripper_pos"][:]], axis=-1
        ).astype(np.float32)
        action = np.concatenate(
            [ep["action/actual/arm"][:], ep["action/actual/gripper"][:]], axis=-1
        ).astype(np.float32)
        force = ep[effort_key][:].astype(np.float32)
        timestamps = ep["timestamps"][:].astype(np.float64)
        joint_vel = ep["obs/state/joint_vel"][:].astype(np.float32) if encoder_kind == "motion_only_v2" else None
    n = state.shape[0]
    if force.shape[1] not in (3, 6):
        raise ValueError(f"{path}: force key {effort_key} must be 3-D or 6-D, got {force.shape}")
    if encoder_kind == "motion_only_v2":
        from force_gap_forecast_v1.data import Episode, segment_limits
        from force_gap_motion_force_v2.metrics import predict_episode
        starts, ends = segment_limits(timestamps)
        episode = Episode(_trajectory_id(path), force[:, :3], state[:, :7], joint_vel, timestamps,
                          np.full(n, "unannotated", object), starts, ends)
        effort = predict_episode(encoder.model, episode, encoder.stats, int(domain == "sim"))["z"].astype(np.float32)
    elif encoder_kind == "force_predictive_v3":
        effort = encoder.batch(force[:, :3], timestamps).astype(np.float32)
    else:
        effort = encoder.latents(force[:, :3], None, timestamps).astype(np.float32)
    if len(effort) != n:
        raise ValueError(f"{path}: encoder returned {len(effort)} latents for {n} frames")
    front = _read_video(path.parent / "front_camera.mp4", n)
    wrist = _read_video(path.parent / "wrist_camera.mp4", n)
    return state, action, effort, front, wrist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data"))
    parser.add_argument("--repo-id", default="local/tavla_single_arm_force_align_latent")
    parser.add_argument("--task", default="peg-in-hole")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--effort-key", default="obs/state/ee_force_base")
    parser.add_argument("--encoder-kind", choices=("force_alignment", "motion_only_v2", "force_predictive_v3"), default="force_alignment")
    parser.add_argument("--domain", choices=("real", "sim"), default="real")
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true", help="Store images instead of encoded videos.")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.raw_dir.glob("traj_*/data.h5"), key=_trajectory_id)
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No traj_*/data.h5 files found under {args.raw_dir}")

    if args.encoder_kind == "motion_only_v2":
        from force_gap_motion_force_v2 import LAYOUT
        from force_gap_motion_force_v2.runtime import JointMotionForceEncoder
        from force_gap_motion_force_v2.policy_contract import read_encoder_manifest
        read_encoder_manifest(args.encoder_path)
        encoder = JointMotionForceEncoder(args.encoder_path, args.domain, device="cpu")
        encoder.model.requires_grad_(False)
        if encoder.model.domain_gains.tolist() != [0., 0.]:
            raise ValueError("Selected motion_only checkpoint must have zero domain gains")
        latent_dim, effort_names = 32, tuple(LAYOUT)
    elif args.encoder_kind == "force_predictive_v3":
        from force_predictive_encoder_v3 import LAYOUT
        from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
        encoder = ForcePredictiveEncoder(args.encoder_path, args.domain, device="cpu")
        encoder.model.requires_grad_(False)
        latent_dim, effort_names = 32, tuple(LAYOUT)
    else:
        encoder = ForceAlignmentLatentEncoder(args.encoder_path, device="cpu")
        latent_dim = encoder.latent_dim
        effort_names = tuple(f"latent_{i}" for i in range(latent_dim))
    dataset = _create_dataset(
        args.repo_id,
        fps=args.fps,
        effort_dim=latent_dim,
        effort_names=effort_names,
        overwrite=args.overwrite,
        use_videos=not args.no_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )
    for path in tqdm.tqdm(files, desc="Converting episodes"):
        state, action, effort, front, wrist = _load_episode(path, args.effort_key, encoder, args.encoder_kind, args.domain)
        with h5py.File(path, "r") as episode_file:
            timestamps = episode_file["timestamps"][:]
        boundaries = (np.r_[0, np.flatnonzero(np.diff(timestamps) > .5) + 1, len(state)]
                      if args.encoder_kind in ("motion_only_v2", "force_predictive_v3") else np.array([0, len(state)]))
        for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
            for i in range(int(lo), int(hi)):
                dataset.add_frame(
                    {
                        "observation.state": state[i],
                        "observation.effort": effort[i],
                        "action": action[i],
                        "observation.images.cam_high": front[i],
                        "observation.images.cam_left_wrist": wrist[i],
                        "observation.images.cam_right_wrist": wrist[i],
                    }
                )
            dataset.save_episode(task=args.task)
    dataset.consolidate(run_compute_stats=False)
    dataset.meta.stats = compute_stats(dataset, batch_size=8, num_workers=0)
    write_json(serialize_dict(dataset.meta.stats), dataset.root / "meta/stats.json")
    if args.encoder_kind in ("motion_only_v2", "force_predictive_v3"):
        manifest = json.loads(args.encoder_path.read_text())
        write_json(dict(encoder_sha256=manifest["encoder_sha256"], domain=args.domain, layout=list(effort_names),
                        completed=True, action_source="domain_native_actual", source_root=str(args.raw_dir.resolve()),
                        source_h5_sha256={str(_trajectory_id(path)): hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in files}),
                   dataset.root / ("meta/force_predictive_v3_encoder.json" if args.encoder_kind == "force_predictive_v3"
                                   else "meta/motion_only_encoder.json"))
    print(f"Wrote LeRobot dataset: {LEROBOT_HOME / args.repo_id} (effort dim {latent_dim})")


if __name__ == "__main__":
    main()
