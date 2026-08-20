"""Convert the current ``tavla_raw_v1`` simulation export to a LeRobot dataset.

The exporter under ``sim_hand_noise_50/tavla_raw`` stores one 30 Hz episode per
HDF5 file with raw RGB images.  TAVLA is trained at 10 Hz, so this converter
keeps one sample every ``--decision-stride`` frames and writes the final
``wrench_final`` stream as ``observation.effort``.

The CSV files are intentionally not read here.  In particular,
``actions.csv`` is the raw PPO source and is not the 8-D TAVLA action label;
the HDF5 ``action`` dataset is the canonical 8-D absolute joint-target label.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import serialize_dict
from lerobot.common.datasets.utils import write_json
import numpy as np


STATE_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)
FORCE_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def _episode_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _image_feature(shape: tuple[int, int, int]) -> dict:
    return {
        "dtype": "image",
        "shape": shape,
        "names": ["channels", "height", "width"],
    }


def _create_dataset(repo_id: str, image_shape: tuple[int, int, int], overwrite: bool) -> LeRobotDataset:
    dataset_path = LEROBOT_HOME / repo_id
    if dataset_path.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_path} exists; pass --overwrite to replace it")
        shutil.rmtree(dataset_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [STATE_NAMES],
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (6,),
            "names": [FORCE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [STATE_NAMES],
        },
        "observation.images.cam_high": _image_feature(image_shape),
        "observation.images.cam_left_wrist": _image_feature(image_shape),
        # The current simulation export has one wrist camera.  Keep the TAVLA
        # three-camera contract by mirroring it to the right-wrist key.
        "observation.images.cam_right_wrist": _image_feature(image_shape),
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type="single_arm",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )


def _check_episode(path: Path, effort_key: str, decision_stride: int, decision_offset: int) -> tuple[int, tuple[int, int, int]]:
    required = (
        "action",
        "timestamp",
        "observations/qpos",
        "observations/images/cam_high",
        "observations/images/cam_left_wrist",
    )
    with h5py.File(path, "r") as h5:
        missing = [key for key in required if key not in h5]
        effort_path = f"observations/{effort_key}"
        if effort_path not in h5:
            missing.append(effort_path)
        if missing:
            raise ValueError(f"{path}: missing datasets: {missing}")

        timestamp = h5["timestamp"][:]
        state = h5["observations/qpos"]
        effort = h5[effort_path]
        action = h5["action"]
        front = h5["observations/images/cam_high"]
        wrist = h5["observations/images/cam_left_wrist"]
        n = len(timestamp)

        expected_shapes = {
            "observations/qpos": (n, 8),
            effort_path: (n, 6),
            "action": (n, 8),
            "observations/images/cam_high": (n, 480, 640, 3),
            "observations/images/cam_left_wrist": (n, 480, 640, 3),
        }
        for key, expected in expected_shapes.items():
            actual = h5[key].shape
            if actual != expected:
                raise ValueError(f"{path}: {key} has shape {actual}, expected {expected}")

        if n == 0:
            raise ValueError(f"{path}: empty episode")
        if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
            raise ValueError(f"{path}: timestamp is not finite and strictly increasing")
        for key, dataset in (("state", state), ("effort", effort), ("action", action)):
            values = dataset[:]
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {key} contains NaN/Inf")
        if not np.isfinite(h5["observations/images/cam_high"][0]).all():
            raise ValueError(f"{path}: front image contains invalid values")
        if not np.isfinite(h5["observations/images/cam_left_wrist"][0]).all():
            raise ValueError(f"{path}: wrist image contains invalid values")
        if not np.issubdtype(front.dtype, np.uint8) or not np.issubdtype(wrist.dtype, np.uint8):
            raise ValueError(f"{path}: images must be uint8 RGB arrays")

        if "observations/wrench_base" in h5 and "observations/wrench_final" in h5:
            base = h5["observations/wrench_base"][:]
            final = h5["observations/wrench_final"][:]
            if not np.allclose(final, -base, atol=1e-6, rtol=0):
                raise ValueError(f"{path}: wrench_final is not -wrench_base")

        indices = np.arange(decision_offset, n, decision_stride)
        if len(indices) == 0:
            raise ValueError(f"{path}: decision stride produced zero frames")
        return len(indices), (3, 480, 640)


def _add_episode(
    dataset: LeRobotDataset,
    path: Path,
    effort_key: str,
    task: str,
    decision_stride: int,
    decision_offset: int,
) -> int:
    with h5py.File(path, "r") as h5:
        n = h5["timestamp"].shape[0]
        indices = np.arange(decision_offset, n, decision_stride)
        states = h5["observations/qpos"]
        effort = h5[f"observations/{effort_key}"]
        actions = h5["action"]
        front = h5["observations/images/cam_high"]
        wrist = h5["observations/images/cam_left_wrist"]
        for index in indices:
            front_chw = np.transpose(front[index], (2, 0, 1)).astype(np.uint8, copy=False)
            wrist_chw = np.transpose(wrist[index], (2, 0, 1)).astype(np.uint8, copy=False)
            dataset.add_frame(
                {
                    "observation.state": states[index].astype(np.float32),
                    "observation.effort": effort[index].astype(np.float32),
                    "action": actions[index].astype(np.float32),
                    "observation.images.cam_high": front_chw,
                    "observation.images.cam_left_wrist": wrist_chw,
                    "observation.images.cam_right_wrist": wrist_chw,
                }
            )
    dataset.save_episode(task=task)
    return len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("sim_hand_noise_50"))
    parser.add_argument(
        "--repo-id",
        default="local/tavla_single_arm_ee_wrench_sim_hand_noise_50",
        help="Output LeRobot repo id under LEROBOT_HOME.",
    )
    parser.add_argument("--task", default="peg-in-hole")
    parser.add_argument("--effort-key", choices=("effort", "wrench_final"), default="wrench_final")
    parser.add_argument("--decision-stride", type=int, default=3)
    parser.add_argument("--decision-offset", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.decision_stride <= 0:
        raise ValueError("--decision-stride must be positive")
    if args.decision_offset < 0 or args.decision_offset >= args.decision_stride:
        raise ValueError("--decision-offset must be in [0, decision_stride)")

    source_dir = args.source_dir
    files = sorted((source_dir / "tavla_raw").glob("episode_*.hdf5"), key=_episode_id)
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No tavla_raw/episode_*.hdf5 under {source_dir}")

    decision_counts = []
    image_shape = None
    for path in files:
        count, shape = _check_episode(path, args.effort_key, args.decision_stride, args.decision_offset)
        decision_counts.append(count)
        if image_shape is None:
            image_shape = shape
        elif shape != image_shape:
            raise ValueError(f"Image shape mismatch in {path}: {shape} != {image_shape}")

    dataset = _create_dataset(args.repo_id, image_shape, args.overwrite)
    total_frames = 0
    for path, expected_count in zip(files, decision_counts, strict=True):
        count = _add_episode(
            dataset,
            path,
            args.effort_key,
            args.task,
            args.decision_stride,
            args.decision_offset,
        )
        if count != expected_count:
            raise RuntimeError(f"{path}: wrote {count} frames, expected {expected_count}")
        total_frames += count
        print(f"{path.stem}: {count} decision frames, task={args.task!r}")

    dataset.consolidate(run_compute_stats=False)
    dataset.meta.stats = compute_stats(dataset, batch_size=8, num_workers=0)
    write_json(serialize_dict(dataset.meta.stats), dataset.root / "meta/stats.json")
    print(f"Wrote LeRobot dataset: {LEROBOT_HOME / args.repo_id}")
    print(f"episodes={len(files)} frames={total_frames} fps=10 effort=observations/{args.effort_key}")
    print("action=HDF5 action (8-D absolute joint target); task=peg-in-hole")


if __name__ == "__main__":
    main()
