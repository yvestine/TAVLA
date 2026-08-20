"""Convert the local single-arm HDF5 + MP4 recordings to LeRobot format.

The raw data layout expected by this converter is:

    data/traj_0/data.h5
    data/traj_0/front_camera.mp4
    data/traj_0/wrist_camera.mp4

It writes a local LeRobot dataset with three image keys expected by TA-VLA:
`cam_high`, `cam_left_wrist`, and `cam_right_wrist`. Since the raw data only has
one wrist camera, the wrist stream is duplicated into both wrist keys.

The raw `action/actual/*` arrays in the current dataset are same-frame state copies.
This is kept by default because the training data loader builds action chunks from
consecutive frames, and the delta-action transform turns the first same-frame step
into a zero delta while preserving future steps in the chunk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import cv2
import h5py
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import serialize_dict, write_json
import numpy as np
import tqdm


JOINT_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
STATE_NAMES = (*JOINT_NAMES, "gripper")
WRENCH_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
EFFORT_KEYS = (
    "obs/state/joint_torque_external",
    "obs/state/joint_torque",
    "obs/state/ee_wrench_base",
    "obs/state/ee_wrench_stiffness",
    "obs/state/ee_wrench",
)


def _trajectory_id(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def _read_video(path: Path, expected_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(np.transpose(frame_rgb, (2, 0, 1)))
    finally:
        cap.release()

    if len(frames) != expected_frames:
        raise ValueError(f"{path} has {len(frames)} frames, expected {expected_frames}")
    return np.asarray(frames, dtype=np.uint8)


def _create_dataset(
    repo_id: str,
    fps: int,
    effort_dim: int,
    effort_names: tuple[str, ...],
    overwrite: bool,
    use_videos: bool,
    image_writer_processes: int,
    image_writer_threads: int,
) -> LeRobotDataset:
    dataset_path = LEROBOT_HOME / repo_id
    if dataset_path.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_path)

    image_feature = {
        "dtype": "video" if use_videos else "image",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    }
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [STATE_NAMES],
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (effort_dim,),
            "names": [effort_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [STATE_NAMES],
        },
        "observation.images.cam_high": image_feature,
        "observation.images.cam_left_wrist": image_feature,
        "observation.images.cam_right_wrist": image_feature,
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="single_arm",
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def _infer_effort_spec(path: Path, effort_key: str) -> tuple[int, tuple[str, ...]]:
    with h5py.File(path, "r") as ep:
        effort_dim = int(ep[effort_key].shape[-1])
    if effort_dim == len(JOINT_NAMES):
        return effort_dim, JOINT_NAMES
    if effort_dim == len(WRENCH_NAMES):
        return effort_dim, WRENCH_NAMES
    return effort_dim, tuple(f"effort_{i}" for i in range(effort_dim))


def _load_episode(
    path: Path,
    effort_key: str,
    action_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as ep:
        raw_state = np.concatenate([ep["obs/state/joint_pos"][:], ep["obs/state/gripper_pos"][:]], axis=-1).astype(np.float32)
        raw_action = np.concatenate([ep["action/actual/arm"][:], ep["action/actual/gripper"][:]], axis=-1).astype(np.float32)
        effort = ep[effort_key][:].astype(np.float32)
        n = raw_state.shape[0]

    front = _read_video(path.parent / "front_camera.mp4", n)
    wrist = _read_video(path.parent / "wrist_camera.mp4", n)

    if action_mode == "next_state":
        state = raw_state[:-1]
        action = raw_state[1:]
        effort = effort[:-1]
        front = front[:-1]
        wrist = wrist[:-1]
    elif action_mode == "actual":
        state = raw_state
        action = raw_action
    else:
        raise ValueError(f"Unsupported action_mode: {action_mode}")

    return state, action, effort, front, wrist


def convert(
    raw_dir: Path,
    repo_id: str,
    task: str,
    fps: int,
    effort_key: str,
    overwrite: bool,
    use_videos: bool,
    max_episodes: int | None,
    image_writer_processes: int,
    image_writer_threads: int,
    action_mode: str,
) -> None:
    files = sorted(raw_dir.glob("traj_*/data.h5"), key=_trajectory_id)
    if max_episodes is not None:
        files = files[:max_episodes]
    if not files:
        raise FileNotFoundError(f"No traj_*/data.h5 files found under {raw_dir}")

    effort_dim, effort_names = _infer_effort_spec(files[0], effort_key)
    dataset = _create_dataset(
        repo_id,
        fps=fps,
        effort_dim=effort_dim,
        effort_names=effort_names,
        overwrite=overwrite,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    for path in tqdm.tqdm(files, desc="Converting episodes"):
        state, action, effort, front, wrist = _load_episode(path, effort_key, action_mode)
        for i in range(state.shape[0]):
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
        dataset.save_episode(task=task)
    # Compute LeRobot stats in-process so conversion also works on hosts where
    # the default multi-worker statistics pass cannot create IPC sockets.
    dataset.consolidate(run_compute_stats=False)
    dataset.meta.stats = compute_stats(dataset, batch_size=8, num_workers=0)
    write_json(serialize_dict(dataset.meta.stats), dataset.root / "meta/stats.json")
    print(f"Wrote LeRobot dataset: {LEROBOT_HOME / repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data"))
    parser.add_argument("--repo-id", default="local/tavla_single_arm")
    parser.add_argument("--task", default="peg-in-hole")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--effort-key",
        default="obs/state/joint_torque_external",
        choices=EFFORT_KEYS,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true", help="Store images instead of encoded videos.")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        default="actual",
        choices=("next_state", "actual"),
        help="`actual` keeps raw action/actual arrays; `next_state` uses next-frame state labels.",
    )
    args = parser.parse_args()

    convert(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        effort_key=args.effort_key,
        overwrite=args.overwrite,
        use_videos=not args.no_videos,
        max_episodes=args.max_episodes,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        action_mode=args.action_mode,
    )


if __name__ == "__main__":
    main()
