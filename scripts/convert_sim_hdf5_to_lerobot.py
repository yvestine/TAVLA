"""Convert ``convert_sim_csv_to_hdf5.py`` output to a local LeRobot dataset.

Only the decision-rate view is exported.  This keeps one training sample per
PPO decision instead of repeating the same label at all 30 Hz control ticks.
The HDF5 files contain one wrist camera, so the wrist image is intentionally
written to both TAVLA wrist keys.
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

from openpi.shared.wrench_adapter import load_adapter


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
ACTION_NAMES = STATE_NAMES
FORCE_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def _episode_id(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def _decode_jpeg(value: np.ndarray) -> np.ndarray:
    encoded = np.asarray(value, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode JPEG image stored in HDF5")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return np.transpose(image_rgb, (2, 0, 1)).astype(np.uint8, copy=False)


def _create_dataset(repo_id: str, image_shape: tuple[int, int, int], effort_dim: int, overwrite: bool) -> LeRobotDataset:
    dataset_path = LEROBOT_HOME / repo_id
    if dataset_path.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_path} exists; pass --overwrite to replace it")
        shutil.rmtree(dataset_path)

    channels, height, width = image_shape
    image_feature = {
        "dtype": "image",
        "shape": (channels, height, width),
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
            "names": [FORCE_NAMES[:effort_dim] if effort_dim == 6 else tuple(f"effort_{i}" for i in range(effort_dim))],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [ACTION_NAMES],
        },
        "observation.images.cam_high": image_feature,
        "observation.images.cam_left_wrist": image_feature,
        "observation.images.cam_right_wrist": image_feature,
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


def _load_hdf5_episode(path: Path, effort_key: str, wrench_adapter=None) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        if "decision" not in h5:
            raise ValueError(f"Missing decision-rate view: {path}")
        task = str(h5.attrs.get("task", "peg-in-hole"))
        state = np.concatenate(
            [h5["decision/obs/state/joint_pos"][:], h5["decision/obs/state/gripper_pos"][:]], axis=-1
        ).astype(np.float32)
        effort = h5[f"decision/obs/state/{effort_key}"][:].astype(np.float32)
        action = h5["decision/action/ppo_joint_targets"][:].astype(np.float32)
        front = [_decode_jpeg(value) for value in h5["decision/images/front_jpeg"][:]]
        wrist = [_decode_jpeg(value) for value in h5["decision/images/wrist_jpeg"][:]]

    if state.shape[0] != action.shape[0] or state.shape[0] != effort.shape[0]:
        raise ValueError(f"Length mismatch in {path}: state={state.shape}, effort={effort.shape}, action={action.shape}")
    if len(front) != state.shape[0] or len(wrist) != state.shape[0]:
        raise ValueError(f"Image length mismatch in {path}")
    for name, value in (("state", state), ("effort", effort), ("action", action)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf in {path}")
    if wrench_adapter is not None:
        effort = wrench_adapter.transform_numpy(effort)
        if not np.isfinite(effort).all():
            raise ValueError(f"Adapted effort contains NaN/Inf in {path}")
    return task, state, effort, action, np.asarray(front), np.asarray(wrist)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data-sim-hdf5"))
    parser.add_argument("--repo-id", default="local/tavla_single_arm_ee_wrench_sim")
    parser.add_argument(
        "--effort-key",
        choices=(
            "effort",
            "wrench_final",
            "wrench_base",
            "force_local",
            "force_world",
            "wrench_model",
            "wrench_parent",
            "wrench_tool",
        ),
        default="wrench_final",
        help="Six-dimensional simulation force stream. Use wrench_final for the sign-corrected TAVLA input.",
    )
    parser.add_argument(
        "--wrench-adapter",
        type=Path,
        help="Optional .pt adapter produced by train_wrench_adapter.py or fit_unpaired_wrench_affine.py",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.source_dir.glob("episode_*/data.h5"), key=_episode_id)
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No episode_*/data.h5 files under {args.source_dir}")

    wrench_adapter = load_adapter(args.wrench_adapter) if args.wrench_adapter else None

    with h5py.File(files[0], "r") as h5:
        first_image = _decode_jpeg(h5["decision/images/front_jpeg"][0])
        effort_dim = int(h5[f"decision/obs/state/{args.effort_key}"].shape[-1])
    dataset = _create_dataset(args.repo_id, tuple(first_image.shape), effort_dim, args.overwrite)

    for path in files:
        task, state, effort, action, front, wrist = _load_hdf5_episode(path, args.effort_key, wrench_adapter)
        for index in range(state.shape[0]):
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
        dataset.save_episode(task=task)
        print(f"{path.parent.name}: {state.shape[0]} decision frames, task={task!r}")

    # The default LeRobot implementation computes stats with eight worker
    # processes.  That is fragile on restricted robot/GPU hosts (and can leave
    # a dataset without meta/stats.json), so compute them in-process here.
    dataset.consolidate(run_compute_stats=False)
    dataset.meta.stats = compute_stats(dataset, batch_size=8, num_workers=0)
    write_json(serialize_dict(dataset.meta.stats), dataset.root / "meta/stats.json")
    print(f"Wrote LeRobot dataset: {LEROBOT_HOME / args.repo_id}")
    adapter_note = f" + adapter {args.wrench_adapter}" if args.wrench_adapter else ""
    print(f"effort source: decision/obs/state/{args.effort_key}{adapter_note}")


if __name__ == "__main__":
    main()
