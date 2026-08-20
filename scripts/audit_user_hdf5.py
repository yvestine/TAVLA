"""Audit the local TA-VLA raw HDF5 + video recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


FIELDS = (
    "obs/state/joint_pos",
    "obs/state/joint_vel",
    "obs/state/joint_torque",
    "obs/state/joint_torque_external",
    "obs/state/ee_force",
    "obs/state/ee_torque",
    "obs/state/ee_wrench",
    "action/actual/arm",
    "action/actual/gripper",
    "action/policy/arm",
    "action/policy/gripper",
)


def _trajectory_id(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def _video_info(path: Path) -> tuple[bool, int, float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    try:
        return (
            cap.isOpened(),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
    finally:
        cap.release()


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values)
    return f"min={arr.min():.6g}, median={np.median(arr):.6g}, max={arr.max():.6g}, mean={arr.mean():.6g}"


def main(raw_dir: Path = Path("data")) -> None:
    files = sorted(raw_dir.glob("traj_*/data.h5"), key=_trajectory_id)
    if not files:
        raise FileNotFoundError(f"No traj_*/data.h5 files found under {raw_dir}")

    lengths: list[int] = []
    dt_values: list[float] = []
    mismatches: list[str] = []
    field_stats = {
        key: {"min": [], "max": [], "std": [], "zero_frac": [], "nan": 0, "inf": 0, "shapes": set()} for key in FIELDS
    }

    for path in files:
        with h5py.File(path, "r") as ep:
            length_map = {key: ep[key].shape[0] for key in (*FIELDS, "timestamps") if key in ep and ep[key].shape}
            if len(set(length_map.values())) != 1:
                mismatches.append(f"{path}: {length_map}")
            n = length_map.get("timestamps") or next(iter(length_map.values()))
            lengths.append(n)

            if "timestamps" in ep:
                timestamps = ep["timestamps"][:]
                if len(timestamps) > 1:
                    dt_values.extend(np.diff(timestamps).tolist())

            for key in FIELDS:
                if key not in ep:
                    continue
                arr = ep[key][:]
                info = field_stats[key]
                info["shapes"].add(tuple(arr.shape[1:]))
                info["nan"] += int(np.isnan(arr).sum())
                info["inf"] += int(np.isinf(arr).sum())
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    info["min"].append(float(np.min(finite)))
                    info["max"].append(float(np.max(finite)))
                    info["std"].append(float(np.std(finite)))
                    info["zero_frac"].append(float(np.mean(np.isclose(finite, 0))))

            for video_name in ("front_camera.mp4", "wrist_camera.mp4"):
                opened, frames, fps, size = _video_info(path.parent / video_name)
                if not opened or frames != n:
                    mismatches.append(
                        f"{path.parent / video_name}: opened={opened}, frames={frames}, h5_frames={n}, fps={fps}, size={size}"
                    )

    print(f"trajectories: {len(files)}")
    print(f"frames: total={sum(lengths)}, {_stats(lengths)}")
    if dt_values:
        dt = np.asarray(dt_values)
        print(f"dt_seconds: {_stats(dt_values)}")
        print(f"fps: median={1 / np.median(dt):.6g}, mean={1 / np.mean(dt):.6g}")
        print(f"dt<=0: {int((dt <= 0).sum())}, dt>0.2s: {int((dt > 0.2).sum())}")

    print("\nfields:")
    for key, info in field_stats.items():
        if not info["min"]:
            print(f"- {key}: missing")
            continue
        print(
            f"- {key}: shape_tail={sorted(info['shapes'])}, "
            f"min={min(info['min']):.6g}, max={max(info['max']):.6g}, "
            f"std_mean={np.mean(info['std']):.6g}, zero_frac_mean={np.mean(info['zero_frac']):.6g}, "
            f"nan={info['nan']}, inf={info['inf']}"
        )

    print("\nknown issue checks:")
    max_action_state = 0.0
    max_policy_actual = 0.0
    for path in files:
        with h5py.File(path, "r") as ep:
            max_action_state = max(
                max_action_state,
                float(np.max(np.abs(ep["action/actual/arm"][:] - ep["obs/state/joint_pos"][:]))),
            )
            max_policy_actual = max(
                max_policy_actual,
                float(np.max(np.abs(ep["action/policy/arm"][:] - ep["action/actual/arm"][:]))),
            )
    print(f"- max |action/actual/arm - obs/state/joint_pos|: {max_action_state:.6g}")
    print(f"- max |action/policy/arm - action/actual/arm|: {max_policy_actual:.6g}")

    if mismatches:
        print("\nwarnings:")
        for item in mismatches[:20]:
            print(f"- {item}")
        if len(mismatches) > 20:
            print(f"- ... {len(mismatches) - 20} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    main(args.raw_dir)
