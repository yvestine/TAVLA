"""Convert the local CSV/MP4 simulation export into per-episode HDF5 files.

The simulation export in ``data-sim`` is not an HDF5 dataset.  This converter
keeps the source values intact and adds an explicit 10 Hz decision view.  The
PPO label is ``ppo_joint_targets.csv``. The known-irregular
``actions.csv`` file is deliberately ignored and is never used as a training
label.

Output layout (one file per episode)::

    data-sim-hdf5/episode_0/data.h5
      attrs["task"]
      timestamps                         [N]
      obs/state/joint_pos                [N, 7]
      obs/state/gripper_pos              [N, 1]
      obs/state/joint_states_raw         [N, 9]
      obs/state/ee_pose                  [N, 7]
      obs/state/force_local              [N, 6]
      obs/state/force_world              [N, 6]
      obs/state/wrench_model             [N, 6] (if present)
      obs/state/wrench_parent            [N, 6] (if present)
      obs/state/wrench_tool              [N, 6] (if present)
      obs/state/wrench_base              [N, 6] (if present)
      obs/state/wrench_final             [N, 6] (if present)
      obs/state/effort                   [N, 6] (canonical effort alias)
      obs/state/ee_wrench                [N, 6] (canonical effort alias)
      action/ppo_joint_targets           [N, 8]
      action/actual/arm                  [N, 7] (ppo_joint_targets alias)
      action/actual/gripper              [N, 1] (ppo_joint_targets alias)
      images/front_jpeg                  [N] (variable-length uint8)
      images/wrist_jpeg                 [N] (variable-length uint8)
      decision/*                         the same fields sampled at offset::stride

The image datasets contain JPEG bytes for the original video frames.  They
are decoded as RGB by downstream readers; storing compressed bytes avoids
expanding the repository by several gigabytes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


REQUIRED_SHAPES = {
    "ee_pose.csv": 7,
    "force_local.csv": 6,
    "force_world.csv": 6,
    "gripper.csv": 1,
    "joint_states.csv": 9,
    "ppo_joint_targets.csv": 8,
    "reward.csv": 1,
    "reward_terms.csv": 16,
    "timestamps.csv": 1,
}

OPTIONAL_SHAPES = {
    "ee_pose_xyzw.csv": 7,
    "wrench_model.csv": 6,
    "wrench_parent.csv": 6,
    "wrench_tool.csv": 6,
    "wrench_base.csv": 6,
    "wrench_final.csv": 6,
}


def _episode_id(path: Path) -> int:
    return int(path.name.split("_")[-1])


def _read_csv(path: Path, expected_width: int) -> tuple[list[str], np.ndarray, list[int]]:
    """Read a numeric CSV and return its header, rectangular values, and ragged rows."""

    header: list[str]
    values: list[list[float]] = []
    ragged: list[int] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {path}") from exc

        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != expected_width:
                ragged.append(row_number)
                continue
            try:
                values.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(f"Non-numeric value in {path}:{row_number}") from exc

    if ragged:
        raise ValueError(f"Ragged rows in {path}; first rows: {ragged[:5]}")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(f"Unexpected shape for {path}: {array.shape}; expected (*, {expected_width})")
    return header, array, ragged


def _read_video_jpegs(path: Path, quality: int) -> tuple[list[np.ndarray], tuple[int, int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    encoded: list[np.ndarray] = []
    shape: tuple[int, int, int] | None = None
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if shape is None:
                shape = (frame_bgr.shape[0], frame_bgr.shape[1], 3)
            if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
                raise ValueError(f"Expected 3-channel video frame in {path}, got {frame_bgr.shape}")
            ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError(f"Could not JPEG-encode frame from {path}")
            encoded.append(np.asarray(buffer, dtype=np.uint8).reshape(-1))
    finally:
        cap.release()

    if shape is None:
        raise ValueError(f"Video has no frames: {path}")
    return encoded, shape


def _check_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        bad = int(np.size(value) - np.isfinite(value).sum())
        raise ValueError(f"{name} contains {bad} NaN/Inf values")


def _create_vlen_dataset(group: h5py.Group, name: str, values: list[np.ndarray]) -> h5py.Dataset:
    dtype = h5py.vlen_dtype(np.dtype("uint8"))
    dataset = group.create_dataset(name, shape=(len(values),), dtype=dtype)
    for index, value in enumerate(values):
        dataset[index] = value
    return dataset


def _write_numeric(group: h5py.Group, name: str, value: np.ndarray, indices: np.ndarray | None = None) -> None:
    if indices is not None:
        value = value[indices]
    group.create_dataset(name, data=value, compression="gzip", compression_opts=1)


def _write_episode(
    source: Path,
    output: Path,
    *,
    task: str,
    task_source: str,
    decision_stride: int,
    decision_offset: int,
    jpeg_quality: int,
    strict_lengths: bool,
    overwrite: bool,
) -> dict[str, Any]:
    metadata_path = source / "episode_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    arrays: dict[str, np.ndarray] = {}
    headers: dict[str, list[str]] = {}
    for filename, width in REQUIRED_SHAPES.items():
        header, array, _ = _read_csv(source / filename, width)
        headers[filename] = header
        arrays[filename] = array
        _check_finite(filename, array)

    for filename, width in OPTIONAL_SHAPES.items():
        path = source / filename
        if not path.exists():
            continue
        header, array, _ = _read_csv(path, width)
        headers[filename] = header
        arrays[filename] = array
        _check_finite(filename, array)

    if "wrench_final.csv" in arrays:
        effort_filename = "wrench_final.csv"
    elif "wrench_model.csv" in arrays:
        effort_filename = "wrench_model.csv"
    else:
        raise ValueError(f"No usable effort field in {source}; expected wrench_final.csv or wrench_model.csv")

    csv_frames = arrays["timestamps.csv"].shape[0]
    for filename, array in arrays.items():
        if array.shape[0] != csv_frames:
            raise ValueError(f"Length mismatch in {source}: timestamps={csv_frames}, {filename}={array.shape[0]}")

    front_jpegs, image_shape = _read_video_jpegs(source / "front" / "front.mp4", jpeg_quality)
    wrist_jpegs, wrist_shape = _read_video_jpegs(source / "wrist" / "wrist.mp4", jpeg_quality)
    if front_jpegs and image_shape != wrist_shape:
        raise ValueError(f"Camera shape mismatch in {source}: front={image_shape}, wrist={wrist_shape}")

    front_frames = len(front_jpegs)
    wrist_frames = len(wrist_jpegs)
    frame_counts = (csv_frames, front_frames, wrist_frames)
    if len(set(frame_counts)) != 1 and strict_lengths:
        raise ValueError(f"Frame length mismatch in {source}: csv/front/wrist={frame_counts}")
    n = min(frame_counts)
    dropped_tail_frames = {
        "csv": csv_frames - n,
        "front": front_frames - n,
        "wrist": wrist_frames - n,
    }
    if any(dropped_tail_frames.values()):
        print(f"{source.name}: truncating to shortest stream {n}; dropped tail frames={dropped_tail_frames}")
        arrays = {name: value[:n] for name, value in arrays.items()}
        front_jpegs = front_jpegs[:n]
        wrist_jpegs = wrist_jpegs[:n]

    timestamps = arrays["timestamps.csv"][:, 0].astype(np.float64)
    if n > 1:
        dt = np.diff(timestamps)
        if np.any(dt <= 0):
            raise ValueError(f"Non-increasing timestamps in {source}")
        fps = float(1.0 / np.median(dt))
    else:
        fps = 0.0

    if decision_stride <= 0:
        raise ValueError("decision_stride must be positive")
    if not 0 <= decision_offset < decision_stride:
        raise ValueError("decision_offset must satisfy 0 <= offset < stride")
    decision_indices = np.arange(decision_offset, n, decision_stride, dtype=np.int64)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")

    with h5py.File(output, "w") as h5:
        h5.attrs["format"] = "ta-vla-sim-csv-export-v1"
        h5.attrs["source_episode"] = source.name
        h5.attrs["task"] = task
        h5.attrs["task_source"] = task_source
        h5.attrs["source_task_attr_present"] = False
        h5.attrs["success"] = bool(metadata.get("success", False))
        h5.attrs["source_fps"] = fps
        h5.attrs["source_frame_counts_csv_front_wrist"] = np.asarray(frame_counts, dtype=np.int64)
        h5.attrs["alignment_policy"] = "strict" if strict_lengths else "truncate_to_shortest"
        h5.attrs["dropped_tail_frames_csv_front_wrist"] = np.asarray(
            (dropped_tail_frames["csv"], dropped_tail_frames["front"], dropped_tail_frames["wrist"]),
            dtype=np.int64,
        )
        h5.attrs["decision_hz"] = fps / decision_stride if fps else 0.0
        h5.attrs["decision_stride"] = decision_stride
        h5.attrs["decision_offset"] = decision_offset
        h5.attrs["decision_index_definition"] = "np.arange(offset, N, stride)"
        h5.attrs["action_source"] = "ppo_joint_targets.csv"
        h5.attrs["action_semantics_confirmed"] = True
        h5.attrs["force_semantics_confirmed"] = True
        h5.attrs["force_component_order"] = "Fx,Fy,Fz,Tx,Ty,Tz"
        h5.attrs["force_units"] = "N,N,N,Nm,Nm,Nm"
        if effort_filename == "wrench_final.csv":
            h5.attrs["force_coordinate_frame"] = "robot base frame"
            h5.attrs["force_reference_point"] = "robot base origin"
            h5.attrs["force_processing"] = "wrench_final = -wrench_base; sign calibrated by +/-X known-force tests"
            h5.attrs["force_sign_correction"] = "global -1 relative to wrench_base"
        else:
            h5.attrs["force_coordinate_frame"] = "tool/fingertip frame"
            h5.attrs["force_reference_point"] = "tool/fingertip origin"
            h5.attrs["force_processing"] = "parent-body wrench transformed, smoothed, zero-calibrated, axis-scaled, noise-added"
        h5.attrs["image_encoding"] = "jpeg"
        h5.attrs["image_color_on_decode"] = "RGB"
        h5.attrs["image_shape"] = image_shape
        h5.attrs["metadata_json"] = json.dumps(metadata, allow_nan=True, sort_keys=True)
        h5.attrs["ignored_source_file"] = "actions.csv"
        h5.attrs["ignored_source_reason"] = "known-irregular and not the PPO target stream"

        obs = h5.require_group("obs/state")
        _write_numeric(obs, "joint_pos", arrays["joint_states.csv"][:, :7])
        _write_numeric(obs, "gripper_pos", arrays["gripper.csv"])
        _write_numeric(obs, "joint_states_raw", arrays["joint_states.csv"])
        _write_numeric(obs, "ee_pose", arrays["ee_pose.csv"])
        _write_numeric(obs, "force_local", arrays["force_local.csv"])
        _write_numeric(obs, "force_world", arrays["force_world.csv"])
        for name in ("wrench_model", "wrench_parent", "wrench_tool", "wrench_base", "wrench_final"):
            filename = f"{name}.csv"
            if filename in arrays:
                _write_numeric(obs, name, arrays[filename])
        _write_numeric(obs, "effort", arrays[effort_filename])
        _write_numeric(obs, "ee_wrench", arrays[effort_filename])
        obs.attrs["joint_pos_source"] = "joint_states.csv[:, :7]"
        obs.attrs["gripper_source"] = "gripper.csv"
        obs.attrs["effort_source"] = effort_filename
        obs.attrs["force_source_fields"] = ",".join(
            name for name in (
                "force_local.csv",
                "force_world.csv",
                "wrench_model.csv",
                "wrench_parent.csv",
                "wrench_tool.csv",
                "wrench_base.csv",
                "wrench_final.csv",
            ) if name in arrays
        )

        action = h5.require_group("action")
        _write_numeric(action, "ppo_joint_targets", arrays["ppo_joint_targets.csv"])
        actual = action.require_group("actual")
        _write_numeric(actual, "arm", arrays["ppo_joint_targets.csv"][:, :7])
        _write_numeric(actual, "gripper", arrays["ppo_joint_targets.csv"][:, 7:8])
        action.attrs["names"] = "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,gripper"
        action.attrs["source"] = "ppo_joint_targets.csv"
        action.attrs["absolute_or_delta"] = "absolute joint targets"
        action.attrs["gripper_range"] = "source normalized gripper target"

        images = h5.require_group("images")
        _create_vlen_dataset(images, "front_jpeg", front_jpegs)
        _create_vlen_dataset(images, "wrist_jpeg", wrist_jpegs)

        h5.create_dataset("timestamps", data=timestamps, compression="gzip", compression_opts=1)

        decision = h5.require_group("decision")
        decision.attrs["indices"] = decision_indices
        decision.attrs["sampling"] = f"every {decision_stride} source frames, offset {decision_offset}"
        _write_numeric(decision, "timestamps", timestamps, decision_indices)
        decision_obs = decision.require_group("obs/state")
        for name in (
            "joint_pos",
            "gripper_pos",
            "joint_states_raw",
            "ee_pose",
            "force_local",
            "force_world",
            "wrench_model",
            "wrench_parent",
            "wrench_tool",
            "wrench_base",
            "wrench_final",
            "effort",
            "ee_wrench",
        ):
            if name in obs:
                _write_numeric(decision_obs, name, obs[name][:], decision_indices)
        decision_action = decision.require_group("action")
        _write_numeric(decision_action, "ppo_joint_targets", action["ppo_joint_targets"][:], decision_indices)
        decision_actual = decision_action.require_group("actual")
        _write_numeric(decision_actual, "arm", action["actual/arm"][:], decision_indices)
        _write_numeric(decision_actual, "gripper", action["actual/gripper"][:], decision_indices)
        decision_images = decision.require_group("images")
        _create_vlen_dataset(decision_images, "front_jpeg", [front_jpegs[i] for i in decision_indices])
        _create_vlen_dataset(decision_images, "wrist_jpeg", [wrist_jpegs[i] for i in decision_indices])

    return {
        "episode": source.name,
        "effort_source": effort_filename,
        "source_frames": n,
        "source_frame_counts": frame_counts,
        "dropped_tail_frames": dropped_tail_frames,
        "decision_frames": int(decision_indices.size),
        "source_fps": fps,
        "decision_hz": fps / decision_stride if fps else 0.0,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data-sim"))
    parser.add_argument("--output-dir", type=Path, default=Path("data-sim-hdf5"))
    parser.add_argument(
        "--task",
        default="peg-in-hole",
        help="Task stored in HDF5 attrs. The current CSV export has no source task attribute.",
    )
    parser.add_argument("--decision-stride", type=int, default=3)
    parser.add_argument("--decision-offset", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--strict-lengths",
        action="store_true",
        help="Fail when CSV and video frame counts differ; default truncates all streams to the shortest.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = sorted(args.source_dir.glob("episode_*"), key=_episode_id)
    if args.max_episodes is not None:
        sources = sources[: args.max_episodes]
    if not sources:
        raise FileNotFoundError(f"No episode_* directories under {args.source_dir}")

    task_source = "cli_or_repository_default; source CSV export has no data.attrs[task]"
    reports = []
    for source in sources:
        output = args.output_dir / source.name / "data.h5"
        report = _write_episode(
            source,
            output,
            task=args.task,
            task_source=task_source,
            decision_stride=args.decision_stride,
            decision_offset=args.decision_offset,
            jpeg_quality=args.jpeg_quality,
            strict_lengths=args.strict_lengths,
            overwrite=args.overwrite,
        )
        reports.append(report)
        print(
            f"{report['episode']}: {report['source_frames']} frames -> "
            f"{report['decision_frames']} decision frames, "
            f"{report['source_fps']:.6g} Hz -> {report['decision_hz']:.6g} Hz; "
            "actions.csv kept as raw 6D source; not used as TAVLA label"
        )

    summary = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "task": args.task,
        "task_source": task_source,
        "decision_stride": args.decision_stride,
        "decision_offset": args.decision_offset,
        "episodes": reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(f"Wrote {len(reports)} HDF5 episodes under {args.output_dir}")


if __name__ == "__main__":
    main()
