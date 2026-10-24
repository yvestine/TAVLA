#!/usr/bin/env python3
"""Build dense four-camera + force/pose review panels for all trajectories.

The generated panels are evidence for human labeling.  This script does not
assign insertion-quality labels automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from insertion_quality_v1_lib import causal_mean
from insertion_quality_v1_lib import read_config


CAMERA_SPECS = (
    ("RF", "data", "front_camera.mp4"),
    ("RW", "data", "wrist_camera.mp4"),
    ("SF", "sim_force", "front_camera.mp4"),
    ("SW", "sim_force", "wrist_camera.mp4"),
)


def read_video_frames(path: Path, frames: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    result: dict[int, np.ndarray] = {}
    wanted = set(frames)
    frame = 0
    while wanted:
        ok, image = capture.read()
        if not ok:
            break
        if frame in wanted:
            result[frame] = image
            wanted.remove(frame)
        frame += 1
    capture.release()
    if wanted:
        raise ValueError(f"{path}: missing requested frames {sorted(wanted)}")
    return result


def annotate_tile(image: np.ndarray, text: str, width: int = 280, height: int = 210) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (width, 25), (0, 0, 0), -1)
    cv2.putText(resized, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    return resized


def build_video_sheet(repo: Path, trajectory: int, frames: list[int], output: Path) -> None:
    camera_frames: dict[str, dict[int, np.ndarray]] = {}
    for short_name, root, filename in CAMERA_SPECS:
        path = repo / root / f"traj_{trajectory}" / filename
        camera_frames[short_name] = read_video_frames(path, frames)
    cells: list[np.ndarray] = []
    for frame in frames:
        top = np.hstack(
            (
                annotate_tile(camera_frames["RF"][frame], f"RF  f{frame}"),
                annotate_tile(camera_frames["RW"][frame], f"RW  f{frame}"),
            )
        )
        bottom = np.hstack(
            (
                annotate_tile(camera_frames["SF"][frame], f"SF  f{frame}"),
                annotate_tile(camera_frames["SW"][frame], f"SW  f{frame}"),
            )
        )
        cells.append(np.vstack((top, bottom)))
    columns = 4
    blank = np.zeros_like(cells[0])
    rows: list[np.ndarray] = []
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    cv2.imwrite(str(output), sheet)


def build_signal_plot(repo: Path, config: dict[str, object], trajectory: int, frames: list[int], output: Path) -> None:
    real_path = repo / "data" / f"traj_{trajectory}" / "data.h5"
    sim_path = repo / "sim_force" / f"traj_{trajectory}" / "data.h5"
    with h5py.File(real_path, "r") as handle:
        real_force = np.asarray(handle[config["data"]["real_force_key"]], dtype=np.float64)[:, :3]
        pose = np.asarray(handle[config["data"]["pose_key"]], dtype=np.float64)[:, :3]
    with h5py.File(sim_path, "r") as handle:
        sim_force = np.asarray(handle[config["data"]["sim_force_key"]], dtype=np.float64)[:, :3]
    free_end = int(config["free_end_frames"][str(trajectory)])
    baseline = np.median(real_force[: free_end + 1], axis=0)
    real = causal_mean(real_force - baseline, 3)
    sim = causal_mean(sim_force, 3)
    real_lateral = np.linalg.norm(real[:, :2], axis=1)
    real_axial = np.abs(real[:, 2])
    sim_lateral = np.linalg.norm(sim[:, :2], axis=1)
    sim_axial = np.abs(sim[:, 2])
    z_progress = (pose[0, 2] - pose[:, 2]) * 1000.0
    xy_step = np.r_[0.0, np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1) * 1000.0]
    z_step = np.r_[0.0, -np.diff(pose[:, 2]) * 1000.0]
    frame_axis = np.arange(len(real))

    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    axes[0].plot(frame_axis, real_lateral, label="Real lateral residual", color="#d62728")
    axes[0].plot(frame_axis, real_axial, label="Real axial residual", color="#1f77b4")
    axes[0].set_ylabel("Real N")
    axes[0].legend(ncol=2, loc="upper left")
    axes[1].plot(frame_axis, sim_lateral, label="Sim lateral", color="#d62728")
    axes[1].plot(frame_axis, sim_axial, label="Sim axial", color="#1f77b4")
    axes[1].set_ylabel("Sim N")
    axes[1].legend(ncol=2, loc="upper left")
    axes[2].plot(frame_axis, z_progress, label="downward Z progress", color="black")
    axes[2].plot(frame_axis, xy_step * 8.0, label="XY step ×8", color="#ff7f0e")
    axes[2].plot(frame_axis, z_step * 8.0, label="downward Z step ×8", color="#2ca02c", alpha=0.8)
    axes[2].set_ylabel("mm / scaled step")
    axes[2].set_xlabel("frame (10 Hz)")
    axes[2].legend(ncol=3, loc="upper left")
    for axis in axes:
        axis.axvspan(0, free_end, color="#bdbdbd", alpha=0.12)
        axis.axvline(free_end, color="black", linestyle="--", linewidth=0.8)
        for frame in frames:
            axis.axvline(frame, color="#999999", linewidth=0.25, alpha=0.35)
        axis.grid(alpha=0.15)
    axes[0].set_title(
        f"traj_{trajectory}: gray=confirmed free-space; video samples every 5 frames after cutoff"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(repo: Path, config_path: Path, output_dir: Path) -> None:
    config = read_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for trajectory in [int(value) for value in config["all_trajectories"]]:
        with h5py.File(repo / "data" / f"traj_{trajectory}" / "data.h5", "r") as handle:
            n_frames = len(handle[config["data"]["timestamp_key"]])
        free_end = int(config["free_end_frames"][str(trajectory)])
        start = max(0, free_end - 10)
        frames = list(range(start, n_frames, 5))
        if frames[-1] != n_frames - 1:
            frames.append(n_frames - 1)
        build_video_sheet(repo, trajectory, frames, output_dir / f"traj_{trajectory}_videos.jpg")
        build_signal_plot(repo, config, trajectory, frames, output_dir / f"traj_{trajectory}_signals.png")
        print(f"traj_{trajectory}: {n_frames} frames, reviewed samples={len(frames)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_v1.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_outputs/insertion_quality_label_review_v2")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, config, output)


if __name__ == "__main__":
    main()
