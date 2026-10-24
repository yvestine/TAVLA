#!/usr/bin/env python3
"""Targeted review aid for transferred phase labels on the aligned Sim data.

This is a read-only audit.  It does NOT assign or change any label, and it does
not touch checkpoints or run training.  It reduces "eyeball 40 trajectories"
down to "confirm or override a ranked shortlist" by doing two things per
trajectory in ``sim-data-aligned``:

1. Render a review sheet (Sim front/wrist frames) plus a signal plot in which
   the transferred ``insertion_quality_labels_v2`` phase bands are drawn on top
   of the *new* Sim force magnitude and insertion-depth progress.  A human can
   see immediately where a band disagrees with the new contact evidence.
2. Score each labeled window for disagreement between its transferred label and
   the new Sim force/depth signal, then write a ranked table.  The score is a
   heuristic prior for *where to look*; it is never a label decision.

The force features reuse the same 20-frame free-space calibration as encoder
training (``build_real_force_trend_features`` with ``calibration_frames``), so
what you review matches what the encoder will actually consume.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.force_features import DEFAULT_CALIBRATION_FRAMES
from openpi.force_features import build_real_force_trend_features
from openpi.force_features import causal_mean

from insertion_quality_v1_lib import QUALITY_LABELS
from insertion_quality_v1_lib import annotation_arrays
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config


PHASE_COLORS = {
    "free": "#bdbdbd",
    "misaligned_contact": "#d62728",
    "aligned_progress": "#2ca02c",
    "terminal": "#9467bd",
    "uncertain": "#ff7f0e",
}
LABEL_SPANS = ("free", "misaligned_contact", "aligned_progress", "terminal", "uncertain")


def label_spans(annotations: list[dict[str, object]], trajectory: int) -> list[tuple[int, int, str, str]]:
    spans: list[tuple[int, int, str, str]] = []
    for row in annotations:
        if int(row["trajectory_id"]) != trajectory:
            continue
        spans.append(
            (int(row["start_frame"]), int(row["end_frame"]), str(row["label"]), str(row["confidence"]))
        )
    return sorted(spans, key=lambda item: item[0])


def sim_force_signals(
    force_xyz: np.ndarray,
    axis: np.ndarray,
    rep: dict[str, object],
    calibration_frames: int,
) -> dict[str, np.ndarray]:
    """Return the same smoothed Sim force magnitude/axial/lateral the encoder sees."""
    features, meta = build_real_force_trend_features(
        force_xyz,
        free_end=calibration_frames - 1,
        insertion_axis=axis,
        smoothing_frames=int(rep["causal_smoothing_frames"]),
        trend_lag_frames=int(rep["trend_lag_frames"]),
        epsilon=float(rep["epsilon"]),
        active_noise_mad_multiplier=float(rep["active_noise_mad_multiplier"]),
        active_noise_floor=float(rep["active_noise_floor_newton"]),
    )
    names = [
        "force_dir_x", "force_dir_y", "force_dir_z", "lateral_force_ratio",
        "axial_force_fraction", "trend_dir_x", "trend_dir_y", "trend_dir_z",
        "force_norm", "trend_norm",
    ]
    index = {name: i for i, name in enumerate(names)}
    return {
        "force_norm": features[:, index["force_norm"]],
        "lateral_ratio": features[:, index["lateral_force_ratio"]],
        "axial_fraction": features[:, index["axial_force_fraction"]],
        "trend_norm": features[:, index["trend_norm"]],
        "gate_threshold": float(meta["activity_gate_threshold"]),
    }


def score_windows(
    spans: list[tuple[int, int, str, str]],
    signals: dict[str, np.ndarray],
    depth_mm: np.ndarray,
) -> list[dict[str, object]]:
    """Flag transferred quality windows whose new-Sim force/depth looks inconsistent.

    Heuristics (priors for where a human should look, never label decisions):
      - misaligned_contact windows whose median force is below the free-space
        gate look under-contacted (may be mislabeled free/uncertain).
      - aligned_progress windows that make no net downward depth progress look
        stalled (may be mislabeled misaligned).
      - either quality window with a large in-window force spike is flagged.
    """
    gate = signals["gate_threshold"]
    force = signals["force_norm"]
    rows: list[dict[str, object]] = []
    for start, end, label, confidence in spans:
        if label not in QUALITY_LABELS:
            continue
        end = min(end, len(force) - 1)
        if end < start:
            continue
        window_force = force[start : end + 1]
        window_depth = depth_mm[start : end + 1]
        median_force = float(np.median(window_force))
        peak_force = float(np.max(window_force))
        net_progress = float(window_depth[-1] - window_depth[0])
        flags: list[str] = []
        if label == "misaligned_contact" and median_force < gate:
            flags.append("misaligned_below_gate")
        if label == "aligned_progress" and net_progress <= 0.5:
            flags.append("aligned_no_progress")
        if peak_force > 5.0:
            flags.append(f"force_spike_{peak_force:.1f}N")
        rows.append(
            {
                "start_frame": start,
                "end_frame": end,
                "label": label,
                "confidence": confidence,
                "n_frames": end - start + 1,
                "median_force_N": round(median_force, 4),
                "peak_force_N": round(peak_force, 4),
                "gate_threshold_N": round(gate, 4),
                "net_depth_progress_mm": round(net_progress, 3),
                "flags": ";".join(flags) if flags else "",
                "priority": len(flags),
            }
        )
    return rows


def read_video_frames(path: Path, frames: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    result: dict[int, np.ndarray] = {}
    wanted = set(frames)
    index = 0
    while wanted:
        ok, image = capture.read()
        if not ok:
            break
        if index in wanted:
            result[index] = image
            wanted.remove(index)
        index += 1
    capture.release()
    return result


def annotate_tile(image: np.ndarray, text: str, width: int = 300, height: int = 225) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (width, 24), (0, 0, 0), -1)
    cv2.putText(resized, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return resized


def build_video_sheet(traj_dir: Path, frames: list[int], output: Path) -> None:
    front = read_video_frames(traj_dir / "front_camera.mp4", frames)
    wrist = read_video_frames(traj_dir / "wrist_camera.mp4", frames)
    cells: list[np.ndarray] = []
    for frame in frames:
        if frame not in front or frame not in wrist:
            continue
        cells.append(
            np.vstack(
                (
                    annotate_tile(front[frame], f"SimFront f{frame}"),
                    annotate_tile(wrist[frame], f"SimWrist f{frame}"),
                )
            )
        )
    if not cells:
        return
    columns = 6
    blank = np.zeros_like(cells[0])
    rows: list[np.ndarray] = []
    for chunk in range(0, len(cells), columns):
        row = cells[chunk : chunk + columns]
        row = row + [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output), np.vstack(rows))


def build_signal_plot(
    trajectory: int,
    spans: list[tuple[int, int, str, str]],
    signals: dict[str, np.ndarray],
    depth_mm: np.ndarray,
    calibration_frames: int,
    output: Path,
) -> None:
    frame_axis = np.arange(len(signals["force_norm"]))
    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    axes[0].plot(frame_axis, signals["force_norm"], color="#1f77b4", label="Sim |force| (encoder input)")
    axes[0].axhline(signals["gate_threshold"], color="black", linestyle=":", linewidth=0.9, label="free-space gate")
    axes[0].set_ylabel("N")
    axes[1].plot(frame_axis, signals["lateral_ratio"], color="#d62728", label="lateral ratio")
    axes[1].plot(frame_axis, signals["axial_fraction"], color="#2ca02c", label="axial fraction")
    axes[1].set_ylabel("ratio")
    axes[2].plot(frame_axis, depth_mm, color="black", label="downward Z progress (mm)")
    axes[2].set_ylabel("mm")
    axes[2].set_xlabel("frame (10 Hz)")
    for axis in axes:
        for start, end, label, _ in spans:
            axis.axvspan(start, min(end, len(frame_axis) - 1), color=PHASE_COLORS.get(label, "#cccccc"), alpha=0.18)
        axis.axvline(calibration_frames, color="black", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.15)
        axis.legend(ncol=3, loc="upper left", fontsize=8)
    bands = "  ".join(f"{name}={PHASE_COLORS[name]}" for name in LABEL_SPANS)
    axes[0].set_title(
        f"traj_{trajectory}: transferred phase bands over NEW Sim force; dashed=calib end ({calibration_frames}f)   [{bands}]"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def downward_depth_mm(pose_xyz: np.ndarray, calibration_frames: int) -> np.ndarray:
    reference = np.median(pose_xyz[:calibration_frames, 2])
    return (reference - pose_xyz[:, 2]) * 1000.0


def run(repo: Path, config_path: Path, output_dir: Path, no_video: bool) -> None:
    config = read_config(config_path)
    if "force_config" in config:
        force_config_path = Path(config["force_config"])
        if not force_config_path.is_absolute():
            force_config_path = repo / force_config_path
        force_config = read_config(force_config_path)
        config["free_end_frames"] = force_config["free_end_frames"]
        config["representation"] = {**force_config["representation"], **config.get("representation", {})}
    rep = config["representation"]
    calibration_frames = int(rep.get("calibration_frames", DEFAULT_CALIBRATION_FRAMES))
    axis = np.asarray(rep["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    annotations = read_annotations(repo / "annotations/insertion_quality_labels_v2.csv")
    sim_root = repo / str(config["data"]["sim_root"])
    sim_force_key = str(config["data"]["sim_force_key"])
    pose_key = str(config["data"]["pose_key"])
    trajectories = [int(value) for value in config["evaluation_trajectories"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for trajectory in trajectories:
        traj_dir = sim_root / f"traj_{trajectory}"
        with h5py.File(traj_dir / "data.h5", "r") as handle:
            force_xyz = np.asarray(handle[sim_force_key], dtype=np.float64)[:, :3]
            pose_xyz = np.asarray(handle[pose_key], dtype=np.float64)[:, :3]
        signals = sim_force_signals(force_xyz, axis, rep, calibration_frames)
        depth_mm = downward_depth_mm(pose_xyz, calibration_frames)
        spans = label_spans(annotations, trajectory)
        rows = score_windows(spans, signals, depth_mm)
        for row in rows:
            row["trajectory"] = f"traj_{trajectory}"
            all_rows.append(row)
        build_signal_plot(
            trajectory, spans, signals, depth_mm, calibration_frames,
            output_dir / f"traj_{trajectory}_signals.png",
        )
        if not no_video:
            sample_frames = sorted({0} | set(range(calibration_frames, len(force_xyz), 5)) | {len(force_xyz) - 1})
            build_video_sheet(traj_dir, sample_frames, output_dir / f"traj_{trajectory}_frames.jpg")
        flagged = sum(1 for row in rows if row["priority"] > 0)
        print(f"traj_{trajectory}: {len(rows)} quality windows, {flagged} flagged for review", flush=True)

    all_rows.sort(key=lambda row: (-int(row["priority"]), str(row["trajectory"]), int(row["start_frame"])))
    fieldnames = [
        "priority", "trajectory", "start_frame", "end_frame", "label", "confidence",
        "n_frames", "median_force_N", "peak_force_N", "gate_threshold_N",
        "net_depth_progress_mm", "flags",
    ]
    with (output_dir / "label_review_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    flagged_rows = [row for row in all_rows if row["priority"] > 0]
    summary = [
        "# Aligned Sim label review (read-only audit)",
        "",
        f"- Trajectories audited: `{len(trajectories)}` (encoder evaluation set).",
        f"- Calibration window: first `{calibration_frames}` frames (matches encoder training and deployment).",
        f"- Quality windows scored: `{len(all_rows)}`; flagged for human review: `{len(flagged_rows)}`.",
        "- Flags are priors for where to look; they never change a label. Confirm or override by watching the frame sheets.",
        "",
        "## Flagged windows (most suspicious first)",
        "",
        "| trajectory | frames | label | conf | median N | gate N | net depth mm | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in flagged_rows:
        summary.append(
            f"| {row['trajectory']} | {row['start_frame']}–{row['end_frame']} | {row['label']} | "
            f"{row['confidence']} | {row['median_force_N']} | {row['gate_threshold_N']} | "
            f"{row['net_depth_progress_mm']} | {row['flags']} |"
        )
    if not flagged_rows:
        summary.append("| — | — | — | — | — | — | — | (no windows flagged) |")
    summary.extend([
        "",
        "## How to use this",
        "",
        "1. Open `label_review_ranking.csv` and start at the top (highest priority).",
        "2. For each flagged window, open `traj_<id>_frames.jpg` and `traj_<id>_signals.png`.",
        "3. If the video/force disagree with the transferred band, edit "
        "`annotations/insertion_quality_labels_v2.csv` (start/end/label/confidence) for that span.",
        "4. traj_12 (~17 N failure), traj_20/traj_34 (few aligned windows) and phase boundaries deserve a look "
        "even if unflagged.",
        "",
        "This audit does not modify labels, checkpoints, or run training.",
    ])
    (output_dir / "label_review_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nWrote ranking + summary + per-trajectory panels to {output_dir}")
    print(f"Flagged {len(flagged_rows)} windows for human confirmation.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_sim_aligned_force.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/sim_aligned_label_review"))
    parser.add_argument("--no-video", action="store_true", help="Skip frame sheets; write signal plots + ranking only")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, config, output, args.no_video)


if __name__ == "__main__":
    main()
