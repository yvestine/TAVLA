#!/usr/bin/env python3
"""Plot force, reviewed phases, motion, and held-out LOTO predictions together."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from insertion_quality_v1_lib import causal_mean
from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config
from insertion_quality_v1_lib import write_csv


PHASE_COLORS = {
    "free": "#bdbdbd",
    "uncertain": "#ffcc66",
    "misaligned_contact": "#d62728",
    "aligned_progress": "#2ca02c",
    "terminal": "#9467bd",
}


def read_predictions(path: Path) -> dict[int, list[dict[str, str]]]:
    result: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trajectory = int(row["trajectory"].split("_")[-1])
            result.setdefault(trajectory, []).append(row)
    return result


def shade_phases(axes: list[plt.Axes], rows: list[dict[str, object]]) -> None:
    for row in rows:
        label = str(row["label"])
        color = PHASE_COLORS[label]
        for axis in axes:
            axis.axvspan(int(row["start_frame"]), int(row["end_frame"]), color=color, alpha=0.09)


def run(repo: Path, config_path: Path, annotations_path: Path, predictions_path: Path, output_dir: Path) -> None:
    config = read_config(config_path)
    annotations = read_annotations(annotations_path)
    ids = [int(value) for value in config["manual_trajectories"]]
    free_ends = {int(key): int(value) for key, value in config["free_end_frames"].items()}
    smoothing = int(config["representation"]["causal_smoothing_frames"])
    data = load_trajectories(repo, config, ids)
    predictions = read_predictions(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# Insertion Quality V1 visualizations",
        "",
        "每张图把同一条 replay 的 Real/Sim 三维力拆成 lateral/axial、EE 未来运动、人工视频审阅阶段和 held-out LOTO 输出。Real 力先减本轨迹 free-space median；Sim 使用 `contact_total_base`。两域 Newton 量级不同，因此分开画轴，不用幅值相等解释对齐。",
        "",
    ]
    phase_confidence: dict[str, list[float]] = {label: [] for label in PHASE_COLORS}
    phase_aligned: dict[str, list[float]] = {label: [] for label in PHASE_COLORS}
    for trajectory_id in ids:
        trajectory = data[trajectory_id]
        free_end = free_ends[trajectory_id]
        baseline = np.median(trajectory.real_force[: free_end + 1], axis=0)
        real = causal_mean(trajectory.real_force - baseline, smoothing)
        sim = causal_mean(trajectory.sim_force, smoothing)
        frame = np.arange(len(real))
        real_lateral = np.linalg.norm(real[:, :2], axis=1)
        real_axial = np.abs(real[:, 2])
        sim_lateral = np.linalg.norm(sim[:, :2], axis=1)
        sim_axial = np.abs(sim[:, 2])
        z_progress_mm = (trajectory.pose[0, 2] - trajectory.pose[:, 2]) * 1000.0
        xy_step = np.linalg.norm(np.diff(trajectory.pose[:, :2], axis=0), axis=1)
        xy_path_mm = np.r_[0.0, np.cumsum(xy_step)] * 1000.0
        pred_rows = predictions[trajectory_id]
        pred_frame = np.asarray([int(row["frame"]) for row in pred_rows])
        p_misaligned = np.asarray([float(row["p_misaligned"]) for row in pred_rows])
        p_aligned = np.asarray([float(row["p_aligned_progress"]) for row in pred_rows])
        p_terminal = np.asarray([float(row["p_terminal"]) for row in pred_rows])
        trajectory_annotations = [row for row in annotations if int(row["trajectory_id"]) == trajectory_id]
        for row in pred_rows:
            prediction_frame = int(row["frame"])
            for annotation in trajectory_annotations:
                if int(annotation["start_frame"]) <= prediction_frame <= int(annotation["end_frame"]):
                    label = str(annotation["label"])
                    phase_confidence[label].append(float(row["confidence"]))
                    phase_aligned[label].append(float(row["p_aligned_progress"]))
                    break

        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        shade_phases(list(axes), trajectory_annotations)
        axes[0].plot(frame, real_lateral, label="Real lateral residual", color="#d62728", linewidth=1.2)
        axes[0].plot(frame, real_axial, label="Real axial residual |Fz|", color="#1f77b4", linewidth=1.2)
        axes[0].set_ylabel("Real force (N)")
        axes[0].legend(loc="upper left", ncol=2)
        axes[1].plot(frame, sim_lateral, label="Sim lateral", color="#d62728", linewidth=1.2)
        axes[1].plot(frame, sim_axial, label="Sim axial |Fz|", color="#1f77b4", linewidth=1.2)
        axes[1].set_ylabel("Sim force (N)")
        axes[1].legend(loc="upper left", ncol=2)
        axes[2].plot(frame, z_progress_mm, label="downward Z progress", color="#222222")
        axes[2].plot(frame, xy_path_mm, label="cumulative XY path", color="#ff7f0e")
        axes[2].set_ylabel("motion (mm)")
        axes[2].legend(loc="upper left", ncol=2)
        axes[3].plot(pred_frame, p_misaligned, label="p_misaligned", color="#d62728")
        axes[3].plot(pred_frame, p_aligned, label="p_aligned_progress", color="#2ca02c")
        axes[3].plot(pred_frame, p_terminal, label="p_terminal", color="#9467bd")
        axes[3].axhline(0.5, color="black", linestyle=":", linewidth=0.7)
        axes[3].set_ylim(-0.03, 1.03)
        axes[3].set_ylabel("LOTO probability")
        axes[3].set_xlabel("frame (nominal 10 Hz)")
        axes[3].legend(loc="upper left", ncol=3)
        axes[0].set_title(
            f"traj_{trajectory_id}: force + pose + held-out prediction | "
            "gray=free, yellow=uncertain, red=misaligned, green=aligned, purple=terminal"
        )
        fig.tight_layout()
        path = output_dir / f"traj_{trajectory_id}_insertion_quality.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        index_lines.append(f"- `traj_{trajectory_id}`: `{path.relative_to(repo)}`")
    phase_rows = []
    for label in PHASE_COLORS:
        confidence = np.asarray(phase_confidence[label], dtype=np.float64)
        aligned = np.asarray(phase_aligned[label], dtype=np.float64)
        phase_rows.append(
            {
                "phase": label,
                "n_frames": len(confidence),
                "confidence_mean": float(np.mean(confidence)),
                "confidence_median": float(np.median(confidence)),
                "p_aligned_mean": float(np.mean(aligned)),
                "aligned_prediction_ratio": float(np.mean(aligned >= 0.5)),
            }
        )
    write_csv(output_dir.parent / "phase_output_summary.csv", phase_rows)
    (repo / "insertion_quality_v1_visualization_index.md").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(ids)} trajectory diagnostics to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_v1.json"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/insertion_quality_labels_v1.csv"))
    parser.add_argument(
        "--predictions", type=Path, default=Path("eval_outputs/insertion_quality_v1/loto_predictions.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_outputs/insertion_quality_v1/trajectory_diagnostics")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    resolve = lambda path: path if path.is_absolute() else repo / path
    run(repo, resolve(args.config), resolve(args.annotations), resolve(args.predictions), resolve(args.output_dir))


if __name__ == "__main__":
    main()
