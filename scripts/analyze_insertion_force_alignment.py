#!/usr/bin/env python3
"""Compare paired Real/Sim force only during labeled active insertion.

This is a read-only diagnostic. It never changes the HDF5 data, encoder input,
or policy/checkpoint files. The axial comparison is reported alongside the
original 3-D force norm so a change of signal cannot masquerade as improved
full-wrench agreement.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from insertion_quality_v1_lib import load_trajectory, read_annotations, read_config
from openpi.force_features import causal_mean


ACTIVE_LABELS = {"misaligned_contact", "aligned_progress"}


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 5 or np.std(first) < 1e-8 or np.std(second) < 1e-8:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(repo: Path, output: Path, *, calibration_frames: int = 20) -> None:
    config = read_config(repo / "configs/insertion_quality_sim_aligned_force_pose.json")
    force_config = read_config(repo / config["force_config"])
    axis = np.asarray(force_config["representation"]["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    annotations = read_annotations(repo / "annotations/insertion_quality_labels_v2.csv")
    segments: dict[int, list[dict[str, object]]] = {}
    for row in annotations:
        segments.setdefault(int(row["trajectory_id"]), []).append(row)

    rows: list[dict[str, object]] = []
    signals: dict[int, dict[str, np.ndarray]] = {}
    for trajectory in range(40):
        data = load_trajectory(repo, config, trajectory)
        n_frames = len(data.timestamps)
        if n_frames <= calibration_frames:
            raise ValueError(f"traj_{trajectory}: shorter than calibration window")
        mask = np.zeros(n_frames, dtype=bool)
        for segment in segments[trajectory]:
            if segment["label"] in ACTIVE_LABELS:
                mask[int(segment["start_frame"]) : int(segment["end_frame"]) + 1] = True
        row: dict[str, object] = {"trajectory": f"traj_{trajectory}", "active_frames": int(mask.sum())}
        entry = {"time": data.timestamps, "active": mask}
        for domain, force in (("real", data.real_force), ("sim", data.sim_force)):
            baseline = np.median(force[:calibration_frames], axis=0)
            residual = force - baseline
            for width in (3, 7):
                smooth = causal_mean(residual, width)
                entry[f"{domain}_norm_{width}"] = np.linalg.norm(smooth, axis=1)
                entry[f"{domain}_axial_{width}"] = smooth @ axis
            if mask.any():
                row[f"{domain}_norm_3_p95_N"] = float(np.quantile(entry[f"{domain}_norm_3"][mask], 0.95))
        signals[trajectory] = entry
        for name in ("norm_3", "norm_7", "axial_3", "axial_7"):
            row[f"r_{name}"] = (
                correlation(entry[f"real_{name}"][mask], entry[f"sim_{name}"][mask])
                if mask.any() else float("nan")
            )
        row["sim_real_norm_3_p95_ratio"] = (
            float(row["sim_norm_3_p95_N"]) / float(row["real_norm_3_p95_N"])
            if mask.any() else float("nan")
        )
        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "methods_by_trajectory.csv", rows)
    eligible = [row for row in rows if int(row["active_frames"]) >= 20]
    if not eligible:
        raise ValueError("No trajectories have at least 20 active insertion frames")
    metrics = {name: np.asarray([float(row[f"r_{name}"]) for row in eligible]) for name in
               ("norm_3", "norm_7", "axial_3", "axial_7")}

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    box = axes[0].boxplot(list(metrics.values()), patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ("#60a5fa", "#93c5fd", "#4ade80", "#86efac")):
        patch.set_facecolor(color)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_xticks(range(1, 5), ("3D norm\n3 frames", "3D norm\n7 frames",
                                       "signed axial\n3 frames", "signed axial\n7 frames"))
    axes[0].set(ylabel="Same-frame Pearson r", ylim=(-1, 1),
                title=f"Active insertion only; {len(eligible)} trajectories with >=20 frames")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].scatter(metrics["norm_3"], metrics["axial_7"], color="#2563eb", s=45)
    axes[1].plot([-1, 1], [-1, 1], "--", color="gray", linewidth=1)
    axes[1].set(xlim=(-1, 1), ylim=(-1, 1), xlabel="Original 3D norm r (3 frames)",
                ylabel="Signed insertion-axis r (7 frames)", title="Signal choice changes the question")
    for row in eligible:
        if int(row["trajectory"].split("_")[-1]) in (0, 7, 12, 25, 34):
            axes[1].annotate(row["trajectory"].split("_")[-1],
                             (float(row["r_norm_3"]), float(row["r_axial_7"])),
                             xytext=(4, 4), textcoords="offset points", fontsize=8)
    fig.savefig(output / "method_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    representative = (0, 7, 12, 25, 34)
    fig, axes = plt.subplots(len(representative), 1, figsize=(12, 12), constrained_layout=True)
    for ax, trajectory in zip(axes, representative):
        entry = signals[trajectory]
        mask = entry["active"]
        frames = np.flatnonzero(mask)
        time = entry["time"][frames] - entry["time"][frames[0]]
        ax.plot(time, entry["real_axial_7"][frames], color="#2563eb", label="Real")
        ax.plot(time, entry["sim_axial_7"][frames], color="#ea580c", label="Sim")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set(title=f"traj_{trajectory}: axial r={rows[trajectory]['r_axial_7']:.2f}, "
                     f"3D norm r={rows[trajectory]['r_norm_3']:.2f}",
               xlabel="Time since first active insertion sample (s)", ylabel="Signed axial force (N)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(output / "axial_traces.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    report = ["# 插入阶段真机/仿真力的可比口径", "",
              f"40 条轨迹全部检查；{sum(int(row['active_frames']) > 0 for row in rows)} 条有活动插入标签；"
              f"下表使用其中 {len(eligible)} 条至少有 20 帧活动插入数据的轨迹。",
              "每条轨迹分别用前 20 帧三轴中位数扣偏置，再做因果平滑；评估只用插歪接触/持续插入帧。",
              "轴向力是沿配置中插入轴的有符号投影。更高的轴向相关性不意味着完整三维力已对齐。", "",
              "| 比较信号 | 平滑窗口 | 逐轨迹 r 中位数 | r 为正的条数 |",
              "|---|---:|---:|---:|"]
    for name, label, width in (("norm_3", "三维力模长", 3), ("norm_7", "三维力模长", 7),
                               ("axial_3", "有符号插入轴力", 3), ("axial_7", "有符号插入轴力", 7)):
        values = metrics[name]
        report.append(f"| {label} | {width} 帧 | {np.median(values):.3f} | {int(np.sum(values > 0))}/{len(values)} |")
    report += ["", "图：`method_comparison.png`、`axial_traces.png`；逐轨迹数据：`methods_by_trajectory.csv`。",
               "只改分析口径，不修改训练数据、encoder 特征、权重或 checkpoint。"]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)
    for name, values in metrics.items():
        print(f"{name}: median r={np.median(values):.3f}, positive={np.sum(values > 0)}/{len(values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/real_sim_force_comparison_20260929/insertion_alignment"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, output)


if __name__ == "__main__":
    main()
