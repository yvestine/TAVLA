#!/usr/bin/env python3
"""Visual audit of traj_0 insertion phases for Real and Sim force signals.

This is an audit/visualization only. It does not create training labels and
does not modify TAVLA training code.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHASE_COLORS = {
    "free_space": "#d9f0d3",
    "initial_contact_alignment": "#fee8c8",
    "final_insertion": "#fddbc7",
}

HIGH_LEVEL_PHASES = [
    ("free_space", 0, 95, "A  free-space"),
    ("initial_contact_alignment", 96, 220, "B  initial contact / alignment search"),
    ("final_insertion", 221, 266, "C  final insertion / terminal hold"),
]

DETAILED_PHASES = [
    ("free_space_high_confidence", 0, 95, "free-space"),
    ("near_contact_uncertain", 96, 150, "near-contact / uncertain"),
    ("contact_onset_impact", 151, 156, "first contact impact"),
    ("contact_alignment_search", 157, 220, "alignment search / correction"),
    ("insertion_descent", 221, 255, "insertion descent"),
    ("terminal_contact_peak", 256, 260, "terminal contact peak"),
    ("terminal_hold", 261, 266, "terminal hold"),
]


def read_force(path: Path, key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        timestamps = np.asarray(handle["timestamps"], dtype=np.float64)
        force = np.asarray(handle[key], dtype=np.float64)
        pose = np.asarray(handle["obs/state/ee_pose"], dtype=np.float64)
    time_s = timestamps - timestamps[0]
    return time_s, force, pose


def read_existing_labels(path: Path) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[int(row["frame"])] = row["phase"]
    return labels


def phase_rows(
    domain: str,
    time_s: np.ndarray,
    force: np.ndarray,
    pose: np.ndarray,
    phases: Iterable[Tuple[str, int, int, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    norm = np.linalg.norm(force, axis=1)
    for name, start, end, display_name in phases:
        values = force[start : end + 1]
        norms = norm[start : end + 1]
        rows.append(
            {
                "domain": domain,
                "phase": name,
                "display_name": display_name,
                "start_frame": start,
                "end_frame": end,
                "start_time_s": float(time_s[start]),
                "end_time_s": float(time_s[end]),
                "n_frames": int(len(values)),
                "Fx_mean": float(np.mean(values[:, 0])),
                "Fy_mean": float(np.mean(values[:, 1])),
                "Fz_mean": float(np.mean(values[:, 2])),
                "norm_p50": float(np.quantile(norms, 0.50)),
                "norm_p95": float(np.quantile(norms, 0.95)),
                "norm_max": float(np.max(norms)),
                "delta_norm_p50": float(np.quantile(np.abs(np.diff(norms)), 0.50))
                if len(norms) > 1
                else 0.0,
                "delta_norm_p95": float(np.quantile(np.abs(np.diff(norms)), 0.95))
                if len(norms) > 1
                else 0.0,
                "pose_z_start": float(pose[start, 2]),
                "pose_z_end": float(pose[end, 2]),
                "pose_z_change": float(pose[end, 2] - pose[start, 2]),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_force_comparison(
    output: Path,
    real_time: np.ndarray,
    real_force: np.ndarray,
    sim_time: np.ndarray,
    sim_force: np.ndarray,
) -> None:
    real_norm = np.linalg.norm(real_force, axis=1)
    sim_norm = np.linalg.norm(sim_force, axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True, constrained_layout=True)

    for name, start, end, display_name in HIGH_LEVEL_PHASES:
        left = real_time[start]
        right = real_time[end]
        for ax in axes:
            ax.axvspan(left, right, color=PHASE_COLORS[name], alpha=0.52, lw=0)
        axes[0].text(
            (left + right) / 2,
            1.03,
            display_name,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    # Detailed boundaries are intentionally dashed: they are audit boundaries,
    # not new automatic labels.
    for _, frame, _, text in [
        ("near-contact", 96, 0, "near-contact start"),
        ("first-impact", 151, 0, "first impact"),
        ("final-descent", 221, 0, "final descent"),
        ("terminal-peak", 256, 0, "terminal peak"),
    ]:
        x = real_time[frame]
        for ax in axes:
            ax.axvline(x, color="#555555", linestyle="--", linewidth=0.9, alpha=0.75)
        axes[0].text(x, 0.98, text, rotation=90, va="top", ha="right", fontsize=8)

    for idx, name in enumerate(("Fx", "Fy", "Fz")):
        axes[0].plot(real_time, real_force[:, idx], label=f"Real {name}", linewidth=1.4)
        axes[1].plot(sim_time, sim_force[:, idx], label=f"Sim {name}", linewidth=1.4)
    axes[0].set_ylabel("Real force [N]")
    axes[1].set_ylabel("Sim force [N]")
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].legend(ncol=3, loc="upper left")
    axes[0].set_title(
        "traj_0 insertion phase audit: force components and annotated video phases\n"
        "Real: obs/state/ee_force_base    |    Sim: sim/force/contact_total_base"
    )

    axes[2].plot(real_time, real_norm, color="#1f77b4", label="Real ||F||", linewidth=1.7)
    axes[2].plot(sim_time, sim_norm, color="#d62728", label="Sim ||F||", linewidth=1.7)
    axes[2].set_ylabel("||F|| [N]")
    axes[2].legend(loc="upper left")
    axes[2].set_title("Same-time-axis force magnitude; Real baseline is nonzero, Sim is zero before first event")

    axes[3].plot(real_time, real_norm, color="#1f77b4", label="Real ||F||", linewidth=1.5)
    axes[3].plot(sim_time, sim_norm, color="#d62728", label="Sim ||F||", linewidth=1.5)
    axes[3].set_yscale("symlog", linthresh=1.0)
    axes[3].set_ylabel("||F|| [N]\nsymlog")
    axes[3].set_xlabel("time from video start [s] (10 Hz)")
    axes[3].legend(loc="upper left")
    axes[3].set_title("Same signal with symlog scale to expose Sim sparse impulses")
    axes[3].grid(True, alpha=0.2)
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.set_xlim(float(real_time[0]), float(real_time[-1]))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_keyframe_sheet(output: Path, video_paths: Dict[str, Path], frames: List[int]) -> None:
    """Create a frame sheet using ffmpeg, keeping the audit self-contained."""
    import subprocess
    import tempfile

    font = ImageFont.load_default()
    with tempfile.TemporaryDirectory(prefix="traj0_keyframes_") as temp_dir:
        temp = Path(temp_dir)
        for key, video in video_paths.items():
            sheet = Image.new("RGB", (320 * 5, 240 * 5), (0, 0, 0))
            for j, frame in enumerate(frames):
                image_path = temp / f"{key}_{frame}.png"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(video),
                        "-vf",
                        f"select=eq(n\\,{frame})",
                        "-frames:v",
                        "1",
                        str(image_path),
                    ],
                    check=True,
                )
                image = Image.open(image_path).convert("RGB").resize((320, 240))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 130, 18), fill=(0, 0, 0))
                draw.text((4, 3), f"frame {frame}  {frame / 10:.1f}s", font=font, fill=(255, 255, 0))
                sheet.paste(image, ((j % 5) * 320, (j // 5) * 240))
            sheet.save(output.with_name(f"{output.stem}_{key}.jpg"), quality=92)


def write_report(path: Path, rows: List[Dict[str, object]]) -> None:
    def row(domain: str, phase: str) -> Dict[str, object]:
        return next(item for item in rows if item["domain"] == domain and item["phase"] == phase)

    real_free = row("Real", "free_space")
    sim_free = row("Sim", "free_space")
    real_initial = row("Real", "initial_contact_alignment")
    sim_initial = row("Sim", "initial_contact_alignment")
    real_final = row("Real", "final_insertion")
    sim_final = row("Sim", "final_insertion")
    lines = [
        "# traj_0 insertion phase force audit",
        "",
        "本文件只做视频/H5 对齐和可视化，不新增训练标签，不修改 TAVLA。",
        "",
        "## 数据来源",
        "",
        "- Real video: `data/traj_0/front_camera.mp4`, `data/traj_0/wrist_camera.mp4`",
        "- Real force: `data/traj_0/data.h5::obs/state/ee_force_base`",
        "- Sim video: `sim_force/traj_0/front_camera.mp4`, `sim_force/traj_0/wrist_camera.mp4`",
        "- Sim force: `sim_force/traj_0/data.h5::sim/force/contact_total_base`",
        "- 视频/H5: 267 帧，10 Hz，约 26.7 s；Sim 视频是 Real rollout 的 replay 画面。",
        "- 阶段边界使用已有 `data/traj_0/traj0_contact_phase_labels_v1.csv`，没有用单一 force threshold 重新造标签。",
        "",
        "## 观测到的三个大阶段",
        "",
        "| 阶段 | 帧 / 时间 | 视频和力信号解释 |",
        "|---|---:|---|",
        "| A free-space | 0–95 / 0.0–9.5 s | 视觉上仍在接近目标；Real 有传感器/重力补偿后的非零背景力，Sim `contact_total_base` 严格为 0。Real 力方向缓慢变化，主要对应运动和基线变化。 |",
        "| B initial contact / alignment | 96–220 / 9.6–22.0 s | 不能简单叫“错误标签”，但视频和 Real 力都显示进入接触后的试探/对准阶段。Real 在 96–150 先出现分量重排，151–156 出现第一次明显冲击，之后 157–220 有横向力和方向反复变化，符合边缘接触/姿态修正。Sim 116 帧左右才出现第一小力事件，151 附近出现离散 impulse，190–220 后力快速增大且更尖锐。 |",
        "| C final insertion / terminal | 221–266 / 22.1–26.6 s | Real 221–255 继续下压，255–257 出现最大的真实末端力峰，随后保持；Sim 同一时间段产生多次大幅、方向变化明显的接触脉冲，幅值显著高于 Real。 |",
        "",
        "## 关键数值",
        "",
        f"- Real free-space `||F||`: p50={real_free['norm_p50']:.3f} N, p95={real_free['norm_p95']:.3f} N, max={real_free['norm_max']:.3f} N。",
        f"- Sim free-space `||F||`: p50={sim_free['norm_p50']:.3f} N, p95={sim_free['norm_p95']:.3f} N, max={sim_free['norm_max']:.3f} N。",
        f"- Real initial-contact/alignment `||F||`: p50={real_initial['norm_p50']:.3f} N, p95={real_initial['norm_p95']:.3f} N, max={real_initial['norm_max']:.3f} N。",
        f"- Sim initial-contact/alignment `||F||`: p50={sim_initial['norm_p50']:.3f} N, p95={sim_initial['norm_p95']:.3f} N, max={sim_initial['norm_max']:.3f} N。",
        f"- Real final-insertion/terminal `||F||`: p50={real_final['norm_p50']:.3f} N, p95={real_final['norm_p95']:.3f} N, max={real_final['norm_max']:.3f} N。",
        f"- Sim final-insertion/terminal `||F||`: p50={sim_final['norm_p50']:.3f} N, p95={sim_final['norm_p95']:.3f} N, max={sim_final['norm_max']:.3f} N。",
        "",
        "## 结论",
        "",
        "1. Real 的有效信息不是简单的 contact state，而是接触后的连续动力学：初始边缘接触时横向分量重排和方向反复，最终插入时出现轴向冲击并进入保持。",
        "2. Sim 的视频动作与 Real replay 对齐，但 Sim force 的语义更像碰撞/接触 impulse：free-space 全零，早期接触稀疏，后期出现大幅尖峰；不能直接把 Sim 的单帧大力峰当作 Real 的“错误插入”标签。",
        "3. 对后续 TAVLA，更合适的目标是学习“插入动力学/插入质量”而不是单独学习 contact。当前这张图支持先构造连续 force-dynamics representation，再用更多轨迹的成功/修正/失败证据建立 outcome 标签。",
        "",
        "## 输出图",
        "",
        "- `traj0_force_phase_comparison.png`: Real/Sim 三维 force、force norm、阶段标记。",
        "- `traj0_video_keyframes_real_wrist.jpg` / `traj0_video_keyframes_sim_wrist.jpg`: wrist video 关键帧。",
        "- `traj0_video_keyframes_real_front.jpg` / `traj0_video_keyframes_sim_front.jpg`: front video 关键帧。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output or repo / "eval_outputs" / "traj0_insertion_phase_audit"
    output.mkdir(parents=True, exist_ok=True)

    real_time, real_force, real_pose = read_force(
        repo / "data/traj_0/data.h5", "obs/state/ee_force_base"
    )
    sim_time, sim_force, sim_pose = read_force(
        repo / "sim_force/traj_0/data.h5", "sim/force/contact_total_base"
    )
    labels = read_existing_labels(repo / "data/traj_0/traj0_contact_phase_labels_v1.csv")
    if len(real_time) != len(sim_time) or len(real_time) != 267:
        raise ValueError("Expected synchronized 267-frame traj_0 Real/Sim data")
    if set(labels) != set(range(267)):
        raise ValueError("Existing traj_0 phase labels must cover frames 0..266")

    rows = phase_rows("Real", real_time, real_force, real_pose, HIGH_LEVEL_PHASES)
    rows.extend(phase_rows("Sim", sim_time, sim_force, sim_pose, HIGH_LEVEL_PHASES))
    write_csv(output / "traj0_phase_force_stats.csv", rows)
    plot_force_comparison(
        output / "traj0_force_phase_comparison.png",
        real_time,
        real_force,
        sim_time,
        sim_force,
    )
    make_keyframe_sheet(
        output / "traj0_video_keyframes",
        {
            "real_front": repo / "data/traj_0/front_camera.mp4",
            "real_wrist": repo / "data/traj_0/wrist_camera.mp4",
            "sim_front": repo / "sim_force/traj_0/front_camera.mp4",
            "sim_wrist": repo / "sim_force/traj_0/wrist_camera.mp4",
        },
        [0, 40, 80, 95, 110, 130, 150, 151, 170, 200, 220, 240, 255, 257, 266],
    )
    write_report(output / "traj0_insertion_phase_audit.md", rows)
    print(json.dumps({"output_dir": str(output), "files": sorted(p.name for p in output.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
