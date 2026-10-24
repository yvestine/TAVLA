#!/usr/bin/env python3
"""Export and validate the manually reviewed insertion-quality labels.

These labels were produced by reviewing, for every trajectory, the Real and
Sim front/wrist videos together with Real/Sim force and EE-pose traces.  No
force threshold is used to assign a quality label.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from insertion_quality_v1_lib import causal_mean, read_config


# Inclusive frame intervals.  Ambiguous boundaries deliberately remain
# uncertain instead of being converted into artificial quality supervision.
PHASES: dict[int, list[tuple[int, int, str, str]]] = {
    0: [(0, 95, "free", "high"), (96, 150, "uncertain", "high"), (151, 220, "misaligned_contact", "high"), (221, 255, "aligned_progress", "high"), (256, 266, "terminal", "high")],
    1: [(0, 96, "free", "high"), (97, 112, "uncertain", "medium"), (113, 128, "misaligned_contact", "medium"), (129, 178, "aligned_progress", "medium"), (179, 187, "terminal", "high")],
    2: [(0, 103, "free", "high"), (104, 145, "uncertain", "medium"), (146, 176, "aligned_progress", "medium"), (177, 185, "terminal", "high")],
    3: [(0, 95, "free", "high"), (96, 155, "uncertain", "medium"), (156, 160, "aligned_progress", "medium"), (161, 166, "terminal", "high")],
    4: [(0, 114, "free", "high"), (115, 143, "uncertain", "medium"), (144, 169, "aligned_progress", "medium"), (170, 177, "terminal", "high")],
    5: [(0, 105, "free", "high"), (106, 133, "uncertain", "medium"), (134, 160, "aligned_progress", "medium"), (161, 171, "terminal", "high")],
    6: [(0, 100, "free", "high"), (101, 117, "uncertain", "medium"), (118, 140, "misaligned_contact", "medium"), (141, 167, "aligned_progress", "medium"), (168, 176, "terminal", "high")],
    7: [(0, 109, "free", "high"), (110, 119, "uncertain", "medium"), (120, 229, "misaligned_contact", "high"), (230, 244, "aligned_progress", "high"), (245, 252, "terminal", "high")],
    8: [(0, 88, "free", "high"), (89, 109, "uncertain", "medium"), (110, 129, "misaligned_contact", "medium"), (130, 175, "aligned_progress", "medium"), (176, 183, "terminal", "high")],
    9: [(0, 93, "free", "high"), (94, 134, "uncertain", "medium"), (135, 314, "misaligned_contact", "high"), (315, 359, "aligned_progress", "high"), (360, 367, "terminal", "high")],
    10: [(0, 92, "free", "high"), (93, 135, "uncertain", "medium"), (136, 285, "misaligned_contact", "high"), (286, 313, "aligned_progress", "high"), (314, 319, "terminal", "high")],
    11: [(0, 103, "free", "high"), (104, 120, "uncertain", "medium"), (121, 159, "misaligned_contact", "medium"), (160, 178, "aligned_progress", "medium"), (179, 184, "terminal", "high")],
    12: [(0, 91, "free", "high"), (92, 109, "uncertain", "medium"), (110, 224, "misaligned_contact", "high"), (225, 255, "uncertain", "high")],
    13: [(0, 97, "free", "high"), (98, 134, "uncertain", "medium"), (135, 259, "misaligned_contact", "high"), (260, 282, "aligned_progress", "high"), (283, 290, "terminal", "high")],
    14: [(0, 68, "free", "high"), (69, 119, "uncertain", "medium"), (120, 145, "aligned_progress", "medium"), (146, 152, "terminal", "high")],
    15: [(0, 73, "free", "high"), (74, 109, "uncertain", "medium"), (110, 127, "aligned_progress", "medium"), (128, 132, "terminal", "high")],
    16: [(0, 70, "free", "high"), (71, 91, "uncertain", "medium"), (92, 207, "misaligned_contact", "high"), (208, 224, "aligned_progress", "high"), (225, 229, "terminal", "high")],
    17: [(0, 94, "free", "high"), (95, 109, "uncertain", "medium"), (110, 238, "misaligned_contact", "high"), (239, 258, "aligned_progress", "high"), (259, 264, "terminal", "high")],
    18: [(0, 66, "free", "high"), (67, 129, "uncertain", "medium"), (130, 137, "terminal", "high")],
    19: [(0, 50, "free", "high"), (51, 57, "uncertain", "medium"), (58, 139, "aligned_progress", "medium"), (140, 147, "terminal", "high")],
    20: [(0, 107, "free", "high"), (108, 128, "uncertain", "medium"), (129, 219, "misaligned_contact", "high"), (220, 229, "aligned_progress", "medium"), (230, 236, "terminal", "high")],
    21: [(0, 99, "free", "high"), (100, 119, "uncertain", "medium"), (120, 153, "misaligned_contact", "medium"), (154, 190, "aligned_progress", "medium"), (191, 196, "terminal", "high")],
    22: [(0, 66, "free", "high"), (67, 92, "uncertain", "medium"), (93, 125, "misaligned_contact", "high"), (126, 162, "aligned_progress", "high"), (163, 168, "terminal", "high")],
    23: [(0, 63, "free", "high"), (64, 89, "uncertain", "medium"), (90, 139, "misaligned_contact", "high"), (140, 160, "aligned_progress", "high"), (161, 166, "terminal", "high")],
    24: [(0, 66, "free", "high"), (67, 84, "uncertain", "medium"), (85, 150, "misaligned_contact", "high"), (151, 166, "aligned_progress", "medium"), (167, 172, "terminal", "high")],
    25: [(0, 77, "free", "high"), (78, 113, "uncertain", "medium"), (114, 124, "misaligned_contact", "medium"), (125, 188, "aligned_progress", "medium"), (189, 192, "terminal", "high")],
    26: [(0, 62, "free", "high"), (63, 67, "uncertain", "medium"), (68, 96, "aligned_progress", "medium"), (97, 102, "terminal", "high")],
    27: [(0, 44, "free", "high"), (45, 77, "uncertain", "medium"), (78, 125, "misaligned_contact", "high"), (126, 143, "aligned_progress", "medium"), (144, 149, "terminal", "high")],
    28: [(0, 62, "free", "high"), (63, 87, "uncertain", "medium"), (88, 137, "misaligned_contact", "medium"), (138, 157, "aligned_progress", "medium"), (158, 163, "terminal", "high")],
    29: [(0, 57, "free", "high"), (58, 83, "uncertain", "medium"), (84, 139, "misaligned_contact", "high"), (140, 161, "aligned_progress", "high"), (162, 167, "terminal", "high")],
    30: [(0, 53, "free", "high"), (54, 97, "uncertain", "medium"), (98, 104, "terminal", "high")],
    31: [(0, 51, "free", "high"), (52, 73, "uncertain", "medium"), (74, 95, "aligned_progress", "medium"), (96, 101, "terminal", "high")],
    32: [(0, 81, "free", "high"), (82, 98, "uncertain", "medium"), (99, 124, "misaligned_contact", "medium"), (125, 149, "aligned_progress", "medium"), (150, 152, "terminal", "high")],
    33: [(0, 65, "free", "high"), (66, 79, "uncertain", "medium"), (80, 107, "misaligned_contact", "medium"), (108, 121, "aligned_progress", "medium"), (122, 127, "terminal", "high")],
    34: [(0, 70, "free", "high"), (71, 94, "uncertain", "medium"), (95, 195, "misaligned_contact", "high"), (196, 207, "aligned_progress", "medium"), (208, 213, "terminal", "high")],
    35: [(0, 54, "free", "high"), (55, 77, "uncertain", "medium"), (78, 110, "aligned_progress", "medium"), (111, 116, "terminal", "high")],
    36: [(0, 57, "free", "high"), (58, 72, "uncertain", "medium"), (73, 94, "aligned_progress", "medium"), (95, 99, "terminal", "high")],
    37: [(0, 52, "free", "high"), (53, 69, "uncertain", "medium"), (70, 93, "aligned_progress", "medium"), (94, 98, "terminal", "high")],
    38: [(0, 66, "free", "high"), (67, 91, "uncertain", "medium"), (92, 109, "aligned_progress", "medium"), (110, 114, "terminal", "high")],
    39: [(0, 43, "free", "high"), (44, 69, "uncertain", "medium"), (70, 74, "terminal", "high")],
}


LABEL_COLORS = {
    "free": "#bdbdbd",
    "uncertain": "#fdd835",
    "misaligned_contact": "#e53935",
    "aligned_progress": "#43a047",
    "terminal": "#5e35b1",
}


def evidence(trajectory: int, label: str) -> tuple[str, str, str]:
    if label == "free":
        return (
            "Real and Sim front/wrist views show approach before the reviewed contact transition.",
            "The interval is retained from the previously confirmed per-trajectory free-space review.",
            "EE is still in the approach portion of the rollout.",
        )
    if label == "uncertain":
        suffix = " This trajectory never provides a defensible aligned-progress segment." if trajectory in {12, 18, 30, 39} else ""
        return (
            "Near-hole geometry or transition timing is not visually decisive." + suffix,
            "Force changes do not uniquely separate edge loading from aligned insertion.",
            "The motion lies near a phase boundary; it is masked from quality training.",
        )
    if label == "misaligned_contact":
        if trajectory == 12:
            video = "The peg remains in a prolonged unsuccessful contact/search episode and later withdraws; no centered insertion is visible."
        else:
            video = "Front/wrist review shows contact at the rim or continued search/correction before the final centered descent."
        return (
            video,
            "Sustained or repeated lateral/axial force events occur before successful downward progress resumes.",
            "Insertion-axis progress stalls or rebounds while lateral correction/search continues.",
        )
    if label == "aligned_progress":
        return (
            "After the final correction/transition, the peg remains centered and visibly proceeds into the hole.",
            "The response evolves as a sustained insertion segment rather than an isolated terminal impulse.",
            "EE makes continued insertion-axis progress with substantially less correction or rebound.",
        )
    if label == "terminal":
        return (
            "The reviewed videos show the final seated, impact, or hold portion of the rollout.",
            "A late axial/terminal response is present and is kept separate from aligned progress.",
            "EE reaches its lowest/final region and subsequent progress stops.",
        )
    raise KeyError(label)


def validate(repo: Path, config: dict[str, object]) -> dict[int, int]:
    lengths: dict[int, int] = {}
    expected = set(int(value) for value in config["all_trajectories"])
    if set(PHASES) != expected:
        raise ValueError(f"phase trajectories {set(PHASES)} != config trajectories {expected}")
    for trajectory in sorted(PHASES):
        with h5py.File(repo / "data" / f"traj_{trajectory}" / "data.h5", "r") as handle:
            n_frames = len(handle[config["data"]["timestamp_key"]])
        lengths[trajectory] = n_frames
        expected_start = 0
        for start, end, label, confidence in PHASES[trajectory]:
            if start != expected_start or end < start:
                raise ValueError(f"traj_{trajectory}: invalid/gapped interval {start}-{end}; expected {expected_start}")
            if label not in LABEL_COLORS or confidence not in {"low", "medium", "high"}:
                raise ValueError(f"traj_{trajectory}: invalid label/confidence {label}/{confidence}")
            expected_start = end + 1
        if expected_start != n_frames:
            raise ValueError(f"traj_{trajectory}: labels end at {expected_start - 1}, data ends at {n_frames - 1}")
        free_end = int(config["free_end_frames"][str(trajectory)])
        if PHASES[trajectory][0][:3] != (0, free_end, "free"):
            raise ValueError(f"traj_{trajectory}: confirmed free interval was changed")
    return lengths


def write_csv(output: Path) -> list[dict[str, object]]:
    fields = ["trajectory", "start_frame", "end_frame", "label", "confidence", "video_evidence", "force_evidence", "pose_evidence"]
    rows: list[dict[str, object]] = []
    for trajectory in sorted(PHASES):
        for start, end, label, confidence in PHASES[trajectory]:
            video, force, pose = evidence(trajectory, label)
            rows.append({
                "trajectory": f"traj_{trajectory}",
                "start_frame": start,
                "end_frame": end,
                "label": label,
                "confidence": confidence,
                "video_evidence": video,
                "force_evidence": force,
                "pose_evidence": pose,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_trajectory(repo: Path, config: dict[str, object], trajectory: int, output: Path) -> None:
    with h5py.File(repo / "data" / f"traj_{trajectory}" / "data.h5", "r") as handle:
        real_force = np.asarray(handle[config["data"]["real_force_key"]], dtype=np.float64)[:, :3]
        pose = np.asarray(handle[config["data"]["pose_key"]], dtype=np.float64)[:, :3]
    with h5py.File(repo / "sim_force" / f"traj_{trajectory}" / "data.h5", "r") as handle:
        sim_force = np.asarray(handle[config["data"]["sim_force_key"]], dtype=np.float64)[:, :3]
    free_end = int(config["free_end_frames"][str(trajectory)])
    baseline = np.median(real_force[: free_end + 1], axis=0)
    real = causal_mean(real_force - baseline, 3)
    sim = causal_mean(sim_force, 3)
    x = np.arange(len(real))
    z_progress = (pose[0, 2] - pose[:, 2]) * 1000.0
    xy_step = np.r_[0.0, np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1) * 1000.0]

    fig, axes = plt.subplots(3, 1, figsize=(15, 7.5), sharex=True)
    axes[0].plot(x, np.linalg.norm(real[:, :2], axis=1), label="Real lateral", color="#d62728")
    axes[0].plot(x, np.abs(real[:, 2]), label="Real axial", color="#1f77b4")
    axes[1].plot(x, np.linalg.norm(sim[:, :2], axis=1), label="Sim lateral", color="#d62728")
    axes[1].plot(x, np.abs(sim[:, 2]), label="Sim axial", color="#1f77b4")
    axes[2].plot(x, z_progress, label="downward Z progress (mm)", color="black")
    axes[2].plot(x, xy_step * 8.0, label="XY step x8", color="#ff7f0e")
    for axis in axes:
        for start, end, label, _ in PHASES[trajectory]:
            axis.axvspan(start, end + 1, color=LABEL_COLORS[label], alpha=0.14)
        axis.grid(alpha=0.15)
        axis.legend(loc="upper left", ncol=2)
    axes[0].set_ylabel("Real N")
    axes[1].set_ylabel("Sim N")
    axes[2].set_ylabel("mm / scaled")
    axes[2].set_xlabel("frame (10 Hz)")
    axes[0].set_title(f"traj_{trajectory} reviewed insertion-quality phases")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_report(output: Path, rows: list[dict[str, object]], lengths: dict[int, int], review_dir: Path) -> None:
    frame_counts: Counter[str] = Counter()
    trajectory_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    for trajectory, phases in PHASES.items():
        for start, end, label, _ in phases:
            frame_counts[label] += end - start + 1
            trajectory_counts[label] += 1
            # A quality window is usable only when all ten history frames stay
            # inside the same reviewed phase.
            window_counts[label] += max(0, end - start + 1 - 9)
    lines = [
        "# Insertion Quality Label Audit V2",
        "",
        "## 结论",
        "",
        "已逐条复核 40 条轨迹的 Real/Sim front、wrist 视频，并与 Real/Sim 三维力和 EE pose 同步检查。标签是人工多证据审阅结果，不是由固定力阈值自动生成。",
        "",
        "标签语义：`misaligned_contact` 表示已经接触但插入进展停滞/反弹且仍在纠偏；`aligned_progress` 表示最后一次纠偏后持续向孔内推进；`terminal` 与正确插入分开；无法证明质量的边界统一标为 `uncertain` 并在训练时 mask。",
        "",
        f"- 总轨迹：{len(PHASES)}",
        f"- 总帧数：{sum(lengths.values())}",
        f"- 包含明确 misaligned 片段的轨迹：{sum(any(p[2] == 'misaligned_contact' for p in phases) for phases in PHASES.values())}",
        f"- 包含明确 aligned-progress 片段的轨迹：{sum(any(p[2] == 'aligned_progress' for p in phases) for phases in PHASES.values())}",
        f"- 同时包含两种质量标签且可做轨迹内对照的轨迹：{sum(any(p[2] == 'misaligned_contact' and p[1] - p[0] + 1 >= 10 for p in phases) and any(p[2] == 'aligned_progress' and p[1] - p[0] + 1 >= 10 for p in phases) for phases in PHASES.values())}",
        f"- 没有可信 aligned-progress、不可作为正类的轨迹：traj_12、traj_18、traj_30、traj_39",
        "",
        "## 帧数分布",
        "",
        "| label | frames | fraction | trajectories/segments |",
        "|---|---:|---:|---:|",
    ]
    total = sum(frame_counts.values())
    for label in LABEL_COLORS:
        lines.append(f"| {label} | {frame_counts[label]} | {frame_counts[label] / total:.1%} | {trajectory_counts[label]} |")
    lines += [
        "",
        "按 10 帧历史且窗口不能跨阶段边界计算，可得到 `misaligned_contact=1471`、`aligned_progress=664` 个候选窗口；23 条轨迹同时含有两类可用窗口。训练时仍需按 trajectory/label 配平采样，不能直接按帧全量采样。",
        "",
        "## 每条轨迹的最终边界",
        "",
        "| trajectory | reviewed phases (inclusive frames) | video sheet | labeled signal |",
        "|---|---|---|---|",
    ]
    for trajectory, phases in PHASES.items():
        phase_text = "; ".join(f"{label} {start}-{end} ({confidence})" for start, end, label, confidence in phases)
        lines.append(
            f"| traj_{trajectory} | {phase_text} | "
            f"[videos](eval_outputs/insertion_quality_label_review_v2/traj_{trajectory}_videos.jpg) | "
            f"[signals](eval_outputs/insertion_quality_label_review_v2/labeled/traj_{trajectory}_labeled.png) |"
        )
    lines += [
        "",
        "## 使用限制",
        "",
        "- 这是一版严格保守标签：`uncertain` 不可转成 aligned 或 misaligned。",
        "- Real/Sim 视频用于共同复核运动阶段，但两域 force timing 不一致，因此标签不等于同帧力学状态完全一致。",
        "- `medium` 表示片段语义可信但边界可能有约 3-5 帧误差；训练窗口若跨越边界应继续 mask。",
        "- `traj_12` 是重要的错误插入/撤回负例；不能因为轨迹结束就自动添加 aligned-progress。",
        "- 下一步应先用这些标签重新统计每折可用正负窗口与类别平衡，再决定是否训练 Encoder。",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = read_config(repo / "configs/insertion_quality_v1.json")
    lengths = validate(repo, config)
    rows = write_csv(repo / "annotations/insertion_quality_labels_v2.csv")
    plot_dir = repo / "eval_outputs/insertion_quality_label_review_v2/labeled"
    for trajectory in sorted(PHASES):
        plot_trajectory(repo, config, trajectory, plot_dir / f"traj_{trajectory}_labeled.png")
    write_report(
        repo / "insertion_quality_label_audit_v2.md",
        rows,
        lengths,
        repo / "eval_outputs/insertion_quality_label_review_v2",
    )
    print(f"validated {len(PHASES)} trajectories / {sum(lengths.values())} frames")
    print(f"wrote {len(rows)} phase rows")


if __name__ == "__main__":
    main()
