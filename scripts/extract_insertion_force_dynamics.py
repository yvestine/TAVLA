#!/usr/bin/env python3
"""Extract paired Real/Sim insertion-dynamics features.

This script is intentionally label-free. It extracts candidate signatures of
edge contact/correction and axial insertion from paired rollouts; it does not
claim that any trajectory is a ground-truth success or failure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


PHASES = ("free", "transition", "contact_event", "insertion", "terminal")


def load_domain(repo: Path, trajectory: int, domain: str):
    if domain == "Real":
        path = repo / "data" / f"traj_{trajectory}" / "data.h5"
        force_key = "obs/state/ee_force_base"
    else:
        path = repo / "sim_force" / f"traj_{trajectory}" / "data.h5"
        force_key = "sim/force/contact_total_base"
    with h5py.File(path, "r") as handle:
        timestamp = np.asarray(handle["timestamps"], dtype=np.float64)
        force = np.asarray(handle[force_key], dtype=np.float64)[:, :3]
        pose = np.asarray(handle["obs/state/ee_pose"], dtype=np.float64)[:, :3]
    return timestamp - timestamp[0], force, pose


def read_annotations(path: Path) -> dict[int, dict[str, str]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[int(row["trajectory"].split("_")[-1])] = row
    return result


def moving_average(x: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 1:
        return x.copy()
    pad = width // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(x.shape[1])], axis=1)


def phase_ranges(n: int, free_end: int, onset: int, terminal: int) -> dict[str, tuple[int, int]]:
    ranges = {
        "free": (0, free_end),
        "transition": (min(free_end + 1, n - 1), min(max(onset - 1, 0), n - 1)),
        "contact_event": (min(onset, n - 1), min(onset + 10, n - 1)),
        "insertion": (min(onset + 11, n - 1), min(max(terminal - 3, onset + 11), n - 1)),
        "terminal": (max(min(terminal - 2, n - 1), 0), min(terminal + 2, n - 1)),
    }
    return {name: pair for name, pair in ranges.items() if pair[0] <= pair[1]}


def percentile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if len(x) else float("nan")


def count_reversals(x: np.ndarray) -> int:
    if len(x) < 3:
        return 0
    sign = np.sign(x)
    sign[sign == 0] = np.nan
    valid = ~np.isnan(sign)
    if valid.sum() < 3:
        return 0
    sign = sign[valid]
    return int(np.sum(sign[1:] * sign[:-1] < 0))


def feature_array(force: np.ndarray, real_baseline: np.ndarray | None) -> dict[str, np.ndarray]:
    residual = force - real_baseline if real_baseline is not None else force.copy()
    smoothed = moving_average(residual, 3)
    norm = np.linalg.norm(residual, axis=1)
    lateral = np.linalg.norm(residual[:, :2], axis=1)
    axial = np.abs(residual[:, 2])
    ratio = lateral / (norm + 1e-8)
    unit = residual / (norm[:, None] + 1e-8)
    delta = np.linalg.norm(np.diff(smoothed, axis=0), axis=1) if len(smoothed) > 1 else np.zeros(1)
    direction_delta = np.zeros(len(smoothed), dtype=np.float64)
    if len(smoothed) > 1:
        cosine = np.sum(unit[1:] * unit[:-1], axis=1)
        direction_delta[1:] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return {
        "residual": residual,
        "smoothed": smoothed,
        "norm": norm,
        "lateral": lateral,
        "axial": axial,
        "lateral_ratio": ratio,
        "delta_norm": np.r_[0.0, delta],
        "direction_delta": direction_delta,
    }


def summarize_phase(features: dict[str, np.ndarray], force: np.ndarray, pose: np.ndarray, start: int, end: int) -> dict[str, object]:
    sl = slice(start, end + 1)
    norm = features["norm"][sl]
    lateral = features["lateral"][sl]
    axial = features["axial"][sl]
    ratio = features["lateral_ratio"][sl]
    delta = features["delta_norm"][sl]
    direction_delta = features["direction_delta"][sl]
    return {
        "start_frame": start,
        "end_frame": end,
        "norm_p50_N": percentile(norm, 0.50),
        "norm_p95_N": percentile(norm, 0.95),
        "norm_max_N": float(np.max(norm)),
        "lateral_p50_N": percentile(lateral, 0.50),
        "lateral_p95_N": percentile(lateral, 0.95),
        "axial_p50_N": percentile(axial, 0.50),
        "axial_p95_N": percentile(axial, 0.95),
        "lateral_ratio_p50": percentile(ratio, 0.50),
        "lateral_ratio_p95": percentile(ratio, 0.95),
        "delta_norm_p95_N": percentile(delta, 0.95),
        "direction_delta_p95_rad": percentile(direction_delta, 0.95),
        "Fx_reversals": count_reversals(force[start : end + 1, 0]),
        "Fy_reversals": count_reversals(force[start : end + 1, 1]),
        "xy_path_m": float(np.linalg.norm(np.diff(pose[start : end + 1, :2], axis=0), axis=1).sum()) if end > start else 0.0,
        "z_progress_m": float(pose[start, 2] - pose[end, 2]),
    }


def extract(repo: Path, annotations: dict[int, dict[str, str]], output: Path) -> list[dict[str, object]]:
    rows = []
    for trajectory in sorted(annotations):
        annotation = annotations[trajectory]
        free_end = int(annotation["free_end_frame"])
        onset = int(annotation["contact_onset_frame"])
        terminal = int(annotation["terminal_review_frame"])
        for domain in ("Real", "Sim"):
            time, force, pose = load_domain(repo, trajectory, domain)
            baseline = np.median(force[: free_end + 1], axis=0) if domain == "Real" else None
            features = feature_array(force, baseline)
            for phase, (start, end) in phase_ranges(len(force), free_end, onset, terminal).items():
                summary = summarize_phase(features, force, pose, start, end)
                rows.append(
                    {
                        "trajectory": f"traj_{trajectory}",
                        "domain": domain,
                        "phase": phase,
                        "baseline_subtraction": domain == "Real",
                        **summary,
                    }
                )
            # Store the complete frame-level representation for the plotting stage.
            np.savez_compressed(
                output / f"traj_{trajectory}_{domain.lower()}_dynamics.npz",
                time=time,
                force=force,
                **features,
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(output: Path, annotations: dict[int, dict[str, str]]) -> None:
    ids = sorted(annotations)
    fig, axes = plt.subplots(len(ids), 2, figsize=(15, 2.2 * len(ids)), sharex=False)
    if len(ids) == 1:
        axes = np.asarray([axes])
    for row_idx, trajectory in enumerate(ids):
        annotation = annotations[trajectory]
        free_end = int(annotation["free_end_frame"])
        onset = int(annotation["contact_onset_frame"])
        terminal = int(annotation["terminal_review_frame"])
        for col_idx, domain in enumerate(("Real", "Sim")):
            data = np.load(output / f"traj_{trajectory}_{domain.lower()}_dynamics.npz")
            time = data["time"]
            axes[row_idx, col_idx].plot(time, data["lateral"], label="lateral ||Fxy||", lw=1.2)
            axes[row_idx, col_idx].plot(time, data["axial"], label="axial |Fz|", lw=1.2)
            axes[row_idx, col_idx].plot(time, data["norm"], label="||F||", lw=1.0, alpha=0.65)
            axes[row_idx, col_idx].axvline(time[free_end], color="black", ls="--", lw=0.8)
            axes[row_idx, col_idx].axvline(time[onset], color="tab:red", ls=":", lw=0.9)
            axes[row_idx, col_idx].axvline(time[terminal], color="tab:purple", ls="-.", lw=0.8)
            axes[row_idx, col_idx].set_title(f"traj_{trajectory} {domain}: dashed free_end, red onset, purple terminal")
            axes[row_idx, col_idx].grid(alpha=0.2)
            axes[row_idx, col_idx].legend(fontsize=7, ncol=3, loc="upper left")
            axes[row_idx, col_idx].set_ylabel("N")
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output / "paired_insertion_force_dynamics.png", dpi=160)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    by_key = {(r["trajectory"], r["domain"], r["phase"]): r for r in rows}
    lines = [
        "# Paired insertion force dynamics extraction",
        "",
        "## 目标",
        "",
        "从同一 rollout 的 Real/Sim 配对轨迹中提取‘插歪候选模式’和‘插对候选模式’，不训练网络，也不把启发式结果当作 correct/wrong ground truth。Sim torque 没有参与；只使用三维 translational force。",
        "",
        "## 特征定义",
        "",
        "- `F_lateral = sqrt(Fx^2 + Fy^2)`：把 xy 方向作为横向/偏心受力代理。",
        "- `F_axial = abs(Fz)`：把 z 方向作为插入轴向受力代理；这是基于当前 base-frame 和视频运动方向的工程假设。",
        "- `lateral_ratio = F_lateral / (||F|| + eps)`：区分横向边缘接触和轴向插入。",
        "- `delta_norm`：三帧平滑后的力模长变化。",
        "- `direction_delta`：相邻力方向的夹角变化。",
        "- `Fx/Fy_reversals`：横向力分量符号反转次数，作为纠偏/振荡候选信号。",
        "- Real 使用各自 free-space 段的向量 median 做 baseline subtraction；Sim 保持 `contact_total_base` 原值，因为其 free-space 基本为零。",
        "",
        "## 如何解释‘插歪’和‘插对’",
        "",
        "### 插歪/边缘接触的候选力学模式",
        "",
        "通常应同时出现：横向力 `F_lateral` 上升、`lateral_ratio` 较高、力方向快速旋转或 Fx/Fy 反复换向、EE xy 出现纠偏路径；若随后重新对准，接触阶段会持续一段时间而不是单个 impulse。单个大力峰不够作为 wrong-insertion 证据。",
        "",
        "### 插对/稳定插入的候选力学模式",
        "",
        "通常更接近：z 方向持续推进、轴向力 `F_axial` 占主导、横向比例下降或保持较低、力方向变化减小，最后出现轴向 terminal event 并保持。末端大轴向峰表示插入/到底事件，不单独等价于成功。",
        "",
        "## 当前 10 条结果",
        "",
        "|轨迹|Real 接触后横向/轴向|Real 末端|Sim 接触后横向/轴向|当前判断|",
        "|---|---|---|---|---|",
    ]
    trajectories = sorted(
        {r["trajectory"] for r in rows},
        key=lambda item: int(str(item).split("_")[-1]),
    )
    for trajectory in trajectories:
        r_real = by_key[(trajectory, "Real", "contact_event")]
        r_terminal = by_key[(trajectory, "Real", "terminal")]
        r_sim = by_key[(trajectory, "Sim", "contact_event")]
        if float(r_real["lateral_ratio_p95"]) > 0.7 and float(r_real["Fx_reversals"]) + float(r_real["Fy_reversals"]) > 1:
            judgment = "存在横向纠偏/边缘接触候选"
        else:
            judgment = "可见接触/插入事件，但无足够 wrong 证据"
        lines.append(
            f"|{trajectory}|ratio p95={float(r_real['lateral_ratio_p95']):.3f}, Δ||F|| p95={float(r_real['delta_norm_p95_N']):.3f} N|axial p95={float(r_terminal['axial_p95_N']):.3f} N, z进度={float(r_terminal['z_progress_m']):.4f} m|ratio p95={float(r_sim['lateral_ratio_p95']):.3f}, ||F|| p95={float(r_sim['norm_p95_N']):.2f} N|{judgment}|"
        )
    lines += [
        "",
        "## 重要限制",
        "",
        "1. 这 10 条轨迹目前没有人工 success/failure 或 wrong-insertion 标签，所以本报告只能提取阶段和候选信号，不能训练‘插歪/插对’分类器。",
        "2. 这些轨迹的 Sim 是对应 Real rollout 的视觉 replay，配对主要用于比较事件顺序和趋势；不能要求 Real 与 Sim 的 Newton 数值相等。",
        "3. 当前最有价值的跨域公共信息不是绝对力幅值，而是‘横向/轴向比例、方向变化、持续时间、纠偏路径、末端轴向事件’。",
        "4. 真正要学习 correct/wrong，下一步应在这些事件窗口上补 outcome 标签，或者利用明确的纠偏动作作为 weak label；不能仅靠 free/contact onset 代替。",
        "",
        "## 输出",
        "",
        "- `insertion_force_dynamics.csv`：每条轨迹、每个域、每个阶段的统计。",
        "- `eval_outputs/insertion_force_dynamics/paired_insertion_force_dynamics.png`：Real/Sim 横向力、轴向力和总力的对齐图。",
        "- `eval_outputs/insertion_force_dynamics/traj_*_{real,sim}_dynamics.npz`：逐帧提取结果。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--annotations", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    annotations_path = args.annotations or repo / "representative_trajectory_audit.csv"
    output = repo / "eval_outputs" / "insertion_force_dynamics"
    output.mkdir(parents=True, exist_ok=True)
    annotations = read_annotations(annotations_path)
    rows = extract(repo, annotations, output)
    write_csv(repo / "insertion_force_dynamics.csv", rows)
    plot_overview(output, annotations)
    write_report(repo / "insertion_force_dynamics.md", rows)
    print("wrote insertion_force_dynamics.csv, insertion_force_dynamics.md, and overview plot")


if __name__ == "__main__":
    main()
