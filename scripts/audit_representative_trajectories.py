#!/usr/bin/env python3
"""Summarize the manual video/force review of representative trajectories.

This is an audit tool only. It does not alter TAVLA training data or create
automatic contact labels. The phase boundaries below are review annotations;
post-cutoff frames remain transition/uncertain unless the video, pose, and
force response support a stronger interpretation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


TRAJECTORIES = {
    0: {
        "free_end": 95,
        "contact_onset": 151,
        "confidence": "high",
        "transition": "96-150",
        "post_contact": "151-156 onset impact; 157-220 alignment/correction search; 221-255 insertion descent; 256-266 terminal peak/hold",
        "terminal_frame": 257,
        "evidence": "Existing detailed annotation; both cameras, pose stop/rebound, and wrench impulse agree.",
        "wrong_evidence": "No binary wrong-insertion label; visible correction/alignment search is present.",
    },
    3: {
        "free_end": 95,
        "contact_onset": 156,
        "confidence": "medium",
        "transition": "96-155",
        "post_contact": "156-160 impact/insertion; 161-166 pose locked/terminal",
        "terminal_frame": 158,
        "evidence": "Peg is aligned in both views; z reaches its minimum and the force impulse persists after frame 156.",
        "wrong_evidence": "No clear correction loop; do not label as wrong insertion.",
    },
    8: {
        "free_end": 88,
        "contact_onset": 110,
        "confidence": "medium",
        "transition": "89-109",
        "post_contact": "110-113 first contact response; 114-175 continued insertion; 176-183 terminal event/hold",
        "terminal_frame": 177,
        "evidence": "Centered approach, short z plateau/rebound around 110, multi-frame wrench response, and later terminal peak.",
        "wrong_evidence": "No reliable visual evidence of a wrong insertion state.",
    },
    14: {
        "free_end": 68,
        "contact_onset": 94,
        "confidence": "medium",
        "transition": "69-93",
        "post_contact": "94-152 sustained low/moderate contact response; no isolated terminal impulse comparable to other tracks",
        "terminal_frame": 147,
        "evidence": "Peg and hole align in both views; z and lateral motion change near 94, followed by a stable shallow insertion response.",
        "wrong_evidence": "No clear wrong-insertion or correction event visible.",
    },
    18: {
        "free_end": 66,
        "contact_onset": 130,
        "confidence": "medium",
        "transition": "67-129",
        "post_contact": "130-131 onset impact; 132-137 pose locked/terminal",
        "terminal_frame": 130,
        "evidence": "Earlier force is elevated but ambiguous; first unambiguous lock/impact occurs at the insertion floor near 130.",
        "wrong_evidence": "No reliable visual evidence of a wrong insertion state.",
    },
    19: {
        "free_end": 50,
        "contact_onset": 58,
        "confidence": "medium",
        "transition": "51-57",
        "post_contact": "58-139 continued approach/insertion; 140-144 terminal axial peak; 145-147 hold",
        "terminal_frame": 141,
        "evidence": "Target enters alignment around 55-60; Real force changes at the reviewed onset and a later terminal axial event is clear.",
        "wrong_evidence": "The transition is short; no defensible wrong-insertion label from these views.",
    },
    25: {
        "free_end": 77,
        "contact_onset": 114,
        "confidence": "medium",
        "transition": "78-113",
        "post_contact": "114-124 first response/rebound; 125-188 continued insertion; 189-192 terminal axial peak",
        "terminal_frame": 191,
        "evidence": "Peg remains visually aligned; first sustained response is 114-120, followed by rebound and a distinct terminal peak.",
        "wrong_evidence": "Possible small post-contact adjustment, but not enough evidence to call it wrong insertion.",
    },
    30: {
        "free_end": 53,
        "contact_onset": 98,
        "confidence": "medium",
        "transition": "54-97",
        "post_contact": "98-102 strong onset/terminal impact; 103-104 hold",
        "terminal_frame": 101,
        "evidence": "Final approach is visible; force rises sharply at 98-101 while z reaches its minimum and pose locks.",
        "wrong_evidence": "No visible correction loop; do not label as wrong insertion.",
    },
    32: {
        "free_end": 81,
        "contact_onset": 99,
        "confidence": "medium",
        "transition": "82-98",
        "post_contact": "99-149 contact/insertion evolution; 150-152 terminal axial peak/hold",
        "terminal_frame": 151,
        "evidence": "Approach alignment is visible; the first pose/force transition is near 99, while the larger terminal event is near 150.",
        "wrong_evidence": "Some lateral adjustment is visible, but no reliable wrong-insertion class can be assigned.",
    },
    39: {
        "free_end": 43,
        "contact_onset": 70,
        "confidence": "medium",
        "transition": "44-69",
        "post_contact": "70 onset; 71-73 large terminal axial peak; 74 hold",
        "terminal_frame": 72,
        "evidence": "Peg reaches the hole during final descent; force and pose change at 70 and peak at 71-73.",
        "wrong_evidence": "No reliable wrong-insertion state; the event is abrupt but visually aligned.",
    },
}


def read_domain(repo: Path, trajectory: int, domain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if domain == "Real":
        path = repo / "data" / f"traj_{trajectory}" / "data.h5"
        key = "obs/state/ee_force_base"
    else:
        path = repo / "sim_force" / f"traj_{trajectory}" / "data.h5"
        key = "sim/force/contact_total_base"
    with h5py.File(path, "r") as handle:
        timestamps = np.asarray(handle["timestamps"], dtype=np.float64)
        force = np.asarray(handle[key], dtype=np.float64)[:, :3]
        pose = np.asarray(handle["obs/state/ee_pose"], dtype=np.float64)[:, :3]
    return timestamps - timestamps[0], force, pose


def summarize(repo: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectory, annotation in TRAJECTORIES.items():
        real_time, real_force, real_pose = read_domain(repo, trajectory, "Real")
        sim_time, sim_force, _ = read_domain(repo, trajectory, "Sim")
        if len(real_force) != len(sim_force):
            raise ValueError(f"traj_{trajectory}: Real/Sim frame counts differ")
        real_norm = np.linalg.norm(real_force, axis=1)
        sim_norm = np.linalg.norm(sim_force, axis=1)
        free_end = annotation["free_end"]
        onset = annotation["contact_onset"]
        terminal = annotation["terminal_frame"]
        first_sim = np.flatnonzero(sim_norm > 1e-8)
        real_peak = int(np.argmax(real_norm))
        sim_peak = int(np.argmax(sim_norm))
        transition_slice = slice(free_end + 1, onset)
        post_slice = slice(onset, None)
        post_real = real_norm[post_slice]
        post_sim = sim_norm[post_slice]
        pose_path = np.linalg.norm(np.diff(real_pose[onset:, :2], axis=0), axis=1).sum()
        rows.append(
            {
                "trajectory": f"traj_{trajectory}",
                "n_frames": len(real_force),
                "free_end_frame": free_end,
                "contact_onset_frame": onset,
                "free_end_time_s": round(float(real_time[free_end]), 4),
                "contact_onset_time_s": round(float(real_time[onset]), 4),
                "transition_frames": annotation["transition"],
                "confidence": annotation["confidence"],
                "sim_first_nonzero_frame": int(first_sim[0]) if len(first_sim) else "none",
                "sim_first_nonzero_time_s": round(float(sim_time[first_sim[0]]), 4) if len(first_sim) else "",
                "real_peak_frame": real_peak,
                "real_peak_norm_N": round(float(real_norm[real_peak]), 4),
                "sim_peak_frame": sim_peak,
                "sim_peak_norm_N": round(float(sim_norm[sim_peak]), 4),
                "real_post_onset_p50_N": round(float(np.quantile(post_real, 0.5)), 4),
                "real_post_onset_p95_N": round(float(np.quantile(post_real, 0.95)), 4),
                "sim_post_onset_p50_N": round(float(np.quantile(post_sim, 0.5)), 4),
                "sim_post_onset_p95_N": round(float(np.quantile(post_sim, 0.95)), 4),
                "terminal_review_frame": terminal,
                "real_onset_to_end_xy_path_m": round(float(pose_path), 5),
                "post_contact_interpretation": annotation["post_contact"],
                "video_force_evidence": annotation["evidence"],
                "wrong_insertion_evidence": annotation["wrong_evidence"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# 10 条代表轨迹人工辅助审阅",
        "",
        "本审计把 Real front/wrist 视频、Sim front/wrist replay、Real EE force、Sim `contact_total_base` 和 EE pose 放在同一帧轴上查看。已有 free-space cutoff 只表示高置信度 free-space；cutoff 之后先保留为 transition/uncertain，不能直接当作 contact，更不能据此生成 wrong-insertion ground truth。",
        "",
        "## 总结结论",
        "",
        "1. 10 条轨迹都表现出相同的宏观流程：free-space approach → 目标孔附近过渡 → 接触/对准后的继续下压或保持。视频中的 Sim 是对应 Real rollout 的视觉 replay，画面运动大体一致；这说明不能用视频画面本身解释 Sim force 的大幅差异。",
        "2. Real 的接触信息主要表现为时间连续的过程：接触 onset 时力分量重新分配，部分轨迹随后有轻微横向调整，最终插入/末端保持时出现明显轴向峰值。它不是一个单帧 contact bit。",
        "3. Sim 的 `contact_total_base` 在 free-space 基本严格为 0，接触后出现稀疏且明显放大的 impulse；很多轨迹的 Sim 第一非零帧早于人工确认的 Real stable-contact onset，但通常落在 Real 的 transition/uncertain 区间或之前，不能把它直接解释成错误插入。",
        "4. 这 10 条视频没有提供足够可靠的‘插对/插歪’结果标签。能较有把握标的是阶段边界和 terminal force event；除 traj_0 之外，没有一条能仅凭当前视频+力数据安全地标为 wrong insertion。",
        "5. 因此当前数据更适合先构造‘接触后力变化/插入阶段’的时序特征，再寻找或补充真正的 outcome 标签；不建议把这 10 条的 contact onset 直接当作 correct/wrong supervision。",
        "",
        "## 逐条审阅",
        "",
        "|轨迹|free end|人工审阅 onset|Real terminal peak|Sim 第一非零|Sim 峰值|置信度|判定|",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        judgment = "阶段边界可用；wrong 不可判"
        if row["trajectory"] == "traj_0":
            judgment = "有明确 alignment/correction search；仍不是二分类 wrong 标签"
        lines.append(
            f"|{row['trajectory']}|{row['free_end_frame']}|{row['contact_onset_frame']}|{row['real_peak_frame']} ({row['real_peak_norm_N']} N)|{row['sim_first_nonzero_frame']}|{row['sim_peak_frame']} ({row['sim_peak_norm_N']} N)|{row['confidence']}|{judgment}|"
        )
    lines += [
        "",
        "## 解释口径",
        "",
        "- `free_end`: 用户确认的 high-confidence free-space 末帧，包含该帧。",
        "- `contact_onset`: 视频、EE z/xy 运动变化和 Real wrench 时间响应共同支持的首次明显接触/插入转折；它不是自动阈值结果。",
        "- `transition`: free_end 与 contact_onset 之间的帧，默认 near-contact/uncertain。",
        "- `terminal peak`: 后续最大的 Real axial/force event，用于描述插入末端变化，不等同于成功标签。",
        "- `Sim 第一非零`: 仅表示 `contact_total_base` 第一次离开严格零；它不是 Real stable-contact onset，也不是 wrong-insertion onset。",
        "",
        "## 可复查文件",
        "",
        "- 逐轨迹数值表：`representative_trajectory_audit.csv`",
        "- 生成脚本：`scripts/audit_representative_trajectories.py`",
        "- 关键帧和力曲线：`eval_outputs/representative_trajectory_audit/`（本地生成目录，已被 gitignore 忽略）",
        "",
        "## 最终判断",
        "",
        "这 10 条可以支持‘不同阶段的力变化模式’审计，但不能单独支持‘正确/错误插入’监督。当前最可靠的共同信号是：接触 onset 后的力方向/分量重排、持续时间、以及末端轴向峰值；Sim 的幅值和脉冲形态不能直接与 Real 一一对应。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    rows = summarize(repo)
    write_csv(repo / "representative_trajectory_audit.csv", rows)
    write_report(repo / "representative_trajectory_audit.md", rows)
    print("wrote representative_trajectory_audit.csv and representative_trajectory_audit.md")


if __name__ == "__main__":
    main()
