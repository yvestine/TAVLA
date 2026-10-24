#!/usr/bin/env python3
"""Validate motion-derived insertion-quality weak labels before encoder training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from insertion_quality_v1_lib import LABEL_NAMES
from insertion_quality_v1_lib import MOTION_NAMES
from insertion_quality_v1_lib import annotation_arrays
from insertion_quality_v1_lib import fit_quantile_motion_rule
from insertion_quality_v1_lib import history_consistent_labels
from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import predict_weak_labels
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config
from insertion_quality_v1_lib import save_json
from insertion_quality_v1_lib import write_csv


def safe_precision(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    selected = y_pred == label
    if not np.any(selected):
        return float("nan")
    return float(np.mean(y_true[selected] == label))


def safe_recall(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    selected = y_true == label
    if not np.any(selected):
        return float("nan")
    return float(np.mean(y_pred[selected] == label))


def run(repo: Path, config_path: Path, annotations_path: Path, output_dir: Path) -> bool:
    config = read_config(config_path)
    annotations = read_annotations(annotations_path)
    reviewed_ids = [
        int(value) for value in config.get("reviewed_trajectories", config["manual_trajectories"])
    ]
    evaluation_ids = [int(value) for value in config.get("evaluation_trajectories", reviewed_ids)]
    all_ids = [int(value) for value in config["all_trajectories"]]
    free_ends = {int(key): int(value) for key, value in config["free_end_frames"].items()}
    data = load_trajectories(repo, config, all_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, object]] = []
    threshold_rows: dict[str, object] = {}
    aggregate_true: list[np.ndarray] = []
    aggregate_pred: list[np.ndarray] = []
    post_count = 0
    accepted_count = 0
    plot_predictions: dict[int, np.ndarray] = {}

    history = int(config["representation"]["history_frames"])
    for held_out in evaluation_ids:
        train_ids = [trajectory for trajectory in reviewed_ids if trajectory != held_out]
        rule = fit_quantile_motion_rule(data, annotations, train_ids, config)
        weak, motion = predict_weak_labels(rule, data[held_out], config, free_ends[held_out])
        human, terminal, confidence = annotation_arrays(annotations, held_out, len(weak))
        human_window = history_consistent_labels(human, history)
        weak_window = history_consistent_labels(weak, history)
        confidence_window = np.zeros_like(confidence)
        for frame in range(history - 1, len(confidence)):
            confidence_window[frame] = np.min(confidence[frame - history + 1 : frame + 1])
        evaluated = (human_window >= 0) & (confidence_window >= 0.65)
        aggregate_true.append(human_window[evaluated])
        aggregate_pred.append(weak_window[evaluated])
        post = np.arange(len(weak)) > free_ends[held_out]
        post &= terminal < 0.5
        post_count += int(np.sum(post))
        accepted_count += int(np.sum(post & (weak_window >= 0)))
        plot_predictions[held_out] = weak_window
        fold_rows.append(
            {
                "held_out_trajectory": f"traj_{held_out}",
                "n_human_quality_frames": int(np.sum(evaluated)),
                "misaligned_precision": safe_precision(human_window[evaluated], weak_window[evaluated], 0),
                "aligned_precision": safe_precision(human_window[evaluated], weak_window[evaluated], 1),
                "misaligned_recall": safe_recall(human_window[evaluated], weak_window[evaluated], 0),
                "aligned_recall": safe_recall(human_window[evaluated], weak_window[evaluated], 1),
                "post_contact_coverage": float(np.mean(weak_window[post] >= 0)) if np.any(post) else float("nan"),
                "selected_quantile": rule.quantile,
                "rule_type": rule.as_dict().get("rule_type", "quantile_projection"),
                "stall_step_threshold_m": rule.stall_step_threshold,
                "train_precision_misaligned": rule.train_precision_misaligned,
                "train_precision_aligned": rule.train_precision_aligned,
                "train_coverage": rule.train_coverage,
            }
        )
        threshold_rows[f"held_out_traj_{held_out}"] = rule.as_dict()

    y_true = np.concatenate(aggregate_true)
    y_pred = np.concatenate(aggregate_pred)
    aggregate = {
        "misaligned_precision": safe_precision(y_true, y_pred, 0),
        "aligned_precision": safe_precision(y_true, y_pred, 1),
        "misaligned_recall": safe_recall(y_true, y_pred, 0),
        "aligned_recall": safe_recall(y_true, y_pred, 1),
        "human_quality_coverage": float(np.mean(y_pred >= 0)),
        "post_contact_coverage": accepted_count / max(post_count, 1),
        "evaluated_human_frames": int(len(y_true)),
        "reviewed_trajectories": len(reviewed_ids),
        "evaluation_folds": len(evaluation_ids),
    }
    weak_cfg = config["weak_labels"]
    gate_passed = (
        aggregate["misaligned_precision"] >= float(weak_cfg["minimum_precision"])
        and aggregate["aligned_precision"] >= float(weak_cfg["minimum_precision"])
        and aggregate["post_contact_coverage"] >= float(weak_cfg["minimum_post_contact_coverage"])
    )
    aggregate["gate_passed"] = gate_passed
    aggregate["required_precision"] = float(weak_cfg["minimum_precision"])
    aggregate["required_post_contact_coverage"] = float(weak_cfg["minimum_post_contact_coverage"])

    write_csv(output_dir / "weak_label_loto_metrics.csv", fold_rows)
    save_json(output_dir / "weak_label_loto_thresholds.json", threshold_rows)
    save_json(output_dir / "weak_label_gate.json", aggregate)

    # Fit the final diagnostic rule on all ten reviewed trajectories.  The CSV
    # is emitted even if the gate fails, but is marked diagnostic and training
    # code refuses to consume it unless gate_passed=true.
    final_rule = fit_quantile_motion_rule(data, annotations, reviewed_ids, config)
    weak_rows: list[dict[str, object]] = []
    for trajectory_id in all_ids:
        weak, motion = predict_weak_labels(final_rule, data[trajectory_id], config, free_ends[trajectory_id])
        for frame in range(len(weak)):
            weak_rows.append(
                {
                    "trajectory": f"traj_{trajectory_id}",
                    "frame": frame,
                    "label": LABEL_NAMES[int(weak[frame])],
                    "label_id": int(weak[frame]),
                    "source": "future_ee_motion_quantile_rule",
                    "training_eligible": int(gate_passed and weak[frame] >= 0),
                    **{name: float(motion[frame, index]) for index, name in enumerate(MOTION_NAMES)},
                }
            )
    weak_labels_path = repo / str(
        config.get("weak_labels_filename", "annotations/insertion_quality_weak_labels_v1.csv")
    )
    write_csv(weak_labels_path, weak_rows)
    save_json(output_dir / "final_weak_label_rule.json", final_rule.as_dict())

    fig, axes = plt.subplots(len(evaluation_ids), 1, figsize=(15, 2.0 * len(evaluation_ids)), sharex=False)
    if len(evaluation_ids) == 1:
        axes = [axes]
    color = {-1: "#b8b8b8", 0: "#d62728", 1: "#2ca02c"}
    for axis, trajectory_id in zip(axes, evaluation_ids):
        n_frames = len(data[trajectory_id].pose)
        human, terminal, _ = annotation_arrays(annotations, trajectory_id, n_frames)
        weak = plot_predictions[trajectory_id]
        for frame in range(n_frames):
            axis.axvspan(frame - 0.5, frame + 0.5, ymin=0.52, ymax=0.96, color=color[int(human[frame])], alpha=0.9)
            axis.axvspan(frame - 0.5, frame + 0.5, ymin=0.04, ymax=0.48, color=color[int(weak[frame])], alpha=0.9)
        terminal_frames = np.flatnonzero(terminal > 0.5)
        if len(terminal_frames):
            axis.axvspan(terminal_frames[0], terminal_frames[-1], color="#9467bd", alpha=0.22)
        axis.axvline(free_ends[trajectory_id], color="black", linestyle="--", linewidth=0.8)
        axis.set_xlim(0, n_frames - 1)
        axis.set_yticks([0.25, 0.75])
        axis.set_yticklabels(["weak", "human"])
        axis.set_title(f"traj_{trajectory_id}", loc="left", fontsize=9)
    axes[-1].set_xlabel("frame")
    fig.suptitle("Insertion quality: human intervals (top) vs LOTO motion weak labels (bottom)")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_dir / "weak_label_loto_coverage.png", dpi=180)
    plt.close(fig)

    report = [
        "# Insertion Quality V1 — weak-label gate",
        "",
        "## 结论",
        "",
        f"**Gate: {'PASS' if gate_passed else 'FAIL'}**。该闸门只验证由未来 EE 运动产生的标签；force 没有参与标签生成。",
        "",
        f"- Misaligned precision: `{aggregate['misaligned_precision']:.3f}`（要求 ≥ `{aggregate['required_precision']:.2f}`）",
        f"- Aligned precision: `{aggregate['aligned_precision']:.3f}`（要求 ≥ `{aggregate['required_precision']:.2f}`）",
        f"- Post-contact coverage: `{aggregate['post_contact_coverage']:.3f}`（要求 ≥ `{aggregate['required_post_contact_coverage']:.2f}`）",
        f"- Misaligned recall: `{aggregate['misaligned_recall']:.3f}`",
        f"- Aligned recall: `{aggregate['aligned_recall']:.3f}`",
        "",
        "## 方法",
        "",
        f"每个 LOTO fold 使用其余 `{len(reviewed_ids) - 1}` 条人工审阅轨迹拟合 motion rule，并在 `{len(evaluation_ids)}` 条同时包含两类有效窗口的轨迹上评估。规则使用 `future_z_progress / future_xy_path / stall / rebound / resumed progress`，先做 train-only robust scaling 和轻量 Logistic 判别，再做配置化的因果概率平滑；两个达到 train-only precision 要求的 acceptance threshold 之间全部保留为 uncertain。held-out 轨迹不参与拟合。",
        "",
        "`free` 只用于 baseline/noise；`terminal` 从 quality 弱标签中剔除。低置信度人工区间不参与 gate precision 统计。",
        "",
        "## 限制",
        "",
        "这里的 precision 只在有人工 binary quality 标签的帧上可计算；人工 uncertain 区间没有真值。即便 Gate 通过，它也只说明该规则值得用于下一阶段弱监督，不代表 40 条数据获得了物理 ground truth。",
        "",
        "## 产物",
        "",
        "- `weak_label_loto_metrics.csv`",
        "- `weak_label_loto_thresholds.json`",
        "- `weak_label_gate.json`",
        "- `weak_label_loto_coverage.png`",
        f"- `{weak_labels_path.relative_to(repo)}`",
    ]
    (output_dir / "weak_label_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return gate_passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_v1.json"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/insertion_quality_labels_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/insertion_quality_v1"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    annotations = args.annotations if args.annotations.is_absolute() else repo / args.annotations
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, config, annotations, output_dir)


if __name__ == "__main__":
    main()
