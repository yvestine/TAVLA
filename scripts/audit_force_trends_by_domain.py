#!/usr/bin/env python3
"""Audit whether Real and Sim force trends separately correlate with quality labels.

This is a diagnostic only.  It never trains a model and never compares Real and
Sim force magnitudes directly.  The same reviewed intervals are evaluated
inside each domain so that the question is: does this domain's force trend
contain information about misaligned versus aligned progress?
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config


TARGET_LABELS = ("misaligned_contact", "aligned_progress")
FEATURES = (
    "norm_median",
    "norm_p95",
    "lateral_ratio_median",
    "lateral_ratio_p95",
    "axial_fraction_median",
    "axial_fraction_p95",
    "force_delta_median",
    "force_delta_p95",
    "direction_delta_median",
    "direction_delta_p95",
    "trend_norm_median",
    "trend_lateral_median",
    "trend_axial_median",
    "lateral_reversals",
)


def causal_mean(values: np.ndarray, width: int) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    for index in range(len(values)):
        start = max(0, index - width + 1)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return result


def robust_percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def count_sign_reversals(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    signs = np.sign(values)
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0


def force_features(force: np.ndarray, baseline: np.ndarray | None) -> dict[str, np.ndarray]:
    residual = force - baseline if baseline is not None else force.copy()
    smooth = np.column_stack([causal_mean(residual[:, axis], 3) for axis in range(3)])
    norm = np.linalg.norm(residual, axis=1)
    lateral = np.linalg.norm(residual[:, :2], axis=1)
    axial = np.abs(residual[:, 2])
    denominator = np.maximum(norm, 1e-8)
    lateral_ratio = lateral / denominator
    axial_fraction = axial / denominator
    force_delta = np.zeros(len(residual), dtype=np.float64)
    if len(residual) > 1:
        force_delta[1:] = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
    direction_delta = np.zeros(len(residual), dtype=np.float64)
    valid = norm > 1e-8
    unit = residual / denominator[:, None]
    if len(residual) > 1:
        both_valid = valid[1:] & valid[:-1]
        cosine = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
        direction_delta[1:][both_valid] = np.arccos(cosine[both_valid])
    smooth_norm = np.linalg.norm(smooth, axis=1)
    smooth_lateral = np.linalg.norm(smooth[:, :2], axis=1)
    smooth_axial = np.abs(smooth[:, 2])
    lag = 5
    trend_norm = np.zeros(len(residual), dtype=np.float64)
    trend_lateral = np.zeros(len(residual), dtype=np.float64)
    trend_axial = np.zeros(len(residual), dtype=np.float64)
    if len(residual) > lag:
        trend_norm[lag:] = smooth_norm[lag:] - smooth_norm[:-lag]
        trend_lateral[lag:] = smooth_lateral[lag:] - smooth_lateral[:-lag]
        trend_axial[lag:] = smooth_axial[lag:] - smooth_axial[:-lag]
    return {
        "norm": norm,
        "lateral_ratio": lateral_ratio,
        "axial_fraction": axial_fraction,
        "force_delta": force_delta,
        "direction_delta": direction_delta,
        "trend_norm": trend_norm,
        "trend_lateral": trend_lateral,
        "trend_axial": trend_axial,
        "fx": residual[:, 0],
        "fy": residual[:, 1],
    }


def summarize_segment(features: dict[str, np.ndarray], start: int, end: int) -> dict[str, float]:
    sl = slice(max(0, start), min(len(next(iter(features.values()))) - 1, end) + 1)
    result = {
        "norm_median": float(np.median(features["norm"][sl])),
        "norm_p95": robust_percentile(features["norm"][sl], 0.95),
        "lateral_ratio_median": float(np.median(features["lateral_ratio"][sl])),
        "lateral_ratio_p95": robust_percentile(features["lateral_ratio"][sl], 0.95),
        "axial_fraction_median": float(np.median(features["axial_fraction"][sl])),
        "axial_fraction_p95": robust_percentile(features["axial_fraction"][sl], 0.95),
        "force_delta_median": float(np.median(features["force_delta"][sl])),
        "force_delta_p95": robust_percentile(features["force_delta"][sl], 0.95),
        "direction_delta_median": float(np.median(features["direction_delta"][sl])),
        "direction_delta_p95": robust_percentile(features["direction_delta"][sl], 0.95),
        "trend_norm_median": float(np.median(features["trend_norm"][sl])),
        "trend_lateral_median": float(np.median(features["trend_lateral"][sl])),
        "trend_axial_median": float(np.median(features["trend_axial"][sl])),
        "lateral_reversals": float(
            count_sign_reversals(features["fx"][sl]) + count_sign_reversals(features["fy"][sl])
        ),
    }
    return result


def auc_for_misaligned(values: np.ndarray, labels: np.ndarray) -> float:
    positive = values[labels == "misaligned_contact"]
    negative = values[labels == "aligned_progress"]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )


def paired_rows(rows: list[dict[str, object]], domain: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["domain"] == domain and row["label"] in TARGET_LABELS:
            grouped.setdefault((str(row["trajectory"]), str(row["label"])), []).append(row)
    result: list[dict[str, object]] = []
    for trajectory in sorted({key[0] for key in grouped}, key=lambda value: int(value.split("_")[-1])):
        mis = grouped.get((trajectory, "misaligned_contact"), [])
        aligned = grouped.get((trajectory, "aligned_progress"), [])
        if not mis or not aligned:
            continue
        output: dict[str, object] = {"trajectory": trajectory, "domain": domain}
        for feature in FEATURES:
            mis_value = float(np.median([float(item[feature]) for item in mis]))
            aligned_value = float(np.median([float(item[feature]) for item in aligned]))
            output[f"{feature}_misaligned"] = mis_value
            output[f"{feature}_aligned"] = aligned_value
            output[f"{feature}_diff"] = mis_value - aligned_value
        result.append(output)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_v3_transformer.json"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/force_trend_domain_audit"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    annotations_path = args.annotations if args.annotations.is_absolute() else repo / args.annotations
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = read_config(config_path)
    annotations = read_annotations(annotations_path)
    all_ids = [int(value) for value in config["all_trajectories"]]
    data = load_trajectories(repo, config, all_ids)
    free_ends = {int(key): int(value) for key, value in config["free_end_frames"].items()}
    segment_rows: list[dict[str, object]] = []
    for trajectory in all_ids:
        item = data[trajectory]
        for domain, force in (("Real", item.real_force), ("Sim", item.sim_force)):
            baseline = np.median(force[: free_ends[trajectory] + 1], axis=0) if domain == "Real" else None
            features = force_features(force, baseline)
            for annotation in annotations:
                if int(annotation["trajectory_id"]) != trajectory or str(annotation["label"]) not in TARGET_LABELS:
                    continue
                start = max(0, int(annotation["start_frame"]))
                end = min(len(force) - 1, int(annotation["end_frame"]))
                summary = summarize_segment(features, start, end)
                segment_rows.append(
                    {
                        "trajectory": f"traj_{trajectory}",
                        "domain": domain,
                        "label": str(annotation["label"]),
                        "confidence": str(annotation["confidence"]),
                        "start_frame": start,
                        "end_frame": end,
                        **summary,
                    }
                )
    write_csv(output / "segment_force_trends.csv", segment_rows)

    stats: list[dict[str, object]] = []
    for domain in ("Real", "Sim"):
        domain_rows = [row for row in segment_rows if row["domain"] == domain]
        paired = paired_rows(segment_rows, domain)
        for feature in FEATURES:
            values = np.asarray([float(row[feature]) for row in domain_rows], dtype=np.float64)
            labels = np.asarray([str(row["label"]) for row in domain_rows])
            mis = values[labels == "misaligned_contact"]
            aligned = values[labels == "aligned_progress"]
            diffs = np.asarray([float(row[f"{feature}_diff"]) for row in paired], dtype=np.float64)
            stats.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "n_misaligned_segments": len(mis),
                    "n_aligned_segments": len(aligned),
                    "n_paired_trajectories": len(paired),
                    "misaligned_median": float(np.median(mis)) if len(mis) else float("nan"),
                    "aligned_median": float(np.median(aligned)) if len(aligned) else float("nan"),
                    "median_difference": float(np.median(mis) - np.median(aligned)) if len(mis) and len(aligned) else float("nan"),
                    "segment_auc_misaligned_high": auc_for_misaligned(values, labels),
                    "paired_median_difference": float(np.median(diffs)) if len(diffs) else float("nan"),
                    "paired_misaligned_higher_fraction": float(np.mean(diffs > 0)) if len(diffs) else float("nan"),
                }
            )
        write_csv(output / f"paired_trajectory_differences_{domain.lower()}.csv", paired)
    write_csv(output / "feature_statistics.csv", stats)

    selected = ("lateral_ratio_median", "axial_fraction_median", "force_delta_p95", "direction_delta_p95", "trend_norm_median")
    fig, axes = plt.subplots(2, len(selected), figsize=(19, 7), sharey=False)
    for row_index, domain in enumerate(("Real", "Sim")):
        domain_rows = [row for row in segment_rows if row["domain"] == domain]
        for col_index, feature in enumerate(selected):
            groups = [
                [float(row[feature]) for row in domain_rows if row["label"] == "misaligned_contact"],
                [float(row[feature]) for row in domain_rows if row["label"] == "aligned_progress"],
            ]
            axes[row_index, col_index].boxplot(groups, labels=["mis", "aligned"], showfliers=False)
            axes[row_index, col_index].set_title(feature, fontsize=9)
            axes[row_index, col_index].grid(alpha=0.2)
        axes[row_index, 0].set_ylabel(domain)
    fig.suptitle("Force-trend distributions evaluated separately in Real and Sim")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "force_trend_distributions_by_domain.png", dpi=180)
    plt.close(fig)

    representative = [0, 3, 8, 14, 18, 19, 25, 30, 32, 39]
    fig, axes = plt.subplots(len(representative), 2, figsize=(15, 2.0 * len(representative)), sharex=False)
    for row_index, trajectory in enumerate(representative):
        item = data[trajectory]
        for col_index, (domain, force) in enumerate((("Real", item.real_force), ("Sim", item.sim_force))):
            baseline = np.median(force[: free_ends[trajectory] + 1], axis=0) if domain == "Real" else None
            values = force_features(force, baseline)
            time = item.timestamps - item.timestamps[0]
            axes[row_index, col_index].plot(time, values["lateral_ratio"], label="lateral ratio", lw=1.0)
            axes[row_index, col_index].plot(time, values["axial_fraction"], label="axial fraction", lw=1.0)
            axes[row_index, col_index].plot(time, np.clip(values["force_delta"] / (np.quantile(values["force_delta"], 0.95) + 1e-8), 0, 3), label="delta norm / p95", lw=0.8)
            for annotation in annotations:
                if int(annotation["trajectory_id"]) != trajectory:
                    continue
                if str(annotation["label"]) in TARGET_LABELS:
                    color = "tab:red" if annotation["label"] == "misaligned_contact" else "tab:green"
                    axes[row_index, col_index].axvspan(time[int(annotation["start_frame"])], time[min(int(annotation["end_frame"]), len(time) - 1)], color=color, alpha=0.10)
            axes[row_index, col_index].axvline(time[free_ends[trajectory]], color="black", ls="--", lw=0.7)
            axes[row_index, col_index].set_ylim(-0.05, 3.05)
            axes[row_index, col_index].set_title(f"traj_{trajectory} {domain}", fontsize=9)
            axes[row_index, col_index].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=3)
    fig.suptitle("Separate-domain force trends; red=misaligned, green=aligned")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output / "representative_force_trends_by_domain.png", dpi=160)
    plt.close(fig)

    report_lines = [
        "# Separate-domain force trend audit",
        "",
        "This is a diagnostic, not a trained classifier. Real and Sim are analyzed separately using the same reviewed quality intervals. Sim labels are transferred from the paired rollout only as a diagnostic target; they are not independent Sim ground truth.",
        "",
        "## Definitions",
        "",
        "- Real force subtracts each trajectory's free-space vector median; Sim uses `sim/force/contact_total_base` without Real baseline subtraction.",
        "- `lateral_ratio = ||Fxy|| / (||F|| + eps)` and `axial_fraction = |Fz| / (||F|| + eps)`.",
        "- `force_delta` is the causal-smoothed vector difference; `trend_*` is a 5-frame change.",
        "- Red intervals are `misaligned_contact`; green intervals are `aligned_progress`.",
        "",
        "## Interpretation rule",
        "",
        "A force feature is considered a candidate signal only if its segment-level AUC and paired trajectory direction are both reasonably consistent. A difference in pooled medians alone is not enough because windows within one trajectory are correlated.",
        "",
    ]
    for domain in ("Real", "Sim"):
        report_lines.extend([f"## {domain}", ""])
        domain_stats = [row for row in stats if row["domain"] == domain]
        ranked = sorted(domain_stats, key=lambda row: abs(float(row["segment_auc_misaligned_high"]) - 0.5) if math.isfinite(float(row["segment_auc_misaligned_high"])) else -1, reverse=True)
        for row in ranked[:6]:
            report_lines.append(
                f"- `{row['feature']}`: AUC(misaligned-high)={float(row['segment_auc_misaligned_high']):.3f}, paired median diff={float(row['paired_median_difference']):.4g}, misaligned-higher in {float(row['paired_misaligned_higher_fraction']):.1%} of {int(row['n_paired_trajectories'])} paired trajectories."
            )
        report_lines.append("")
    report_lines.extend([
        "## Limitations",
        "",
        "1. The quality intervals were reviewed using video, force, and pose evidence; this audit tests correlation with force trends, not force-only ground truth.",
        "2. The same Real-derived phase intervals are applied to Sim only because the rollouts share the trajectory timeline. This does not establish Sim semantic labels.",
        "3. This audit does not justify a shared Real/Sim encoder. It only asks whether each domain individually contains a usable force trend.",
        "",
        "## Outputs",
        "",
        "- `feature_statistics.csv`",
        "- `segment_force_trends.csv`",
        "- `paired_trajectory_differences_real.csv` and `paired_trajectory_differences_sim.csv`",
        "- `force_trend_distributions_by_domain.png`",
        "- `representative_force_trends_by_domain.png`",
    ])
    (output / "force_trend_domain_audit.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {output / 'force_trend_domain_audit.md'}")


if __name__ == "__main__":
    main()
