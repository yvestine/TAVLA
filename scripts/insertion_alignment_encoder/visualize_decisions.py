#!/usr/bin/env python3
"""Create a direct, held-out view of encoder decisions and errors.

Writes the same replaceable PDF as visualize_pairs.py. Page one compares the
label and Real/Sim predictions on one held-out trajectory. Page two summarizes
held-out confusion matrices and paired Real-vs-Sim probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from .visualize_pairs import (
    PHASE_COLORS,
    REAL_COLOR,
    SIM_COLOR,
    _summaries,
    read_labels,
)


LABEL_TO_ID = {"misaligned_contact": 0, "aligned_progress": 1}
LABEL_NAMES = ["Misaligned contact", "Aligned insertion"]
LABEL_COLORS = [PHASE_COLORS["misaligned_contact"], PHASE_COLORS["aligned_progress"]]
EVAL_BOUNDARY_GUARD = 5


def _evaluation_mask(trajectory: int, entry, intervals) -> tuple[np.ndarray, np.ndarray]:
    """Match the held-out windows used by training (two-class labels, ±5 guard, cutoff)."""
    n = len(entry["frames"])
    target = np.full(n, -1, dtype=np.int8)
    mask = np.zeros(n, dtype=bool)
    cutoff = int(entry["cutoff"])
    for row in intervals[trajectory]:
        label = str(row["label"])
        if label not in LABEL_TO_ID:
            continue
        start = int(row["start"]) + EVAL_BOUNDARY_GUARD
        end = min(int(row["end"]) - EVAL_BOUNDARY_GUARD, cutoff - 1)
        if end < start:
            continue
        target[start : end + 1] = LABEL_TO_ID[label]
        mask[start : end + 1] = True
    # The runtime waits for a complete causal force history before returning a prediction.
    mask &= entry["valid"]
    for domain in entry["domains"]:
        mask &= np.isfinite(domain["probability"]).all(axis=1)
    return mask, target


def _predict(probability: np.ndarray) -> np.ndarray:
    return np.asarray(probability).argmax(axis=1)


def _class_stats(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    if not len(truth):
        return float("nan"), float("nan"), float("nan")
    f1s = []
    for class_id in (0, 1):
        tp = np.sum((truth == class_id) & (prediction == class_id))
        fp = np.sum((truth != class_id) & (prediction == class_id))
        fn = np.sum((truth == class_id) & (prediction != class_id))
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom else 0.0)
    return float(np.mean(truth == prediction)), f1s[0], f1s[1]


def _shade_labels(axis, intervals, cutoff: int) -> None:
    for row in intervals:
        color = PHASE_COLORS.get(str(row["label"]))
        if color:
            axis.axvspan(int(row["start"]) - 0.5, int(row["end"]) + 0.5,
                         color=color, alpha=0.10, linewidth=0)
    axis.axvline(cutoff - 0.5, color="#333333", linestyle="--", linewidth=1.2)


def _page_example(pdf, trajectory: int, entry, intervals, payload) -> None:
    cutoff = min(int(entry["cutoff"]), len(entry["frames"]))
    x = np.arange(cutoff)
    mask, truth = _evaluation_mask(trajectory, entry, intervals)
    pred_real = _predict(entry["domains"][0]["probability"])
    pred_sim = _predict(entry["domains"][1]["probability"])
    plot_mask = mask[:cutoff]
    plot_truth = truth[:cutoff]
    valid_frames = np.flatnonzero(plot_mask)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.25, 1.0, 0.75])
    ax_force = fig.add_subplot(gs[0, 0])
    ax_prob = fig.add_subplot(gs[1, 0], sharex=ax_force)
    ax_state = fig.add_subplot(gs[2, 0], sharex=ax_force)

    # Show lateral residual force in physical units, rather than normalized model features.
    for domain_index, domain_name, color, style in (
        (0, "Real", REAL_COLOR, "-"), (1, "Sim", SIM_COLOR, "--")
    ):
        force = entry["domains"][domain_index]["force"]
        baseline = np.median(force[: int(payload["config"]["baseline_frames"])], axis=0)
        residual = force - baseline
        ax_force.plot(x, residual[:cutoff, 0], color=color, linestyle=style,
                      linewidth=1.5, label=f"{domain_name} Fx")
        ax_force.plot(x, residual[:cutoff, 1], color=color, linestyle=style,
                      linewidth=1.2, alpha=0.68, label=f"{domain_name} Fy")
    ax_force.axhline(0, color="#777777", linewidth=0.7)
    ax_force.set_ylabel("Lateral force change (N)")
    ax_force.set_title("Force input trend: solid = Real, dashed = Sim")
    ax_force.legend(ncol=4, fontsize=9, loc="upper right")
    _shade_labels(ax_force, intervals[trajectory], cutoff)

    for domain_index, name, color, style in (
        (0, "Real", REAL_COLOR, "-"), (1, "Sim", SIM_COLOR, "--")
    ):
        probability = entry["domains"][domain_index]["probability"][:cutoff, 1]
        ax_prob.plot(x, probability, color=color, linestyle=style, linewidth=1.8,
                     label=f"{name} P(aligned)")
    ax_prob.axhline(0.5, color="#333333", linestyle=":", linewidth=1.0,
                    label="Decision threshold = 0.5")
    ax_prob.set_ylim(-0.04, 1.04)
    ax_prob.set_ylabel("P(aligned)")
    ax_prob.legend(ncol=3, fontsize=9, loc="upper right")
    _shade_labels(ax_prob, intervals[trajectory], cutoff)

    state_rows = np.full((3, cutoff), -1, dtype=np.int8)
    state_rows[0, plot_mask] = plot_truth[plot_mask]
    state_rows[1, plot_mask] = pred_real[:cutoff][plot_mask]
    state_rows[2, plot_mask] = pred_sim[:cutoff][plot_mask]
    state_cmap = ListedColormap(["#e7e7e7", LABEL_COLORS[0], LABEL_COLORS[1]])
    ax_state.imshow(state_rows, origin="upper", aspect="auto", interpolation="nearest",
                    extent=(-0.5, cutoff - 0.5, 2.5, -0.5), cmap=state_cmap, vmin=-1, vmax=1)
    ax_state.set_yticks([0, 1, 2], ["Label", "Real prediction", "Sim prediction"])
    ax_state.set_xlabel("Frame (10 Hz)")
    ax_state.set_xlim(-0.5, cutoff - 0.5)
    ax_state.set_xticks(np.arange(0, cutoff, 25))
    ax_state.grid(axis="x", color="white", linewidth=0.5, alpha=0.7)

    # Mark every wrong prediction with a black cross on the corresponding prediction row.
    wrong_real = valid_frames[pred_real[valid_frames] != truth[valid_frames]]
    wrong_sim = valid_frames[pred_sim[valid_frames] != truth[valid_frames]]
    ax_state.scatter(wrong_real, np.ones(len(wrong_real)), marker="x", s=25, linewidths=1.3,
                     color="#111111", zorder=4)
    ax_state.scatter(wrong_sim, np.full(len(wrong_sim), 2), marker="x", s=25, linewidths=1.3,
                     color="#111111", zorder=4)
    ax_state.legend(handles=[
        Patch(facecolor=LABEL_COLORS[0], label=LABEL_NAMES[0]),
        Patch(facecolor=LABEL_COLORS[1], label=LABEL_NAMES[1]),
        Patch(facecolor="#e7e7e7", label="Not evaluated / history unavailable"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="Incorrect prediction"),
    ], ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)

    _, real_mis_f1, real_aligned_f1 = _class_stats(plot_truth[plot_mask], pred_real[:cutoff][plot_mask])
    _, sim_mis_f1, sim_aligned_f1 = _class_stats(plot_truth[plot_mask], pred_sim[:cutoff][plot_mask])
    fig.suptitle(
        f"Held-out trajectory {trajectory}: predictions vs labels  |  "
        f"Real F1 (misaligned/aligned)={real_mis_f1:.2f}/{real_aligned_f1:.2f}, "
        f"Sim={sim_mis_f1:.2f}/{sim_aligned_f1:.2f}, evaluated frames={int(mask.sum())}",
        fontsize=14,
    )
    pdf.savefig(fig, dpi=170)
    plt.close(fig)


def _draw_confusion(axis, matrix, title, metrics, cmap) -> None:
    matrix = np.asarray(matrix, dtype=int)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals > 0)
    axis.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    axis.set_xticks([0, 1], LABEL_NAMES)
    axis.set_yticks([0, 1], LABEL_NAMES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Annotated label")
    axis.set_title(
        f"{title}  |  Macro-F1 {metrics['macro_f1']:.3f}\n"
        f"Misaligned F1 {metrics['per_class']['misaligned_contact']['f1']:.3f}  ·  "
        f"Aligned F1 {metrics['per_class']['aligned_progress']['f1']:.3f}"
    )
    for i in range(2):
        for j in range(2):
            axis.text(j, i, f"{matrix[i, j]} frames\n{normalized[i, j]:.0%}",
                      ha="center", va="center", fontsize=11,
                      color="white" if normalized[i, j] > 0.55 else "#222222")


def _page_summary(pdf, data, intervals, payload) -> None:
    test = payload["metrics"]["held_out_test"]
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.3])
    ax_real = fig.add_subplot(gs[0, 0])
    ax_sim = fig.add_subplot(gs[1, 0])
    ax_pair = fig.add_subplot(gs[:, 1])
    _draw_confusion(ax_real, test["real"]["class"]["confusion_matrix"], "Real held-out set",
                    test["real"]["class"], "Blues")
    _draw_confusion(ax_sim, test["sim"]["class"]["confusion_matrix"], "Sim held-out set",
                    test["sim"]["class"], "Oranges")

    true_values, real_values, sim_values = [], [], []
    for trajectory, entry in sorted(data.items()):
        mask, truth = _evaluation_mask(trajectory, entry, intervals)
        true_values.extend(truth[mask].tolist())
        real_values.extend(entry["domains"][0]["probability"][mask, 1].tolist())
        sim_values.extend(entry["domains"][1]["probability"][mask, 1].tolist())
    truth = np.asarray(true_values, dtype=np.int8)
    real_p = np.asarray(real_values, dtype=np.float32)
    sim_p = np.asarray(sim_values, dtype=np.float32)
    for class_id, name, color in ((0, LABEL_NAMES[0], LABEL_COLORS[0]),
                                  (1, LABEL_NAMES[1], LABEL_COLORS[1])):
        selection = truth == class_id
        ax_pair.scatter(real_p[selection], sim_p[selection], s=24, alpha=0.52,
                        color=color, label=f"{name} ({int(selection.sum())} frames)",
                        edgecolors="none")
    ax_pair.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1,
                 label="Real and Sim agree")
    ax_pair.axvline(0.5, color="#888888", linestyle=":", linewidth=0.9)
    ax_pair.axhline(0.5, color="#888888", linestyle=":", linewidth=0.9)
    ax_pair.set_xlim(-0.03, 1.03)
    ax_pair.set_ylim(-0.03, 1.03)
    ax_pair.set_aspect("equal", adjustable="box")
    ax_pair.set_xlabel("Real P(aligned)")
    ax_pair.set_ylabel("Sim P(aligned)")
    # Probability threshold is 0.5; class 1 is aligned_progress.
    agreement = float(np.mean((real_p >= 0.5) == (sim_p >= 0.5))) if len(real_p) else float("nan")
    ax_pair.set_title(
        f"Paired Real and Sim predictions at the same trajectory and frame "
        f"({len(real_p)} held-out test frames)\n"
        f"Decision agreement: {agreement:.1%}. Upper right = both aligned; "
        "lower left = both misaligned; other quadrants = disagreement."
    )
    ax_pair.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.suptitle("Does the encoder identify insertion state? Held-out trajectories only", fontsize=15)
    pdf.savefig(fig, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/paired_latent_inspection.pdf"))
    args = parser.parse_args()

    plt.rcParams["font.family"] = "DejaVu Sans"
    payload = torch.load(args.artifact, map_location="cpu", weights_only=True)
    labels = read_labels(args.labels)
    _, data = _summaries(payload, labels, (args.real_root, args.sim_root), args.force_key, heldout_only=True)
    if args.trajectory not in data:
        raise ValueError(f"trajectory {args.trajectory} is not in the held-out test set")

    valid_count = 0
    for trajectory, entry in data.items():
        mask, _ = _evaluation_mask(trajectory, entry, labels)
        valid_count += int(mask.sum())
    expected = int(payload["metrics"]["test_pair_windows"])
    if valid_count != expected:
        raise RuntimeError(
            f"held-out visualization has {valid_count} evaluated frames, but artifact reports {expected}; "
            "check the saved boundary guard and annotations"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        _page_example(pdf, args.trajectory, data[args.trajectory], labels, payload)
        _page_summary(pdf, data, labels, payload)
    print(f"Wrote {args.output} (2 pages; trajectory {args.trajectory}; {valid_count} held-out paired frames)")
    for domain in ("real", "sim"):
        metrics = payload["metrics"]["held_out_test"][domain]["class"]
        print(
            f"{domain}: macro_f1={metrics['macro_f1']:.3f}, "
            f"misaligned_f1={metrics['per_class']['misaligned_contact']['f1']:.3f}, "
            f"aligned_f1={metrics['per_class']['aligned_progress']['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
