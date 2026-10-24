#!/usr/bin/env python3
"""Visualize paired Real/Sim force histories and frozen-encoder representations.

Writes one two-page PDF, overwriting the same path each run. Page one explains
one trajectory through force input, classifier output, 32 latent channels, and
same-frame cosine. Page two compares held-out paired similarity with phase-
matched unpaired baselines and summarizes all trajectory pairs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import torch

from .model import ForceAlignmentEncoder
from .preprocess import force_history_features, initial_baseline, normalize_features


PHASE_COLORS = {"misaligned_contact": "#d95f5f", "aligned_progress": "#48a868"}
REAL_COLOR = "#2369bd"
SIM_COLOR = "#e07a22"


def read_labels(path: Path) -> dict[int, list[dict[str, object]]]:
    rows: dict[int, list[dict[str, object]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            trajectory = int(row["trajectory"].removeprefix("traj_"))
            rows.setdefault(trajectory, []).append({
                "start": int(row["start_frame"]),
                "end": int(row["end_frame"]),
                "label": row["label"],
            })
    return rows


def _load_domain(root: Path, trajectory: int, force_key: str):
    path = root / f"traj_{trajectory}" / "data.h5"
    with h5py.File(path, "r") as h5:
        force = np.asarray(h5[force_key], dtype=np.float32)
        time = np.asarray(h5["timestamps"], dtype=np.float64)
    if force.ndim != 2 or force.shape[1] != 3 or len(time) != len(force):
        raise ValueError(f"{path}: expected force (T,3) and matching timestamps")
    return force, time


def _infer_episode(force, timestamps, model, payload):
    baseline = initial_baseline(force, int(payload["config"]["baseline_frames"]))
    feature = force_history_features(force, timestamps, baseline)
    feature = normalize_features(feature, np.asarray(payload["preprocess"]["feature_scales"], dtype=np.float32))
    history = int(payload["config"]["history_length"])
    count = max(0, len(force) - history + 1)
    x = np.stack([feature[end - history + 1:end + 1] for end in range(history - 1, len(force))]) if count else np.empty((0, history, 6), np.float32)
    latent = np.full((len(force), int(payload["config"]["latent_dim"])), np.nan, dtype=np.float32)
    probability = np.full((len(force), 2), np.nan, dtype=np.float32)
    if count:
        zs, ps = [], []
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                z, logits, _ = model(torch.from_numpy(x[start:start + 512]))
                zs.append(z.numpy())
                ps.append(torch.softmax(logits, dim=-1).numpy())
        latent[history - 1:] = np.concatenate(zs)
        probability[history - 1:] = np.concatenate(ps)
    return feature, latent, probability


def _phase_at_frames(frames: np.ndarray, intervals: list[dict[str, object]]) -> np.ndarray:
    phase = np.full(len(frames), "other", dtype=object)
    for row in intervals:
        label = str(row["label"])
        if label not in PHASE_COLORS:
            continue
        mask = (frames >= int(row["start"])) & (frames <= int(row["end"]))
        phase[mask] = label
    return phase


def _summaries(payload, labels, roots, force_key, heldout_only=True):
    model = ForceAlignmentEncoder(latent_dim=int(payload["config"]["latent_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    ids = sorted(labels)
    heldout = set(payload["metrics"]["held_out_trajectories"])
    if heldout_only:
        ids = [t for t in ids if t in heldout]
    data = {}
    for trajectory in ids:
        domains = []
        times = []
        for root in roots:
            force, timestamp = _load_domain(root, trajectory, force_key)
            feature, latent, probability = _infer_episode(force, timestamp, model, payload)
            domains.append({"force": force, "feature": feature, "latent": latent, "probability": probability})
            times.append(timestamp)
        if len(domains[0]["force"]) != len(domains[1]["force"]):
            raise ValueError(f"traj_{trajectory}: Real/Sim force lengths differ")
        time_error = float(np.max(np.abs(times[0] - times[1])))
        if time_error > 1e-8:
            raise ValueError(f"traj_{trajectory}: timestamps are not aligned (max difference {time_error})")
        cutoff = int(payload["geometry"]["first_hit_cutoff_frames"][f"traj_{trajectory}"])
        frames = np.arange(len(domains[0]["force"]))
        phase = _phase_at_frames(frames, labels[trajectory])
        valid = (frames < cutoff) & np.isfinite(domains[0]["latent"]).all(axis=1)
        latent_real = domains[0]["latent"]
        latent_sim = domains[1]["latent"]
        cosine = np.full(len(frames), np.nan, dtype=np.float32)
        denom = np.linalg.norm(latent_real, axis=1) * np.linalg.norm(latent_sim, axis=1)
        cosine[valid] = np.sum(latent_real[valid] * latent_sim[valid], axis=1) / np.maximum(denom[valid], 1e-8)
        agreement = np.full(len(frames), np.nan, dtype=np.float32)
        agreement[valid] = (
            domains[0]["probability"][valid].argmax(axis=1)
            == domains[1]["probability"][valid].argmax(axis=1)
        ).astype(np.float32)
        data[trajectory] = {
            "domains": domains, "frames": frames, "phase": phase, "cutoff": cutoff,
            "valid": valid, "cosine": cosine, "agreement": agreement,
            "timestamps": times[0], "heldout": trajectory in heldout,
        }
    return model, data


def _matched_distributions(data: dict[int, dict[str, object]], seed: int = 17):
    rng = np.random.default_rng(seed)
    paired: list[float] = []
    shifted: list[float] = []
    cross_trajectory: list[float] = []
    for trajectory, entry in data.items():
        phase = entry["phase"]
        valid = entry["valid"]
        frames = entry["frames"]
        zr = entry["domains"][0]["latent"]
        zs = entry["domains"][1]["latent"]
        for class_name in PHASE_COLORS:
            indices = np.flatnonzero(valid & (phase == class_name))
            if len(indices):
                paired.extend(entry["cosine"][indices].tolist())
            if len(indices) > 5:
                candidates = []
                for _ in range(min(1500, 4 * len(indices))):
                    first, second = rng.choice(indices, 2, replace=False)
                    if abs(int(frames[first]) - int(frames[second])) >= 5:
                        candidates.append(float(_cosine(zr[first], zs[second])))
                shifted.extend(candidates)
            other_ids = [key for key in data if key != trajectory]
            for _ in range(min(1500, 4 * len(indices))):
                if not len(indices):
                    break
                other_id = int(rng.choice(other_ids))
                other = data[other_id]
                other_indices = np.flatnonzero(other["valid"] & (other["phase"] == class_name))
                if not len(other_indices):
                    continue
                i = int(rng.choice(indices))
                j = int(rng.choice(other_indices))
                cross_trajectory.append(float(_cosine(zr[i], other["domains"][1]["latent"][j])))
    return {
        "same_frame_paired": np.asarray(paired),
        "same_trajectory_shifted": np.asarray(shifted),
        "different_trajectory_same_phase": np.asarray(cross_trajectory),
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / max(denominator, 1e-8))


def _shade_phases(axis, intervals, cutoff):
    for row in intervals:
        color = PHASE_COLORS.get(str(row["label"]))
        if color:
            axis.axvspan(int(row["start"]), int(row["end"]) + 1, color=color, alpha=0.09, linewidth=0)
    axis.axvline(cutoff, color="#333333", linestyle="--", linewidth=1.1)


def _page_trajectory(pdf, trajectory: int, entry, labels, payload):
    domains = entry["domains"]
    frames = entry["frames"]
    cutoff = entry["cutoff"]
    end = max(1, min(cutoff, len(frames)))
    x = np.arange(end)
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1.25, 1.0, 1.0, 1.3, 1.3])
    ax_force = fig.add_subplot(gs[0, :])
    ax_prob = fig.add_subplot(gs[1, :], sharex=ax_force)
    ax_cos = fig.add_subplot(gs[2, :], sharex=ax_force)
    ax_real = fig.add_subplot(gs[3:, 0], sharex=ax_force)
    ax_sim = fig.add_subplot(gs[3:, 1], sharex=ax_force)
    force_names = ("Fx", "Fy", "Fz")
    for channel, name in enumerate(force_names):
        ax_force.plot(x, domains[0]["feature"][:end, channel], color=(REAL_COLOR, "#58a46a", "#8b5fbf")[channel],
                      linewidth=1.2, label=f"Real {name}")
        ax_force.plot(x, domains[1]["feature"][:end, channel], color=(REAL_COLOR, "#58a46a", "#8b5fbf")[channel],
                      linewidth=1.2, linestyle="--", alpha=0.82, label=f"Sim {name}")
    ax_force.set_ylabel("Force residual (N)")
    ax_force.set_title("Encoder input: baseline-subtracted force (solid = Real, dashed = Sim)")
    ax_force.legend(ncol=6, fontsize=8, loc="upper right")
    _shade_phases(ax_force, labels[trajectory], cutoff)

    ax_prob.plot(x, domains[0]["probability"][:end, 1], color=REAL_COLOR, label="Real P(aligned)")
    ax_prob.plot(x, domains[1]["probability"][:end, 1], color=SIM_COLOR, linestyle="--", label="Sim P(aligned)")
    ax_prob.set_ylim(-0.03, 1.03)
    ax_prob.set_ylabel("Probability")
    ax_prob.legend(ncol=2, fontsize=8, loc="upper right")
    _shade_phases(ax_prob, labels[trajectory], cutoff)

    ax_cos.plot(x, entry["cosine"][:end], color="#777777", alpha=0.45, linewidth=0.7, label="frame cosine")
    smooth = np.convolve(np.nan_to_num(entry["cosine"][:end], nan=0.0), np.ones(7) / 7, mode="same")
    valid_count = np.convolve(np.isfinite(entry["cosine"][:end]).astype(float), np.ones(7), mode="same")
    smooth = np.divide(smooth * 7, valid_count, out=np.full_like(smooth, np.nan), where=valid_count > 0)
    ax_cos.plot(x, smooth, color="#242424", linewidth=1.8, label="7-frame mean")
    ax_cos.set_ylim(-1.03, 1.03)
    ax_cos.set_ylabel("Cosine")
    ax_cos.legend(ncol=2, fontsize=8, loc="upper right")
    _shade_phases(ax_cos, labels[trajectory], cutoff)

    latent_cmap = plt.get_cmap("coolwarm").copy()
    latent_cmap.set_bad("#eeeeee")
    for axis, domain_index, name in ((ax_real, 0, "Real latent (32 channels)"), (ax_sim, 1, "Sim latent (32 channels)")):
        z = domains[domain_index]["latent"][:end].T.copy()
        masked = np.ma.masked_invalid(z)
        im = axis.imshow(masked, origin="lower", aspect="auto", interpolation="nearest",
                         extent=(-0.5, end - 0.5, -0.5, z.shape[0] - 0.5), cmap=latent_cmap, vmin=-1, vmax=1)
        axis.set_title(name)
        axis.set_ylabel("Latent channel")
        axis.set_xlabel("Frame (10 Hz)")
        _shade_phases(axis, labels[trajectory], cutoff)
    fig.colorbar(im, ax=[ax_real, ax_sim], shrink=0.82, label="Latent value")
    ax_sim.tick_params(labelleft=False)
    heldout = "held-out" if entry["heldout"] else "training"
    active = np.isfinite(entry["cosine"][:end])
    cosine_median = float(np.nanmedian(entry["cosine"][:end]))
    agreement = float(np.nanmean(entry["agreement"][:end]))
    fig.suptitle(
        f"Trajectory {trajectory} ({heldout}) — paired cosine median {cosine_median:.3f}; "
        f"classifier agreement {agreement:.1%}; {active.sum()} encoded frames",
        fontsize=14,
    )
    pdf.savefig(fig, dpi=170)
    plt.close(fig)


def _page_summary(pdf, data, labels):
    distributions = _matched_distributions({k: v for k, v in data.items() if v["heldout"]})
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    names = list(distributions)
    values = [distributions[name] for name in names]
    labels_short = ["same frame\npaired", "same traj, shifted\n≥5 frames", "different traj,\nsame phase"]
    bp = axes[0].boxplot(values, tick_labels=labels_short, showfliers=False, patch_artist=True, whis=(10, 90))
    for patch, color in zip(bp["boxes"], ("#68a98a", "#e6bc64", "#7a9cc6")):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_title("Held-out Real/Sim latent similarity — same phase controls for label similarity")
    axes[0].grid(axis="y", alpha=0.2)
    text = []
    for name, label in zip(names, labels_short):
        arr = distributions[name]
        text.append(f"{label.replace(chr(10), ' ')}: n={len(arr)}, median={np.median(arr):.3f}, mean={np.mean(arr):.3f}")
    axes[0].text(0.015, 0.04, "\n".join(text), transform=axes[0].transAxes,
                 fontsize=9, va="bottom", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})

    ids = sorted(data)
    medians, low, high, colors = [], [], [], []
    for trajectory in ids:
        entry = data[trajectory]
        mask = entry["valid"] & np.isin(entry["phase"], list(PHASE_COLORS))
        vals = entry["cosine"][mask]
        medians.append(np.nanmedian(vals) if len(vals) else np.nan)
        low.append(np.nanquantile(vals, 0.25) if len(vals) else np.nan)
        high.append(np.nanquantile(vals, 0.75) if len(vals) else np.nan)
        colors.append("#5c9c7c" if entry["heldout"] else "#b7bdc5")
    centers = np.asarray(medians)
    errors = np.vstack((centers - np.asarray(low), np.asarray(high) - centers))
    x = np.arange(len(ids))
    for i in range(len(ids)):
        axes[1].errorbar(x[i], centers[i], yerr=errors[:, i:i + 1], fmt="o", color=colors[i],
                         ecolor=colors[i], capsize=2, markersize=5)
    axes[1].set_xticks(x, [str(t) for t in ids], fontsize=8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("Trajectory (green = held-out test; gray = training)")
    axes[1].set_ylabel("Paired cosine, median [IQR]")
    axes[1].set_title("Similarity varies by trajectory; cosine alone does not show useful control information")
    axes[1].grid(axis="y", alpha=0.2)
    pdf.savefig(fig, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/paired_latent_inspection.pdf"))
    args = parser.parse_args()
    payload = torch.load(args.artifact, map_location="cpu", weights_only=True)
    if payload.get("artifact_format") != "tavla_force_alignment_encoder_terminal_reference_v1":
        raise ValueError("This visualizer expects the terminal-reference encoder artifact")
    labels = read_labels(args.labels)
    roots = (args.real_root, args.sim_root)
    _, data = _summaries(payload, labels, roots, args.force_key, heldout_only=False)
    if args.trajectory not in data:
        raise ValueError(f"trajectory {args.trajectory} is missing from paired data")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        _page_trajectory(pdf, args.trajectory, data[args.trajectory], labels, payload)
        _page_summary(pdf, data, labels)
    heldout_data = {k: v for k, v in data.items() if v["heldout"]}
    summaries = _matched_distributions(heldout_data)
    print(f"Wrote {args.output} (2 pages; selected trajectory {args.trajectory})")
    for name, values in summaries.items():
        print(f"{name}: n={len(values)} mean={np.mean(values):.3f} median={np.median(values):.3f} p10={np.quantile(values,.1):.3f}")


if __name__ == "__main__":
    main()
