"""Inspect pre-contact calibration candidates without changing data or models.

Phase annotations locate the leading reviewed free-space interval. Real robot
motion identifies candidate windows; copied Sim poses are not independent
evidence that the simulated robot was stationary. Alternative features start at
the candidate's first frame and are compared only after its calibration ends.
No later calibration samples are applied to earlier observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .features import force_gap_features


PHASES = ("free", "uncertain", "misaligned_contact", "aligned_progress", "terminal")


def distribution(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return dict(zip(
        ("min", "p05", "median", "p95", "max"),
        map(float, np.quantile(values, [0, 0.05, 0.5, 0.95, 1])),
        strict=True,
    ))


def read_annotations(path: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["trajectory"], []).append(row)
    if not result:
        raise ValueError(f"No annotations: {path}")
    return result


def annotation_arrays(rows: list[dict[str, str]], length: int):
    phases = np.full(length, "unannotated", dtype=object)
    reviewed = np.zeros(length, dtype=bool)
    cursor, free_end = 0, 0
    for row in sorted(rows, key=lambda item: int(item["start_frame"])):
        start, end = int(row["start_frame"]), int(row["end_frame"]) + 1
        if start != cursor or not start < end <= length:
            raise ValueError(f"Invalid annotation coverage at {row['trajectory']}:{start}:{end}")
        label, confidence = row["label"], row["confidence"].strip().lower()
        if label not in PHASES or confidence not in ("high", "medium", "low"):
            raise ValueError(f"Unknown annotation: {row}")
        phases[start:end] = label
        reviewed[start:end] = confidence != "low"
        if start == free_end and label == "free" and confidence != "low":
            free_end = end
        cursor = end
    if cursor != length:
        raise ValueError(f"Annotations cover {cursor} frames, data contain {length}")
    return phases, reviewed, free_end


def load_episode(path: Path, force_key: str):
    with h5py.File(path, "r") as h5:
        episode = {
            "force": np.asarray(h5[force_key], dtype=np.float32),
            "pose": np.asarray(h5["obs/state/ee_pose"], dtype=np.float64),
            "joints": np.asarray(h5["obs/state/joint_pos"], dtype=np.float64),
            "timestamps": np.asarray(h5["timestamps"], dtype=np.float64),
        }
        # These attributes describe the recorded source; they do not verify it.
        episode["provenance"] = {
            key: str(h5.attrs[key]) for key in (
                "data_source", "force_source", "force_alignment", "paired_real_source"
            ) if key in h5.attrs
        }
    force, times = episode["force"], episode["timestamps"]
    if force.ndim != 2 or force.shape[1] < 3 or times.shape != (len(force),):
        raise ValueError(f"{path}: invalid force/timestamp shape")
    episode["force"] = force[:, :3]
    for key in ("pose", "joints"):
        if episode[key].ndim != 2 or len(episode[key]) != len(force) or episode[key].shape[1] < 3:
            raise ValueError(f"{path}: invalid {key} shape")
    if not all(np.isfinite(episode[key]).all() for key in ("force", "pose", "joints", "timestamps")):
        raise ValueError(f"{path}: non-finite data")
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    return episode


def motion_window(episode, start: int, frames: int):
    xyz = episode["pose"][start:start + frames, :3]
    joints = episode["joints"][start:start + frames]
    times = episode["timestamps"][start:start + frames]
    return {
        "start_frame": start,
        "end_frame": start + frames - 1,
        "duration_s": float(times[-1] - times[0]),
        "max_displacement_from_start_mm": float(np.linalg.norm(xyz - xyz[0], axis=1).max() * 1000),
        "max_joint_displacement_from_start_rad": float(np.abs(joints - joints[0]).max()),
    }


def timing_summary(times: np.ndarray, smoothing: int, lag: int):
    periods = np.diff(times)
    mean_span = smoothing - 1
    return {
        "period_s": distribution(periods),
        "period_outside_80_120ms_fraction": float(np.mean((periods < 0.08) | (periods > 0.12))),
        "smoothing_sample_span_s": distribution(times[mean_span:] - times[:-mean_span]) if mean_span else None,
        "trend_sample_span_s": distribution(times[lag:] - times[:-lag]),
    }


def stage_summary(features, phases, reviewed, eligible):
    result = {}
    for phase in PHASES:
        mask = (phases == phase) & reviewed & eligible
        if not mask.any():
            continue
        result[phase] = {
            "frames": int(mask.sum()),
            "force_valid_fraction": float(features[mask, 10].mean()),
            "trend_valid_fraction": float(features[mask, 11].mean()),
            "smoothed_residual_force_norm_n": distribution(np.expm1(features[mask, 8])),
            "trend_norm_n": distribution(np.expm1(features[mask, 9])),
        }
    return result


def mismatch_summary(real, sim, phases, reviewed, eligible):
    return {
        phase: {
            "frames": int(mask.sum()),
            "force_valid_mismatch_fraction": float(np.mean(real[mask, 10] != sim[mask, 10])),
            "trend_valid_mismatch_fraction": float(np.mean(real[mask, 11] != sim[mask, 11])),
        }
        for phase in PHASES
        if (mask := (phases == phase) & reviewed & eligible).any()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/encoder.pt"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--quiet-displacement-mm", type=float, default=1.0)
    parser.add_argument("--quiet-joint-displacement-rad", type=float, default=0.005)
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/calibration_audit.json"))
    args = parser.parse_args()
    if args.quiet_displacement_mm <= 0 or args.quiet_joint_displacement_rad <= 0:
        raise ValueError("Quiet-window diagnostic tolerances must be positive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("artifact_format") != "tavla_force_gap_encoder_v2":
        raise ValueError("Expected a force_gap_encoder_v2 checkpoint")
    frames = int(payload["config"]["baseline_frames"])
    preprocess = payload["preprocess"]
    options = {key: preprocess[key] for key in (
        "smoothing_frames", "trend_lag_frames", "active_noise_mad_multiplier", "active_noise_floor_newton"
    )}
    options.update(baseline_frames=frames, insertion_axis=np.asarray(preprocess["insertion_axis_base"]))
    annotations = read_annotations(args.labels)
    rows = []
    for trajectory in sorted(annotations, key=lambda name: int(name.removeprefix("traj_"))):
        episodes = {domain: load_episode(root / trajectory / "data.h5", args.force_key)
                    for domain, root in (("real", args.real_root), ("sim", args.sim_root))}
        real, sim = episodes["real"], episodes["sim"]
        length = len(real["force"])
        if length < frames or len(sim["force"]) != length:
            raise ValueError(f"{trajectory}: incompatible episode lengths")
        phases, reviewed, free_end = annotation_arrays(annotations[trajectory], length)
        windows = [motion_window(real, start, frames) for start in range(max(0, free_end - frames + 1))]
        quiet = [w for w in windows if w["max_displacement_from_start_mm"] <= args.quiet_displacement_mm
                 and w["max_joint_displacement_from_start_rad"] <= args.quiet_joint_displacement_rad]
        candidate = quiet[0] if quiet else None  # earliest by motion, never by later phase performance
        best_motion = min(windows, key=lambda w: max(
            w["max_displacement_from_start_mm"] / args.quiet_displacement_mm,
            w["max_joint_displacement_from_start_rad"] / args.quiet_joint_displacement_rad
        )) if windows else None
        features, metadata = {}, {}
        for domain, episode in episodes.items():
            features[domain], metadata[domain] = force_gap_features(episode["force"], **options)
        eligible = np.arange(length) >= frames
        row = {
            "trajectory": trajectory,
            "leading_reviewed_free_end_frame": free_end - 1,
            "first_calibration_all_reviewed_free": free_end >= frames,
            "first_calibration_motion": motion_window(real, 0, frames),
            "quiet_candidate_window_count": len(quiet),
            "earliest_quiet_candidate": candidate,
            "best_motion_window_in_leading_free": best_motion,
            "max_real_sim_timestamp_difference_s": float(np.max(np.abs(real["timestamps"] - sim["timestamps"]))),
            "real_sim_pose_arrays_identical": bool(np.array_equal(real["pose"], sim["pose"])),
            "domains": {
                domain: {
                    "provenance": episode["provenance"],
                    "timebase": timing_summary(episode["timestamps"], options["smoothing_frames"], options["trend_lag_frames"]),
                    "current_first_calibration": metadata[domain],
                    "current_stage_statistics": stage_summary(features[domain], phases, reviewed, eligible),
                } for domain, episode in episodes.items()
            },
            "current_validity_mismatch_by_transferred_phase": mismatch_summary(features["real"], features["sim"], phases, reviewed, eligible),
        }
        if candidate is not None:
            start = candidate["start_frame"]
            eligible_after_candidate = np.arange(length - start) >= frames
            alternatives = {}
            for domain, episode in episodes.items():
                alternative, meta = force_gap_features(episode["force"][start:], **options)
                alternatives[domain] = alternative
                row["domains"][domain]["quiet_candidate_comparison"] = {
                    "comparison_starts_at_frame": start + frames,
                    "candidate_calibration": meta,
                    "current_on_common_frames": stage_summary(features[domain][start:], phases[start:], reviewed[start:], eligible_after_candidate),
                    "candidate_on_common_frames": stage_summary(alternative, phases[start:], reviewed[start:], eligible_after_candidate),
                }
            row["candidate_validity_mismatch_by_transferred_phase"] = mismatch_summary(
                alternatives["real"], alternatives["sim"], phases[start:], reviewed[start:], eligible_after_candidate
            )
        rows.append(row)
    stage_medians = {}
    for phase in PHASES:
        stage_rows = [row for row in rows if phase in row["current_validity_mismatch_by_transferred_phase"]]
        if not stage_rows:
            continue
        stage_medians[phase] = {
            "trajectory_count": len(stage_rows),
            **{
                f"{domain}_force_valid_fraction_median": float(np.median([
                    row["domains"][domain]["current_stage_statistics"][phase]["force_valid_fraction"]
                    for row in stage_rows
                ])) for domain in ("real", "sim")
            },
            "force_valid_mismatch_fraction_median": float(np.median([
                row["current_validity_mismatch_by_transferred_phase"][phase]["force_valid_mismatch_fraction"]
                for row in stage_rows
            ])),
        }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "baseline_frames": frames,
        "summary": {
            "trajectory_count": len(rows),
            "first_calibration_all_reviewed_free_count": sum(r["first_calibration_all_reviewed_free"] for r in rows),
            "trajectories_with_quiet_candidate": sum(r["earliest_quiet_candidate"] is not None for r in rows),
            "first_calibration_displacement_mm": distribution(np.array([r["first_calibration_motion"]["max_displacement_from_start_mm"] for r in rows])),
            "stage_statistics_scope": "all reviewed frames after calibration; includes frames beyond the training first-hit cutoff",
            "current_stage_trajectory_medians": stage_medians,
            "current_force_noise_gate_n": {
                domain: distribution(np.array([
                    row["domains"][domain]["current_first_calibration"]["force_noise_gate_n"] for row in rows
                ])) for domain in ("real", "sim")
            },
        },
        "quiet_window_diagnostic_tolerances": {
            "displacement_mm": args.quiet_displacement_mm,
            "joint_displacement_rad": args.quiet_joint_displacement_rad,
        },
        "interpretation": [
            "Motion tolerances are diagnostic heuristics, not verified stationary calibration requirements.",
            "Leading free-space annotations are transferred to Sim; they do not prove absence of simulated contact.",
            "Sim poses/timestamps may be inherited from Real; agreement does not validate simulated motion or the sensor clock.",
            "force_valid/trend_valid suppress directions and ratios; log-magnitude channels are retained.",
            "Candidate selection is offline diagnosis, not an implemented online calibration detector.",
            "No candidate baseline is applied before its calibration finishes. This report does not change training preprocessing.",
            "Do not resample force/actions solely because timestamp periods vary; verify timestamp provenance first.",
            "This audit does not establish force-gap reduction or action-prediction gains."
        ],
        "trajectories": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告：{args.output.resolve()}")
    print("仅完成校准诊断；未修改数据、checkpoint 或训练/部署预处理。")


if __name__ == "__main__":
    main()
