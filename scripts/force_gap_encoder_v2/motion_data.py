"""Input provenance and trajectory-grouped arrays for the motion_v1 recipe."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch

from insertion_alignment_encoder.preprocess import force_history_features, initial_baseline
from insertion_alignment_encoder.train_terminal_geometry import terminal_reference_geometry

from .features import fit_normalizer
from .motion_features import motion_features
from .train import LABEL_IDS, read_labels


@dataclass
class MotionEpisode:
    trajectory: int
    domain: int
    force: np.ndarray
    q: np.ndarray
    velocity: np.ndarray
    timestamps: np.ndarray
    pose: np.ndarray
    phase: np.ndarray
    confidence: np.ndarray
    y: np.ndarray
    human_weight: np.ndarray
    cutoff: int
    geometry: np.ndarray
    old_features: np.ndarray


@dataclass
class MotionWindows:
    features: np.ndarray
    vectors: np.ndarray
    scales: np.ndarray
    elapsed: np.ndarray
    state: np.ndarray
    raw_force: np.ndarray
    background: np.ndarray
    old: np.ndarray
    geometry: np.ndarray
    y: np.ndarray
    human_weight: np.ndarray
    signal_confidence: np.ndarray
    domain: np.ndarray
    trajectory: np.ndarray
    frame: np.ndarray
    phase: np.ndarray
    future: np.ndarray
    future_valid: np.ndarray
    future_elapsed: np.ndarray

    def take(self, index):
        return MotionWindows(**{field.name: getattr(self, field.name)[index] for field in fields(self)})

    def __len__(self):
        return len(self.y)


def history(values, frame: int, length: int = 32):
    window = values[max(0, frame - length + 1):frame + 1]
    if len(window) < length:
        window = np.concatenate((np.repeat(window[:1], length - len(window), axis=0), window), axis=0)
    return window


def grouped_inner_ids(ids, seed=42):
    order = np.random.default_rng(seed).permutation(sorted(ids))
    valid = list(map(int, order[:max(2, len(ids) // 4)]))
    return sorted(set(ids) - set(valid)), sorted(valid)


def _annotations(rows, count, guard):
    phase = np.full(count, "unannotated", dtype=object)
    confidence = np.full(count, "low", dtype=object)
    y = np.full(count, -1, np.int64)
    weight = np.zeros(count, np.float32)
    cursor = 0
    for row in sorted(rows, key=lambda r: r["start"]):
        start, stop = row["start"], row["end"] + 1
        if start != cursor or stop <= start or stop > count:
            raise ValueError("Annotation intervals must cover every frame contiguously")
        phase[start:stop], confidence[start:stop] = row["label"], row["confidence"]
        if row["label"] in LABEL_IDS and row["confidence"] in ("high", "medium"):
            lo, hi = start + guard, stop - guard
            if hi > lo:
                y[lo:hi] = LABEL_IDS[row["label"]]
                weight[lo:hi] = 1.0 if row["confidence"] == "high" else 0.65
        cursor = stop
    if cursor != count:
        raise ValueError("Annotations do not cover the episode")
    return phase, confidence, y, weight


def load_motion_episodes(args):
    labels = read_labels(args.labels)
    old_payload = torch.load(args.cutoff_checkpoint, map_location="cpu", weights_only=False)
    if old_payload.get("artifact_format") != "tavla_force_alignment_encoder_terminal_reference_v1":
        raise ValueError("Expected original terminal-reference encoder as cutoff/legacy recipe source")
    cutoffs = old_payload["geometry"]["first_hit_cutoff_frames"]
    offset = np.asarray(old_payload["config"]["tool_offset_local_mm"], np.float64)
    result, diagnostics = {}, {}
    for trajectory in sorted(labels):
        pairs, source_info = [], []
        for domain, root in enumerate((args.real_root, args.sim_root)):
            path = root / f"traj_{trajectory}" / "data.h5"
            with h5py.File(path, "r") as h5:
                force = np.asarray(h5[args.force_key], np.float32)[:, :3]
                q = np.asarray(h5["obs/state/joint_pos"], np.float32)
                velocity = np.asarray(h5["obs/state/joint_vel"], np.float32)
                times = np.asarray(h5["timestamps"], np.float64)
                pose = np.asarray(h5["obs/state/ee_pose"], np.float64)
                attrs = {key: str(h5.attrs[key]) for key in ("data_source", "force_source", "force_alignment", "paired_real_source") if key in h5.attrs}
                if domain == 1:
                    declared = str(h5.attrs.get("force_field", "obs/state/ee_wrench_base"))
                    if declared not in h5:
                        raise ValueError(f"{path}: no declared force provenance field")
                    field_attrs = h5[declared].attrs
                    contact_isolated = str(field_attrs.get("counterfactual_no_contact_subtracted", "")).lower() in ("true", "1")
                    if not contact_isolated or str(field_attrs.get("frame", "")) != "robot_base_O":
                        raise ValueError(f"{path}: expected contact-isolated base-frame Sim force provenance")
                    if not np.array_equal(force, np.asarray(h5[declared], np.float32)[:, :3]):
                        raise ValueError(f"{path}: chosen force differs from its declared calibrated source")
                    attrs.update({key: str(value) for key, value in field_attrs.items()})
                actual_is_q = np.array_equal(np.asarray(h5["action/actual/arm"]), q)
                policy_is_q = np.array_equal(np.asarray(h5["action/policy/arm"]), q)
            count = len(force)
            if force.shape != (count, 3) or q.shape != (count, 7) or velocity.shape != q.shape or pose.shape != (count, 7) or times.shape != (count,):
                raise ValueError(f"{path}: incompatible force/state/timestamp shapes")
            if count <= args.baseline_frames + 5 or not all(np.isfinite(a).all() for a in (force, q, velocity, times, pose)):
                raise ValueError(f"{path}: short or non-finite episode")
            if np.any(np.diff(times) <= 0) or not 0.08 <= np.median(np.diff(times)) <= 0.12:
                raise ValueError(f"{path}: invalid nominal 10 Hz timebase")
            phase, confidence, y, weight = _annotations(labels[trajectory], count, args.boundary_guard)
            if not np.all((phase[:args.baseline_frames] == "free") & (confidence[:args.baseline_frames] == "high")):
                raise ValueError(f"{path}: leading calibration must be reviewed free-space; motion is allowed")
            cutoff = int(cutoffs[f"traj_{trajectory}"])
            if not args.baseline_frames + 5 < cutoff <= count:
                raise ValueError(f"{path}: invalid stored first-hit cutoff {cutoff}")
            old_features = force_history_features(force, times, initial_baseline(force, args.baseline_frames))
            pairs.append(MotionEpisode(trajectory, domain, force, q, velocity, times, pose, phase,
                                       confidence, y, weight, cutoff, np.empty((count, 3)), old_features))
            digest = hashlib.sha256()
            for array in (force, q, velocity, times):
                digest.update(np.ascontiguousarray(array).tobytes())
            source_info.append({"path": str(path.resolve()), "selected_arrays_sha256": digest.hexdigest(),
                                "provenance": attrs, "same_frame_actual_action_equals_q": actual_is_q,
                                "same_frame_policy_action_equals_q": policy_is_q,
                                "period_outside_80_120ms_fraction": float(np.mean((np.diff(times) < .08) | (np.diff(times) > .12)))})
        if len(pairs[0].force) != len(pairs[1].force) or not np.allclose(pairs[0].timestamps, pairs[1].timestamps, atol=1e-6, rtol=0):
            raise ValueError(f"traj_{trajectory}: incompatible teacher-trajectory correspondence")
        geometry, _ = terminal_reference_geometry(pairs[1].pose, offset)
        for episode in pairs:
            episode.geometry = geometry
            result[(trajectory, episode.domain)] = episode
        diagnostics[str(trajectory)] = {"sources": source_info,
            "real_sim_q_identical": bool(np.array_equal(pairs[0].q, pairs[1].q)),
            "real_sim_velocity_identical": bool(np.array_equal(pairs[0].velocity, pairs[1].velocity)),
            "sim_target_is_paired_demonstration_not_independent_dynamics": True}
    return result, old_payload, diagnostics


def build_motion_context(episodes, ids, background, args):
    signals = {}
    for trajectory in ids:
        for domain in (0, 1):
            episode = episodes[(trajectory, domain)]
            signals[(trajectory, domain)] = motion_features(episode.force, episode.q, episode.velocity,
                episode.timestamps, background, "real" if domain == 0 else "sim_contact_isolated", args.baseline_frames)
    rows = {field.name: [] for field in fields(MotionWindows)}
    for trajectory in ids:
        for domain in (0, 1):
            episode, signal = episodes[(trajectory, domain)], signals[(trajectory, domain)]
            state = np.concatenate((episode.q, episode.velocity), axis=1)
            for frame in range(args.baseline_frames - 1, episode.cutoff):
                for key in ("features", "vectors", "elapsed", "background"):
                    rows[key].append(history(signal[key], frame, args.history_length))
                rows["scales"].append(signal["scales"])
                rows["state"].append(history(state, frame, args.history_length).reshape(-1))
                rows["raw_force"].append(history(episode.force, frame, args.history_length).reshape(-1))
                rows["old"].append(history(episode.old_features, frame, args.history_length))
                rows["geometry"].append(episode.geometry[frame])
                rows["y"].append(episode.y[frame])
                rows["human_weight"].append(episode.human_weight[frame])
                rows["signal_confidence"].append(signal["signal_confidence"][frame])
                rows["domain"].append(domain)
                rows["trajectory"].append(trajectory)
                rows["frame"].append(frame)
                rows["phase"].append(episode.phase[frame])
                valid = frame + 5 < episode.cutoff
                rows["future_valid"].append(valid)
                rows["future"].append(np.concatenate([episode.q[frame + lag] - episode.q[frame] for lag in (3, 5)]) if valid else np.zeros(14))
                rows["future_elapsed"].append([episode.timestamps[frame + lag] - episode.timestamps[frame] for lag in (3, 5)] if valid else [0., 0.])
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    for key in ("features", "vectors", "scales", "elapsed", "state", "raw_force", "background", "old", "geometry", "human_weight", "signal_confidence", "future", "future_elapsed"):
        arrays[key] = arrays[key].astype(np.float32)
    return MotionWindows(**arrays), signals


def normalizers(episodes, signals, fit_ids, calibration):
    center, scale = fit_normalizer([signals[(t, d)]["features"][calibration - 1:episodes[(t, d)].cutoff]
                                   for t in fit_ids for d in (0, 1)])
    old = np.concatenate([episodes[(t, d)].old_features[:episodes[(t, d)].cutoff] for t in fit_ids for d in (0, 1)])
    old_scale = np.maximum(np.quantile(np.abs(old), .95, axis=0), [0.5, 0.5, 0.5, 1., 1., 1.]).astype(np.float32)
    return center, scale, old_scale
