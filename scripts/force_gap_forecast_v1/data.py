"""Full trajectories; motion provenance; train-only scales; causal paired windows."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import FUTURE, GAP_S, HISTORY, PAST
from .common import finite_array, sha256

HOLDOUT = [0, 2, 21, 22, 23, 33, 34, 39]
FOLDS = [[38, 32, 19, 28, 30, 10, 29, 7], [12, 25, 5, 20, 11, 31, 24, 8],
         [35, 9, 15, 17, 36, 3, 6, 14], [4, 27, 16, 37, 13, 18, 26, 1]]
TRAIN_IDS = sorted(x for fold in FOLDS for x in fold)
FORCE_KEY = "obs/state/ee_force_base"


@dataclass
class Episode:
    trajectory: int
    force: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    time: np.ndarray
    phase: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def segment_limits(times):
    boundaries = np.flatnonzero(np.diff(times) > GAP_S) + 1
    starts, ends = np.zeros(len(times), np.int64), np.zeros(len(times), np.int64)
    for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, len(times)]):
        starts[lo:hi], ends[lo:hi] = lo, hi
    return starts, ends


def raw_features(force, q, dq, time):
    delta = np.zeros_like(force)
    delta[1:] = np.diff(force, axis=0)
    dt = np.r_[0.1, np.diff(time)].astype(np.float32)
    resets = dt > GAP_S
    delta[resets], dt[resets] = 0, 0.1
    return np.concatenate((force, delta, q, dq, dt[:, None]), axis=1).astype(np.float32)


def phases(path):
    result = {}
    if path and Path(path).is_file():
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = int(row["trajectory"].removeprefix("traj_"))
                result.setdefault(key, []).append(row)
    return result


def load_pairs(real_root, sim_root, labels=None):
    roots = [Path(real_root).resolve(), Path(sim_root).resolve()]
    files = [{int(p.parent.name.split("_")[-1]): p for p in root.glob("traj_*/data.h5")} for root in roots]
    if set(files[0]) != set(range(40)) or set(files[1]) != set(range(40)):
        raise ValueError("Expected the same 40 paired trajectories (0..39) in both roots")
    annotations = phases(labels)
    episodes, audit = {}, {"roots": [str(x) for x in roots], "trajectories": {}, "force_key": FORCE_KEY,
                           "full_trajectories": True, "startup_baseline_subtraction": False}
    all_actual = True
    for trajectory in range(40):
        pair, records = [], []
        for domain in range(2):
            path = files[domain][trajectory]
            with h5py.File(path, "r") as handle:
                force = np.asarray(handle[FORCE_KEY], np.float32)
                n = len(force)
                force = finite_array(force, (n, 3), "force")
                q = finite_array(handle["obs/state/joint_pos"][:], (n, 7), "q")
                dq = finite_array(handle["obs/state/joint_vel"][:], (n, 7), "dq")
                time = finite_array(handle["timestamps"][:], (n,), "timestamps", np.float64)
                if n < 2 or np.any(np.diff(time) <= 0):
                    raise ValueError(f"{path}: timestamps must increase and have >=2 frames")
                if domain:
                    actual = (handle.attrs.get("state_source") == "sim_observed"
                              and bool(handle.attrs.get("synchronized_same_replay", False)))
                    if actual:
                        ids = handle["pairing/real_frame_id"][:]
                        if not np.array_equal(ids, np.arange(n)):
                            raise ValueError("Actual-Sim replay must log one observation per reference frame in order")
                        if handle.attrs.get("force_contract") != "aligned_base_xyz_n_v1":
                            raise ValueError("Actual-Sim capture must declare the aligned calibrated force contract")
                        if handle.attrs.get("paired_real_sha256") != records[0]["sha256"]:
                            raise ValueError("Actual-Sim capture references a different Real trajectory")
                else:
                    actual = True
            phase = np.full(n, "unannotated", dtype=object)
            for row in annotations.get(trajectory, []):
                lo, hi = int(row["start_frame"]), int(row["end_frame"]) + 1
                if not 0 <= lo < hi <= n:
                    raise ValueError("Invalid phase bounds")
                phase[lo:hi] = row["label"]
            starts, ends = segment_limits(time)
            pair.append(Episode(trajectory, force, q, dq, time, phase, starts, ends))
            records.append({"path": str(path), "sha256": sha256(path), "frames": n,
                            "median_period_s": float(np.median(np.diff(time))), "actual_sim_state": bool(domain and actual)})
        real, sim = pair
        if len(real.force) != len(sim.force):
            raise ValueError(f"traj_{trajectory}: paired frame count mismatch")
        copied = np.array_equal(real.q, sim.q) and np.array_equal(real.dq, sim.dq)
        actual = records[1]["actual_sim_state"] and not copied
        all_actual &= actual
        if not records[1]["actual_sim_state"] and not np.allclose(real.time, sim.time, rtol=0, atol=1e-6):
            raise ValueError("Legacy copied-state dataset must have matched reference timestamps")
        # Pair by frame ID, not by future-aware DTW. Each input keeps its own causal resets.
        for domain, ep in enumerate(pair):
            episodes[(trajectory, domain)] = ep
        audit["trajectories"][str(trajectory)] = {"sources": records, "state_is_real_copy": copied,
                                                "actual_sim_state_verified": actual,
                                                "q_tracking_rmse_rad": float(np.sqrt(np.mean((real.q - sim.q)**2))),
                                                "dq_tracking_rmse_rad_s": float(np.sqrt(np.mean((real.dq - sim.dq)**2)))}
    audit["actual_sim_state_verified"] = bool(all_actual)
    audit["warning"] = None if all_actual else "Sim q/dq are copied or provenance is unverified; prototype only."
    return episodes, audit


def fit_stats(episodes, ids):
    stats = {"fit_trajectory_ids": sorted(ids)}
    raw = [np.concatenate([raw_features(ep.force, ep.q, ep.dq, ep.time)
                           for i in ids for ep in [episodes[(i, d)]]]) for d in range(2)]
    motion = np.concatenate([x[:, 6:] for x in raw])
    for d, name in enumerate(("real", "sim")):
        mean, scale = raw[d].mean(axis=0), np.maximum(raw[d].std(axis=0), 1e-3)
        mean[6:], scale[6:] = motion.mean(axis=0), np.maximum(motion.std(axis=0), 1e-3)
        stats[name] = {"mean": mean.tolist(), "scale": scale.tolist()}
    stats["target"] = {"mean": raw[0][:, :3].mean(axis=0).tolist(),
                       "scale": np.maximum(raw[0][:, :3].std(axis=0), 1e-3).tolist()}
    return stats


def normalized_features(ep, stats, domain, motion_only=False):
    values = raw_features(ep.force, ep.q, ep.dq, ep.time)
    values = (values - np.asarray(stats[domain]["mean"], np.float32)) / np.asarray(stats[domain]["scale"], np.float32)
    if motion_only:
        values[:, :6] = 0
    return values.astype(np.float32)


def window(values, t, start, length=HISTORY):
    index = np.maximum(np.arange(t - length + 1, t + 1), start)
    return values[index]


class Blocks(Dataset):
    def __init__(self, episodes, ids, stats, motion_only=False):
        self.episodes, self.ids, self.stats = episodes, ids, stats
        self.features = {(i, d): normalized_features(episodes[(i, d)], stats, ("real", "sim")[d], motion_only)
                         for i in ids for d in range(2)}
        self.keys, weights = [], []
        for i in ids:
            real = episodes[(i, 0)]
            delta = np.linalg.norm(np.diff(real.force, axis=0, prepend=real.force[:1]), axis=1)
            amplitude = np.linalg.norm(real.force, axis=1)
            event = np.minimum(4., 1. + delta / max(np.median(delta), 0.1)
                               + amplitude / max(np.quantile(amplitude, .75), 0.1))
            probability = .5 / len(delta) + .5 * event / event.sum()
            self.keys.extend((i, t) for t in range(len(delta)))
            weights.extend(probability / len(ids))
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        i, anchor = self.keys[item]
        real = self.episodes[(i, 0)]
        sim = self.episodes[(i, 1)]
        start, end = max(real.starts[anchor], sim.starts[anchor]), min(real.ends[anchor], sim.ends[anchor])
        original = np.arange(anchor - PAST, anchor + 1)
        # The extra predecessor must remain available even if only the OTHER
        # domain reset at this anchor. Mask supervision, not observed inputs.
        endpoints = np.maximum(original, 0)
        inputs = np.stack([np.stack([window(self.features[(i, d)], t, self.episodes[(i, d)].starts[t])
                                     for t in endpoints]) for d in range(2)])
        past_ids = endpoints[:, None] + np.arange(-PAST + 1, 1)[None]
        future_ids = endpoints[:, None] + np.arange(1, FUTURE + 1)[None]
        valid_endpoint = original >= start
        past_valid = (past_ids >= start) & valid_endpoint[:, None]
        future_valid = (future_ids < end) & valid_endpoint[:, None]
        past = real.force[np.clip(past_ids, start, end - 1)]
        future = real.force[np.clip(future_ids, start, end - 1)]
        delta = real.force[endpoints] - real.force[np.maximum(endpoints - 1, real.starts[endpoints])]
        return {"x": inputs, "current": real.force[endpoints], "delta": delta,
                "past": past, "future": future, "past_valid": past_valid,
                "future_valid": future_valid, "valid": valid_endpoint,
                "delta_valid": valid_endpoint & (original > start),
                "real_reset": (original < 0) | (endpoints == real.starts[endpoints]),
                "sim_reset": (original < 0) | (endpoints == sim.starts[endpoints])}
