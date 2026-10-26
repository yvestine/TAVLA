"""Force-only multi-task data, portable splits, and task-balanced causal blocks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from force_gap_forecast_v1.common import sha256
from force_gap_forecast_v1.data import segment_limits
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import BLOCK
from . import FUTURE
from . import GAP_S
from . import HISTORY
from . import PAST
from .data import force_window


@dataclass
class ForceEpisode:
    task: str
    trajectory: int
    force: np.ndarray
    time: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    path: Path
    sha256: str


def task_config(path, data_root, selected=None):
    config = json.loads(Path(path).read_text())
    if config.get("version") != 1 or not config.get("tasks"):
        raise ValueError("Expected version=1 and a nonempty tasks list")
    tasks = {}
    for entry in config["tasks"]:
        name = entry["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", "..") or name in tasks:
            raise ValueError(f"Invalid or duplicate task name: {name}")
        task = dict(entry)
        for field in ("real_root", "sim_root"):
            if field in task:
                task[field] = str((Path(data_root) / task[field]).resolve())
        tasks[name] = task
    if selected:
        unknown = set(selected) - tasks.keys()
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        tasks = {name: task for name, task in tasks.items() if name in selected}
    return tasks


def trajectory_files(root):
    result = {}
    for path in Path(root).glob("traj_*/data.h5"):
        tid = int(path.parent.name.removeprefix("traj_"))
        if tid in result:
            raise ValueError(f"Duplicate trajectory ID {tid} in {root}")
        result[tid] = path
    if not result:
        raise ValueError(f"No traj_*/data.h5 files in {root}")
    return dict(sorted(result.items()))


def read_episode(path, task, tid, force_key, sign=1.0):
    with h5py.File(path, "r") as handle:
        force = np.asarray(handle[force_key][:], np.float32)
        time = np.asarray(handle["timestamps"][:], np.float64)
    if force.shape == (len(time), 6):
        force = force[:, :3]
    if time.ndim != 1 or len(time) < 2 or force.shape != (len(time), 3):
        raise ValueError(f"{path}: expected force [N,3] (or wrench [N,6]) and timestamps [N], N>=2")
    if not np.isfinite(force).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: finite forces and strictly increasing timestamps required")
    starts, ends = segment_limits(time)
    return ForceEpisode(task, tid, force * sign, time, starts, ends, path, sha256(path))


def load_real(tasks):
    episodes = {}
    for name, task in tasks.items():
        for tid, path in trajectory_files(task["real_root"]).items():
            episodes[f"{name}/traj_{tid}"] = read_episode(
                path, name, tid, task.get("real_force_key", "obs/state/ee_force_base")
            )
    return episodes


def load_sim(tasks, real):
    """Pair only by declared reference frames; never infer alignment from force curves."""
    episodes, pairing = {}, {}
    for name, task in tasks.items():
        if "sim_root" not in task:
            raise ValueError(f"{name}: add sim_root to the task config before student training")
        files = trajectory_files(task["sim_root"])
        expected = {ep.trajectory for ep in real.values() if ep.task == name}
        if set(files) != expected:
            raise ValueError(
                f"{name}: Sim trajectory IDs must match Real; missing={sorted(expected-set(files))}, "
                f"extra={sorted(set(files)-expected)}"
            )
        source = task.get("sim_force_source", "aligned_base_xyz_n_v1")
        if source not in ("aligned_base_xyz_n_v1", "calibrated_link7"):
            raise ValueError(f"{name}: unsupported sim_force_source {source}")
        for tid, path in files.items():
            key = f"{name}/traj_{tid}"
            reference = real[key]
            ep = read_episode(
                path,
                name,
                tid,
                task.get("sim_force_key", "obs/state/ee_force_base"),
                -1.0 if source == "calibrated_link7" else 1.0,
            )
            if len(ep.force) != len(reference.force):
                raise ValueError(f"{key}: Real/Sim frame counts differ")
            with h5py.File(path, "r") as handle:
                if "pairing/real_frame_id" in handle:
                    ids = handle["pairing/real_frame_id"][:]
                    if not np.array_equal(ids, np.arange(len(ep.force))):
                        raise ValueError(f"{key}: real_frame_id must be 0..N-1 in reference order")
                    method = "real_frame_id"
                elif task.get("pairing") == "frame_index":
                    method = "declared_frame_index"
                else:
                    raise ValueError(
                        f"{key}: missing pairing/real_frame_id; set pairing=frame_index only "
                        "when the replay has a known one-observation-per-reference-frame contract"
                    )
                if "pairing/real_timestamp_s" in handle and not np.allclose(
                    handle["pairing/real_timestamp_s"][:], reference.time, rtol=0, atol=1e-6
                ):
                    raise ValueError(f"{key}: reference timestamps do not match Real")
                linked = handle.attrs.get("paired_real_sha256")
                if isinstance(linked, bytes):
                    linked = linked.decode()
                if linked is not None and linked != reference.sha256:
                    raise ValueError(f"{key}: paired_real_sha256 references different Real data")
            episodes[key], pairing[key] = ep, method
    return episodes, pairing


def make_split(real, seed, validation_fraction):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for task in dict.fromkeys(ep.task for ep in real.values()):
        keys = [key for key, ep in real.items() if ep.task == task]
        if len(keys) < 2:
            raise ValueError(f"{task}: at least two trajectories are required for a fixed holdout")
        order = rng.permutation(keys).tolist()
        count = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
        validation.extend(order[:count])
        train.extend(order[count:])
    return {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train": sorted(train),
        "validation": sorted(validation),
    }


def data_identity(episodes):
    return {key: {"sha256": ep.sha256, "frames": len(ep.force)} for key, ep in sorted(episodes.items())}


def data_audit(episodes):
    return {
        key: {
            **data_identity({key: ep})[key],
            "path": str(ep.path),
            "median_dt_s": float(np.median(np.diff(ep.time))),
            "gaps_gt_0_5_s": int((np.diff(ep.time) > GAP_S).sum()),
        }
        for key, ep in episodes.items()
    }


def fit_domain_stats(episodes, keys):
    values = np.concatenate([episodes[key].force for key in keys])
    increments = np.concatenate(
        [np.diff(episodes[key].force, axis=0)[np.diff(episodes[key].time) <= GAP_S] for key in keys]
    )
    if not len(increments):
        raise ValueError("No continuous training frames for force increment statistics")
    return (
        {"mean": values.mean(0).tolist(), "scale": np.maximum(values.std(0), 1e-3).tolist()},
        max(float(np.quantile(np.linalg.norm(increments, axis=1), 0.9)), 0.1),
    )


def prepare_arrays(episodes, stats, domain):
    arrays = {}
    for key, ep in episodes.items():
        inputs, components = [], []
        for t in range(len(ep.force)):
            lo = max(int(ep.starts[t]), t - HISTORY + 1)
            x, c = force_window(ep.force[lo : t + 1], ep.time[lo : t + 1], stats, domain)
            inputs.append(x)
            components.append(c)
        arrays[key] = {"x": np.stack(inputs), "components": np.stack(components)}
    return arrays


class MultiTaskBlocks(Dataset):
    def __init__(self, real, sim, keys, arrays, threshold):
        self.real, self.sim, self.arrays = real, sim, arrays
        counts = {task: sum(real[key].task == task for key in keys) for task in {real[key].task for key in keys}}
        self.keys, weights = [], []
        for key in keys:
            ep = real[key]
            delta = np.linalg.norm(np.diff(ep.force, axis=0, prepend=ep.force[:1]), axis=1)
            delta[np.unique(ep.starts)] = 0
            event = np.minimum(4.0, 1 + delta / threshold)
            self.keys.extend((key, t) for t in range(len(ep.force)))
            weights.extend((0.5 / len(event) + 0.5 * event / event.sum()) / (len(counts) * counts[ep.task]))
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key, anchor = self.keys[item]
        real = self.real[key]
        domains = [self.real] if self.sim is None else [self.real, self.sim]
        start = max(episodes[key].starts[anchor] for episodes in domains)
        end = min(episodes[key].ends[anchor] for episodes in domains)
        original = np.arange(anchor - BLOCK, anchor + 1)
        indices = np.maximum(original, start)
        valid = original >= start
        past = indices[:, None] + np.arange(-PAST + 1, 1)
        future = indices[:, None] + np.arange(1, FUTURE + 1)
        return {
            "x": np.stack([self.arrays[d][key]["x"][indices] for d in range(len(domains))]),
            "components": np.stack([self.arrays[d][key]["components"][indices] for d in range(len(domains))]),
            "future": real.force[np.clip(future, start, end - 1)],
            "valid": valid,
            "past_mask": (past >= start) & valid[:, None],
            "future_mask": (future < end) & valid[:, None],
        }
