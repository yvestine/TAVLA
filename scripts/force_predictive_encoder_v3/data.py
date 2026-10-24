"""Causal force windows, reversible decomposition, and trajectory-balanced blocks."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from force_gap_forecast_v1.data import FOLDS, HOLDOUT, TRAIN_IDS, fit_stats, load_pairs  # noqa: F401
from . import BLOCK, GAP_S, HISTORY, PAST, FUTURE, TAU_S


def force_window(force, time, stats, domain):
    """Only the supplied causal window is used; raw = slow + fast exactly up to roundoff."""
    force, time = np.asarray(force, np.float32), np.asarray(time, np.float64)
    if not 1 <= len(force) <= HISTORY or force.shape != (len(time), 3):
        raise ValueError("Expected 1..32 synchronized force samples")
    if not np.isfinite(force).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Finite forces and increasing timestamps required")
    if np.any(np.diff(time) > GAP_S):
        raise ValueError("A window may not cross a reset")
    dt = np.r_[.1, np.diff(time)].astype(np.float32)
    values = (force - np.asarray(stats[domain]["mean"][:3], np.float32)) / np.asarray(
        stats[domain]["scale"][:3], np.float32)
    padding = HISTORY - len(values)
    values = np.pad(values, ((padding, 0), (0, 0)), mode="edge")
    dt = np.r_[np.full(padding, .1, np.float32), dt]
    slow = np.empty_like(values)
    slow[0] = values[0]
    for k in range(1, HISTORY):
        alpha = np.float32(-np.expm1(-dt[k] / TAU_S))
        slow[k] = slow[k - 1] + alpha * (values[k] - slow[k - 1])
    fast = values - slow
    delta = np.diff(values, axis=0, prepend=values[:1])
    slow_delta = np.diff(slow, axis=0, prepend=slow[:1])
    inputs = np.stack((np.column_stack((slow, slow_delta, dt / .1)),
                       np.column_stack((fast, delta, dt / .1)))).astype(np.float32)
    components = np.stack((slow[-PAST:], fast[-PAST:]))
    return inputs, components


def motion_window(q, dq, time, stats):
    """Separate optional decoder input. Never concatenated into the force encoder."""
    q, dq, time = np.asarray(q, np.float32), np.asarray(dq, np.float32), np.asarray(time, np.float64)
    n = len(time)
    if not 1 <= n <= HISTORY or q.shape != (n, 7) or dq.shape != (n, 7):
        raise ValueError("Expected synchronized q/dq history of shape [1..32,7]")
    if not all(np.isfinite(a).all() for a in (q, dq, time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Finite motion and increasing timestamps required")
    if np.any(np.diff(time) > GAP_S):
        raise ValueError("Motion history crosses a reset")
    dt = np.r_[.1, np.diff(time)].astype(np.float32)
    x = np.column_stack((q, dq, dt))
    x = (x - np.asarray(stats["real"]["mean"][6:], np.float32)) / np.asarray(
        stats["real"]["scale"][6:], np.float32)
    return np.pad(x, ((HISTORY - n, 0), (0, 0)), mode="edge").astype(np.float32)


def episode_arrays(ep, stats, domain):
    inputs, components, motion = [], [], []
    for t in range(len(ep.force)):
        lo = max(int(ep.starts[t]), t - HISTORY + 1)
        x, c = force_window(ep.force[lo:t + 1], ep.time[lo:t + 1], stats, domain)
        inputs.append(x)
        components.append(c)
        motion.append(motion_window(ep.q[lo:t + 1], ep.dq[lo:t + 1], ep.time[lo:t + 1], stats))
    return dict(x=np.stack(inputs), components=np.stack(components), motion=np.stack(motion))


def event_threshold(episodes, ids, domain):
    deltas = []
    for tid in ids:
        ep = episodes[(tid, domain)]
        valid = ep.starts[1:] == ep.starts[:-1]
        deltas.extend(np.linalg.norm(np.diff(ep.force, axis=0)[valid], axis=1).tolist())
    return max(float(np.quantile(deltas, .9)), .1) if deltas else .1


def make_stats(episodes, ids):
    stats = fit_stats(episodes, ids)
    stats["event_threshold_n"] = {name: event_threshold(episodes, ids, d)
                                  for d, name in enumerate(("real", "sim"))}
    return stats


class Blocks(Dataset):
    def __init__(self, episodes, ids, stats):
        self.episodes, self.stats = episodes, stats
        self.arrays = {(i, d): episode_arrays(episodes[(i, d)], stats, name)
                       for i in ids for d, name in enumerate(("real", "sim"))}
        self.keys, weights = [], []
        for i in ids:
            ep = episodes[(i, 0)]
            delta = np.linalg.norm(np.diff(ep.force, axis=0, prepend=ep.force[:1]), axis=1)
            delta[np.unique(ep.starts)] = 0
            event = np.minimum(4., 1 + delta / stats["event_threshold_n"]["real"])
            self.keys.extend((i, t) for t in range(len(ep.force)))
            weights.extend((.5 / len(event) + .5 * event / event.sum()) / len(ids))
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        i, anchor = self.keys[item]
        real, sim = self.episodes[(i, 0)], self.episodes[(i, 1)]
        start = max(real.starts[anchor], sim.starts[anchor])
        end = min(real.ends[anchor], sim.ends[anchor])
        original = np.arange(anchor - BLOCK, anchor + 1)
        index = np.maximum(original, start)
        valid = original >= start
        past_index = index[:, None] + np.arange(-PAST + 1, 1)
        future_index = index[:, None] + np.arange(1, FUTURE + 1)
        past_mask = (past_index >= start) & valid[:, None]
        future_mask = (future_index < end) & valid[:, None]
        return dict(x=np.stack([self.arrays[(i, d)]["x"][index] for d in (0, 1)]),
                    components=np.stack([self.arrays[(i, d)]["components"][index] for d in (0, 1)]),
                    motion=np.stack([self.arrays[(i, d)]["motion"][index] for d in (0, 1)]),
                    future=real.force[np.clip(future_index, start, end - 1)],
                    current=real.force[index], valid=valid, past_mask=past_mask, future_mask=future_mask)
