"""Causal v3 input features, train-only target scales, and paired real-force labels."""

from __future__ import annotations

from force_predictive_encoder_v3.data import force_window
import numpy as np
import torch
from torch.utils.data import Dataset

from . import BLOCK
from . import FUTURE
from . import HISTORY


def make_stats(episodes, train_ids):
    stats = {"fit_trajectory_ids": sorted(train_ids)}
    for domain, name in enumerate(("real", "sim")):
        values = np.concatenate([episodes[(i, domain)].force for i in train_ids])
        stats[name] = {"mean": values.mean(0).tolist(), "scale": np.maximum(values.std(0), 0.1).tolist()}
    increments = []
    for tid in train_ids:
        ep = episodes[(tid, 0)]
        continuous = ep.starts[1:] == ep.starts[:-1]
        increments.append(np.diff(ep.force, axis=0)[continuous])
    increments = np.concatenate(increments)
    if not len(increments):
        raise ValueError("Training data needs continuous pairs for increment statistics")
    stats["target"] = {
        "mean": stats["real"]["mean"],
        "scale": stats["real"]["scale"],
        "increment_scale_n": np.maximum(increments.std(0), 0.05).tolist(),
        "event_threshold_n": max(float(np.quantile(np.linalg.norm(increments, axis=1), 0.9)), 0.1),
        "domain": "real",
    }
    return stats


def episode_inputs(ep, stats, domain):
    return np.stack(
        [
            force_window(
                ep.force[max(int(ep.starts[t]), t - HISTORY + 1) : t + 1],
                ep.time[max(int(ep.starts[t]), t - HISTORY + 1) : t + 1],
                stats,
                domain,
            )[0]
            for t in range(len(ep.force))
        ]
    )


def targets(episodes, tid, indices):
    real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
    fi = np.asarray(indices)[:, None] + np.arange(FUTURE + 1)
    ends = np.minimum(real.ends[indices], sim.ends[indices])
    return real.force[np.minimum(fi, len(real.force) - 1)], fi < ends[:, None]


class Blocks(Dataset):
    def __init__(self, episodes, ids, arrays, stats):
        self.episodes, self.arrays = episodes, arrays
        self.keys, weights = [], []
        for tid in ids:
            ep = episodes[(tid, 0)]
            delta = np.linalg.norm(np.diff(ep.force, axis=0, prepend=ep.force[:1]), axis=1)
            delta[np.unique(ep.starts)] = 0
            event = np.minimum(4.0, 1 + delta / stats["target"]["event_threshold_n"])
            self.keys.extend((tid, t) for t in range(len(ep.force)))
            weights.extend((0.5 / len(event) + 0.5 * event / event.sum()) / len(ids))
        self.weights = torch.tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        tid, anchor = self.keys[index]
        start = max(self.episodes[(tid, d)].starts[anchor] for d in (0, 1))
        original = np.arange(anchor - BLOCK + 1, anchor + 1)
        ii = np.maximum(original, start)
        truth, mask = targets(self.episodes, tid, ii)
        mask &= (original >= start)[:, None]
        return {
            "x": torch.stack([self.arrays[(tid, d)][ii] for d in (0, 1)]),
            "target": torch.from_numpy(truth),
            "mask": torch.from_numpy(mask),
        }
