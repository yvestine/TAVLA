"""Fixed-alpha, force-only ridge baselines fitted on training trajectories only."""

from __future__ import annotations

import numpy as np

from . import FUTURE
from . import HISTORY
from .data import targets
from .metrics import aggregate
from .metrics import trajectory_metrics


def design(ep, stats, domain, kind):
    values = (ep.force - np.asarray(stats[domain]["mean"])) / np.asarray(stats[domain]["scale"])
    if kind == "ar":
        values = np.stack(
            [
                values[np.maximum(np.arange(t - HISTORY + 1, t + 1), ep.starts[t])].reshape(-1)
                for t in range(len(values))
            ]
        )
    return np.column_stack((values, np.ones(len(values)))).astype(np.float64)


def fit(episodes, ids, stats, domain, kind, alpha=1.0):
    if kind == "hold" and domain == 0:
        return None
    name = ("real", "sim")[domain]
    matrices = {i: design(episodes[(i, domain)], stats, name, kind) for i in ids}
    coefficients = []
    for h in [0] if kind == "hold" else range(FUTURE + 1):
        xs, ys, weights = [], [], []
        for tid in ids:
            truth, mask = targets(episodes, tid, np.arange(len(matrices[tid])))
            keep = mask[:, h]
            if keep.any():
                xs.append(matrices[tid][keep])
                ys.append(truth[keep, h])
                weights.append(np.full(keep.sum(), 1.0 / keep.sum()))
        if not xs:
            raise ValueError("No valid baseline targets")
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
        w *= len(w) / w.sum()
        penalty = np.eye(x.shape[1]) * alpha
        penalty[-1, -1] = 0
        coefficients.append(np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T @ (w[:, None] * y)))
    return np.stack(coefficients)


def evaluate(coefficients, episodes, ids, stats, domain, kind):
    rows = {}
    for tid in ids:
        real, source = episodes[(tid, 0)], episodes[(tid, domain)]
        if coefficients is None:
            pred = source.force[:, None]
        else:
            x = design(source, stats, ("real", "sim")[domain], kind)
            pred = np.einsum("nf,hfc->nhc", x, coefficients).astype(np.float32)
        if kind == "hold":
            pred = np.repeat(pred, FUTURE + 1, axis=1)
        rows[str(tid)] = trajectory_metrics(pred, real, source, stats["target"]["event_threshold_n"])
    return {"summary": aggregate([row["all"] for row in rows.values()]), "trajectories": rows}
