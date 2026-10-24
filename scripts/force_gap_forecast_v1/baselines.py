"""Train-fold-only affine / temporal ridge baselines with real-force targets."""
from __future__ import annotations

import numpy as np
from scipy import linalg

from . import FUTURE
from .data import HISTORY, normalized_features, window
from .metrics import aggregate, score_prediction


def design(ep, stats, kind):
    if kind == "affine":
        values = ep.force
    else:
        raw = normalized_features(ep, stats, "sim")
        # History of measured force, q, dq, dt; omit redundant delta for ARX.
        raw = raw[:, np.r_[0:3, 6:21]]
        values = np.stack([window(raw, t, ep.starts[t]).reshape(-1) for t in range(len(raw))])
    return np.column_stack((values, np.ones(len(values)))).astype(np.float64)


def fit(episodes, ids, stats, kind, alpha=1.):
    matrices = [design(episodes[(i, 1)], stats, kind) for i in ids]
    out = []
    horizons = [0] if kind == "affine" else list(range(FUTURE + 1))
    for step in horizons:
        xs, ys, ws = [], [], []
        for i, matrix in zip(ids, matrices):
            real, sim = episodes[(i, 0)], episodes[(i, 1)]
            index = np.arange(len(matrix))
            index = index[(index + step < real.ends) & (index + step < sim.ends)]
            xs.append(matrix[index])
            ys.append(real.force[index + step])
            ws.append(np.full(len(index), 1 / max(len(index), 1)))
        x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        # Equal mass per trajectory; ridge strength independent of frame count.
        weight *= len(x) / weight.sum()
        gram = x.T @ (weight[:, None] * x)
        penalty = np.eye(x.shape[-1]) * alpha
        penalty[-1, -1] = 0
        coef = linalg.solve(gram + penalty, x.T @ (weight[:, None] * y), assume_a="pos")
        out.append(coef)
    return {"kind": kind, "alpha": alpha, "coefficients": out}


def predict(model, sim, stats):
    if model["kind"] == "raw":
        current = sim.force.copy()
        future = np.repeat(current[:, None], FUTURE, axis=1)
    else:
        matrix = design(sim, stats, model["kind"])
        current = matrix @ model["coefficients"][0]
        future = (np.repeat(current[:, None], FUTURE, axis=1) if model["kind"] == "affine"
                  else np.stack([matrix @ coef for coef in model["coefficients"][1:]], axis=1))
    return {"current": current.astype(np.float32), "future": future.astype(np.float32)}


def evaluate(model, episodes, ids, stats):
    scale = np.asarray(stats["target"]["scale"])
    return {str(i): score_prediction(predict(model, episodes[(i, 1)], stats), episodes[(i, 0)], scale, episodes[(i, 1)])
            for i in ids}


def choose_alpha(episodes, train_ids, valid_ids, stats):
    candidates = []
    for alpha in (.1, 1., 10.):
        model = fit(episodes, train_ids, stats, "arx", alpha)
        rows = evaluate(model, episodes, valid_ids, stats)
        candidates.append((aggregate([r["all"] for r in rows.values()])["score"], alpha))
    return min(candidates)[1]
