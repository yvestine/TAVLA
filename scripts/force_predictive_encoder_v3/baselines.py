"""Force-only linear baselines. Sim hold is calibrated on the training trajectories."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from . import HISTORY, FUTURE
from .metrics import aggregate, forecast_metrics, masks


def design(ep, stats, domain, kind):
    values = (ep.force - np.asarray(stats[domain]["mean"][:3])) / np.asarray(stats[domain]["scale"][:3])
    if kind == "ar":
        values = np.stack([values[np.maximum(np.arange(t - HISTORY + 1, t + 1), ep.starts[t])].reshape(-1)
                           for t in range(len(values))])
    return np.column_stack((values, np.ones(len(values)))).astype(np.float64)


def fit(episodes, ids, stats, domain, kind, alpha):
    if kind == "hold" and domain == 0:
        return dict(kind="hold", domain=domain, coefficients=[])
    steps = [0] if kind == "hold" else range(1, FUTURE + 1)
    matrices = {i: design(episodes[(i, domain)], stats, ("real", "sim")[domain], kind) for i in ids}
    coefficients = []
    for step in steps:
        xs, ys, weights = [], [], []
        for tid in ids:
            real, source = episodes[(tid, 0)], episodes[(tid, domain)]
            ii = np.arange(len(real.force))
            ii = ii[(ii + step < real.ends) & (ii + step < source.ends)]
            if len(ii):
                xs.append(matrices[tid][ii])
                ys.append(real.force[ii + step])
                weights.append(np.full(len(ii), 1 / len(ii)))
        if not xs:
            raise ValueError("No valid baseline forecast targets")
        x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
        weight *= len(x) / weight.sum()
        penalty = np.eye(x.shape[1]) * alpha
        penalty[-1, -1] = 0
        coefficients.append(solve(x.T @ (weight[:, None] * x) + penalty, x.T @ (weight[:, None] * y), assume_a="pos"))
    return dict(kind=kind, domain=domain, coefficients=coefficients)


def predict(model, ep, stats):
    if model["kind"] == "hold" and model["domain"] == 0:
        return np.repeat(ep.force[:, None], FUTURE, axis=1)
    x = design(ep, stats, ("real", "sim")[model["domain"]], model["kind"])
    pred = np.stack([x @ c for c in model["coefficients"]], axis=1).astype(np.float32)
    return np.repeat(pred, FUTURE, axis=1) if model["kind"] == "hold" else pred


def evaluate(model, episodes, ids, stats):
    rows = {}
    for tid in ids:
        real, source = episodes[(tid, 0)], episodes[(tid, model["domain"])]
        pred = predict(model, source, stats)
        rows[str(tid)] = {phase: forecast_metrics(pred, real, source, stats["event_threshold_n"]["real"], mask)
                          for phase, mask in masks(real).items()}
    return rows


def select(episodes, train_ids, valid_ids, stats, domain):
    results = []
    for alpha in (.1, 1., 10.):
        model = fit(episodes, train_ids, stats, domain, "ar", alpha)
        rows = evaluate(model, episodes, valid_ids, stats)
        results.append((aggregate([row["all"] for row in rows.values()])["forecast_score_n"], alpha))
    return min(results)[1]
