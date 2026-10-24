from __future__ import annotations

import copy

import numpy as np
import torch

from force_gap_forecast_v1.data import normalized_features, window
from force_gap_forecast_v1.metrics import aggregate, score_prediction as legacy_score


def score_prediction(pred, real, scale, input_ep):
    result = legacy_score(pred, real, scale, input_ep)
    n = len(real.force)
    for phase, row in result.items():
        chosen = (np.ones(n, bool) if phase == "all" else
                  np.arange(n) >= n - 8 if phase == "last_8_frames" else real.phase == phase)
        for step in (1, 3, 5):
            origins = np.flatnonzero(chosen & (np.arange(n) + step < real.ends) &
                                      (np.arange(n) + step < input_ep.ends))
            error = pred["future"][origins, step - 1] - real.force[origins + step]
            row[f"future_{step}_z_rmse_n"] = float(np.sqrt(np.mean(error[:, 2] ** 2))) if len(error) else None
        origins = np.flatnonzero(chosen & (np.arange(n) + 5 < real.ends) &
                                  (np.arange(n) + 5 < input_ep.ends))
        if len(origins):
            truth = np.abs(real.force[origins[:, None] + np.arange(1, 6), 2])
            forecast = np.abs(pred["future"][origins, :, 2])
            row["future_z_peak_mae_n"] = float(np.abs(truth.max(1) - forecast.max(1)).mean())
            event = np.ptp(truth, axis=1) > .1 * float(np.mean(scale))
            if event.any():
                indices = origins[event]
                pi = forecast[event].argmax(1) + 1
                ti = truth[event].argmax(1) + 1
                row["future_z_peak_timing_mae_s"] = float(np.abs(real.time[indices + pi] - real.time[indices + ti]).mean())
        if all(row.get(f"future_{step}_rmse_n") is not None for step in (1, 3, 5)):
            # Actual-N future errors only: current sensor bypass cannot improve this score.
            row["forecast_score_n"] = sum(weight * row[f"future_{step}_rmse_n"]
                                          for step, weight in ((1, .25), (3, .35), (5, .40)))
    return result


@torch.inference_mode()
def prediction_parts(model, ep, stats, domain, batch_size=128):
    model.eval()
    features = normalized_features(ep, stats, ("real", "sim")[domain])
    device = next(model.parameters()).device
    buckets = {}
    for lo in range(0, len(ep.force), batch_size):
        x = np.stack([window(features, t, ep.starts[t]) for t in range(lo, min(lo + batch_size, len(ep.force)))])
        parts = model.encode(torch.from_numpy(x).to(device), domain)
        for key, value in parts.items():
            buckets.setdefault(key, []).append(value)
    return {key: torch.cat(values)[None] for key, values in buckets.items()}


@torch.inference_mode()
def compose_prediction(model, parts, ep, domain, gain=None):
    device = next(model.parameters()).device
    reset = torch.zeros((1, len(ep.force)), dtype=torch.bool, device=device)
    reset[:, np.unique(ep.starts)] = True
    output = model.compose(parts, reset, torch.as_tensor(ep.force, device=device)[None], domain, gain)
    return {key: output[source][0].cpu().numpy() for key, source in
            dict(h="h", z="z", history="past", current="current", future="future").items()}


def predict_episode(model, ep, stats, domain, gain=None):
    return compose_prediction(model, prediction_parts(model, ep, stats, domain), ep, domain, gain)


def evaluate_gains(model, episodes, ids, stats, domain, gains):
    result = {float(gain): {} for gain in gains}
    for tid in ids:
        ep = episodes[(tid, domain)]
        parts = prediction_parts(model, ep, stats, domain)
        for gain in result:
            pred = compose_prediction(model, parts, ep, domain, gain)
            result[gain][str(tid)] = score_prediction(pred, episodes[(tid, 0)], stats["target"]["scale"], ep)
    return result


def evaluate(model, episodes, ids, stats, domain=1, save_dir=None, gain=None):
    result = {}
    for tid in ids:
        ep = episodes[(tid, domain)]
        pred = predict_episode(model, ep, stats, domain, gain)
        result[str(tid)] = score_prediction(pred, episodes[(tid, 0)], stats["target"]["scale"], ep)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_dir / f"traj_{tid}.npz", **pred, real_force=episodes[(tid, 0)].force,
                                input_force=ep.force, time=ep.time, real_time=episodes[(tid, 0)].time)
    return result


def persistence(episodes, ids, stats):
    rows = {}
    for tid in ids:
        ep = episodes[(tid, 0)]
        pred = dict(current=ep.force, future=np.repeat(ep.force[:, None], 5, axis=1))
        rows[str(tid)] = score_prediction(pred, ep, stats["target"]["scale"], ep)
    return rows


def force_reliance(model, episodes, ids, stats, prediction_dir):
    """Sensitivity diagnostic only: constant force may be outside the training distribution."""
    result = {}
    for tid in ids:
        original = episodes[(tid, 1)]
        changed = copy.copy(original)
        changed.force = np.broadcast_to(np.asarray(stats["sim"]["mean"][:3], np.float32), original.force.shape).copy()
        prediction = predict_episode(model, changed, stats, 1)
        with np.load(prediction_dir / f"traj_{tid}.npz", allow_pickle=False) as saved:
            response = {f"{key}_change_rms": float(np.sqrt(np.mean((saved[key] - prediction[key]) ** 2)))
                        for key in ("current", "future", "h")}
        result[str(tid)] = dict(response, force_at_train_mean_metrics=score_prediction(
            prediction, episodes[(tid, 0)], stats["target"]["scale"], changed)["all"])
    return result
