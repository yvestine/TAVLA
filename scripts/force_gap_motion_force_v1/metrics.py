from __future__ import annotations

import numpy as np
import torch

from force_gap_forecast_v1.data import normalized_features, window
from force_gap_forecast_v1.metrics import aggregate, score_prediction


@torch.inference_mode()
def prediction_parts(model, ep, stats, domain, batch_size=128):
    """One causal forward pass supports the whole validation gain grid."""
    features = normalized_features(ep, stats, ("real", "sim")[domain])
    device = next(model.parameters()).device
    model.eval()
    pieces = [[], [], [], []]
    for lo in range(0, len(ep.force), batch_size):
        x = np.stack([window(features, t, ep.starts[t]) for t in range(lo, min(lo + batch_size, len(ep.force)))])
        for bucket, value in zip(pieces, model.encode(torch.from_numpy(x).to(device), domain)):
            bucket.append(value)
    h, history, dp, df = [torch.cat(bucket) for bucket in pieces]
    current = history[:, -1]
    delta = torch.zeros_like(current)
    delta[1:] = current[1:] - current[:-1]
    delta[torch.as_tensor(np.unique(ep.starts), device=device)] = 0
    future = model.motion.future(torch.cat((current, delta, h), -1))
    return {key: value.cpu().numpy() for key, value in
            dict(h=h, base_history=history, base_future=future, dp=dp, df=df).items()}


def compose_prediction(parts, ep, domain, gain):
    history = parts["base_history"] + gain * parts["dp"]
    current = ep.force.copy() if domain == 0 else history[:, -1]
    delta = np.zeros_like(current)
    delta[1:] = current[1:] - current[:-1]
    delta[np.unique(ep.starts)] = 0
    return {"h": parts["h"], "history": history, "current": current,
            "z": np.concatenate((current, delta, parts["h"]), -1),
            "future": parts["base_future"] + gain * parts["df"]}


def predict_episode(model, ep, stats, domain):
    return compose_prediction(prediction_parts(model, ep, stats, domain), ep, domain,
                              float(model.residual_gain.item()))


def evaluate_gains(model, episodes, ids, stats, domain, gains):
    result = {float(gain): {} for gain in gains}
    scale = np.asarray(stats["target"]["scale"])
    for tid in ids:
        ep = episodes[(tid, domain)]
        parts = prediction_parts(model, ep, stats, domain)
        for gain in result:
            pred = compose_prediction(parts, ep, domain, gain)
            result[gain][str(tid)] = score_prediction(pred, episodes[(tid, 0)], scale, ep)
    return result


def evaluate(model, episodes, ids, stats, domain=1, save_dir=None):
    result = {}
    scale = np.asarray(stats["target"]["scale"])
    for tid in ids:
        ep = episodes[(tid, domain)]
        pred = predict_episode(model, ep, stats, domain)
        result[str(tid)] = score_prediction(pred, episodes[(tid, 0)], scale, ep)
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_dir / f"traj_{tid}.npz", **pred,
                                real_force=episodes[(tid, 0)].force, input_force=ep.force,
                                time=ep.time, real_time=episodes[(tid, 0)].time)
    return result
