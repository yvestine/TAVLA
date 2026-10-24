"""Physical metrics, including the current-to-next increment and streaming changes."""

from __future__ import annotations

import numpy as np
import torch

from . import FUTURE
from .data import targets


def rmse(error):
    return float(np.sqrt(np.mean(np.square(error)))) if np.size(error) else None


def aggregate(rows):
    return {
        key: float(np.mean(values)) if values else None
        for key in rows[0]
        for values in [[row[key] for row in rows if row[key] is not None]]
    }


def score_prediction(prediction, real, source, threshold, chosen=None):
    n = len(real.force)
    chosen = np.ones(n, bool) if chosen is None else chosen
    fi = np.arange(n)[:, None] + np.arange(FUTURE + 1)
    valid = (fi < np.minimum(real.ends, source.ends)[:, None]) & chosen[:, None]
    truth = real.force[np.minimum(fi, n - 1)]
    error = prediction - truth
    row = {"frames": int(chosen.sum()), "current_rmse_n": rmse(error[valid[:, 0], 0])}
    for h in range(FUTURE + 1):
        prefix = "current" if h == 0 else f"future_{h}"
        if h:
            row[f"{prefix}_rmse_n"] = rmse(error[valid[:, h], h])
            ii = np.flatnonzero(valid[:, h])
            row[f"{prefix}_elapsed_s"] = float(np.mean(real.time[ii + h] - real.time[ii])) if len(ii) else None
            active = np.linalg.norm(truth[:, h] - truth[:, h - 1], axis=-1) >= threshold
            row[f"{prefix}_event_rmse_n"] = rmse(error[valid[:, h] & active, h])
        for j, axis in enumerate("xyz"):
            row[f"{prefix}_{axis}_rmse_n"] = rmse(error[valid[:, h], h, j])
    scores = [row[f"future_{h}_rmse_n"] for h in (1, 3, 5)]
    row["forecast_score_n"] = float(np.dot(scores, [0.25, 0.35, 0.40])) if all(v is not None for v in scores) else None
    increment_error = np.diff(prediction, axis=1) - np.diff(truth, axis=1)
    increment_mask = valid[:, 1:] & valid[:, :-1]
    row["increment_rmse_n"] = rmse(increment_error[increment_mask])
    row["first_increment_rmse_n"] = rmse(increment_error[increment_mask[:, 0], 0])
    continuous = (
        chosen[1:] & chosen[:-1] & (real.starts[1:] == real.starts[:-1]) & (source.starts[1:] == source.starts[:-1])
    )
    row["stream_current_delta_rmse_n"] = rmse(
        (np.diff(prediction[:, 0], axis=0) - np.diff(real.force, axis=0))[continuous]
    )
    return row


def trajectory_metrics(prediction, real, source, threshold):
    return {
        name: score_prediction(prediction, real, source, threshold, mask)
        for name, mask in {
            "all": np.ones(len(real.force), bool),
            **{str(phase): real.phase == phase for phase in sorted(set(real.phase))},
        }.items()
    }


def latent_diagnostics(latents):
    z = np.concatenate(latents).astype(np.float64)
    eigen = np.maximum(np.linalg.eigvalsh(np.cov(z.T, bias=True)), 0)
    total = eigen.sum()
    p = eigen[eigen > 0] / total if total > 1e-12 else np.array([])
    return {
        "effective_rank": float(np.exp(-np.sum(p * np.log(p)))) if len(p) else 0.0,
        "first_pc_variance_fraction": float(eigen[-1] / total) if total > 1e-12 else 0.0,
        "channel_std": z.std(0).tolist(),
    }


@torch.inference_mode()
def predict_episode(model, inputs, domain, batch_size=128):
    model.eval()
    device = next(model.parameters()).device
    rows = {}
    for lo in range(0, len(inputs), batch_size):
        for key, value in model(inputs[lo : lo + batch_size].to(device), domain).items():
            rows.setdefault(key, []).append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in rows.items()}


def evaluate(model, episodes, ids, arrays, save_dir=None, domains=(0, 1)):
    result = {}
    for domain in domains:
        name = ("real", "sim")[domain]
        rows, latents = {}, []
        for tid in ids:
            real, source = episodes[(tid, 0)], episodes[(tid, domain)]
            out = predict_episode(model, arrays[(tid, domain)], domain)
            rows[str(tid)] = trajectory_metrics(out["force"], real, source, model.stats["target"]["event_threshold_n"])
            latents.append(out["z"])
            if save_dir is not None:
                dest = save_dir / name
                dest.mkdir(parents=True, exist_ok=True)
                truth, mask = targets(episodes, tid, np.arange(len(real.force)))
                np.savez_compressed(
                    dest / f"traj_{tid}.npz",
                    **out,
                    target=truth,
                    valid=mask,
                    input_force=source.force,
                    real_force=real.force,
                    time=source.time,
                    real_time=real.time,
                    starts=source.starts,
                    ends=source.ends,
                    real_starts=real.starts,
                    real_ends=real.ends,
                )
        result[name] = {
            "summary": aggregate([row["all"] for row in rows.values()]),
            "trajectories": rows,
            "latent": latent_diagnostics(latents),
        }
    return result
