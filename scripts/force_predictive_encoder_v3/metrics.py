"""Trajectory-level forecasts and reconstruction, without measured-force bypasses."""
from __future__ import annotations

import copy
import numpy as np
import torch

from force_gap_forecast_v1.metrics import aggregate  # noqa: F401
from . import FUTURE, PAST
from .data import episode_arrays


def rmse(value):
    return float(np.sqrt(np.mean(np.square(value)))) if np.size(value) else None


def masks(ep):
    n = len(ep.force)
    return {"all": np.ones(n, bool), "last_8_frames": np.arange(n) >= n - 8,
            **{str(phase): ep.phase == phase for phase in sorted(set(ep.phase))}}


def forecast_metrics(future, real, source, threshold, chosen=None):
    n = len(real.force)
    chosen = np.ones(n, bool) if chosen is None else chosen
    indices = np.arange(n)
    row = {"frames": int(chosen.sum())}
    for step in range(1, FUTURE + 1):
        ii = indices[chosen & (indices + step < real.ends) & (indices + step < source.ends)]
        error = future[ii, step - 1] - real.force[ii + step]
        row[f"future_{step}_rmse_n"] = rmse(error)
        row[f"future_{step}_elapsed_s"] = float(np.mean(real.time[ii + step] - real.time[ii])) if len(ii) else None
        for axis, name in enumerate("xyz"):
            row[f"future_{step}_{name}_rmse_n"] = rmse(error[:, axis])
        active = np.linalg.norm(real.force[ii + step] - real.force[ii + step - 1], axis=1) >= threshold
        row[f"future_{step}_event_rmse_n"] = rmse(error[active])
    values = [row[f"future_{s}_rmse_n"] for s in (1, 3, 5)]
    row["forecast_score_n"] = float(np.dot(values, [.25, .35, .40])) if all(x is not None for x in values) else None
    ii = indices[chosen & (indices + FUTURE < real.ends) & (indices + FUTURE < source.ends)]
    row["future_delta_rmse_n"] = None
    row["future_z_peak_mae_n"] = None
    row["future_z_peak_timing_mae_s"] = None
    if len(ii):
        truth = real.force[ii[:, None] + np.arange(1, FUTURE + 1)]
        pred = future[ii]
        row["future_delta_rmse_n"] = rmse(np.diff(pred, axis=1) - np.diff(truth, axis=1))
        pz, tz = np.abs(pred[..., 2]), np.abs(truth[..., 2])
        row["future_z_peak_mae_n"] = float(np.abs(pz.max(1) - tz.max(1)).mean())
        active = np.ptp(tz, axis=1) >= threshold
        if active.any():
            origin = ii[active]
            row["future_z_peak_timing_mae_s"] = float(np.abs(
                real.time[origin + 1 + pz[active].argmax(1)] -
                real.time[origin + 1 + tz[active].argmax(1)]).mean())
    return row


def reconstruction_metrics(history, source, stats, domain, chosen=None):
    n = len(source.force)
    chosen = np.ones(n, bool) if chosen is None else chosen
    current = history[:, -1]
    ii = np.flatnonzero(chosen & (np.arange(n) - PAST + 1 >= source.starts))
    row = {"reconstruction_rmse_n": None, "reconstruction_nrmse": None,
           "high_band_mse": None, "stream_high_band_mse": None,
           "high_band_energy_ratio": None, "peak_amplitude_mae_n": None, "peak_timing_mae_s": None,
           "current_reconstruction_rmse_n": rmse(current[chosen] - source.force[chosen])}
    mask = chosen[1:] & chosen[:-1] & (source.starts[1:] == source.starts[:-1])
    row["reconstruction_delta_rmse_n"] = rmse((np.diff(current, axis=0) - np.diff(source.force, axis=0))[mask])
    if not len(ii):
        return row
    index = ii[:, None] + np.arange(-PAST + 1, 1)
    truth, pred = source.force[index], history[ii]
    scale = np.asarray(stats[domain]["scale"][:3])
    row["reconstruction_rmse_n"] = rmse(pred - truth)
    row["reconstruction_nrmse"] = rmse((pred - truth) / scale)
    taper = np.hanning(PAST)[None, :, None]
    keep = np.fft.rfftfreq(PAST) >= .25
    spectrum = lambda x: np.fft.rfft(x / scale * taper, axis=1, norm="ortho")[:, keep]
    target_spectrum, pred_spectrum = spectrum(truth), spectrum(pred)
    row["high_band_mse"] = float(np.mean(np.abs(pred_spectrum - target_spectrum) ** 2))
    row["stream_high_band_mse"] = float(np.mean(np.abs(spectrum(current[index]) - target_spectrum) ** 2))
    energy = float(np.mean(np.abs(target_spectrum) ** 2))
    row["high_band_energy_ratio"] = float(np.mean(np.abs(pred_spectrum) ** 2) / energy) if energy > 1e-12 else None
    p, t = np.abs(pred), np.abs(truth)
    event = np.ptp(t, axis=1) >= stats["event_threshold_n"][domain]
    if event.any():
        row["peak_amplitude_mae_n"] = float(np.abs(p.max(1) - t.max(1))[event].mean())
        time = source.time[index]
        pt = np.take_along_axis(time[..., None], p.argmax(1)[:, None, :], axis=1)[:, 0]
        tt = np.take_along_axis(time[..., None], t.argmax(1)[:, None, :], axis=1)[:, 0]
        row["peak_timing_mae_s"] = float(np.abs(pt - tt)[event].mean())
    return row


@torch.inference_mode()
def predict_episode(model, ep, stats, domain, with_motion=False, batch_size=128, prepared=None):
    model.eval()
    arrays = prepared or episode_arrays(ep, stats, ("real", "sim")[domain])
    device = next(model.parameters()).device
    rows = {}
    for start in range(0, len(ep.force), batch_size):
        x = torch.from_numpy(arrays["x"][start:start + batch_size]).to(device)
        motion = torch.from_numpy(arrays["motion"][start:start + batch_size]).to(device) if with_motion else None
        out = model(x, domain, motion)
        for key, value in out.items():
            rows.setdefault(key, []).append(value.cpu().numpy())
    result = {key: np.concatenate(value) for key, value in rows.items()}
    name = ("real", "sim")[domain]
    result["history"] = result.pop("reconstruction") * np.asarray(stats[name]["scale"][:3], np.float32) + np.asarray(
        stats[name]["mean"][:3], np.float32)
    return result


def evaluate(model, episodes, ids, stats, domains=(0, 1), with_motion=False, prepared=None, save_dir=None):
    rows = {}
    for domain in domains:
        name = ("real", "sim")[domain]
        rows[name] = {}
        for tid in ids:
            real, ep = episodes[(tid, 0)], episodes[(tid, domain)]
            pred = predict_episode(model, ep, stats, domain, with_motion,
                                   prepared=None if prepared is None else prepared[(tid, domain)])
            kinds = {"force": "future", **({"joint": "joint", "control": "control"} if with_motion else {})}
            row = {}
            for kind, field in kinds.items():
                row[kind] = {phase: forecast_metrics(pred[field], real, ep, stats["event_threshold_n"]["real"], mask)
                             for phase, mask in masks(real).items()}
            row["reconstruction"] = {phase: reconstruction_metrics(pred["history"], ep, stats, name, mask)
                                     for phase, mask in masks(real).items()}
            row["latent_std"] = np.std(pred["z"], axis=0).tolist()
            rows[name][str(tid)] = row
            if save_dir is not None:
                dest = save_dir / name
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest / f"traj_{tid}.npz", **pred, input_force=ep.force, real_force=real.force,
                                    time=ep.time, real_time=real.time, starts=ep.starts, ends=ep.ends,
                                    real_starts=real.starts, real_ends=real.ends, phase=real.phase.astype(str))
    return rows


def flatten_evaluation(rows, kind="force"):
    result = {}
    for domain, trajectories in rows.items():
        result[domain] = {}
        for tid, row in trajectories.items():
            result[domain][tid] = {phase: {**value, **row["reconstruction"][phase]}
                                    for phase, value in row[kind].items()}
    return result


def validation_summary(rows, kind="force"):
    return aggregate([value["all"] for tids in flatten_evaluation(rows, kind).values() for value in tids.values()])


def alignment_evaluation(model, episodes, train_ids, valid_ids, stats):
    # The same teacher and train-only whitening define units for all student candidates.
    train_z = np.concatenate([predict_episode(model, episodes[(i, 0)], stats, 0)["z"][:, :16] for i in train_ids])
    mean, scale = train_z.mean(0), np.maximum(train_z.std(0), .1)
    real_rows, sim_rows, distance, change, per_trajectory = [], [], [], [], {}
    for tid in valid_ids:
        pair = [predict_episode(model, episodes[(tid, d)], stats, d)["z"][:, :16] for d in (0, 1)]
        real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
        rows = []
        for size in (4, 8, 16):
            indices = [t for t in range(size - 1, len(real.force), size)
                       if t - size + 1 >= max(real.starts[t], sim.starts[t])]
            if not indices:
                continue
            pooled = [np.stack([(z[t - size + 1:t + 1].mean(0) - mean) / scale for t in indices]) for z in pair]
            d = np.mean((pooled[0] - pooled[1]) ** 2, axis=1)
            rows.extend(d.tolist())
            if size == 16:
                real_rows.extend(pooled[0])
                sim_rows.extend(pooled[1])
            if len(indices) > 1:
                valid = np.array([indices[j] - size + 1 >= max(real.starts[indices[j - 1]], sim.starts[indices[j - 1]])
                                  and real.starts[indices[j]] == real.starts[indices[j - 1]]
                                  and sim.starts[indices[j]] == sim.starts[indices[j - 1]] for j in range(1, len(indices))])
                delta_error = np.diff(pooled[0], axis=0) - np.diff(pooled[1], axis=0)
                if valid.any():
                    change.append(float(np.mean(delta_error[valid] ** 2)))
        if rows:
            per_trajectory[str(tid)] = float(np.mean(rows))
            distance.append(per_trajectory[str(tid)])
    retrieval = None
    if real_rows:
        r, s = np.asarray(real_rows), np.asarray(sim_rows)
        matrix = ((s[:, None] - r[None]) ** 2).mean(-1)
        retrieval = float(np.mean(matrix.argmin(1) == np.arange(len(s))))
    return dict(trend_distance=float(np.mean(distance)) if distance else None,
                trend_change_error=float(np.mean(change)) if change else None,
                retrieval_top1=retrieval, retrieval_candidates=len(real_rows), per_trajectory=per_trajectory,
                teacher_train_std=scale.tolist())


def force_audit(model, episodes, ids, stats):
    result = {}
    for d, name in enumerate(("real", "sim")):
        result[name] = {}
        for tid in ids:
            original = episodes[(tid, d)]
            perturbed = copy.deepcopy(original)
            rng = np.random.default_rng(7300 + tid)
            for start in np.unique(original.starts):
                end = original.ends[start]
                perturbed.force[start:end] = original.force[start:end][rng.permutation(end - start)]
            a = predict_episode(model, original, stats, d, True)
            b = predict_episode(model, perturbed, stats, d, True)
            result[name][str(tid)] = dict(latent_response_rms=rmse(a["z"] - b["z"]),
                force_prediction_response_n=rmse(a["future"] - b["future"]),
                joint_prediction_response_n=rmse(a["joint"] - b["joint"]),
                shuffled_forecast=forecast_metrics(b["future"], episodes[(tid, 0)], perturbed,
                                                  stats["event_threshold_n"]["real"]))
    return result
