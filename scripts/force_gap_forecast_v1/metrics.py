"""Trajectory-equal waveform metrics; event timing uses actual timestamps."""
from __future__ import annotations

import numpy as np
import torch

from . import FUTURE, HISTORY, PAST
from .data import normalized_features, window


@torch.inference_mode()
def predict_episode(model, ep, stats, domain, motion_only=False, batch_size=128):
    features = normalized_features(ep, stats, ("real", "sim")[domain], motion_only)
    device = next(model.parameters()).device
    model.eval()
    enc = (model.real_encoder, model.sim_encoder)[domain]
    hs, histories = [], []
    for lo in range(0, len(ep.force), batch_size):
        x = np.stack([window(features, t, ep.starts[t]) for t in range(lo, min(lo + batch_size, len(ep.force)))])
        h = enc(torch.from_numpy(x).to(device))
        hs.append(h)
        histories.append(model.history(h))
    h, history = torch.cat(hs), torch.cat(histories)
    current = torch.as_tensor(ep.force, device=device) if domain == 0 and not motion_only else history[:, -1]
    delta = torch.zeros_like(current)
    delta[1:] = current[1:] - current[:-1]
    delta[torch.as_tensor(np.unique(ep.starts), device=device)] = 0
    z = torch.cat((current, delta, h), -1)
    return {"current": current.cpu().numpy(), "history": history.cpu().numpy(), "z": z.cpu().numpy(),
            "future": model.future(z).cpu().numpy(), "h": h.cpu().numpy()}


def _rmse(x, scale=None):
    if len(x) == 0:
        return None
    return float(np.sqrt(np.mean(np.square(x if scale is None else x / scale))))


def _events(force, time, starts):
    change = np.zeros_like(force)
    change[1:] = np.diff(force, axis=0)
    change[np.unique(starts)] = 0
    magnitude = np.linalg.norm(change, axis=1)
    threshold = max(float(np.quantile(magnitude, .9)), .1)
    candidates = np.flatnonzero(magnitude >= threshold)
    chosen = []
    for index in sorted(candidates, key=lambda i: -magnitude[i]):
        if all(abs(index - other) > 3 for other in chosen):
            chosen.append(int(index))
    return sorted(chosen)


def waveform_metrics(pred, real, scale, frame_mask=None, input_ep=None):
    n = len(real.force)
    chosen = np.ones(n, bool) if frame_mask is None else frame_mask.copy()
    input_ep = real if input_ep is None else input_ep
    future_errors, future_elapsed = [], []
    for step in range(1, FUTURE + 1):
        indices = np.arange(n)
        keep = chosen & (indices + step < real.ends) & (indices + step < input_ep.ends)
        indices = indices[keep]
        error = pred["future"][indices, step - 1] - real.force[indices + step]
        future_errors.append(error)
        future_elapsed.append((real.time[indices + step] - real.time[indices]).tolist())
    current_error = pred["current"][chosen] - real.force[chosen]
    delta_real = np.diff(real.force, axis=0)
    delta_pred = np.diff(pred["current"], axis=0)
    valid_delta = chosen[1:] & chosen[:-1] & (real.starts[1:] == real.starts[:-1]) & (input_ep.starts[1:] == input_ep.starts[:-1])
    past_errors, band_errors, band_true = [], [], []
    for t in np.flatnonzero(chosen):
        if t - PAST + 1 >= max(real.starts[t], input_ep.starts[t]):
            truth = real.force[t - PAST + 1:t + 1]
            if "history" in pred:
                past_errors.append(pred["history"][t] - truth)
            # Canonical recovered time series, not independently decoded past windows.
            prediction = pred["current"][t - PAST + 1:t + 1]
            bins = np.fft.rfftfreq(PAST)
            keep = (bins >= .25) & (bins <= .5)
            spectrum = np.fft.rfft(prediction / scale, axis=0, norm="ortho")[keep]
            target = np.fft.rfft(truth / scale, axis=0, norm="ortho")[keep]
            band_errors.append(np.abs(spectrum - target)**2)
            band_true.append(np.abs(target)**2)
    peak_error, peak_time = [], []
    # Events are selected from target force by a fixed rule; no fitted time shift.
    for center in _events(real.force, real.time, real.starts):
        if not chosen[center]:
            continue
        lo = max(center - 3, real.starts[center], input_ep.starts[center])
        hi = min(center + 4, real.ends[center], input_ep.ends[center])
        truth, prediction = np.abs(real.force[lo:hi]), np.abs(pred["current"][lo:hi])
        for axis in range(3):
            ti, pi = int(np.argmax(truth[:, axis])), int(np.argmax(prediction[:, axis]))
            peak_error.append(abs(float(prediction[pi, axis] - truth[ti, axis])))
            peak_time.append(abs(float(real.time[lo + pi] - real.time[lo + ti])))
    future_nrmse = [_rmse(error, scale) for error in future_errors]
    current_nrmse = _rmse(current_error, scale)
    valid_future = [x for x in future_nrmse if x is not None]
    future_mean = float(np.mean(valid_future)) if valid_future else None
    metric = {"frames": int(chosen.sum()), "current_rmse_n": _rmse(current_error),
              "current_mae_n": float(np.abs(current_error).mean()) if len(current_error) else None,
              "current_nrmse": current_nrmse,
              "future_nrmse_mean": future_mean,
              "score": .5 * (current_nrmse + future_mean) if current_nrmse is not None and future_mean is not None else None,
              "history_nrmse": _rmse(np.asarray(past_errors), scale),
              "delta_rmse_n": _rmse((delta_pred - delta_real)[valid_delta]),
              "peak_amplitude_mae_n": float(np.mean(peak_error)) if peak_error else None,
              "peak_timing_mae_s": float(np.mean(peak_time)) if peak_time else None,
              "high_band_mse": float(np.mean(band_errors)) if band_errors else None,
              "high_band_true_energy": float(np.mean(band_true)) if band_true else None,
              "predicted_std_n": np.std(pred["current"][chosen], axis=0).tolist() if chosen.any() else None,
              "target_std_n": np.std(real.force[chosen], axis=0).tolist() if chosen.any() else None}
    for axis, name in enumerate(("x", "y", "z")):
        metric[f"current_{name}_rmse_n"] = _rmse(current_error[:, axis])
    for step in (1, 3, 5):
        error = future_errors[step - 1]
        metric[f"future_{step}_rmse_n"] = _rmse(error)
        metric[f"future_{step}_mae_n"] = float(np.abs(error).mean()) if len(error) else None
        metric[f"future_{step}_elapsed_mean_s"] = float(np.mean(future_elapsed[step - 1])) if future_elapsed[step - 1] else None
        events = np.asarray(_events(real.force, real.time, real.starts), np.int64)
        origins = events - step
        keep = (origins >= 0)
        origins, events = origins[keep], events[keep]
        keep = chosen[origins] & (events < real.ends[origins]) & (events < input_ep.ends[origins])
        origins, events = origins[keep], events[keep]
        metric[f"future_{step}_peak_event_rmse_n"] = _rmse(pred["future"][origins, step - 1] - real.force[events])
    if "h" in pred:
        metric["latent_std_mean"] = float(np.std(pred["h"][chosen], axis=0).mean()) if chosen.any() else None
    # Teacher epoch selection does not benefit from its exact-force bypass.
    metric["teacher_score"] = (.5 * (metric["history_nrmse"] + future_mean)
                               if metric["history_nrmse"] is not None and future_mean is not None else None)
    return metric


def aggregate(rows):
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return {key: float(np.mean([row[key] for row in rows if isinstance(row.get(key), (float, int))]))
            for key in sorted(keys) if any(isinstance(row.get(key), (float, int)) for row in rows)}


def score_prediction(pred, real, scale, input_ep=None):
    result = {"all": waveform_metrics(pred, real, scale, input_ep=input_ep)}
    for phase in sorted(set(real.phase)):
        result[phase] = waveform_metrics(pred, real, scale, real.phase == phase, input_ep)
    # Include the final eight samples even if historical labels lack terminal_hold.
    terminal = np.arange(len(real.force)) >= len(real.force) - PAST
    result["last_8_frames"] = waveform_metrics(pred, real, scale, terminal, input_ep)
    return result
