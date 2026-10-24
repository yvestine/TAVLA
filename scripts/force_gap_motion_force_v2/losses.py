"""Forecast-first losses. No labels or observed future motion are model inputs."""
from __future__ import annotations

import torch


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def forecast_loss(model, output, batch, domain, args):
    scale = model.target_scale
    # A physical-N denominator shared across axes avoids downweighting Fz merely because its variance is high.
    unit = scale.mean().clamp_min(.1)
    axis = output["future"].new_tensor([1., 1., args.fz_weight])
    horizon = output["future"].new_tensor([1., 1., 1.5, 1.5, 2.])
    endpoints = batch["valid"].clone()
    endpoints[:, 0] = False  # The extra predecessor exists only to construct a causal delta.
    valid = batch["future_valid"]
    squared = ((output["future"] - batch["future"]) / unit).square()
    error = (squared * axis).mean(-1) * horizon
    future = masked_mean(error, valid & endpoints[..., None])
    current = masked_mean((((output["current"] - batch["current"]) / unit).square() * axis).mean(-1),
                          endpoints)
    past = masked_mean(((output["past"] - batch["past"]) / unit).square().mean(-1),
                       batch["past_valid"] & endpoints[..., None])

    prediction = output["future"]
    truth = batch["future"]
    # Only fully observed five-step windows contribute to peak amplitude and timing.
    complete = valid.all(-1) & endpoints
    pz, tz = prediction[..., 2].abs() / unit, truth[..., 2].abs() / unit
    peak = masked_mean((pz.amax(-1) - tz.amax(-1)).square(), complete)
    positions = torch.arange(1, 6, device=pz.device, dtype=pz.dtype)
    pt = (torch.softmax(pz / .25, -1) * positions).sum(-1)
    tt = (torch.softmax(tz / .25, -1) * positions).sum(-1)
    # Flat windows have no meaningful peak timing. Threshold uses train-fold scale, never held-out quantiles.
    event = complete & ((tz.amax(-1) - tz.amin(-1)) > .1)
    timing = masked_mean(((pt - tt) / 5).square(), event)
    pd = torch.diff(torch.cat((output["current"][..., None, :], prediction), -2), dim=-2)
    td = torch.diff(torch.cat((batch["current"][..., None, :], truth), -2), dim=-2)
    change = masked_mean(((pd - td) / unit).square().mean(-1), valid & endpoints[..., None])
    expected = masked_mean((output["expected_force"] - batch["x"][:, domain, :, -1, :3]).square().mean(-1),
                           endpoints)
    loss = (future + args.current_weight * current + .05 * past + args.peak_weight * peak +
            args.timing_weight * timing + .1 * change + .05 * expected +
            args.adapter_l2 * output["regularizer"])
    row = {name: float(value.detach()) for name, value in
           dict(total=loss, future=future, current=current, past=past, peak=peak, timing=timing,
                change=change, expected_force=expected).items()}
    return loss, row
