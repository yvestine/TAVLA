"""Physical prediction, waveform preservation and trend-only paired alignment."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_mse(pred, target, mask):
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    weight = mask.expand_as(pred)
    return ((pred - target).square() * weight).sum() / weight.sum().clamp_min(1)


def spectral_loss(pred, target, mask):
    losses = []
    for size in (8, 16):
        for start in range(0, pred.shape[-2] - size + 1, size // 2):
            complete = mask[..., start:start + size].all(-1)
            if not complete.any():
                continue
            p = pred[..., start:start + size, :][complete]
            t = target[..., start:start + size, :][complete]
            taper = torch.hann_window(size, periodic=False, device=p.device, dtype=p.dtype)[None, :, None]
            error = torch.fft.rfft((p - t) * taper, dim=-2, norm="ortho")
            bins = torch.fft.rfftfreq(size, device=p.device)
            weight = torch.where(bins >= .25, 2., 1.)[None, :, None]
            losses.append((error.abs().square() * weight).mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0


def variance_covariance(z, valid):
    h = z[valid]
    if len(h) < 2:
        return z.sum() * 0
    variance = torch.relu(.5 - torch.sqrt(h.var(0, unbiased=False) + 1e-4)).square().mean()
    h = h - h.mean(0)
    covariance = h.T @ h / (len(h) - 1)
    offdiag = covariance - torch.diag_embed(covariance.diagonal())
    return variance + offdiag.square().sum() / h.shape[-1]


def trend_alignment(student, teacher, valid):
    """Causal segment means and their changes; no high-frequency phase locking."""
    terms = []
    for size in (4, 8, 16):
        if student.shape[1] < size:
            continue
        s = F.avg_pool1d(student[..., :16].transpose(1, 2), size, stride=1).transpose(1, 2)
        t = F.avg_pool1d(teacher[..., :16].detach().transpose(1, 2), size, stride=1).transpose(1, 2)
        complete = valid.unfold(1, size, 1).all(-1)
        terms.append(masked_mse(s, t, complete))
        if s.shape[1] > 1:
            terms.append(.5 * masked_mse(s[:, 1:] - s[:, :-1], t[:, 1:] - t[:, :-1],
                                        complete[:, 1:] & complete[:, :-1]))
    return torch.stack(terms).mean() if terms else student.sum() * 0


def peak_loss(pred, target, mask):
    complete = mask.all(-1)
    if not complete.any():
        return pred.sum() * 0
    p, t = pred[complete].abs(), target[complete].abs()
    amplitude = (p.amax(-2) - t.amax(-2)).square().mean()
    positions = torch.linspace(0, 1, p.shape[-2], device=p.device)[None, :, None]
    pt = (torch.softmax(p / .25, dim=-2) * positions).sum(-2)
    tt = (torch.softmax(t / .25, dim=-2) * positions).sum(-2)
    active = t.amax(-2) - t.amin(-2) > .1
    timing = ((pt - tt).square() * active).sum() / active.sum().clamp_min(1)
    return amplitude + .5 * timing


def forecast_loss(model, future, batch):
    mask = batch["future_mask"].clone()
    mask[:, 0] = False
    pred, target = model.normalize_target(future), model.normalize_target(batch["future"])
    value = masked_mse(pred, target, mask)
    change = masked_mse(torch.diff(pred, dim=-2), torch.diff(target, dim=-2), mask[..., 1:] & mask[..., :-1])
    return value + .25 * change + .1 * peak_loss(pred, target, mask)


def objective(model, output, batch, domain, align_weight=0., teacher=None):
    valid = batch["valid"].clone()
    valid[:, 0] = False
    mask = batch["past_mask"] & valid[..., None]
    parts = batch["components"][:, domain]
    truth = parts.sum(-3)
    reconstructed = output["reconstruction"]
    reconstruction = (masked_mse(reconstructed, truth, mask) +
                      masked_mse(output["components"][..., 0, :, :], parts[..., 0, :, :], mask)) / 2
    if model.use_fast:
        reconstruction = reconstruction + .5 * masked_mse(output["components"][..., 1, :, :], parts[..., 1, :, :], mask)
    delta = masked_mse(torch.diff(reconstructed, dim=-2), torch.diff(truth, dim=-2), mask[..., 1:] & mask[..., :-1])
    spectral = spectral_loss(reconstructed, truth, mask)
    peak = peak_loss(reconstructed, truth, mask)
    future = forecast_loss(model, output["future"], batch)
    z = output["z"] if model.use_fast else output["z"][..., :16]
    regularizer = variance_covariance(z, valid)
    alignment = (trend_alignment(output["z"], teacher["z"], batch["valid"])
                 if teacher is not None and align_weight else future * 0)
    total = future + .5 * reconstruction + .25 * delta + .1 * spectral + .1 * peak + .01 * regularizer + align_weight * alignment
    return total, {name: float(value.detach()) for name, value in dict(
        total=total, future=future, reconstruction=reconstruction, delta=delta, spectral=spectral,
        peak=peak, regularizer=regularizer, alignment=alignment).items()}
