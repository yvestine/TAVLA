"""L_current + L_future + lambda_delta L_delta, with independent physical scales."""

from __future__ import annotations

import torch
from torch.nn import functional


def masked_huber(error, mask):
    values = functional.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=1.0)
    weight = mask.unsqueeze(-1).expand_as(values)
    return torch.where(weight, values, 0.0).sum() / weight.sum().clamp_min(1)


def objective(model, output, batch, delta_weight=0.5):
    prediction, truth, mask = output["force"], batch["target"], batch["mask"]
    error = (prediction - truth) / model.target_scale
    current = masked_huber(error[..., 0, :], mask[..., 0])
    future = masked_huber(error[..., 1:, :], mask[..., 1:])
    change_error = (torch.diff(prediction, dim=-2) - torch.diff(truth, dim=-2)) / model.increment_scale
    change = masked_huber(change_error, mask[..., 1:] & mask[..., :-1])
    loss = current + future + delta_weight * change
    return loss, {
        "total": float(loss.detach()),
        "current": float(current.detach()),
        "future": float(future.detach()),
        "delta": float(change.detach()),
    }
