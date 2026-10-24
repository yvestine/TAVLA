import torch


def masked_mse(prediction, target, mask):
    expanded = mask[..., None].expand_as(prediction)
    error = (prediction - target).square()
    return (error * expanded).sum() / expanded.sum().clamp_min(1)


def band_loss(pred, target, mask):
    complete = mask.all(dim=-1)
    if not complete.any():
        return pred.sum() * 0
    pred, target = pred[complete], target[complete]
    bins = torch.fft.rfftfreq(pred.shape[-2], device=pred.device)
    keep = (bins >= .25) & (bins <= .5)
    error = torch.fft.rfft(pred, dim=-2, norm="ortho") - torch.fft.rfft(target, dim=-2, norm="ortho")
    return error[:, keep].abs().square().mean()


def objective(model, output, batch, teacher=None, alignment=True, frequency=True):
    scale = model.target_scale
    # Exclude the extra previous endpoint from reconstruction losses.
    pred_past = model.normalize_force(output["past"][:, 1:])
    pred_future = model.normalize_force(output["future"][:, 1:])
    true_past = model.normalize_force(batch["past"][:, 1:])
    true_future = model.normalize_force(batch["future"][:, 1:])
    pm, fm = batch["past_valid"][:, 1:], batch["future_valid"][:, 1:]
    history = masked_mse(pred_past, true_past, pm)
    future = masked_mse(pred_future, true_future, fm)
    history_delta = masked_mse(torch.diff(pred_past, dim=-2), torch.diff(true_past, dim=-2), pm[..., 1:] & pm[..., :-1])
    pred_seq = torch.cat((model.normalize_force(output["z"][:, 1:, None, :3]), pred_future), -2)
    true_seq = torch.cat((model.normalize_force(batch["current"][:, 1:, None]), true_future), -2)
    future_delta = masked_mse(torch.diff(pred_seq, dim=-2), torch.diff(true_seq, dim=-2), fm)
    current_delta = masked_mse(output["z"][:, 1:, 3:6] / scale, batch["delta"][:, 1:] / scale,
                               batch["delta_valid"][:, 1:])
    stream_pred = model.normalize_force(output["z"][:, 1:, :3])
    stream_true = model.normalize_force(batch["current"][:, 1:])
    stream_mask = batch["valid"][:, 1:]
    high = (band_loss(pred_past, true_past, pm) + band_loss(pred_future, true_future, fm)
            + band_loss(stream_pred, stream_true, stream_mask)) / 3
    complete = pm.all(dim=-1)
    if complete.any():
        # Physical absolute amplitudes; remove no mean before identifying peaks.
        peak = ((output["past"][:, 1:][complete].abs().amax(-2)
                 - batch["past"][:, 1:][complete].abs().amax(-2)) / scale).square().mean()
    else:
        peak = history * 0
    stream_complete = stream_mask.all(dim=-1)
    if stream_complete.any():
        stream_peak = ((output["z"][:, 1:, :3][stream_complete].abs().amax(-2)
                        - batch["current"][:, 1:][stream_complete].abs().amax(-2)) / scale).square().mean()
        peak = (peak + stream_peak) / 2
    h = output["h"][:, 1:][batch["valid"][:, 1:]]
    variance = torch.relu(.1 - h.std(dim=0, unbiased=False)).square().mean() if len(h) else history * 0
    pair = trend = history * 0
    if teacher is not None and alignment:
        pair = masked_mse(output["h"][:, 1:], teacher["h"][:, 1:].detach(), batch["valid"][:, 1:])
        trend = masked_mse(torch.diff(output["h"], dim=1), torch.diff(teacher["h"].detach(), dim=1),
                           batch["valid"][:, 1:] & batch["valid"][:, :-1])
    delta = (history_delta + future_delta + current_delta) / 3
    total = history + future + .5 * delta + (.1 * high if frequency else 0) + .1 * peak + .01 * variance + .1 * pair + .05 * trend
    return total, {name: float(value.detach()) for name, value in
                   dict(history=history, future=future, delta=delta, high_band=high, peak=peak,
                        variance=variance, pair=pair, trend=trend, total=total).items()}
