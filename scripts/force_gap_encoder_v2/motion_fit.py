"""Encoder fits, matched legacy fits and source-only future-motion probes."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from insertion_alignment_encoder.train import PairWindows
from insertion_alignment_encoder.train_terminal_geometry import train_one as train_legacy

from .features import normalize
from .motion_features import torch_features
from .motion_model import FutureMotionProbe, MotionForceEncoder, conditional_alignment
from .train import metrics


def infer_encoder(model, windows, center, scale, device, old=False):
    model.eval()
    values = np.clip(windows.old / scale, -10, 10).astype(np.float32) if old else normalize(windows.features, center, scale)
    outputs, predictions, rec_losses = [], [], []
    with torch.inference_mode():
        for start in range(0, len(values), 512):
            x = torch.from_numpy(values[start:start + 512]).to(device)
            latent, logits, reconstruction = model(x)
            if not all(torch.isfinite(value).all() for value in (latent, logits, reconstruction)):
                raise FloatingPointError("Non-finite encoder inference output")
            outputs.append(latent.cpu().numpy())
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
            if not old:
                rec_losses.append(float(F.smooth_l1_loss(reconstruction, x[:, :, :15]).cpu()))
    return np.concatenate(outputs), np.concatenate(predictions), float(np.mean(rec_losses)) if rec_losses else None


def phase_report(windows, prediction):
    result = {}
    for domain in (0, 1):
        keep = (windows.y >= 0) & (windows.domain == domain)
        result["real" if domain == 0 else "sim"] = metrics(windows.y[keep], prediction[keep])
    return result


def _balanced_batch(windows, rng, batch_size):
    chosen = []
    for domain in (0, 1):
        domain_pool = np.flatnonzero(windows.domain == domain)
        for phase in (0, 1):
            pool = np.flatnonzero((windows.domain == domain) & (windows.y == phase))
            pool = pool if len(pool) else domain_pool
            chosen.extend(rng.choice(pool, batch_size // 8, replace=len(pool) < batch_size // 8))
        chosen.extend(rng.choice(domain_pool, batch_size // 4, replace=len(domain_pool) < batch_size // 4))
    chosen = np.asarray(chosen, np.int64)
    rng.shuffle(chosen)
    return chosen


def fit_motion_encoder(train, center, scale, args, device, seed, alignment_weight, validation=None, epochs=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MotionForceEncoder(args.history_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    original = torch.from_numpy(normalize(train.features, center, scale)).to(device)
    vectors = torch.from_numpy(train.vectors).to(device)
    scales = torch.from_numpy(train.scales).to(device)
    elapsed = torch.from_numpy(train.elapsed).to(device)
    labels = torch.as_tensor(train.y, device=device)
    domains = torch.as_tensor(train.domain, device=device)
    confidence = torch.from_numpy(train.signal_confidence).to(device)
    human = torch.from_numpy(train.human_weight).to(device)
    phase_weights = human * torch.where(domains == 1, confidence, torch.ones_like(confidence))
    reliable = (human == 1) & (confidence >= 0.5) & (labels >= 0)
    alignment_weights = human * confidence
    cn = torch.as_tensor(center, device=device)
    sn = torch.as_tensor(scale, device=device)
    best_state, best_epoch, best_score, stale = None, 0, -float("inf"), 0
    logs = []
    for epoch in range(1, (epochs or args.epochs) + 1):
        model.train()
        losses = []
        active_alignment = 0
        for _ in range(max(1, int(np.ceil(len(train) / args.batch_size)))):
            index = torch.as_tensor(_balanced_batch(train, rng, args.batch_size), device=device)
            z, logits, reconstructed = model(original[index])
            gain = torch.empty((len(index), 1, 1, 1), device=device).uniform_(np.log(.5), np.log(2.)).exp()
            raw_gain = torch_features(vectors[index] * gain, scales[index], elapsed[index], (0., 0., -1.))
            augmented = torch.clamp((raw_gain - cn) / sn, -10, 10)
            za, _, reconstructed_aug = model(augmented)
            ce = F.cross_entropy(logits, labels[index], ignore_index=-1, reduction="none")
            phase = (ce * phase_weights[index]).sum() / phase_weights[index].sum().clamp_min(1e-8)
            reconstruction = .5 * (F.smooth_l1_loss(reconstructed, original[index, :, :15]) +
                                    F.smooth_l1_loss(reconstructed_aug, augmented[:, :, :15]))
            gain_consistency = (z[:, :24] - za[:, :24]).square().mean()
            variance = F.relu(.1 - torch.sqrt(z[:, :24].var(dim=0, unbiased=False) + 1e-4)).mean()
            alignment = conditional_alignment(z, labels[index], domains[index], reliable[index],
                                               alignment_weights[index], index)
            active_alignment += int(float(alignment.detach()) > 0)
            loss = phase + .1 * reconstruction + .05 * gain_consistency + .01 * variance + alignment_weight * alignment
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite motion encoder loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            losses.append([float(value.detach()) for value in (loss, phase, reconstruction, gain_consistency, variance, alignment)])
        row = dict(zip(("total", "phase", "reconstruction", "gain_consistency", "variance", "conditional_alignment"),
                       map(float, np.mean(losses, axis=0)), strict=True))
        row.update(epoch=epoch, alignment_batches_with_nonzero_loss=active_alignment)
        if validation is not None:
            _, predicted, rec_loss = infer_encoder(model, validation, center, scale, device)
            phase_metrics = phase_report(validation, predicted)
            score = min(phase_metrics[d]["macro_f1"] for d in ("real", "sim")) - .01 * rec_loss
            row.update(validation=phase_metrics, validation_reconstruction=rec_loss, selection_score=score)
            if score > best_score + 1e-6:
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                best_epoch, best_score, stale = epoch, score, 0
            else:
                stale += 1
        logs.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(f"  encoder epoch={epoch} loss={row['total']:.5f} phase={row['phase']:.5f} "
                  f"rec={row['reconstruction']:.5f} align={row['conditional_alignment']:.5f}", flush=True)
        if validation is not None and stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    elif validation is not None:
        raise RuntimeError("Encoder epoch selection produced no valid candidate")
    return model, best_epoch if validation is not None else len(logs), logs


def legacy_pairs(windows, scales):
    lookup = {(int(t), int(f)): i for i, (t, f, d) in enumerate(zip(windows.trajectory, windows.frame, windows.domain, strict=True)) if d == 1}
    real = np.flatnonzero((windows.domain == 0) & (windows.y >= 0))
    sim = np.asarray([lookup[(int(windows.trajectory[i]), int(windows.frame[i]))] for i in real], np.int64)
    x = np.stack((windows.old[real], windows.old[sim]), axis=1)
    return PairWindows(np.clip(x / scales, -10, 10).astype(np.float32), windows.y[real],
                      windows.human_weight[real], windows.geometry[real], windows.trajectory[real], windows.frame[real])


def fit_legacy_encoder(train, scales, config, args, device, seed, validation=None, epochs=None):
    settings = SimpleNamespace(history_length=args.history_length, batch_size=128,
        geometry_scale_m=float(config.get("geometry_scale_m", .01)),
        xy_threshold_m=float(config.get("xy_threshold_m", .003)), patience=args.patience)
    return train_legacy(legacy_pairs(train, scales), legacy_pairs(validation, scales) if validation is not None else None,
                        settings, device, seed, epochs or args.epochs)


def input_normalizer(x):
    center = np.median(x, axis=0)
    scale = np.maximum(np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0), 1e-3)
    return center.astype(np.float32), scale.astype(np.float32)


def probe_normalizers(x, y):
    center, scale = input_normalizer(x)
    ycenter = y.mean(axis=0).astype(np.float32)
    yscale = np.maximum(y.std(axis=0), 1e-4).astype(np.float32)
    return {"input_center": center, "input_scale": scale, "target_center": ycenter, "target_scale": yscale}


def probe_input(x, stats):
    return np.clip((x - stats["input_center"]) / stats["input_scale"], -10, 10).astype(np.float32)


def probe_predict(model, x, stats, device):
    model.eval()
    normalized = probe_input(x, stats)
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            prediction = model(torch.from_numpy(normalized[start:start + 512]).to(device))
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite future-motion probe output")
            result.append(prediction.cpu().numpy())
    return np.concatenate(result) * stats["target_scale"] + stats["target_center"]


def fit_probe(x, y, args, device, seed, validation=None, epochs=None):
    if not len(x):
        raise ValueError("No valid future-motion training samples")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = probe_normalizers(x, y)
    xn = torch.from_numpy(probe_input(x, stats)).to(device)
    yn = torch.from_numpy(((y - stats["target_center"]) / stats["target_scale"]).astype(np.float32)).to(device)
    model = FutureMotionProbe(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best_state, best_epoch, best_loss, stale = None, 0, float("inf"), 0
    for epoch in range(1, (epochs or args.probe_epochs) + 1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(order), args.batch_size):
            index = torch.as_tensor(order[start:start + args.batch_size], device=device)
            loss = (model(xn[index]) - yn[index]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite probe loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
        if validation is not None:
            vx, vy = validation
            prediction = probe_predict(model, vx, stats, device)
            val_loss = float(np.mean(((prediction - vy) / stats["target_scale"]) ** 2))
            if val_loss < best_loss - 1e-6:
                best_epoch, best_loss, stale = epoch, val_loss, 0
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            else:
                stale += 1
            if stale >= args.probe_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elif validation is not None:
        raise RuntimeError("Probe epoch selection produced no valid candidate")
    return model, stats, best_epoch if validation is not None else epoch
