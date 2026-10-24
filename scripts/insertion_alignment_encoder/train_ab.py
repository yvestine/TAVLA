#!/usr/bin/env python3
"""Controlled A/B experiment: legacy recipe versus extra terminal geometry.

A uses the original pre-first-hit windows, sampler, optimizer and losses. B
adds one geometry-only batch, restricted to reviewed near-terminal phases.
Both share splits, scales and base random draws. Full-budget
CV continues after the *simulated* legacy early stop only to record curves;
the legacy choice is frozen at that stop. A keeps its legacy final epoch; B
selects its own common CV epoch using active-insertion depth error subject to
classification bounds calibrated on A. Held-out data never select epochs.
All outputs are candidates for subsequent policy experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .model import ForceAlignmentEncoder
from .train import classification_metrics, fit_feature_scales, make_windows, read_labels
from .train_terminal_geometry import (
    InsertionWindows,
    _aggregate_validation,
    _depth_masks,
    _domain_report,
    _evaluate,
    _merge_geometry_metrics,
    _save_new_checkpoint,
    _subset,
    _take,
    load_terminal_episodes,
    make_full_insertion_windows,
    window_coverage,
)


RECIPE = "legacy_terminal_ab_v2"
DOMAINS = ("real", "sim")
BANDS = ("0_3mm", "3_5mm", "5_8mm", "8_10mm", "other")


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_signature(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def phase_table(labels, trajectory: int, frames: np.ndarray) -> np.ndarray:
    return np.asarray([
        next(row["label"] for row in labels[trajectory] if row["start"] <= frame <= row["end"])
        for frame in frames
    ], dtype="U32")


def audit_geometry_references(episodes, labels, args):
    """Qualify the recording's terminal reference, not a measured hole bottom.

    load_terminal_episodes has already checked finite force/pose, increasing
    paired timestamps, paired EE positions, quaternion validity and labels.
    An uncertain final phase cannot establish a successful terminal reference.
    The last three terminal frames must stay within the configured XY/Z range.
    """
    result = {}
    for trajectory, intervals in labels.items():
        for interval in intervals:
            if interval["confidence"] not in {"high", "medium"}:
                raise ValueError(f"traj_{trajectory}: unknown confidence {interval['confidence']!r}")
        final = intervals[-1]
        geometry = episodes[(trajectory, 0)].geometry
        tail = geometry[max(final["start"], len(geometry) - 3):]
        xy = float(np.max(np.linalg.norm(tail[:, :2], axis=1)) * 1000)
        z = float(np.max(np.abs(tail[:, 2])) * 1000)
        failures = []
        if final["label"] != "terminal":
            failures.append("final phase is not terminal")
        if len(tail) < 3:
            failures.append("fewer than three terminal reference frames")
        if xy > args.reference_stability_mm or z > args.reference_stability_mm:
            failures.append("terminal reference tail exceeds stability range")
        result[f"traj_{trajectory}"] = {
            "eligible": not failures, "failures": failures,
            "final_phase": final["label"], "tail_frames": len(tail),
            "tail_xy_radius_max_mm": xy, "tail_abs_z_max_mm": z,
            "scope": "recording consistency only; independent hole-bottom depth is unverified",
        }
    return result


def window_tables(episodes, labels, fit_ids, ids, args):
    """Fit scales on training pre-hit frames, preserving the original recipe."""
    scales = fit_feature_scales(episodes, fit_ids)
    old = make_windows(episodes, labels, ids, scales, args.history_length, args.boundary_guard)
    phases = np.empty(len(old.y), dtype="U32")
    for trajectory in ids:
        mask = old.trajectory == trajectory
        phases[mask] = phase_table(labels, trajectory, old.frame[mask])
    base = InsertionWindows(old.x, old.y, old.weight, old.geometry,
                            old.trajectory, old.frame, phases)
    full = make_full_insertion_windows(episodes, labels, ids, scales,
                                       args.history_length, args.boundary_guard)
    # Classification evaluation always uses exactly the legacy windows.
    original = {(int(t), int(f)): int(y) for t, f, y in zip(base.trajectory, base.frame, base.y)}
    full.y = np.asarray([original.get((int(t), int(f)), -1)
                         for t, f in zip(full.trajectory, full.frame)], dtype=np.int64)
    if np.count_nonzero(full.y >= 0) != len(base.y):
        raise ValueError("full evaluation table does not contain every legacy classification frame")
    radius = np.linalg.norm(full.geometry[:, :2], axis=1)
    keep = ((radius < 0.003) & (np.abs(full.geometry[:, 2]) < 0.010)
            & np.isin(full.phase, ["aligned_progress", "terminal"])
            & np.isin(full.trajectory, args.geometry_eligible_trajectories)
            & (full.y < 0))
    if not args.include_medium_geometry:
        keep &= full.weight == 1.0
    extra = _take(full, keep)
    return base, full, extra, scales


def terminal_probability(windows: InsertionWindows) -> np.ndarray:
    """Equal phase, then depth band, then trajectory, then frames within it."""
    if not len(windows.y):
        raise ValueError("no additional qualified near-terminal geometry windows")
    probability = np.zeros(len(windows.y), dtype=np.float64)
    masks = _depth_masks(windows)
    phases = np.unique(windows.phase)
    for phase in phases:
        phase_mask = windows.phase == phase
        bands = [mask & phase_mask for name, mask in masks.items()
                 if name != "other" and np.any(mask & phase_mask)]
        for band_mask in bands:
            trajectories = np.unique(windows.trajectory[band_mask])
            for trajectory in trajectories:
                selected = band_mask & (windows.trajectory == trajectory)
                probability[selected] = 1 / (len(phases) * len(bands) * len(trajectories) * selected.sum())
    if not np.all(probability > 0):
        raise ValueError("additional geometry window falls outside the configured depth bands")
    return probability / probability.sum()


def coverage(windows: InsertionWindows) -> dict[str, Any]:
    report = window_coverage(windows)
    report["input_at_clip_limit_fraction"] = (
        float(np.mean(np.abs(windows.x) >= 10)) if len(windows.y) else None)
    report["trajectory_frame_signature"] = array_signature(
        np.stack((windows.trajectory, windows.frame), axis=1))
    report["by_confidence"] = {
        "high": int(np.count_nonzero(windows.weight == 1.0)),
        "medium": int(np.count_nonzero(windows.weight < 1.0)),
    }
    report["by_phase_and_trajectory"] = {
        str(phase): {f"traj_{trajectory}": int(np.count_nonzero(
            (windows.phase == phase) & (windows.trajectory == trajectory)))
            for trajectory in np.unique(windows.trajectory[windows.phase == phase])}
        for phase in np.unique(windows.phase)
    }
    return report


def add_target_spread(report, windows: InsertionWindows) -> None:
    """Show target spread alongside prediction spread, including empty groups."""
    def annotate(metrics, mask):
        metrics["target_z_std_mm"] = (float(np.std(windows.geometry[mask, 2]) * 1000)
                                       if np.any(mask) else None)
    annotate(report["geometry"], np.ones(len(windows.y), dtype=bool))
    for band, mask in _depth_masks(windows).items():
        annotate(report["geometry_by_depth"][band], mask)
        for phase, bands in report["geometry_by_phase_and_depth"].items():
            annotate(bands[band], mask & (windows.phase == phase))


def reports_from_predictions(windows, probability, estimate):
    reports = {}
    for domain, name in enumerate(DOMAINS):
        prediction = probability[:, domain].argmax(axis=-1)
        report = _domain_report(windows, prediction, estimate[:, domain])
        add_target_spread(report, windows)
        for trajectory in np.unique(windows.trajectory):
            mask = windows.trajectory == trajectory
            local = _take(windows, mask)
            local_report = _domain_report(local, prediction[mask], estimate[mask, domain])
            add_target_spread(local_report, local)
            local_report.pop("by_trajectory")
            report["by_trajectory"][f"traj_{trajectory}"] = local_report
        reports[name] = report
    return reports


def evaluate(model, windows, device, args):
    _, probability, estimate = _evaluate(model, windows, device, args.geometry_scale_m)
    if not np.isfinite(probability).all() or not np.isfinite(estimate).all():
        raise RuntimeError("non-finite encoder prediction")
    return reports_from_predictions(windows, probability, estimate), probability, estimate


def merge_geometry(reports):
    merged = _merge_geometry_metrics(reports)
    nonempty = [report for report in reports if report["count"]]
    count = sum(report["count"] for report in nonempty)
    second = sum(report["count"] * (report["target_z_std_mm"] ** 2
                                    + report["target_z_mean_mm"] ** 2) for report in nonempty)
    merged["target_z_std_mm"] = (float(np.sqrt(max(0, second / count
                                                 - merged["target_z_mean_mm"] ** 2))) if count else None)
    return merged


def pool_reports(reports):
    pooled = _aggregate_validation(reports)
    for domain in DOMAINS:
        local = [report[domain] for report in reports]
        pooled[domain]["geometry"] = merge_geometry([report["geometry"] for report in local])
        pooled[domain]["geometry_by_depth"] = {
            band: merge_geometry([report["geometry_by_depth"][band] for report in local]) for band in BANDS}
        for phase, bands in pooled[domain]["geometry_by_phase_and_depth"].items():
            for band in bands:
                bands[band] = merge_geometry([
                    report["geometry_by_phase_and_depth"][phase][band]
                    for report in local if phase in report["geometry_by_phase_and_depth"]])
    return pooled


def legacy_score(reports):
    return float(sum(
        0.25 * reports[domain]["class"]["per_class"]["aligned_progress"]["f1"]
        + 0.25 * ((reports[domain]["geometry"]["direction_cosine"] or 0.0) + 1.0) / 2
        for domain in DOMAINS))


def class_values(reports):
    result = {}
    for domain in DOMAINS:
        quality = reports[domain]["class"]
        result[f"{domain}.macro_f1"] = quality["macro_f1"]
        for label, values in quality["per_class"].items():
            result[f"{domain}.{label}.recall"] = values["recall"]
    return result


def calibrate_class_bounds(a_curve, args, seed):
    """Trajectory bootstrap of one common A epoch, never held-out predictions.

    The reference epoch maximizes the worst classification metric across both
    domains and both class recalls. A's legacy final-fit epoch is separate.
    Bootstrap bounds describe validation-trajectory variability, while the
    multiple training seeds separately describe training variability.
    """
    reference = max(a_curve, key=lambda row: min(class_values(row["metrics"]).values()))
    reports = reference["metrics"]
    trajectories = sorted(reports["real"]["by_trajectory"])
    if trajectories != sorted(reports["sim"]["by_trajectory"]):
        raise ValueError("A calibration trajectories differ across domains")
    matrices = {
        domain: np.asarray([reports[domain]["by_trajectory"][trajectory]["class"]["confusion_matrix"]
                            for trajectory in trajectories], dtype=np.int64)
        for domain in DOMAINS
    }
    rng = np.random.default_rng(seed + 300_000)
    values = {key: [] for key in class_values(reports)}
    for _ in range(args.selection_bootstrap_samples):
        sampled = rng.integers(0, len(trajectories), size=len(trajectories))
        bootstrap = {}
        for domain in DOMAINS:
            confusion = matrices[domain][sampled].sum(axis=0)
            truth = np.repeat([0, 0, 1, 1], confusion.reshape(-1))
            prediction = np.repeat([0, 1, 0, 1], confusion.reshape(-1))
            bootstrap[domain] = {"class": classification_metrics(truth, prediction)}
        for key, value in class_values(bootstrap).items():
            values[key].append(value)
    return {
        "method": "paired trajectory bootstrap of A's best worst-class common CV epoch",
        "reference_epoch": reference["epoch"], "reference_metrics": class_values(reports),
        "reference_has_both_class_recalls": all(value > 0 for value in class_values(reports).values()),
        "bootstrap_samples": args.selection_bootstrap_samples, "lower_quantile": args.selection_class_quantile,
        "lower_bounds": {key: float(np.quantile(samples, args.selection_class_quantile))
                         for key, samples in values.items()},
        "bootstrap_std": {key: float(np.std(samples, ddof=1)) for key, samples in values.items()},
        "scope": "diagnostic constraint; not a deployment guarantee or training-seed confidence interval",
    }


def active_depth_score(reports, eligible_ids):
    """Equal trajectory weight, then equal final/approach group weight.

    Only aligned_progress contributes. Terminal holds cannot hide active
    insertion error. The worse domain drives selection; geometry references
    that failed the recording audit remain in reports but cannot select epochs.
    """
    eligible = {f"traj_{trajectory}" for trajectory in eligible_ids}
    result = {}
    for domain in DOMAINS:
        band_errors, band_trajectories = {}, {}
        for band in BANDS[:-1]:
            errors = []
            for trajectory, report in reports[domain]["by_trajectory"].items():
                if trajectory not in eligible:
                    continue
                geometry = report["geometry_by_phase_and_depth"].get("aligned_progress", {}).get(band)
                if geometry and geometry["count"]:
                    errors.append(geometry["z_mae_mm"])
            band_errors[band] = float(np.mean(errors)) if errors else None
            band_trajectories[band] = len(errors)
        approach = [band_errors[band] for band in BANDS[1:-1] if band_errors[band] is not None]
        final = band_errors["0_3mm"]
        balanced = 0.5 * (final + float(np.mean(approach))) if final is not None and approach else None
        result[domain] = {"balanced_z_mae_mm": balanced, "by_depth_z_mae_mm": band_errors,
                          "trajectories_by_depth": band_trajectories}
    scores = [result[domain]["balanced_z_mae_mm"] for domain in DOMAINS]
    return {"worst_domain_z_mae_mm": max(scores) if all(value is not None for value in scores) else None,
            "domains": result}


def select_candidate_epoch(b_curve, calibration, args):
    summaries = []
    for row in b_curve:
        metrics = class_values(row["metrics"])
        failures = [key for key, floor in calibration["lower_bounds"].items() if metrics[key] < floor]
        if not calibration["reference_has_both_class_recalls"]:
            failures.append("A reference lacks a two-class validation signal")
        violation = sum(max(0.0, floor - metrics[key]) for key, floor in calibration["lower_bounds"].items())
        depth = active_depth_score(row["metrics"], args.geometry_eligible_trajectories)
        summaries.append({"epoch": row["epoch"], "class_metrics": metrics, "class_failures": failures,
                          "class_total_shortfall": violation, "depth": depth,
                          "eligible": not failures and depth["worst_domain_z_mae_mm"] is not None})
    candidates = [row for row in summaries if row["eligible"]]
    if candidates:
        selected = min(candidates, key=lambda row: (row["depth"]["worst_domain_z_mae_mm"], row["epoch"]))
        status = "cv_class_constraints_met"
    else:
        # Save a clearly labeled diagnostic result instead of losing the run.
        available = [row for row in summaries if row["depth"]["worst_domain_z_mae_mm"] is not None]
        if not available:
            raise ValueError("CV lacks qualified active insertion in both final and approach depth groups")
        selected = min(available, key=lambda row: (row["class_total_shortfall"],
                                                  row["depth"]["worst_domain_z_mae_mm"], row["epoch"]))
        status = "diagnostic_class_tradeoff"
    return {"status": status, "selected_epoch": selected["epoch"], "selected": selected,
            "class_calibration": calibration, "epochs": summaries,
            "rule": "minimize worse-domain active depth error within A-calibrated classification bounds; terminal excluded"}


def constant_baselines(base, extra, valid):
    """Medians fitted only on this training fold; classification is omitted."""
    result = {}
    for name, table in (("legacy_training_median", base), ("terminal_training_median", extra)):
        target = np.median(table.geometry, axis=0).astype(np.float32)
        estimate = np.broadcast_to(target, (len(valid.y), 2, 3)).copy()
        probability = np.zeros((len(valid.y), 2, 2), dtype=np.float32)
        reports = reports_from_predictions(valid, probability, estimate)
        for report in reports.values():
            report.pop("class")
            for trajectory_report in report["by_trajectory"].values():
                trajectory_report.pop("class")
        result[name] = {"training_target_median_mm": (target * 1000).tolist(), "metrics": reports}
    return result


def fit(base, extra, valid_base, valid_full, args, device, seed, epochs, run_dir, variant, context):
    """A follows the legacy trainer; B masks invalid geometry and adds a batch."""
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    terminal_rng = np.random.default_rng(seed + 100_000)
    model = ForceAlignmentEncoder(latent_dim=32, dropout=0.10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x = torch.from_numpy(base.x).to(device)
    y = torch.from_numpy(base.y).to(device)
    weight = torch.from_numpy(base.weight).to(device)
    target = torch.from_numpy(base.geometry / args.geometry_scale_m).to(device)
    qualified_base = torch.from_numpy(np.isin(base.trajectory, args.geometry_eligible_trajectories)).to(device)
    counts = np.bincount(base.y, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"training fold has a missing class: {counts.tolist()}")
    probability = np.asarray([1.0 / counts[label] for label in base.y]) * base.weight
    probability /= probability.sum()
    terminal_p = terminal_probability(extra)
    extra_x = torch.from_numpy(extra.x).to(device) if variant == "B" else None
    extra_target = torch.from_numpy(extra.geometry / args.geometry_scale_m).to(device) if variant == "B" else None
    steps = int(np.ceil(len(base.y) / args.batch_size))
    best_score, best_epoch, stale, stop_epoch = -float("inf"), 0, 0, None
    history = []
    started = time.monotonic()
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    with (run_dir / "history.jsonl").open("x", encoding="utf-8") as trace:
        for epoch in range(1, epochs + 1):
            model.train()
            sampled = rng.choice(len(base.y), size=steps * args.batch_size, replace=True, p=probability)
            sums = dict.fromkeys(("total", "class", "consistency", "geometry", "direction", "terminal_geometry",
                                  "class_batch_accuracy", "gradient_norm"), 0.0)
            for start in range(0, len(sampled), args.batch_size):
                index = torch.as_tensor(sampled[start:start + args.batch_size], device=device)
                gain = torch.empty((len(index), 2, 1, 1), device=device).uniform_(0.9, 1.1)
                inputs = (x[index] * gain).reshape(-1, args.history_length, 6)
                class_target = y[index].repeat_interleave(2)
                class_weight = weight[index].repeat_interleave(2)
                geom_target = target[index].repeat_interleave(2, dim=0)
                latent, logits, geom_hat = model(inputs)
                class_loss = (F.cross_entropy(logits, class_target, reduction="none") * class_weight).mean()
                paired = latent.reshape(-1, 2, 32)
                consistency = (1 - F.cosine_similarity(paired[:, 0], paired[:, 1], dim=-1)).mean()
                geometry_mask = qualified_base[index].repeat_interleave(2) if variant == "B" else None
                if geometry_mask is None:
                    geom_loss = F.smooth_l1_loss(geom_hat, geom_target, beta=0.2)
                else:
                    geom_loss = (F.smooth_l1_loss(geom_hat[geometry_mask], geom_target[geometry_mask], beta=0.2)
                                 if geometry_mask.any() else torch.zeros((), device=device))
                direction_mask = torch.linalg.vector_norm(geom_target[:, :2], dim=-1) >= 0.003 / args.geometry_scale_m
                if geometry_mask is not None:
                    direction_mask &= geometry_mask
                direction_loss = ((1 - F.cosine_similarity(geom_hat[direction_mask, :2],
                                                          geom_target[direction_mask, :2], dim=-1)).mean()
                                  if direction_mask.any() else torch.zeros((), device=device))
                loss = class_loss + 0.15 * consistency + 0.12 * geom_loss + 0.05 * direction_loss
                terminal_loss = torch.zeros((), device=device)
                if variant == "B":
                    selected = terminal_rng.choice(len(extra.y), size=args.terminal_batch_size, replace=True, p=terminal_p)
                    extra_index = torch.as_tensor(selected, device=device)
                    # Restore the base RNG stream after auxiliary gain/dropout,
                    # so B's extra batch cannot change the next base draw.
                    devices = [device.index] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        auxiliary_seed = seed + 200_000 + (epoch - 1) * steps + start // args.batch_size
                        torch.random.default_generator.manual_seed(auxiliary_seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed(auxiliary_seed)
                        extra_gain = torch.empty((len(extra_index), 2, 1, 1), device=device).uniform_(0.9, 1.1)
                        _, _, prediction = model((extra_x[extra_index] * extra_gain).reshape(-1, args.history_length, 6))
                        terminal_loss = F.smooth_l1_loss(prediction, extra_target[extra_index].repeat_interleave(2, dim=0), beta=0.2)
                    loss = loss + args.terminal_loss_weight * terminal_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{run_dir.name}: non-finite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"{run_dir.name}: non-finite gradient at epoch {epoch}")
                optimizer.step()
                for key, value in (("total", loss), ("class", class_loss), ("consistency", consistency),
                                   ("geometry", geom_loss), ("direction", direction_loss),
                                   ("terminal_geometry", terminal_loss), ("gradient_norm", norm)):
                    sums[key] += float(value.detach().item())
                sums["class_batch_accuracy"] += float((logits.argmax(dim=-1) == class_target).float().mean().item())
            validation = legacy_validation = None
            score = None
            if valid_full is not None:
                legacy_validation = evaluate(model, valid_base, device, args)[0]
                validation = evaluate(model, valid_full, device, args)[0]
                score = legacy_score(legacy_validation)
                # Continue diagnostic curves without changing the legacy choice
                # after the original patience rule would have stopped.
                if stop_epoch is None:
                    if score > best_score + 1e-6:
                        best_score, best_epoch, stale = score, epoch, 0
                    else:
                        stale += 1
                        if stale >= args.patience:
                            stop_epoch = epoch
            row = {"epoch": epoch, "optimizer_steps": epoch * steps, "steps_per_epoch": steps,
                   "elapsed_s": time.monotonic() - started, "train": {key: value / steps for key, value in sums.items()},
                   "validation": validation, "legacy_validation": legacy_validation, "legacy_score": score,
                   "legacy_best_epoch": best_epoch or None, "legacy_stop_epoch": stop_epoch}
            checkpoint = checkpoints / f"epoch_{epoch:04d}.pt"
            _save_new_checkpoint({"artifact_format": "tavla_force_alignment_encoder_cv_snapshot_v1",
                                  "training_recipe": RECIPE, "variant": variant, "seed": seed, "epoch": epoch,
                                  "optimizer_steps": epoch * steps, "context": context,
                                  "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, checkpoint)
            row["checkpoint"] = str(checkpoint)
            history.append(row)
            trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            trace.flush()
            if epoch == 1 or epoch % args.log_every == 0 or epoch == epochs:
                message = (f"{variant}/{run_dir.name} epoch={epoch}/{epochs} steps={epoch * steps} "
                           f"loss={row['train']['total']:.4f} terminal_loss={row['train']['terminal_geometry']:.4f}")
                if validation is not None:
                    message += (f" real_f1={validation['real']['class']['macro_f1']:.3f} "
                                f"sim_f1={validation['sim']['class']['macro_f1']:.3f} legacy_best={best_epoch} legacy_stop={stop_epoch}")
                print(message, flush=True)
    return model, history, {"best_epoch": best_epoch if valid_full is not None else epochs,
                            "simulated_stop_epoch": stop_epoch, "epochs_ran": epochs}


def metric_delta(a, b):
    """B minus A; negative depth error is better, positive F1 is better."""
    result = {}
    for domain in DOMAINS:
        first, second = a[domain], b[domain]
        if first["class"]["count"] != second["class"]["count"]:
            raise ValueError("A/B classification evaluation counts differ")
        by_phase = {}
        for phase in first["geometry_by_phase_and_depth"]:
            by_phase[phase] = {}
            for band in BANDS:
                old = first["geometry_by_phase_and_depth"][phase][band]
                new = second["geometry_by_phase_and_depth"][phase][band]
                if old["count"] != new["count"]:
                    raise ValueError("A/B geometry evaluation counts differ")
                by_phase[phase][band] = {
                    "count": old["count"], "z_mae_mm": new["z_mae_mm"] - old["z_mae_mm"] if old["count"] else None}
        result[domain] = {
            "macro_f1": second["class"]["macro_f1"] - first["class"]["macro_f1"],
            "recall": {name: second["class"]["per_class"][name]["recall"] - value["recall"]
                       for name, value in first["class"]["per_class"].items()},
            "z_mae_mm_by_phase_and_depth": by_phase,
        }
    return result


def artifact_config(args, variant, seed, epochs):
    return {
        "training_recipe": RECIPE, "variant": variant, "candidate_status": "diagnostic_requires_policy_validation",
        "history_length": args.history_length, "baseline_frames": args.baseline_frames,
        "sample_period_s": 0.1, "latent_dim": 32, "geometry_scale_m": args.geometry_scale_m,
        "geometry_supervision": True,
        "geometry_target": "peg-tip pose minus terminal peg-tip pose; terminal reference, not independent metrology",
        "xy_threshold_m": 0.003, "z_threshold_m": 0.003,
        "tool_offset_local_mm": args.tool_offset_mm.tolist(),
        "classes": ["misaligned_contact", "aligned_progress"], "input_force_key": args.force_key,
        "seed": seed, "final_training_epochs": epochs,
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip": 3.0},
        "dropout": 0.10, "label_smoothing": 0.0, "force_gain_range": [0.9, 1.1],
        "batch_size_pairs": args.batch_size, "boundary_guard": args.boundary_guard,
        "base_loss_weights": {"classification": 1.0, "pair_cosine": 0.15, "geometry": 0.12, "direction": 0.05},
        "terminal_batch_size_pairs": args.terminal_batch_size if variant == "B" else 0,
        "terminal_loss_weight": args.terminal_loss_weight if variant == "B" else 0.0,
        "include_medium_geometry": args.include_medium_geometry,
        "reference_stability_mm": args.reference_stability_mm,
        "geometry_eligible_trajectories": args.geometry_eligible_trajectories,
        "terminal_sampling": "equal phase / depth band / trajectory / frame; qualified geometry; XY<3mm, absZ<10mm; no base duplicates",
        "geometry_confidence_usage": "high/medium phase confidence is metadata for extra geometry; original class confidence is unchanged",
        "base_geometry_reference_usage": ("legacy targets retained" if variant == "A" else
                                           "base geometry/direction losses exclude unqualified terminal references; class/pair losses retain all base frames"),
        "final_epoch_rule": ("median of A's legacy fold choices" if variant == "A" else
                             "own common CV epoch: active depth error subject to A-calibrated classification bounds"),
        "selection_class_quantile": args.selection_class_quantile,
        "selection_bootstrap_samples": args.selection_bootstrap_samples,
    }


def run_seed(seed, args, episodes, labels, reference, fold_ids, train_ids, test_ids, geometry_meta):
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir()
    prepared = []
    for index, valid_ids in enumerate(fold_ids):
        fit_ids = sorted(set(train_ids) - set(valid_ids))
        base, full, extra, scales = window_tables(episodes, labels, fit_ids, train_ids, args)
        prepared.append({
            "base": _subset(base, fit_ids), "extra": _subset(extra, fit_ids),
            "valid_base": _subset(base, valid_ids), "valid_full": _subset(full, valid_ids),
            "context": {"fold": index + 1, "fit_trajectories": fit_ids, "valid_trajectories": valid_ids,
                        "feature_scales": scales.tolist()},
        })
    final_base, final_full, final_extra, final_scales = window_tables(
        episodes, labels, train_ids, train_ids + test_ids, args)
    train_base, test_base = _subset(final_base, train_ids), _subset(final_base, test_ids)
    train_extra, test_full = _subset(final_extra, train_ids), _subset(final_full, test_ids)
    # Reject absent auxiliary targets before spending time on A's CV.
    terminal_probability(train_extra)
    for fold in prepared:
        terminal_probability(fold["extra"])
    preparation = {
        "seed": seed, "geometry_reference_audit": args.geometry_reference_audit, "folds": [
            {**fold["context"], "base_training": coverage(fold["base"]),
             "B_base_geometry_masked_pairs": int(np.count_nonzero(~np.isin(
                 fold["base"].trajectory, args.geometry_eligible_trajectories))),
             "extra_training": coverage(fold["extra"]), "validation": coverage(fold["valid_full"]),
             "constant_baselines": constant_baselines(fold["base"], fold["extra"], fold["valid_full"])}
            for fold in prepared],
        "final": {"feature_scales": final_scales.tolist(), "base_training": coverage(train_base),
                  "B_base_geometry_masked_pairs": int(np.count_nonzero(~np.isin(
                      train_base.trajectory, args.geometry_eligible_trajectories))),
                  "extra_training": coverage(train_extra), "held_out": coverage(test_full),
                  "legacy_held_out": coverage(test_base)},
        "reference_comparison": {
            "original_train_pair_windows": reference["metrics"]["train_pair_windows"],
            "current_train_pair_windows": len(train_base.y),
            "original_test_pair_windows": reference["metrics"]["test_pair_windows"],
            "current_test_pair_windows": len(test_base.y),
            "feature_scales_max_absolute_difference": float(np.max(np.abs(
                final_scales - np.asarray(reference["preprocess"]["feature_scales"], dtype=np.float32)))),
        },
    }
    write_json(seed_dir / "data_manifest.json", preparation)
    print(f"seed={seed}: base_train={len(train_base.y)}, extra_train={len(train_extra.y)}, "
          f"extra_active={np.count_nonzero(train_extra.phase == 'aligned_progress')}, "
          f"extra_medium={np.count_nonzero(train_extra.weight < 1.0)}, "
          f"held_out_class={len(test_base.y)}, held_out_geometry={len(test_full.y)}", flush=True)
    variants = {}
    variant_curves, final_epochs_by_variant = {}, {}
    calibration = None
    for variant in ("A", "B"):
        variant_dir = seed_dir / variant
        variant_dir.mkdir()
        histories, selections = [], []
        for index, fold in enumerate(prepared):
            model, history, selection = fit(
                fold["base"], fold["extra"], fold["valid_base"], fold["valid_full"], args, args.device,
                seed + index, args.epochs, variant_dir / f"fold_{index + 1}", variant, fold["context"])
            histories.append(history)
            selections.append({**fold["context"], **selection})
            del model
            if args.device.type == "cuda":
                torch.cuda.empty_cache()
        cv_curve = [{"epoch": epoch, "metrics": pool_reports([history[epoch - 1]["validation"] for history in histories])}
                    for epoch in range(1, args.epochs + 1)]
        variant_curves[variant] = cv_curve
        if variant == "A":
            final_epochs = max(1, int(np.median([row["best_epoch"] for row in selections])))
            calibration = calibrate_class_bounds(cv_curve, args, seed)
            write_json(variant_dir / "class_calibration.json", calibration)
            epoch_selection = {"status": "legacy_baseline", "selected_epoch": final_epochs,
                               "rule": "median of legacy fold choices; bootstrap classification reference epoch is separate"}
        else:
            epoch_selection = select_candidate_epoch(cv_curve, calibration, args)
            final_epochs = epoch_selection["selected_epoch"]
        final_epochs_by_variant[variant] = final_epochs
        write_json(variant_dir / "epoch_selection.json", epoch_selection)
        write_json(variant_dir / "cv_common_epoch.json", {"final_epoch": final_epochs, "epochs": cv_curve,
                                                          "scope": "same epoch across four folds; A and B may select different final epochs"})
        independent_best = pool_reports([
            history[selection["best_epoch"] - 1]["validation"] for history, selection in zip(histories, selections)])
        legacy_independent_best = pool_reports([
            history[selection["best_epoch"] - 1]["legacy_validation"] for history, selection in zip(histories, selections)])
        print(f"{variant}: final_epochs={final_epochs}; selection={epoch_selection['status']}", flush=True)
        if variant == "B":
            print(f"B selected active depth={epoch_selection['selected']['depth']['worst_domain_z_mae_mm']:.3f} mm; "
                  f"class failures={epoch_selection['selected']['class_failures']}", flush=True)
        final_context = {"fit_trajectories": train_ids, "feature_scales": final_scales.tolist(), "valid_trajectories": []}
        model, final_history, _ = fit(train_base, train_extra, None, None, args, args.device,
                                      seed + 1000, final_epochs, variant_dir / "final_fit", variant, final_context)
        held_out, probability, estimate = evaluate(model, test_full, args.device, args)
        legacy_held_out = evaluate(model, test_base, args.device, args)[0]
        # Save per-frame predictions without object arrays / pickle.
        prediction_path = variant_dir / "held_out_predictions.npz"
        with prediction_path.open("xb") as handle:
            np.savez_compressed(handle, trajectory=test_full.trajectory, frame=test_full.frame,
                                phase=test_full.phase, class_target=test_full.y, geometry_target_m=test_full.geometry,
                                class_probability=probability, geometry_estimate_m=estimate)
        metrics = {
            "held_out_test": held_out, "legacy_held_out_test": legacy_held_out,
            "cv_common_final_epoch": cv_curve[final_epochs - 1]["metrics"],
            "cv_epoch_selection": {key: value for key, value in epoch_selection.items() if key != "epochs"},
            "cv_independent_legacy_best_epochs": independent_best,
            "cv_independent_legacy_best_epochs_original_windows": legacy_independent_best,
            "inner_folds": selections, "train_trajectories": train_ids, "held_out_trajectories": test_ids,
            "train_pair_windows": len(train_base.y), "test_pair_windows": len(test_base.y),
            "extra_geometry_train_pair_windows": len(train_extra.y), "final_training_epochs": final_epochs,
            "total_cv_optimizer_steps": sum(history[-1]["optimizer_steps"] for history in histories),
            "final_optimizer_steps": final_history[-1]["optimizer_steps"],
            "elapsed_training_s": sum(history[-1]["elapsed_s"] for history in histories) + final_history[-1]["elapsed_s"],
        }
        model.to("cpu")
        payload = {
            "artifact_format": "tavla_force_alignment_encoder_terminal_reference_v1",
            "config": artifact_config(args, variant, seed, final_epochs),
            "preprocess": {"feature_scales": final_scales.tolist(),
                           "feature_order": ["force_x", "force_y", "force_z", "dforce_x_per_s", "dforce_y_per_s", "dforce_z_per_s"],
                           "baseline": f"coordinate-wise median of the first {args.baseline_frames} force samples"},
            "geometry": geometry_meta, "metrics": metrics,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        _save_new_checkpoint(payload, variant_dir / "encoder.pt")
        write_json(variant_dir / "encoder.report.json", {key: value for key, value in payload.items() if key != "model_state_dict"})
        variants[variant] = {"checkpoint": str(variant_dir / "encoder.pt"), "metrics": metrics}
        del model
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
    matched_epoch = final_epochs_by_variant["A"]
    matched_metrics = {variant: curve[matched_epoch - 1]["metrics"] for variant, curve in variant_curves.items()}
    comparison = {
        "seed": seed, "status": "diagnostic_requires_policy_validation", "final_epochs": final_epochs_by_variant,
        "variants": variants,
        "cv_matched_epoch": {"epoch": matched_epoch, "metrics": matched_metrics},
        "delta_B_minus_A": {
            "cv_matched_epoch": metric_delta(matched_metrics["A"], matched_metrics["B"]),
            "cv_individually_selected_epochs": metric_delta(variants["A"]["metrics"]["cv_common_final_epoch"],
                                                              variants["B"]["metrics"]["cv_common_final_epoch"]),
            "held_out": metric_delta(variants["A"]["metrics"]["held_out_test"], variants["B"]["metrics"]["held_out_test"]),
        },
        "held_out_constant_baselines": constant_baselines(train_base, train_extra, test_full),
        "original_reference_metrics": reference["metrics"],
        "limitations": ["Final-tip reference is not independently measured hole-bottom depth.",
                        "Auxiliary depth prediction does not measure closed-loop action quality.",
                        "New encoder requires new latent data, normalization statistics and TAVLA training.",
                        "Held-out trajectories are a fixed comparison set, not an unseen deployment test.",
                        "Individually selected and held-out comparisons include different final training epoch counts; matched CV comparison isolates geometry supervision changes."],
    }
    write_json(seed_dir / "comparison.json", comparison)
    return comparison


def write_summary(path: Path, comparisons):
    lines = ["# Encoder A/B 结果", "", "A：旧配方；B：补几何校验合格的高 / 中置信度末段监督。",
             "A 沿用旧选轮；B 在分类约束下按活动插入深度选自己的共同 CV epoch。候选需进一步验证。", "",
             "| Seed | 组 | 最终轮数 | 域 | CV F1（该组所选轮次） | 留出 F1 | 留出插入召回 | 活动插入 0–3mm Z MAE | terminal 0–3mm Z MAE |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    selection_notes = []
    def number(value):
        return "无样本" if value is None else f"{value:.4f}"
    for comparison in comparisons:
        for variant, result in comparison["variants"].items():
            metrics = result["metrics"]
            for domain in DOMAINS:
                test = metrics["held_out_test"][domain]
                phases = test["geometry_by_phase_and_depth"]
                active = phases.get("aligned_progress", {}).get("0_3mm", {}).get("z_mae_mm")
                terminal = phases.get("terminal", {}).get("0_3mm", {}).get("z_mae_mm")
                lines.append(f"| {comparison['seed']} | {variant} | {metrics['final_training_epochs']} | {domain} | "
                             f"{metrics['cv_common_final_epoch'][domain]['class']['macro_f1']:.4f} | "
                             f"{test['class']['macro_f1']:.4f} | "
                             f"{test['class']['per_class']['aligned_progress']['recall']:.4f} | {number(active)} | {number(terminal)} |")
        b_selection = comparison["variants"]["B"]["metrics"]["cv_epoch_selection"]
        selection_notes.append(f"seed={comparison['seed']}：B 选轮状态 `{b_selection['status']}`；"
                               f"A 的分类校准参考轮次={b_selection['class_calibration']['reference_epoch']}。")
    lines += [""] + [note + "\n" for note in selection_notes]
    lines += ["", "各阶段 0–3 / 3–5 / 5–8 / 8–10 mm 的样本数、目标和预测均值、标准差、",
              "每轨迹指标、训练折常数基线以及 B−A 差值见各 seed 的 comparison.json。",
              "完整逐轮结果见 A/B/cv_common_epoch.json，权重见 A/B/encoder.pt。", "",
              "同一轮次的 A/B 差值见 cv_matched_epoch；各自选轮后的差值另列，不能混作受控同轮次结果。",
              "B/epoch_selection.json 保存每轮分类约束、活动深度分数及选轮理由。",
              "三个种子的均值与标准差见 multiseed_summary.json；不按留出集挑最好种子。", "",
              "几何零点是示教最后的 tip，仍需确认其对应仿真任务达标深度。",
              "辅助头改善不能直接证明插入 action 改善；新的 latent 需配套重新训练 TAVLA。", ""]
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def multiseed_summary(comparisons):
    def statistics(values):
        array = np.asarray([value for value in values if value is not None], dtype=np.float64)
        return {"mean": float(np.mean(array)) if len(array) else None,
                "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0 if len(array) else None,
                "min": float(np.min(array)) if len(array) else None,
                "max": float(np.max(array)) if len(array) else None,
                "count": len(array), "per_seed_values": values}
    def metrics(report):
        active = report["geometry_by_phase_and_depth"].get("aligned_progress", {}).get("0_3mm", {})
        return {"macro_f1": report["class"]["macro_f1"],
                "aligned_recall": report["class"]["per_class"]["aligned_progress"]["recall"],
                "misaligned_recall": report["class"]["per_class"]["misaligned_contact"]["recall"],
                "active_0_3mm_z_mae_mm": active.get("z_mae_mm")}
    result = {"seeds": [comparison["seed"] for comparison in comparisons],
              "final_epochs": {str(comparison["seed"]): comparison["final_epochs"] for comparison in comparisons},
              "B_selection_status": {str(comparison["seed"]): comparison["variants"]["B"]["metrics"]["cv_epoch_selection"]["status"]
                                     for comparison in comparisons},
              "scope": "training-seed variation; matched CV and separately selected epochs are reported separately; no best seed chosen"}
    for dataset in ("cv_matched_epoch", "cv_common_final_epoch", "held_out_test"):
        result[dataset] = {}
        for domain in DOMAINS:
            by_variant = {}
            for variant in ("A", "B"):
                by_variant[variant] = [metrics(comparison["cv_matched_epoch"]["metrics"][variant][domain]
                                               if dataset == "cv_matched_epoch" else
                                               comparison["variants"][variant]["metrics"][dataset][domain])
                                       for comparison in comparisons]
            result[dataset][domain] = {
                variant: {name: statistics([row[name] for row in rows]) for name in rows[0]}
                for variant, rows in by_variant.items()
            }
            result[dataset][domain]["paired_B_minus_A"] = {
                name: statistics([b[name] - a[name] if a[name] is not None and b[name] is not None else None
                                  for a, b in zip(by_variant["A"], by_variant["B"])])
                for name in by_variant["A"][0]
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--reference-encoder", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=60, help="Full CV diagnostic budget; original maximum was 60")
    parser.add_argument("--patience", type=int, default=10, help="Simulated original early stop; curves continue to --epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Legacy batch size in paired Real/Sim frames")
    parser.add_argument("--terminal-batch-size", type=int, default=32)
    parser.add_argument("--terminal-loss-weight", type=float, default=0.03)
    parser.add_argument("--include-medium-geometry", action=argparse.BooleanOptionalAction, default=True,
                        help="Use qualified high/medium-confidence phases for extra geometry; class confidence stays unchanged")
    parser.add_argument("--reference-stability-mm", type=float, default=3.0,
                        help="Max XY radius and abs Z of last three terminal frames relative to the final tip")
    parser.add_argument("--selection-class-quantile", type=float, default=0.10,
                        help="Lower trajectory-bootstrap quantile for A-calibrated classification constraints")
    parser.add_argument("--selection-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.terminal_batch_size, args.log_every) <= 0:
        parser.error("epochs, patience, batch sizes and log-every must be positive")
    if not np.isfinite(args.terminal_loss_weight) or args.terminal_loss_weight <= 0:
        parser.error("terminal-loss-weight must be finite and positive")
    if not np.isfinite(args.reference_stability_mm) or args.reference_stability_mm <= 0:
        parser.error("reference-stability-mm must be finite and positive")
    if not 0 < args.selection_class_quantile < 0.5 or args.selection_bootstrap_samples < 100:
        parser.error("selection-class-quantile must be within (0,0.5); selection-bootstrap-samples must be >=100")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed > 2**32 - 1_000_000 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers below 2**32 - 1,000,000")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A launcher may have created train.log already; no experiment artifact can
    # exist in the output directory. Refuse reuse instead of overwriting files.
    if any(path.name != "train.log" for path in args.output_dir.iterdir()):
        raise FileExistsError(f"output directory contains experiment files: {args.output_dir}")
    started = time.monotonic()
    reference = torch.load(args.reference_encoder, map_location="cpu", weights_only=True)
    if reference.get("artifact_format") != "tavla_force_alignment_encoder_terminal_reference_v1":
        raise ValueError("reference must be the original terminal-reference encoder")
    config = reference["config"]
    args.history_length, args.baseline_frames, args.boundary_guard = 32, 20, 5
    args.geometry_scale_m, args.xy_threshold_m, args.z_threshold_m = 0.01, 0.003, 0.003
    args.force_key = "obs/state/ee_force_base"
    args.tool_offset_mm = np.asarray(config["tool_offset_local_mm"], dtype=np.float64)
    for key, expected in (("history_length", 32), ("baseline_frames", 20), ("latent_dim", 32),
                          ("geometry_scale_m", 0.01), ("input_force_key", args.force_key),
                          ("xy_threshold_m", 0.003), ("z_threshold_m", 0.003)):
        if config[key] != expected:
            raise ValueError(f"reference {key}={config[key]!r} differs from the legacy recipe {expected!r}")
    labels = read_labels(args.labels)
    train_ids = list(reference["metrics"]["train_trajectories"])
    test_ids = list(reference["metrics"]["held_out_trajectories"])
    fold_ids = [list(row["valid_trajectories"]) for row in reference["metrics"]["inner_folds"]]
    all_fold_ids = [trajectory for fold in fold_ids for trajectory in fold]
    if (len(fold_ids) != 4 or any(not fold for fold in fold_ids)
            or len(set(all_fold_ids)) != len(all_fold_ids) or sorted(all_fold_ids) != sorted(train_ids)
            or set(train_ids) & set(test_ids) or sorted(train_ids + test_ids) != sorted(labels)):
        raise ValueError("reference four folds and train/held-out trajectories are not a disjoint partition of labels")
    args.device = torch.device(args.device)
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check the GPU environment and CUDA_VISIBLE_DEVICES")
        torch.cuda.set_device(args.device)
        args.device = torch.device("cuda", torch.cuda.current_device())
    elif args.device.type != "cpu":
        parser.error("device must be cpu or cuda")
    torch.set_num_threads(4)
    source_dir = args.output_dir / "source_snapshot"
    source_dir.mkdir()
    source_signatures = {}
    for name in ("train_ab.py", "train_terminal_geometry.py", "train.py", "model.py", "preprocess.py"):
        path = Path(__file__).with_name(name)
        shutil.copy2(path, source_dir / name)
        source_signatures[name] = sha256_file(source_dir / name)
    launcher = Path(__file__).resolve().parents[1] / "run_insertion_alignment_encoder_ab.sh"
    if launcher.is_file():
        shutil.copy2(launcher, source_dir / launcher.name)
        source_signatures[launcher.name] = sha256_file(source_dir / launcher.name)
    episodes, cutoffs, holes, disagreement = load_terminal_episodes(args, labels)
    args.geometry_reference_audit = audit_geometry_references(episodes, labels, args)
    args.geometry_eligible_trajectories = [trajectory for trajectory in sorted(labels)
                                          if args.geometry_reference_audit[f"traj_{trajectory}"]["eligible"]]
    write_json(args.output_dir / "geometry_reference_audit.json", args.geometry_reference_audit)
    excluded = [trajectory for trajectory in sorted(labels) if trajectory not in args.geometry_eligible_trajectories]
    print(f"Extra geometry includes medium confidence={args.include_medium_geometry}; "
          f"unqualified terminal references={excluded}", flush=True)
    data_signatures = {f"traj_{trajectory}/{DOMAINS[domain]}": {
        "force": array_signature(episode.force), "timestamps": array_signature(episode.time),
        "geometry": array_signature(episode.geometry), "frame_count": len(episode.force),
    } for (trajectory, domain), episode in episodes.items()}
    parameters = {key: (str(value) if isinstance(value, (Path, torch.device)) else
                         value.tolist() if isinstance(value, np.ndarray) else value)
                  for key, value in vars(args).items()}
    write_json(args.output_dir / "run_manifest.json", {
        "recipe": RECIPE, "parameters": parameters, "source_sha256": source_signatures,
        "labels_sha256": sha256_file(args.labels), "reference_sha256": sha256_file(args.reference_encoder),
        "data_array_sha256": data_signatures, "folds": fold_ids, "train_trajectories": train_ids,
        "held_out_trajectories": test_ids, "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "python_version": sys.version,
        "cuda_version": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else "cpu",
        "base_recipe": artifact_config(args, "A", args.seeds[0], 0),
        "candidate_recipe": artifact_config(args, "B", args.seeds[0], 0),
    })
    geometry_meta = {
        "hole_reference_world_m": {f"traj_{key}": value for key, value in holes.items()},
        "first_hit_cutoff_frames": {f"traj_{key}": value for key, value in cutoffs.items()},
        "max_real_sim_pose_disagreement_mm": disagreement,
        "terminal_reference_audit": args.geometry_reference_audit,
        "first_hit_marker_usage": "A/B base stops before first hit; B's additional geometry includes reviewed near-terminal frames",
    }
    comparisons = [run_seed(seed, args, episodes, labels, reference, fold_ids,
                            train_ids, test_ids, geometry_meta) for seed in args.seeds]
    write_json(args.output_dir / "comparison.json", {"recipe": RECIPE, "status": "diagnostic_requires_policy_validation",
                                                     "elapsed_s": time.monotonic() - started, "seeds": comparisons})
    write_summary(args.output_dir / "comparison.md", comparisons)
    write_json(args.output_dir / "multiseed_summary.json", multiseed_summary(comparisons))
    print(f"A/B complete. Candidates and comparison: {args.output_dir}", flush=True)
    print("Inspect A reproduction and B's active-insertion depth/class deltas before selecting a policy experiment.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
