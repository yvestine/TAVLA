"""Single-holdout, cross-validation, and final-refit workers for force encoder v3."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from force_gap_forecast_v1.common import sha256, write_json
from . import BLOCK, FORMAT, LAYOUT
from . import baselines
from .data import Blocks, FOLDS, HOLDOUT, TRAIN_IDS, episode_arrays, load_pairs, make_stats
from .losses import forecast_loss, objective
from .metrics import (aggregate, alignment_evaluation, evaluate, flatten_evaluation, force_audit,
                      forecast_metrics, masks, validation_summary)
from .model import PredictiveForce

RECON_GUARDS = ("reconstruction_rmse_n", "reconstruction_delta_rmse_n", "high_band_mse",
                "stream_high_band_mse", "peak_amplitude_mae_n", "peak_timing_mae_s")


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def digest(model, frozen_only=False):
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not frozen_only or name not in trainable:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_model(path, model, stats, summary, motion_ready=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(dict(artifact_format=FORMAT, layout=LAYOUT, stats=stats, use_fast=model.use_fast,
                    state_dict=cpu_state(model), stage=summary, motion_ready=motion_ready), temporary)
    temporary.replace(path)


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("artifact_format") != FORMAT or payload.get("layout") != LAYOUT:
        raise ValueError(f"Invalid v3 checkpoint: {path}")
    model = PredictiveForce(payload["stats"], payload["use_fast"]).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, payload


def run_stage(episodes, train_ids, valid_ids, stats, args, seed, path, stage,
              parent=None, use_fast=True, align_weight=0., epochs=None):
    """Resume completed stages only. Interrupted stages restart from their immutable parent."""
    seed_all(seed)
    model = copy.deepcopy(parent) if parent is not None else PredictiveForce(stats, use_fast)
    if model.stats != stats:
        raise ValueError("Parent/train-fold normalization mismatch")
    model = model.to(args.device)
    if stage == "student":
        model.initialize_student()
    model.configure(stage)
    identity = dict(train_ids=sorted(train_ids), valid_ids=sorted(valid_ids), stage=stage, seed=seed,
                    use_fast=model.use_fast, align_weight=align_weight, refit_epochs=epochs,
                    initial_state_sha256=digest(model), max_epochs=args.max_epochs, patience=args.patience,
                    batch_size=args.batch_size, lr=args.lr)
    history_path, checkpoint = path / "history.json", path / "model.pt"
    if history_path.is_file() and checkpoint.is_file():
        previous = read_json(history_path)
        if previous.get("completed"):
            restored, payload = load_model(checkpoint, args.device)
            if (previous.get("identity") != identity or payload["stage"] != previous or
                    payload["stats"] != stats or digest(restored) != previous["state_sha256"]):
                raise ValueError(f"Completed stage changed: {path}; start a new run")
            restored.configure(stage)
            return restored, previous
    elif checkpoint.exists() and not args.resume:
        raise FileExistsError(f"Existing stage: {path}; use a matching resume")
    if not valid_ids and (epochs is None or epochs < 1):
        raise ValueError("Refit requires an inner-selected positive epoch count")
    path.mkdir(parents=True, exist_ok=True)
    frozen_sha = digest(model, frozen_only=True)
    dataset = Blocks(episodes, train_ids, stats)
    valid_arrays = {(i, d): episode_arrays(episodes[(i, d)], stats, name)
                    for i in valid_ids for d, name in enumerate(("real", "sim"))}
    steps_per_epoch = max(1, math.ceil(len(dataset) / (args.batch_size * BLOCK)))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    maximum = epochs if epochs is not None else args.max_epochs
    best_score, best_state, best_epoch, best_metric = math.inf, None, 0, None
    records, stale, steps, started = [], 0, 0, time.monotonic()
    domains = (0,) if stage == "teacher" else (1,) if stage == "student" else (0, 1)
    kind = "joint" if stage == "motion" else "control" if stage == "control" else "force"
    for epoch in range(1, maximum + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch * 7919)
        sampler = WeightedRandomSampler(dataset.weights, steps_per_epoch * args.batch_size, True, generator=generator)
        loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=0)
        losses = []
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if stage in ("teacher", "student"):
                d = domains[0]
                output = model.block(batch, d)
                teacher = None
                if stage == "student" and align_weight:
                    with torch.no_grad():
                        teacher = model.block(batch, 0)
                loss, detail = objective(model, output, batch, d, align_weight, teacher)
            else:
                key = "joint" if stage == "motion" else "control"
                loss = sum(forecast_loss(model, model.block(batch, d, True)[key], batch) for d in domains) / 2
                detail = dict(total=float(loss.detach()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {path}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 3., error_if_nonfinite=True)
            optimizer.step()
            losses.append(detail)
            steps += 1
        record = dict(epoch=epoch, optimizer_steps=steps, train=aggregate(losses), elapsed_s=time.monotonic() - started)
        if valid_ids:
            evaluation = evaluate(model, episodes, valid_ids, stats, domains, stage in ("motion", "control"), valid_arrays)
            metric = validation_summary(evaluation, kind)
            score = metric["forecast_score_n"] / float(np.mean(stats["target"]["scale"]))
            if stage in ("teacher", "student"):
                score += .25 * metric.get("reconstruction_nrmse", 0.)
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite validation score")
            record.update(valid=metric, selection_score=float(score))
            if score < best_score:
                best_score, best_state, best_epoch, best_metric, stale = score, cpu_state(model), epoch, metric, 0
            else:
                stale += 1
        records.append(record)
        write_json(history_path, dict(completed=False, identity=identity, records=records))
        if epoch == 1 or epoch % 10 == 0 or epoch == maximum:
            print(f"{path.name}: epoch={epoch}/{maximum} steps={steps} loss={record['train']['total']:.5f} "
                  f"future_N={record.get('valid', {}).get('forecast_score_n')} elapsed={record['elapsed_s']:.1f}s", flush=True)
        if valid_ids and stale >= args.patience:
            break
    save_model(path / "last.pt", model, stats, dict(last_epoch=records[-1]["epoch"]))
    if valid_ids:
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum
    if digest(model, frozen_only=True) != frozen_sha:
        raise RuntimeError(f"Frozen weights changed during {stage}")
    summary = dict(completed=True, identity=identity, selected_epoch=best_epoch, selected_metrics=best_metric,
                   records=records, optimizer_steps=steps, elapsed_s=time.monotonic() - started,
                   frozen_sha256=frozen_sha, state_sha256=digest(model))
    save_model(checkpoint, model, stats, summary, motion_ready=stage in ("motion", "control"))
    write_json(history_path, summary)
    return model, summary


def guard(candidate, reference, keys=RECON_GUARDS, tolerance=.02):
    return all(candidate.get(key) is not None and candidate[key] <= reference[key] * (1 + tolerance) + 1e-8
               for key in keys if reference.get(key) is not None)


def choose_alignment(candidates):
    baseline = candidates["0.0"]["selected_metrics"]
    eligible = [(row["selected_metrics"]["forecast_score_n"], float(weight))
                for weight, row in candidates.items() if guard(row["selected_metrics"], baseline)]
    return min(eligible)[1]  # Includes zero; ties prefer the smaller alignment weight.


def select_force(episodes, train_ids, valid_ids, stats, args, seed, root, use_fast):
    teacher, tc = run_stage(episodes, train_ids, valid_ids, stats, args, seed, root / "teacher_select",
                            "teacher", use_fast=use_fast)
    choices = {}
    for weight in args.align_weights:
        student, choice = run_stage(episodes, train_ids, valid_ids, stats, args, seed + 1000,
                                   root / f"student_lambda{weight}_select", "student", teacher, align_weight=weight)
        choices[str(weight)] = dict(selected_epoch=choice["selected_epoch"], selected_metrics=choice["selected_metrics"])
        del student
    weight = choose_alignment(choices)
    selected, _ = load_model(root / f"student_lambda{weight}_select" / "model.pt", args.device)
    return selected, dict(teacher_epoch=tc["selected_epoch"], align_weight=weight, candidates=choices,
                          student_epoch=choices[str(weight)]["selected_epoch"])


def refit_force(episodes, ids, stats, args, seed, root, choice, use_fast=True, weight=None):
    teacher, _ = run_stage(episodes, ids, [], stats, args, seed, root / "teacher_refit", "teacher",
                            use_fast=use_fast, epochs=choice["teacher_epoch"])
    weight = choice["align_weight"] if weight is None else weight
    student, _ = run_stage(episodes, ids, [], stats, args, seed + 1000, root / f"student_lambda{weight}_refit",
                            "student", teacher, align_weight=weight, epochs=choice["candidates"][str(weight)]["selected_epoch"])
    return student


def old_motion(episodes, ids, stats, args, seed, fold):
    if not args.old_base_run:
        return None
    from force_gap_motion_force_v2.model import JointMotionForceForecast
    from force_gap_motion_force_v2.metrics import predict_episode
    path = Path(args.old_base_run) / "stages" / f"seed{seed}_fold{fold}_motion_only_student_refit" / "model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {k: v for k, v in stats.items() if k != "event_threshold_n"}
    if payload["stats"] != expected or not payload["stage"].get("motion_only"):
        raise ValueError("Old motion baseline does not match this fold's data/normalizer")
    model = JointMotionForceForecast(expected, payload["state_dict"], use_force=False).to(args.device)
    model.domain_gains.zero_()
    rows = {}
    for d, name in enumerate(("real", "sim")):
        rows[name] = {}
        for tid in ids:
            real, source = episodes[(tid, 0)], episodes[(tid, d)]
            future = predict_episode(model, source, expected, d)["future"]
            rows[name][str(tid)] = {phase: forecast_metrics(future, real, source, stats["event_threshold_n"]["real"], mask)
                                   for phase, mask in masks(real).items()}
    return rows


def train_fold(episodes, args, seed, fold, output):
    valid_ids = FOLDS[fold]
    train_ids = sorted(set(TRAIN_IDS) - set(valid_ids))
    order = np.random.default_rng(42 + fold).permutation(train_ids)
    inner_valid = sorted(map(int, order[:6]))
    inner_train = sorted(set(train_ids) - set(inner_valid))
    inner_stats, stats = make_stats(episodes, inner_train), make_stats(episodes, train_ids)
    result = dict(seed=seed, fold=fold, inner_train=inner_train, inner_valid=inner_valid,
                  outer_train=train_ids, outer_valid=valid_ids, models={}, selections={}, alignment={})
    selected, choice = select_force(episodes, inner_train, inner_valid, inner_stats, args, seed, output / "full", True)
    motion, mc = run_stage(episodes, inner_train, inner_valid, inner_stats, args, seed + 2000,
                           output / "motion_select", "motion", selected)
    _, cc = run_stage(episodes, inner_train, inner_valid, inner_stats, args, seed + 2000,
                      output / "control_select", "control", motion)
    choice.update(motion_epoch=mc["selected_epoch"], control_epoch=cc["selected_epoch"])
    result["selections"]["full"] = choice
    model = refit_force(episodes, train_ids, stats, args, seed, output / "full", choice)
    model, _ = run_stage(episodes, train_ids, [], stats, args, seed + 2000, output / "motion_refit", "motion",
                         model, epochs=choice["motion_epoch"])
    model, _ = run_stage(episodes, train_ids, [], stats, args, seed + 2000, output / "control_refit", "control",
                         model, epochs=choice["control_epoch"])
    full = evaluate(model, episodes, valid_ids, stats, with_motion=True, save_dir=output / "predictions" / "full")
    for name, kind in (("full_force", "force"), ("full_joint", "joint"), ("motion_control", "control")):
        result["models"][name] = flatten_evaluation(full, kind)
    result["alignment"]["full"] = alignment_evaluation(model, episodes, train_ids, valid_ids, stats)
    result["force_input_audit"] = force_audit(model, episodes, valid_ids, stats)
    del model, selected, motion
    noalign = refit_force(episodes, train_ids, stats, args, seed, output / "full", choice, weight=0.) if choice["align_weight"] else None
    if noalign is None:
        noalign, _ = load_model(output / "full" / "student_lambda0.0_refit" / "model.pt", args.device)
    result["models"]["no_alignment"] = flatten_evaluation(evaluate(
        noalign, episodes, valid_ids, stats, save_dir=output / "predictions" / "no_alignment"))
    result["alignment"]["no_alignment"] = alignment_evaluation(noalign, episodes, train_ids, valid_ids, stats)
    del noalign
    _, fast_choice = select_force(episodes, inner_train, inner_valid, inner_stats, args, seed, output / "no_fast", False)
    result["selections"]["no_fast"] = fast_choice
    nofast = refit_force(episodes, train_ids, stats, args, seed, output / "no_fast", fast_choice, use_fast=False)
    result["models"]["no_fast"] = flatten_evaluation(evaluate(
        nofast, episodes, valid_ids, stats, save_dir=output / "predictions" / "no_fast"))
    del nofast
    result["baseline_alphas"] = {}
    for kind in ("hold", "ar"):
        result["models"][kind] = {}
        for d, name in enumerate(("real", "sim")):
            alpha = baselines.select(episodes, inner_train, inner_valid, inner_stats, d) if kind == "ar" else .1
            result["baseline_alphas"][f"{kind}_{name}"] = alpha
            baseline = baselines.fit(episodes, train_ids, stats, d, kind, alpha)
            result["models"][kind][name] = baselines.evaluate(baseline, episodes, valid_ids, stats)
    old = old_motion(episodes, valid_ids, stats, args, seed, fold)
    if old is not None:
        result["models"]["old_motion_only"] = old
    result["completed"] = True
    write_json(output / "result.json", result)


def train_single(episodes, args, seed, run_root):
    """Train one fixed-seed model on one fixed train/validation split, with no CV or ablations."""
    fold = args.validation_fold
    valid_ids = sorted(FOLDS[fold])
    train_ids = sorted(set(TRAIN_IDS) - set(valid_ids))
    stats = make_stats(episodes, train_ids)
    output = run_root / "single"

    teacher, teacher_choice = run_stage(
        episodes, train_ids, valid_ids, stats, args, seed, output / "teacher", "teacher")
    student, student_choice = run_stage(
        episodes, train_ids, valid_ids, stats, args, seed + 1000, output / "student", "student",
        teacher, align_weight=args.single_align_weight)
    del teacher
    motion, motion_choice = run_stage(
        episodes, train_ids, valid_ids, stats, args, seed + 2000, output / "motion", "motion", student)
    del student
    model, control_choice = run_stage(
        episodes, train_ids, valid_ids, stats, args, seed + 2000, output / "control", "control", motion)
    del motion

    validation = evaluate(model, episodes, valid_ids, stats, with_motion=True,
                          save_dir=output / "predictions" / "validation")
    result = dict(
        completed=True,
        mode="single_fixed_holdout",
        seed=seed,
        validation_fold=fold,
        train_ids=train_ids,
        valid_ids=valid_ids,
        align_weight=args.single_align_weight,
        selected_epochs=dict(teacher=teacher_choice["selected_epoch"],
                             student=student_choice["selected_epoch"],
                             motion=motion_choice["selected_epoch"],
                             control=control_choice["selected_epoch"]),
        models={name: flatten_evaluation(validation, kind) for name, kind in
                (("full_force", "force"), ("full_joint", "joint"), ("motion_control", "control"))},
        alignment=alignment_evaluation(model, episodes, train_ids, valid_ids, stats),
        force_input_audit=force_audit(model, episodes, valid_ids, stats),
    )
    checkpoint = run_root / f"encoder_seed{seed}.pt"
    summary = dict(completed=True, mode=result["mode"], seed=seed, validation_fold=fold,
                   align_weight=args.single_align_weight, selected_epochs=result["selected_epochs"])
    save_model(checkpoint, model, stats, summary, motion_ready=True)

    lines = ["# Force predictive encoder v3: single-seed run", "",
             f"Seed: {seed}; validation fold: {fold}; fixed trend alignment weight: {args.single_align_weight}.",
             f"Train trajectories: {len(train_ids)}; validation trajectories: {len(valid_ids)}.",
             "This is one fixed holdout run. Validation metrics guide early stopping and are not an independent cross-validation estimate.",
             "", "## Selected epochs", "",
             "| Stage | Epoch |", "|---|---:|"]
    lines.extend(f"| {name} | {epoch} |" for name, epoch in result["selected_epochs"].items())
    lines += ["", "## Validation metrics", "", "| Domain | Model | Future score (N) | +5 RMSE (N) | Reconstruction RMSE (N) | High-band MSE |",
              "|---|---|---:|---:|---:|---:|"]
    for domain in ("real", "sim"):
        for name in ("full_force", "full_joint", "motion_control"):
            rows = result["models"][name][domain].values()
            row = aggregate([trajectory["all"] for trajectory in rows])
            lines.append(f"| {domain} | {name} | {row['forecast_score_n']:.5f} | "
                         f"{row['future_5_rmse_n']:.5f} | {row['reconstruction_rmse_n']:.5f} | "
                         f"{row['high_band_mse']:.5f} |")
    lines += ["", f"Trend alignment distance: {result['alignment']['trend_distance']}",
              f"Trend change error: {result['alignment']['trend_change_error']}",
              "", "Per-trajectory metrics and waveform arrays are in `result.json` and `predictions/validation/`.",
              "The manifest marks this as a single-holdout diagnostic, not a cross-validated acceptance result."]
    (output / "single_report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "result.json", result)
    return result


def final_choices(results, weights):
    choices = [row["selections"]["full"] for row in results]
    median_weight = float(np.median([c["align_weight"] for c in choices]))
    weight = min(weights, key=lambda w: (abs(w - median_weight), w))
    result = {key: max(1, int(np.median([c[key] for c in choices])))
              for key in ("teacher_epoch", "motion_epoch", "control_epoch")}
    result.update(align_weight=weight, candidates={str(weight): dict(
        selected_epoch=max(1, int(np.median([c["candidates"][str(weight)]["selected_epoch"] for c in choices]))))})
    return result


def train_final(episodes, args, seed, root):
    results = [read_json(root / "jobs" / f"seed{seed}_fold{fold}" / "result.json") for fold in range(4)]
    choice = final_choices(results, args.align_weights)
    stats = make_stats(episodes, TRAIN_IDS)
    output = root / "final" / f"seed{seed}"
    model = refit_force(episodes, TRAIN_IDS, stats, args, seed, output / "full", choice)
    model, _ = run_stage(episodes, TRAIN_IDS, [], stats, args, seed + 2000, output / "motion", "motion", model,
                         epochs=choice["motion_epoch"])
    model, _ = run_stage(episodes, TRAIN_IDS, [], stats, args, seed + 2000, output / "control", "control", model,
                         epochs=choice["control_epoch"])
    diagnostic = evaluate(model, episodes, HOLDOUT, stats, with_motion=True, save_dir=output / "diagnostic_predictions")
    write_json(output / "diagnostic_metrics.json", diagnostic)
    save_model(root / f"encoder_seed{seed}.pt", model, stats,
               dict(completed=True, selection="medians of inner selections only", choices=choice, train_ids=TRAIN_IDS), True)
    write_json(output / "result.json", dict(completed=True, seed=seed, choices=choice,
                                           checkpoint_sha256=sha256(root / f"encoder_seed{seed}.pt")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()
    recipe = read_json(cli.run_dir / "recipe.json")
    scripts = Path(__file__).resolve().parent.parent
    for relative, expected_sha in recipe["source_sha256"].items():
        if sha256(scripts / relative) != expected_sha:
            raise ValueError(f"Source changed after run preparation: {relative}")
    if sha256(recipe["options"]["labels"]) != recipe["label_sha256"]:
        raise ValueError("Labels changed after run preparation")
    args = SimpleNamespace(**recipe["options"], resume=cli.resume, device=cli.device)
    torch.set_num_threads(args.threads)
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on the external GPU host")
    episodes, audit = load_pairs(args.real_root, args.sim_root, args.labels)
    if audit != recipe["data"]:
        raise ValueError("Worker observed changed data")
    if cli.final:
        train_final(episodes, args, cli.seed, cli.run_dir)
    elif cli.single:
        train_single(episodes, args, cli.seed, cli.run_dir)
    else:
        if cli.fold not in range(4):
            parser.error("--fold must be 0..3")
        train_fold(episodes, args, cli.seed, cli.fold, cli.run_dir / "jobs" / f"seed{cli.seed}_fold{cli.fold}")


if __name__ == "__main__":
    main()
