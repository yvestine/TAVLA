"""Train a real-only multi-task v3 teacher, then a paired Sim student on one split."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

from force_gap_forecast_v1.common import sha256
from force_gap_forecast_v1.common import write_json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from . import BLOCK
from . import FORMAT
from . import FUTURE
from . import HISTORY
from . import LAYOUT
from . import PAST
from .losses import objective
from .metrics import aggregate
from .metrics import forecast_metrics
from .metrics import reconstruction_metrics
from .model import PredictiveForce
from .multitask_data import MultiTaskBlocks
from .multitask_data import data_audit
from .multitask_data import data_identity
from .multitask_data import fit_domain_stats
from .multitask_data import load_real
from .multitask_data import load_sim
from .multitask_data import make_split
from .multitask_data import prepare_arrays
from .multitask_data import task_config
from .train import cpu_state
from .train import digest
from .train import load_model
from .train import save_model
from .train import seed_all


def source_identity():
    directory = Path(__file__).parent
    names = (
        "__init__.py",
        "data.py",
        "model.py",
        "losses.py",
        "metrics.py",
        "train.py",
        "multitask.py",
        "multitask_data.py",
    )
    identities = {name: sha256(directory / name) for name in names}
    for name in ("common.py", "data.py", "metrics.py"):
        path = directory.parent / "force_gap_forecast_v1" / name
        identities[f"force_gap_forecast_v1/{name}"] = sha256(path)
    return identities


def emit(output, message):
    print(message, flush=True)
    with (output / "train.log").open("a") as handle:
        handle.write(message + "\n")


def prepare(args):
    tasks = task_config(args.config, args.data_root, args.tasks)
    real = load_real(tasks)
    real_contract = {name: task.get("real_force_key", "obs/state/ee_force_base") for name, task in tasks.items()}
    parent, sim, pairing, teacher_sha = None, None, {}, None
    if args.stage == "teacher":
        split = make_split(real, args.seed, args.validation_fraction)
        domain, threshold = fit_domain_stats(real, split["train"])
        stats = {
            "real": domain,
            "target": copy.deepcopy(domain),
            "fit_trajectory_ids": split["train"],
            "event_threshold_n": {"real": threshold},
            "fit_method": "training_frames_only",
        }
    else:
        if args.teacher_run is None:
            raise ValueError("student requires --teacher-run pointing to a completed teacher directory")
        teacher_recipe = json.loads((args.teacher_run / "recipe.json").read_text())
        manifest = json.loads((args.teacher_run / "manifest.json").read_text())
        if teacher_recipe.get("stage") != "teacher" or manifest.get("ready_domains") != ["real"]:
            raise ValueError("--teacher-run must contain a completed multi-task Real teacher")
        if teacher_recipe["real_data"] != data_identity(real) or teacher_recipe["real_contract"] != real_contract:
            raise ValueError("Real data/task/force fields changed since teacher training; use the same originals")
        if teacher_recipe["source_sha256"] != source_identity():
            raise ValueError("Teacher training code differs; use the same Git revision for both stages")
        checkpoint = args.teacher_run / manifest["checkpoint"]
        teacher_sha = sha256(checkpoint)
        if teacher_sha != manifest["encoder_sha256"]:
            raise ValueError("Teacher checkpoint SHA mismatch")
        parent, _ = load_model(checkpoint)
        split = json.loads((args.teacher_run / "split.json").read_text())
        args.seed = split["seed"]
        stats = copy.deepcopy(parent.stats)
        sim, pairing = load_sim(tasks, real)
        stats["sim"], stats["event_threshold_n"]["sim"] = fit_domain_stats(sim, split["train"])
        # Only Sim input statistics are added; the teacher's physical target and Real input stay fixed.
        parent.stats = copy.deepcopy(stats)
    recipe = {
        "format": "force_predictive_v3_multitask_v1",
        "stage": args.stage,
        "seed": args.seed,
        "split": split,
        "real_contract": real_contract,
        "real_data": data_identity(real),
        "sim_data": data_identity(sim) if sim else None,
        "sim_contract": {
            name: {
                "force_key": task.get("sim_force_key", "obs/state/ee_force_base"),
                "source": task.get("sim_force_source", "aligned_base_xyz_n_v1"),
            }
            for name, task in tasks.items()
        }
        if sim
        else None,
        "pairing": pairing,
        "parent_teacher_sha256": teacher_sha,
        "hyperparameters": {
            name: getattr(args, name) for name in ("max_epochs", "patience", "batch_size", "lr", "align_weight")
        },
        "normalization": stats,
        "sampling": "equal_task_equal_trajectory_half_uniform_half_event",
        "selection": "task-equal forecast_score_N / mean(real_train_std) + 0.25 * reconstruction_nrmse",
        "source_sha256": source_identity(),
    }
    output = args.output_dir
    if (output / "recipe.json").exists():
        if not args.resume:
            raise FileExistsError(f"Existing run {output}; use --resume or a new output directory")
        if json.loads((output / "recipe.json").read_text()) != recipe:
            raise ValueError("Resume requires identical data, split, code, teacher and hyperparameters")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty output directory without recipe: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "recipe.json", recipe)
    write_json(output / "split.json", split)
    write_json(output / "normalization.json", stats)
    write_json(
        output / "data_audit.json",
        {
            "real": data_audit(real),
            "sim": data_audit(sim) if sim else None,
            "pairing": pairing,
            "actual_sim_state_verified": False,
        },
    )
    return tasks, real, sim, stats, split, parent


@torch.inference_mode()
def evaluate(model, real, source, keys, arrays, stats, domain, save_dir=None):
    model.eval()
    device = next(model.parameters()).device
    tasks = {}
    for key in keys:
        ep = source[key]
        outputs = {}
        for lo in range(0, len(ep.force), 128):
            x = torch.from_numpy(arrays[key]["x"][lo : lo + 128]).to(device)
            for name, value in model(x, int(domain == "sim")).items():
                outputs.setdefault(name, []).append(value.cpu().numpy())
        pred = {name: np.concatenate(rows) for name, rows in outputs.items()}
        history = pred["reconstruction"] * np.asarray(stats[domain]["scale"], np.float32) + np.asarray(
            stats[domain]["mean"], np.float32
        )
        row = {
            **forecast_metrics(pred["future"], real[key], ep, stats["event_threshold_n"]["real"]),
            **reconstruction_metrics(history, ep, stats, domain),
        }
        if row.get("forecast_score_n") is None or row.get("reconstruction_nrmse") is None:
            raise ValueError(f"{key}: validation needs continuous segments of >=16 frames and paired future targets")
        covariance = np.cov(pred["z"], rowvar=False)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
        row["latent_effective_rank"] = (
            float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12)))) if eigenvalues.sum() > 1e-12 else 0.0
        )
        row["latent_std"] = pred["z"].std(0).tolist()
        tasks.setdefault(ep.task, {})[str(ep.trajectory)] = row
        if save_dir is not None:
            path = save_dir / ep.task / f"traj_{ep.trajectory}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                z=pred["z"],
                future=pred["future"],
                history=history,
                input_force=ep.force,
                real_force=real[key].force,
                time=ep.time,
                real_time=real[key].time,
                starts=ep.starts,
                ends=ep.ends,
                real_starts=real[key].starts,
                real_ends=real[key].ends,
            )
    by_task = {task: aggregate(list(rows.values())) for task, rows in tasks.items()}
    return {"summary": aggregate(list(by_task.values())), "by_task": by_task, "trajectories": tasks}


def atomic_torch_save(path, payload):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args, tasks, real, sim, stats, split, parent):
    output = args.output_dir
    if args.resume and (output / "manifest.json").is_file():
        manifest = json.loads((output / "manifest.json").read_text())
        if sha256(output / manifest["checkpoint"]) != manifest["encoder_sha256"]:
            raise ValueError("Completed checkpoint SHA mismatch")
        emit(output, f"Reusing completed {args.stage}: {output / 'manifest.json'}")
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a GPU-enabled PyTorch environment or use --device cpu")
    seed_all(args.seed if args.stage == "teacher" else args.seed + 1000)
    model = PredictiveForce(stats) if parent is None else copy.deepcopy(parent)
    if args.stage == "student":
        model.initialize_student()
    model.configure(args.stage)
    model.to(args.device)
    frozen = digest(model, frozen_only=True)
    domain = "real" if sim is None else "sim"
    arrays = [prepare_arrays(real, stats, "real")]
    if sim is not None:
        arrays.append(prepare_arrays(sim, stats, "sim"))
    data = MultiTaskBlocks(real, sim, split["train"], arrays, stats["event_threshold_n"]["real"])
    steps = max(1, math.ceil(len(data) / (args.batch_size * BLOCK)))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    records, best_state, best_score, best_epoch, stale, start_epoch = [], None, math.inf, 0, 0, 1
    elapsed_before = 0.0
    if args.resume and (output / "last.pt").is_file():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        records, best_state, best_score = saved["records"], saved["best_state"], saved["best_score"]
        best_epoch, stale, start_epoch = saved["best_epoch"], saved["stale"], saved["epoch"] + 1
        elapsed_before = saved["elapsed_s"]
        torch.set_rng_state(saved["rng_state"])
        if args.device.startswith("cuda") and saved["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        if digest(model, frozen_only=True) != frozen:
            raise ValueError("Frozen parent weights changed in the resume checkpoint")
    started = time.monotonic()
    emit(
        output,
        f"{args.stage}: tasks={len(tasks)} train={len(split['train'])} validation={len(split['validation'])} "
        f"device={args.device} steps_per_epoch={steps}; starting epoch {start_epoch}",
    )
    for epoch in range(start_epoch, args.max_epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        generator = torch.Generator().manual_seed(args.seed + epoch * 7919)
        sampler = WeightedRandomSampler(data.weights, steps * args.batch_size, replacement=True, generator=generator)
        loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        losses = []
        for raw_batch in loader:
            batch = {key: value.to(args.device) for key, value in raw_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model.block(batch, int(sim is not None))
            teacher = None
            if sim is not None and args.align_weight:
                with torch.no_grad():
                    teacher = model.block(batch, 0)
            loss, detail = objective(model, prediction, batch, int(sim is not None), args.align_weight, teacher)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(detail)
        metrics = evaluate(model, real, real if sim is None else sim, split["validation"], arrays[-1], stats, domain)
        summary = metrics["summary"]
        if "forecast_score_n" not in summary or "reconstruction_nrmse" not in summary:
            raise ValueError("Validation needs continuous segments of >=16 frames for prediction and reconstruction")
        score = (
            summary["forecast_score_n"] / float(np.mean(stats["target"]["scale"]))
            + 0.25 * summary["reconstruction_nrmse"]
        )
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation score")
        if score < best_score:
            best_state, best_score, best_epoch, stale = cpu_state(model), score, epoch, 0
            save_model(output / "best.pt", model, stats, {"stage": args.stage, "epoch": epoch})
        else:
            stale += 1
        elapsed = elapsed_before + time.monotonic() - started
        records.append(
            {
                "epoch": epoch,
                "optimizer_steps": epoch * steps,
                "elapsed_s": elapsed,
                "train": aggregate(losses),
                "validation": summary,
                "by_task": metrics["by_task"],
                "selection_score": score,
            }
        )
        write_json(output / "history.json", {"completed": False, "records": records, "selected_epoch": best_epoch})
        atomic_torch_save(
            output / "last.pt",
            {
                "epoch": epoch,
                "state_dict": cpu_state(model),
                "optimizer": optimizer.state_dict(),
                "best_state": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale": stale,
                "records": records,
                "elapsed_s": elapsed,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
            },
        )
        if epoch == 1 or epoch % 10 == 0 or stale >= args.patience or epoch == args.max_epochs:
            emit(
                output,
                f"epoch={epoch} future_N={summary['forecast_score_n']:.5f} "
                f"reconstruction_nrmse={summary['reconstruction_nrmse']:.5f} "
                f"best_epoch={best_epoch} elapsed={elapsed/60:.1f}min",
            )
    model.load_state_dict(best_state)
    if digest(model, frozen_only=True) != frozen:
        raise RuntimeError("Frozen weights changed during training")
    metrics = evaluate(
        model,
        real,
        real if sim is None else sim,
        split["validation"],
        arrays[-1],
        stats,
        domain,
        output / "predictions" / "validation",
    )
    summary = {"stage": args.stage, "completed": True, "selected_epoch": best_epoch, "selection_score": best_score}
    checkpoint = output / f"encoder_seed{args.seed}.pt"
    save_model(checkpoint, model, stats, summary, motion_ready=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "history.json", {**summary, "records": records})
    lines = [
        "# Multi-task force predictive v3",
        "",
        f"Stage: {args.stage}; selected epoch: {best_epoch}.",
        "One fixed holdout; validation selects the checkpoint. No CV, motion heads or control training.",
        "",
        "| Task | Future score (N) | Reconstruction RMSE (N) |",
        "|---|---:|---:|",
    ]
    for task, row in metrics["by_task"].items():
        lines.append(f"| {task} | {row['forecast_score_n']:.5f} | {row['reconstruction_rmse_n']:.5f} |")
    lines += [
        "",
        "Metrics weight tasks equally and trajectories equally inside each task.",
        "The force-only trainer does not certify Sim state/action/video provenance for policy training.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(
        output / "manifest.json",
        {
            "artifact_format": FORMAT,
            "layout": LAYOUT,
            "checkpoint": checkpoint.name,
            "encoder_sha256": sha256(checkpoint),
            "stage": args.stage,
            "ready_domains": ["real"] if sim is None else ["real", "sim"],
            "history": HISTORY,
            "past": PAST,
            "future": FUTURE,
            "policy_history": 10,
            "selected_seed": args.seed,
            "motion_ready": False,
            "tasks": list(tasks),
            "input": "force_xyz_N_and_timestamps",
            "force_contract": "aligned_base_xyz_n_v1",
            "status": "real_teacher_only" if sim is None else "single_holdout_diagnostic",
            "actual_sim_state_verified": False,
        },
    )
    emit(output, f"Complete: {output / 'manifest.json'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("teacher", "student"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/force_predictive_v3_host9_multitask.json"))
    parser.add_argument("--data-root", type=Path, default=Path())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--align-weight", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("epochs, patience, batch size and threads must be positive")
    if not 0 <= args.seed < 2**32 - 10000:
        parser.error("seed must be a nonnegative integer below 2**32-10000")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be between 0 and 1")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.align_weight) or args.align_weight < 0:
        parser.error("positive finite lr and nonnegative finite align-weight required")
    for field in ("config", "data_root", "output_dir", "teacher_run"):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    prepared = prepare(args)
    if args.audit_only:
        emit(args.output_dir, f"Audit complete: {len(prepared[1])} Real trajectories; no training started")
        return
    train(args, *prepared)


if __name__ == "__main__":
    main()
