"""GPU worker queue, strict run identities, nested CV aggregation and final refits."""
from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import subprocess
import sys
import time

from force_gap_forecast_v1.common import sha256, write_json
from . import FORMAT, HISTORY, PAST, FUTURE, LAYOUT, TAU_S
from .data import FOLDS, TRAIN_IDS, HOLDOUT, load_pairs
from .report import write_report
from .train import read_json


def run_queue(tasks, root, gpu_ids, resume, device="cuda:0"):
    """At most one worker per explicitly supplied GPU; only our own child processes are stopped."""
    pending, active = deque(tasks), {}
    try:
        while pending or active:
            for gpu in gpu_ids:
                if gpu in active or not pending:
                    continue
                task = pending.popleft()
                done = root / task["done"]
                if resume and done.is_file() and read_json(done).get("completed"):
                    if task.get("final"):
                        checkpoint = root / f"encoder_seed{task['seed']}.pt"
                        if not checkpoint.is_file() or sha256(checkpoint) != read_json(done)["checkpoint_sha256"]:
                            raise ValueError(f"Final checkpoint changed: {checkpoint}")
                    print(f"Reusing completed {task['name']}", flush=True)
                    continue
                command = [sys.executable, "-m", "force_predictive_encoder_v3.train", "--run-dir", str(root),
                           "--seed", str(task["seed"]), "--device", device]
                if task.get("final"):
                    command.append("--final")
                elif task.get("single"):
                    command.append("--single")
                else:
                    command += ["--fold", str(task["fold"])]
                if resume:
                    command.append("--resume")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["PYTHONUNBUFFERED"] = "1"
                log = (root / "logs" / f"{task['name']}.log").open("a")
                try:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active[gpu] = (process, log, task, time.monotonic())
                print(f"GPU {gpu}: starting {task['name']}; log={log.name}", flush=True)
            for gpu, (process, log, task, started) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[gpu]
                if status:
                    raise RuntimeError(f"{task['name']} failed (exit {status}); inspect {log.name}. Completed stages are retained.")
                done = root / task["done"]
                if not done.is_file() or not read_json(done).get("completed"):
                    raise RuntimeError(f"Worker did not produce a completed result: {task['name']}")
                print(f"GPU {gpu}: finished {task['name']} in {(time.monotonic() - started) / 60:.1f} min", flush=True)
            if active:
                time.sleep(1)
    finally:
        for process, log, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, log, _, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS", "0"))
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--old-base-run", type=Path, default=Path("eval_outputs/force_gap_forecast_v1/20261012_155026_551478"))
    parser.add_argument("--no-old-baseline", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--align-weights", nargs="+", type=float, default=[0., .03, .1, .3])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--single-run", dest="single_run", action="store_true",
                      help="one seed and one fixed holdout; skip CV, weight search, ablations and final refits (default)")
    mode.add_argument("--full-cv", dest="single_run", action="store_false",
                      help="run the full multi-seed, multi-fold comparison protocol")
    parser.set_defaults(single_run=True)
    parser.add_argument("--seed", type=int, default=42, help="seed used by --single-run")
    parser.add_argument("--validation-fold", type=int, choices=range(4), default=0,
                        help="fixed held-out group used by --single-run")
    parser.add_argument("--single-align-weight", type=float, default=.1,
                        help="fixed trend-alignment weight used by --single-run")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--require-actual-sim-state", action="store_true")
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    args = parser.parse_args()
    import math
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Positive finite learning rate, epochs, patience, batch size and threads required")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 10000 for s in args.seeds):
        parser.error("Seeds must be unique nonnegative integers below 2**32-10000")
    if len(set(args.folds)) != len(args.folds) or not set(args.folds) <= set(range(4)):
        parser.error("Unique fold indices 0..3 required")
    if (0. not in args.align_weights or len(set(args.align_weights)) != len(args.align_weights) or
            any(not math.isfinite(x) or x < 0 for x in args.align_weights)):
        parser.error("Alignment weights must be finite, distinct, nonnegative and contain zero")
    if args.single_run:
        if args.seed < 0 or args.seed >= 2**32 - 10000:
            parser.error("Single-run seed must be a nonnegative integer below 2**32-10000")
        if not math.isfinite(args.single_align_weight) or args.single_align_weight < 0:
            parser.error("Single-run alignment weight must be finite and nonnegative")
        args.seeds = [args.seed]
        args.folds = [args.validation_fold]
        args.align_weights = [args.single_align_weight]
        args.skip_final = True
        args.no_old_baseline = True
    args.gpu_ids = [s.strip() for s in args.gpu_ids.split(",")]
    if not all(s.isdigit() for s in args.gpu_ids) or len(set(args.gpu_ids)) != len(args.gpu_ids):
        parser.error("GPU IDs must be distinct comma-separated physical indices")
    args.gpu_ids = [str(int(s)) for s in args.gpu_ids]
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        parser.error("GPU IDs refer to the same physical device")
    for field in ("output_dir", "real_root", "sim_root", "labels", "old_base_run"):
        setattr(args, field, getattr(args, field).resolve())
    return args


def prepare(args):
    root = args.output_dir
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k not in ("output_dir", "gpu_ids", "resume", "no_plots", "skip_final", "device", "no_old_baseline")}
    if args.no_old_baseline:
        options["old_base_run"] = None
    _, audit = load_pairs(args.real_root, args.sim_root, args.labels)
    if args.require_actual_sim_state and not audit["actual_sim_state_verified"]:
        raise ValueError("Sim observed-state provenance is not verified")
    scripts = Path(__file__).resolve().parent.parent
    sources = list(Path(__file__).parent.glob("*.py")) + [Path(__file__).parent / "README.md", scripts / "run_force_predictive_encoder_v3.sh"]
    for package in ("force_gap_forecast_v1", "force_gap_motion_force_v2"):
        sources.extend((scripts / package).glob("*.py"))
    old_checkpoints = {}
    if options["old_base_run"]:
        old_recipe = read_json(args.old_base_run / "recipe.json")
        if old_recipe["data"] != audit or old_recipe["label_sha256"] != sha256(args.labels):
            raise ValueError("Old baseline used different data/labels; use --no-old-baseline for a fresh dataset")
        for seed in args.seeds:
            for fold in args.folds:
                path = args.old_base_run / "stages" / f"seed{seed}_fold{fold}_motion_only_student_refit" / "model.pt"
                old_checkpoints[str(path)] = sha256(path)
    recipe = dict(artifact_format=FORMAT, options=options, data=audit, label_sha256=sha256(args.labels),
                  folds=FOLDS, train_ids=TRAIN_IDS, diagnostic_ids=HOLDOUT, layout=LAYOUT,
                  history=HISTORY, past=PAST, future=FUTURE, ema_tau_s=TAU_S,
                  old_checkpoints=old_checkpoints, source_sha256={str(p.relative_to(scripts)): sha256(p) for p in sources})
    if args.resume:
        if not (root / "recipe.json").is_file() or read_json(root / "recipe.json") != recipe:
            raise ValueError("Resume requires identical code, data, labels and training parameters; use a new RUN_DIR")
    elif root.exists() and any(p.name != "train.log" for p in root.iterdir()):
        raise FileExistsError("RUN_DIR already contains a run; use matching RESUME=1 or a new directory")
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    write_json(root / "recipe.json", recipe)
    write_json(root / "data_audit.json", audit)
    for source in sources:
        dest = root / "source_snapshot" / source.relative_to(scripts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    return recipe


def main():
    args = parse_args()
    recipe = prepare(args)
    root = args.output_dir
    print(f"V3 run: {root}\nGPU queue: {','.join(args.gpu_ids)}; actual Sim state: {recipe['data']['actual_sim_state_verified']}", flush=True)
    if args.single_run:
        task = dict(name=f"seed{args.seed}_single", seed=args.seed, single=True, done="single/result.json")
        run_queue([task], root, args.gpu_ids, args.resume, args.device)
        result = read_json(root / task["done"])
        checkpoint = root / f"encoder_seed{args.seed}.pt"
        manifest = dict(artifact_format=FORMAT, layout=LAYOUT, checkpoint=checkpoint.name,
                        encoder_sha256=sha256(checkpoint), status="single_holdout_diagnostic",
                        selected_seed=args.seed, validation_fold=args.validation_fold,
                        selection="fixed seed, one fixed holdout, no cross-validation",
                        history=HISTORY, past=PAST, future=FUTURE, policy_history=10,
                        motion_ready=True, input="force_xyz_N_and_timestamps",
                        force_contract="aligned_base_xyz_n_v1",
                        actual_sim_state_verified=recipe["data"]["actual_sim_state_verified"])
        write_json(root / "manifest.json", manifest)
        write_json(root / "run_summary.json", dict(completed=True, status="single_holdout_diagnostic",
                                                   final_encoder_created=True, report=str(root / "single" / "single_report.md"),
                                                   result=str(root / "single" / "result.json")))
        print(f"Complete: {root / 'single' / 'single_report.md'}\nEncoder: {checkpoint}\nStatus: single_holdout_diagnostic", flush=True)
        return
    tasks = [dict(name=f"seed{seed}_fold{fold}", seed=seed, fold=fold,
                  done=f"jobs/seed{seed}_fold{fold}/result.json") for seed in args.seeds for fold in args.folds]
    run_queue(tasks, root, args.gpu_ids, args.resume, args.device)
    results = [read_json(root / task["done"]) for task in tasks]
    gate = write_report(root, results, recipe)
    status = ("waveform_validated" if recipe["data"]["actual_sim_state_verified"] else "replay_validated") if gate["passed"] else "diagnostic"
    final_created = set(args.folds) == set(range(4)) and not args.skip_final
    if final_created:
        final_tasks = [dict(name=f"seed{seed}_final", seed=seed, final=True, done=f"final/seed{seed}/result.json") for seed in args.seeds]
        run_queue(final_tasks, root, args.gpu_ids, args.resume, args.device)
        default_seed = 42 if 42 in args.seeds else min(args.seeds)
        for seed in args.seeds:
            checkpoint = root / f"encoder_seed{seed}.pt"
            manifest = dict(artifact_format=FORMAT, layout=LAYOUT, checkpoint=checkpoint.name,
                            encoder_sha256=sha256(checkpoint), status=status, selected_seed=seed,
                            selection="fixed seed, no outer-score selection", history=HISTORY, past=PAST,
                            future=FUTURE, policy_history=10, motion_ready=True, input="force_xyz_N_and_timestamps",
                            force_contract="aligned_base_xyz_n_v1", actual_sim_state_verified=recipe["data"]["actual_sim_state_verified"],
                            acceptance="acceptance.json", acceptance_sha256=sha256(root / "acceptance.json"),
                            audit="data_audit.json", audit_sha256=sha256(root / "data_audit.json"))
            write_json(root / f"manifest_seed{seed}.json", manifest)
            if seed == default_seed:
                write_json(root / "manifest.json", manifest)
    if not args.no_plots:
        from .visualize import visualize
        visualize(root)
    write_json(root / "run_summary.json", dict(completed=True, status=status, final_encoder_created=final_created,
                                               report=str(root / "report.md")))
    print(f"Complete: {root / 'report.md'}\nStatus: {status}; final encoder: {final_created}", flush=True)


if __name__ == "__main__":
    main()
