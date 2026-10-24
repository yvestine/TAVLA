"""Four-GPU TAVLA training with precomputed, frozen motion-only effort features."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from force_gap_forecast_v1.common import write_json
from .policy_contract import CONFIG_NAME, read_encoder_manifest, read_json, sha256, validate_policy_manifest


def command(root, python, exp_name, batch_size=8, steps=30000, save_interval=10000,
            log_interval=100, num_workers=2, resume=False):
    result = [python, str(Path(root) / "scripts/train.py"), CONFIG_NAME, "--exp-name", exp_name,
              "--batch-size", str(batch_size), "--fsdp-devices", "4", "--num-train-steps", str(steps),
              "--save-interval", str(save_interval), "--log-interval", str(log_interval),
              "--num-workers", str(num_workers), "--no-wandb-enabled"]
    if resume:
        result.append("--resume")
    return result


def attach_contracts(checkpoint_root, policy_path, policy):
    encoder, checkpoint, _ = read_encoder_manifest(policy["encoder_manifest"])
    original_dir = Path(policy["encoder_manifest"]).parent
    attached = []
    for directory in sorted(checkpoint_root.iterdir()):
        if not directory.name.isdecimal() or not (directory / "params").is_dir():
            continue
        assets = directory / "assets"
        if not (assets / "norm_stats.json").is_file():
            continue
        target = assets / "motion_only_v2_policy_manifest.json"
        if target.exists() and read_json(target) != policy:
            raise ValueError("Checkpoint already carries a different encoder/policy contract")
        shutil.copy2(policy_path, target)
        # Inference can load the frozen encoder locally without retraining or fetching its source run.
        shutil.copy2(checkpoint, assets / "motion_only_v2_encoder.pt")
        for key in ("audit", "acceptance"):
            shutil.copy2(original_dir / encoder[key], assets / encoder[key])
        portable = dict(encoder, checkpoint="motion_only_v2_encoder.pt", source_manifest=policy["encoder_manifest"])
        write_json(assets / "motion_only_v2_encoder_manifest.json", portable)
        attached.append(directory.name)
    return attached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    ids = args.gpu_ids.split(",")
    if len(ids) != 4 or any(not value.isdecimal() for value in ids) or len(set(map(int, ids))) != 4:
        parser.error("Exactly four distinct physical GPU indices are required")
    if min(args.batch_size, args.steps, args.save_interval, args.log_interval) < 1 or args.batch_size % 4:
        parser.error("Positive sizes/intervals and a global batch divisible by four are required")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if not args.exp_name or Path(args.exp_name).name != args.exp_name or args.exp_name in (".", ".."):
        parser.error("exp-name must be one directory component")
    args.policy_manifest = args.policy_manifest.resolve()
    home = os.environ.get("LEROBOT_HOME")
    if not home:
        parser.error("Set LEROBOT_HOME to the feature cache directory used during preparation")
    policy = validate_policy_manifest(args.policy_manifest, home)
    root = Path(__file__).resolve().parents[2]
    checkpoint_root = root / "checkpoints" / CONFIG_NAME / args.exp_name
    run_dir = args.policy_manifest.parent / "policy_runs" / args.exp_name
    recipe = dict(policy_manifest_sha256=sha256(args.policy_manifest), config_name=CONFIG_NAME,
                  encoder_sha256=policy["encoder_sha256"], feature_id=policy["feature_id"],
                  batch_size=args.batch_size, fsdp_devices=4, num_train_steps=args.steps,
                  save_interval=args.save_interval, log_interval=args.log_interval, num_workers=args.num_workers,
                  config_sha256=sha256(root / "src/openpi/training/config.py"),
                  train_source_sha256=sha256(root / "scripts/train.py"),
                  wrapper_source_sha256=sha256(Path(__file__)),
                  contract_source_sha256=sha256(Path(__file__).parent / "policy_contract.py"))
    recipe_path = run_dir / "training_recipe.json"
    if args.resume or args.finalize_only:
        if not recipe_path.is_file() or read_json(recipe_path) != recipe:
            raise ValueError("Resume must use the original policy preparation, experiment, code and training parameters")
    elif checkpoint_root.exists() or run_dir.exists():
        raise FileExistsError("Experiment exists; choose a new EXP_NAME or use matching RESUME=1")
    else:
        run_dir.mkdir(parents=True)
        write_json(recipe_path, recipe)
        shutil.copy2(args.policy_manifest, run_dir / "policy_manifest.json")
    env = os.environ.copy()
    env.update(MOTION_ONLY_V2_POLICY_MANIFEST=str(args.policy_manifest), CUDA_VISIBLE_DEVICES=args.gpu_ids,
               XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS="cuda", PYTHONUNBUFFERED="1")
    print(f"Frozen encoder seed={policy['selected_seed']}; effort=(10,32); Real/Sim=50:50; "
          f"GPUs={args.gpu_ids}; global batch={args.batch_size}; per GPU={args.batch_size // 4}", flush=True)
    print(f"Checkpoints: {checkpoint_root}; run metadata: {run_dir}", flush=True)
    if not args.finalize_only:
        cmd = command(root, sys.executable, args.exp_name, args.batch_size, args.steps,
                      args.save_interval, args.log_interval, args.num_workers, args.resume)
        write_json(run_dir / "command.json", dict(argv=cmd, policy_manifest=str(args.policy_manifest)))
        subprocess.run(cmd, cwd=root, env=env, check=True)
    final = checkpoint_root / str(args.steps - 1)
    if not (final / "params").is_dir() or not (final / "assets/norm_stats.json").is_file():
        raise FileNotFoundError(f"Final policy checkpoint is incomplete: {final}")
    attached = attach_contracts(checkpoint_root, args.policy_manifest, policy)
    write_json(run_dir / "completed.json", dict(final_checkpoint=str(final), feature_id=policy["feature_id"],
                                               encoder_sha256=policy["encoder_sha256"], attached_checkpoints=attached))
    print(f"Completed: {final}", flush=True)


if __name__ == "__main__":
    main()
