"""Train the accepted fresh policy; persist its feature contract with checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import FORMAT, LAYOUT
from .common import read_manifest, sha256
from .prepare_policy import CONFIG_NAME


def validate_policy_manifest(path):
    value = json.loads(Path(path).read_text())
    if value.get("artifact_format") != FORMAT or value.get("layout") != LAYOUT or value.get("status") != "policy_train_ready":
        raise ValueError("Unaccepted feature contract")
    encoder, _ = read_manifest(value["encoder_manifest"], require_ready=True)
    if encoder["encoder_sha256"] != value["encoder_sha256"]:
        raise ValueError("Encoder contract mismatch")
    stats = Path(value["norm_stats_dir"]) / value["norm_asset_id"] / "norm_stats.json"
    if sha256(stats) != value["norm_stats_sha256"] or value.get("use_quantile_norm") is not False:
        raise ValueError("Shared normalization mismatch")
    for domain, root, digest in zip(("real", "sim"), value["dataset_roots"], value["dataset_contract_sha256"]):
        contract = Path(root) / "meta/force_gap_contract.json"
        if sha256(contract) != digest:
            raise ValueError("Dataset contract changed")
        data = json.loads(contract.read_text())
        if data["encoder_sha256"] != value["encoder_sha256"] or data["layout"] != LAYOUT or data["domain"] != domain:
            raise ValueError("Dataset encoder / layout mismatch")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    manifest = validate_policy_manifest(args.policy_manifest)
    ids = args.gpu_ids.split(",")
    if len(ids) != 4 or len(set(ids)) != 4 or any(not x.isdecimal() for x in ids):
        parser.error("Exactly four distinct physical GPU IDs required")
    if not args.exp_name or Path(args.exp_name).name != args.exp_name:
        parser.error("exp-name must be one directory component")
    root = Path(__file__).resolve().parents[2]
    checkpoint_root = root / "checkpoints" / CONFIG_NAME / args.exp_name
    if checkpoint_root.exists() and not args.resume and not args.finalize_only:
        raise FileExistsError("Checkpoint experiment exists; use --resume or a new exp-name")
    env = os.environ.copy()
    env.update(FORCE_GAP_FORECAST_POLICY_MANIFEST=str(args.policy_manifest.resolve()), CUDA_VISIBLE_DEVICES=args.gpu_ids,
               XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS="cuda", PYTHONUNBUFFERED="1")
    if not args.finalize_only:
        cmd = [sys.executable, str(root / "scripts/train.py"), CONFIG_NAME, "--exp-name", args.exp_name,
               "--batch-size", "8", "--fsdp-devices", "4", "--num-train-steps", "30000",
               "--save-interval", "30000", "--no-wandb-enabled"]
        if args.resume:
            cmd.append("--resume")
        subprocess.run(cmd, cwd=root, env=env, check=True)
    final = checkpoint_root / "29999"
    if not (final / "params").is_dir() or not (final / "assets/norm_stats.json").is_file():
        raise FileNotFoundError("Final checkpoint 29999 not complete; diagnostic files retained")
    target = final / "assets/force_gap_policy_manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Checkpoint already carries a different feature contract")
    shutil.copy2(args.policy_manifest, target)
    print(f"Policy ready: {final}")
    print(f"Serve: {sys.executable} -m force_gap_forecast_v1.serve --policy-manifest {args.policy_manifest} --checkpoint-dir {final} --port 8006")


if __name__ == "__main__":
    main()
