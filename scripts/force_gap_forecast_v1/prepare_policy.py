"""Waveform gate -> immutable exports -> exact 50:50 shared policy statistics."""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np

from . import FORMAT, LAYOUT
from .common import read_manifest, sha256, write_json
from .convert import convert

CONFIG_NAME = "pi0_lora_user_single_arm_force_gap_forecast_cotrain_50_50"


def shared_stats(repo_ids, assets_root):
    from openpi.training import config, data_loader
    from openpi.shared import normalize
    import openpi.transforms as transforms
    from compute_norm_stats import RemoveStrings, optimize_stats_dataset
    # Build transforms from the legacy shape-compatible config without its old normalization.
    cfg = config.get_config("pi0_lora_user_single_arm_force_align_latent_cotrain_50_50")
    moments = []
    for repo in repo_ids:
        factory = dataclasses.replace(cfg.data, repo_id=repo, assets=config.AssetsConfig(), padding_stat=False)
        dc = factory.create(cfg.assets_dirs, cfg.model)
        dataset = data_loader.create_dataset(dc, cfg.model)
        skip_images = optimize_stats_dataset(dataset)
        dataset = data_loader.TransformedDataset(dataset, [*skip_images, *dc.repack_transforms.inputs,
                                                           *dc.data_transforms.inputs, RemoveStrings()])
        sums, squares, counts = {}, {}, {}
        # Include partial last rows; stats-loader drop_last would silently omit frames.
        for index in range(len(dataset)):
            row = dataset[index]
            for key in ("state", "actions", "effort"):
                values = np.asarray(row[key], np.float64)
                values = values.reshape(-1, values.shape[-1])
                sums[key] = sums.get(key, 0) + values.sum(0)
                squares[key] = squares.get(key, 0) + np.square(values).sum(0)
                counts[key] = counts.get(key, 0) + len(values)
        moments.append({key: (sums[key] / counts[key], squares[key] / counts[key]) for key in sums})
    result = {}
    for key in moments[0]:
        mean = .5 * (moments[0][key][0] + moments[1][key][0])
        second = .5 * (moments[0][key][1] + moments[1][key][1])
        scale = np.maximum(np.sqrt(np.maximum(second - mean**2, 0)), 1e-6)
        result[key] = normalize.NormStats(mean=mean, std=scale)
    normalize.save(assets_root / "shared", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, checkpoint = read_manifest(args.manifest, require_ready=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("New policy preparation directory required; no overwrite")
    # Re-audit all pairs before creating any export.
    from .data import load_pairs
    _, audit = load_pairs(manifest["real_root"], manifest["sim_root"])
    if not audit["actual_sim_state_verified"]:
        raise ValueError("Actual-Sim provenance check failed")
    output.mkdir(parents=True)
    sha = manifest["encoder_sha256"]
    repos = [f"local/tavla_force_gap_forecast_{sha[:12]}_{domain}" for domain in ("real", "sim")]
    dataset_roots = [convert(args.manifest, manifest[f"{domain}_root"], repo, domain)
                     for repo, domain in zip(repos, ("real", "sim"))]
    assets = output / "assets"
    shared_stats(repos, assets)
    policy_manifest = {"artifact_format": FORMAT, "layout": LAYOUT, "status": "policy_train_ready",
                       "encoder_sha256": sha, "checkpoint": str(checkpoint),
                       "encoder_manifest": str(args.manifest.resolve()), "config_name": CONFIG_NAME,
                       "repo_ids": repos, "dataset_roots": [str(p) for p in dataset_roots],
                       "dataset_contract_sha256": [sha256(p / "meta/force_gap_contract.json") for p in dataset_roots],
                       "sampling_weights": [.5, .5], "norm_stats_dir": str(assets), "norm_asset_id": "shared",
                       "norm_stats_sha256": sha256(assets / "shared/norm_stats.json"),
                       "use_quantile_norm": False, "actions_are_absolute": True,
                       "effort_shape": [10, 32], "action_shape": [50, 8]}
    write_json(output / "policy_manifest.json", policy_manifest)
    print(output / "policy_manifest.json")


if __name__ == "__main__":
    main()
