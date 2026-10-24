#!/usr/bin/env python3
"""Compute norm stats for a config but with repo_id / assets_dir overridden.

The stock scripts/compute_norm_stats.py uses ``tyro.cli(main)`` whose main()
only accepts ``config_name/max_frames/batch_size/num_workers`` -- it does NOT
accept the ``--data.repo-id`` / ``--data.assets.assets-dir`` nested overrides
that train.py's ``_config.cli()`` understands.  This standalone helper reuses
that script's exact stats logic (imported, not modified) but lets us point an
existing config at a different dataset and assets directory.

Writes to ``<assets-dir>/<repo-id>/norm_stats.json``, which is exactly where
train.py's LeRobotTavlaDataConfig looks when given the same two overrides.

    PYTHONPATH=scripts:src .venv/bin/python \
      scripts/insertion_alignment_encoder/compute_norm_stats_for_repo.py \
      --config-name pi0_lora_user_single_arm_control_latent \
      --repo-id local/tavla_single_arm_force_align_latent \
      --assets-dir assets/pi0_lora_user_single_arm_force_align_latent
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from etils import epath
import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

# Reused, unmodified, from the stock stats script.
from compute_norm_stats import RemoveStrings, optimize_stats_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    # Redirect the factory to our dataset + assets dir without mutating any file.
    data_factory = dataclasses.replace(
        config.data,
        repo_id=args.repo_id,
        assets=dataclasses.replace(config.data.assets, assets_dir=args.assets_dir),
    )
    config = dataclasses.replace(config, data=data_factory)

    data_config = config.data.create(config.assets_dirs, config.model)
    if not isinstance(data_config.repo_id, str):
        raise ValueError("this helper handles a single repo_id only")

    dataset = _data_loader.create_dataset(data_config, config.model)
    stats_only_transforms = optimize_stats_dataset(dataset)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *stats_only_transforms,
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    num_frames = len(dataset)
    shuffle = False
    if args.max_frames is not None and args.max_frames < num_frames:
        num_frames = args.max_frames
        shuffle = True
    num_batches = num_frames // args.batch_size
    if num_batches < 1:
        raise ValueError("The dataset contains no frames to compute statistics from.")

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    keys = ["state", "actions"]
    if getattr(config.data, "effort_history", False):
        keys.append("effort")
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key])
            stats[key].update(values.reshape(-1, values.shape[-1]))
    norm_stats = {key: running.get_statistics() for key, running in stats.items()}

    output_path = epath.Path(args.assets_dir) / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    main()
