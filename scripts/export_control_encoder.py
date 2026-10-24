#!/usr/bin/env python3
"""Export ONE deployable control-supervised encoder trained on all trajectories.

The leave-one-out fold checkpoints under ``insertion_quality_control_encoder``
exist only to measure sim->real transfer honestly; each holds out a trajectory,
so none of them is the encoder you deploy.  This script trains a single
``ControlEncoder`` on ALL trajectories (Real+Sim) with the same config and freezes
it, so the downstream converter can turn each Real force sample into the 32-D
latent that TAVLA consumes as ``observation.effort``.

It writes ONLY under ``--output-dir`` (a frozen ``encoder.pt`` plus the feature
normalization metadata the converter needs).  It does not touch any TAVLA/pi0
checkpoint.  This is offline and runs in ~1 minute on one GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_insertion_quality_v3_transformer as v3
from train_insertion_quality_control_encoder import ControlEncoder
from train_insertion_quality_control_encoder import build_control_arrays
from train_insertion_quality_control_encoder import train_model
from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_sim_real_control_encoder.json"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--variant", default="force_pose_control_domain_invariant")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/control_encoder_deploy"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    annotations_path = args.annotations if args.annotations.is_absolute() else repo / args.annotations
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir

    config = read_config(config_path)
    if "force_config" in config:
        fc_path = Path(config["force_config"])
        fc_path = fc_path if fc_path.is_absolute() else repo / fc_path
        force_config = read_config(fc_path)
        config["free_end_frames"] = force_config["free_end_frames"]
        config["representation"] = {**force_config["representation"], **config.get("representation", {})}
    annotations = read_annotations(annotations_path)
    all_ids = [int(v) for v in config.get("all_trajectories", sorted(int(k) for k in config["free_end_frames"]))]
    variant = config["variants"][args.variant]
    device = torch.device(args.device)
    data = load_trajectories(repo, config, all_ids)

    # Build features/windows/control targets on every trajectory (no held-out).
    features, feature_metadata = v3.build_features(data, all_ids, all_ids, config, None, variant["use_pose"])
    arrays = v3.build_windows(data, all_ids, set(all_ids), annotations, config, None, features)
    control = build_control_arrays(data, all_ids, arrays, config)
    input_dim = arrays.x.shape[-1]
    v3.set_seed(int(config["seed"]))
    model, history = train_model(arrays, control, input_dim, config, variant, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": args.variant,
            "config": config,
            "feature_metadata": feature_metadata,
            "input_dim": int(input_dim),
            "latent_dim": int(config["model"]["latent_dim"]),
        },
        output_dir / "encoder.pt",
    )
    # The converter needs the exact force feature normalization used at train time
    # so a single Real force sample maps into the same feature space the encoder saw.
    (output_dir / "feature_metadata.json").write_text(json.dumps(feature_metadata, indent=2), encoding="utf-8")
    latent = np.concatenate(
        [
            model.encode(torch.from_numpy(arrays.x[s : s + 512]).to(device)).detach().cpu().numpy()
            for s in range(0, len(arrays.x), 512)
        ]
    )
    print(f"Trained on {len(all_ids)} trajectories, {len(arrays.x)} windows.")
    print(f"Latent shape {latent.shape}; wrote {output_dir/'encoder.pt'}")


if __name__ == "__main__":
    main()
