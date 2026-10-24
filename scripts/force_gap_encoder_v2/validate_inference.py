#!/usr/bin/env python3
"""Check that the exported artifact loads and streaming matches batch inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from force_gap_encoder_v2.deploy_infer import ForceGapLatentEncoder, StreamingForceGapEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/encoder.pt"))
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--trajectories", type=int, nargs="+", default=(0, 2, 21, 22, 23, 33, 34, 39))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    encoder = ForceGapLatentEncoder(args.checkpoint, device=args.device)
    for trajectory in args.trajectories:
        for domain, root in (("real", args.real_root), ("sim", args.sim_root)):
            path = root / f"traj_{trajectory}" / "data.h5"
            with h5py.File(path, "r") as episode:
                force = np.asarray(episode[args.force_key], dtype=np.float32)[:, :3]
                timestamps = np.asarray(episode["timestamps"], dtype=np.float64)
            batch = encoder.latents(force, timestamps)
            stream = StreamingForceGapEncoder(encoder, output_history=10)
            streamed = []
            for sample, timestamp in zip(force, timestamps, strict=True):
                latent = stream.push(sample, float(timestamp))
                if latent is not None:
                    streamed.append(latent)
            streamed = np.asarray(streamed, dtype=np.float32)
            expected = batch[encoder.calibration_frames - 1 :]
            if streamed.shape != expected.shape or not np.isfinite(streamed).all():
                raise ValueError(f"traj_{trajectory} {domain}: invalid stream output shape/content {streamed.shape}")
            max_error = float(np.max(np.abs(streamed - expected)))
            if max_error > args.tolerance:
                raise ValueError(
                    f"traj_{trajectory} {domain}: batch/stream max error {max_error:.8g} exceeds {args.tolerance}"
                )
            history = stream.effort_history()
            if history.shape != (10, encoder.latent_dim) or not np.isfinite(history).all():
                raise ValueError(f"traj_{trajectory} {domain}: malformed 10-frame effort history {history.shape}")
            print(
                f"traj_{trajectory} {domain}: frames={len(force)} latent_dim={encoder.latent_dim} "
                f"stream_batch_max_abs_error={max_error:.3g} effort_history_shape={history.shape}",
                flush=True,
            )

    print("INFERENCE CHECK: PASS")


if __name__ == "__main__":
    main()
