#!/usr/bin/env python3
"""Read-only provenance check for the older contact-force Sim training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def check_episode(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("sim_force_saved", False)):
            raise ValueError(f"{path}: missing sim_force_saved provenance")
        if handle.attrs.get("sim_force_training_source") != "contact_total_base":
            raise ValueError(f"{path}: unexpected training force source")
        contact = np.asarray(handle["sim/force/contact_total_base"])
        training = np.asarray(handle["sim/force/contact_training_wrench"])
        effort = np.asarray(handle["obs/state/ee_wrench_base"])
        force = np.asarray(handle["obs/state/ee_force_base"])
    if contact.ndim != 2 or contact.shape[1] != 3:
        raise ValueError(f"{path}: contact force must be [frames, 3], got {contact.shape}")
    if training.shape != (len(contact), 6) or effort.shape != training.shape or force.shape != contact.shape:
        raise ValueError(f"{path}: force field shapes do not match")
    if not all(np.isfinite(values).all() for values in (contact, training, effort, force)):
        raise ValueError(f"{path}: force contains NaN or Inf")
    if not np.array_equal(training[:, :3], contact) or not np.all(training[:, 3:] == 0):
        raise ValueError(f"{path}: contact_training_wrench is not [contact_total_base, zeros(3)]")
    if not np.array_equal(effort, training) or not np.array_equal(force, contact):
        raise ValueError(f"{path}: training effort is not synchronized with contact force")
    return effort


def check_lerobot(repo: Path, efforts: list[np.ndarray]) -> None:
    import pyarrow.parquet as parquet

    info_path = repo / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["total_episodes"] != len(efforts) or info["total_frames"] != sum(map(len, efforts)):
        raise ValueError(f"{repo}: episode/frame counts differ from raw Sim force")
    shape = info["features"]["observation.effort"]["shape"]
    if shape != [6]:
        raise ValueError(f"{repo}: expected six-dimensional effort, got {shape}")
    for episode_index, expected in enumerate(efforts):
        relative = info["data_path"].format(
            episode_chunk=episode_index // info["chunks_size"],
            episode_index=episode_index,
        )
        path = repo / relative
        actual = np.asarray(
            parquet.read_table(path, columns=["observation.effort"])
            .column("observation.effort")
            .to_pylist(),
            dtype=np.float32,
        )
        if not np.array_equal(actual, expected):
            raise ValueError(f"{path}: LeRobot effort differs from raw Sim force")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--lerobot-repo", type=Path)
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()
    paths = sorted(
        raw_dir.glob("traj_*/data.h5"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"No traj_*/data.h5 under {raw_dir}")
    indices = [int(path.parent.name.split("_")[-1]) for path in paths]
    if indices != list(range(len(paths))):
        raise ValueError(f"{raw_dir}: trajectory indices must be contiguous from zero")
    efforts = [check_episode(path) for path in paths]
    if args.lerobot_repo is not None:
        check_lerobot(args.lerobot_repo.resolve(), efforts)
    print(f"Validated {len(paths)} Sim contact-force episodes, {sum(map(len, efforts))} frames")
    if args.lerobot_repo is not None:
        print(f"LeRobot effort matches raw Sim force: {args.lerobot_repo}")


if __name__ == "__main__":
    main()
