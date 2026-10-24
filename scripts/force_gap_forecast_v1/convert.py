"""New hash-pinned LeRobot exports; never overwrite a legacy dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from . import LAYOUT
from .common import read_manifest, sha256, write_json
from .data import load_pairs
from .runtime import ForceGapEncoder


def convert(manifest_path, root, repo_id, domain):
    from convert_user_hdf5_to_lerobot import _create_dataset, _read_video, _trajectory_id
    from lerobot.common.datasets.compute_stats import compute_stats
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    from lerobot.common.datasets.utils import serialize_dict
    import tqdm
    manifest, _ = read_manifest(manifest_path, require_ready=True)
    root = Path(root).resolve()
    expected = Path(manifest[f"{domain}_root"]).resolve()
    if root != expected:
        raise ValueError("Conversion must use the accepted source root, including actual Sim state")
    dataset_root = Path(LEROBOT_HOME) / repo_id
    audit = json.loads((Path(manifest_path).resolve().parent / manifest["audit"]).read_text())
    domain_id = 0 if domain == "real" else 1
    sources = {str(i): audit["trajectories"][str(i)]["sources"][domain_id]["sha256"] for i in range(40)}
    expected_contract = {"encoder_sha256": manifest["encoder_sha256"], "domain": domain,
                         "layout": LAYOUT, "source_h5_sha256": sources, "status": manifest["status"]}
    contract_path = dataset_root / "meta/force_gap_contract.json"
    if dataset_root.exists():
        if contract_path.is_file() and json.loads(contract_path.read_text()) == expected_contract:
            print(f"Reuse completed, identical export: {repo_id}")
            return dataset_root
        raise FileExistsError("Dataset exists with missing/different contract; use a new repo ID. No overwrite.")
    encoder = ForceGapEncoder(manifest_path, domain)
    torch.set_num_threads(1)
    dataset = _create_dataset(repo_id, fps=10, effort_dim=32, effort_names=tuple(LAYOUT), overwrite=False,
                              use_videos=False, image_writer_processes=0, image_writer_threads=0)
    files = sorted(root.glob("traj_*/data.h5"), key=_trajectory_id)
    for path in tqdm.tqdm(files, desc=f"Export {domain}"):
        i = _trajectory_id(path)
        if sha256(path) != sources[str(i)]:
            raise ValueError("HDF source changed since accepted encoder training")
        with h5py.File(path, "r") as h:
            if domain == "sim" and (h.attrs.get("state_source") != "sim_observed"
                    or not h.attrs.get("cameras_synchronized_same_replay", False)
                    or h.attrs.get("action_source") != "paired_real_reference_targets"):
                raise ValueError("Sim conversion requires actual same-replay state/images and reference action targets")
            state = np.concatenate((h["obs/state/joint_pos"][:], h["obs/state/gripper_pos"][:]), -1).astype(np.float32)
            action = np.concatenate((h["action/actual/arm"][:], h["action/actual/gripper"][:]), -1).astype(np.float32)
            effort = encoder.batch(h["obs/state/ee_force_base"][:], h["obs/state/joint_pos"][:],
                                   h["obs/state/joint_vel"][:], h["timestamps"][:])
        if domain == "sim":
            with h5py.File(Path(manifest["real_root"]) / path.parent.name / "data.h5", "r") as h:
                target = np.concatenate((h["action/actual/arm"][:], h["action/actual/gripper"][:]), -1).astype(np.float32)
            if not np.array_equal(action, target):
                raise ValueError("Sim action labels differ from their paired Real reference")
        n = len(state)
        front, wrist = _read_video(path.parent / "front_camera.mp4", n), _read_video(path.parent / "wrist_camera.mp4", n)
        for t in range(n):
            dataset.add_frame({"observation.state": state[t], "observation.effort": effort[t], "action": action[t],
                               "observation.images.cam_high": front[t], "observation.images.cam_left_wrist": wrist[t],
                               "observation.images.cam_right_wrist": wrist[t]})
        dataset.save_episode(task="peg-in-hole")
    dataset.consolidate(run_compute_stats=False)
    dataset.meta.stats = compute_stats(dataset, batch_size=8, num_workers=0)
    write_json(dataset.root / "meta/stats.json", serialize_dict(dataset.meta.stats))
    write_json(contract_path, expected_contract)
    return dataset_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--domain", choices=["real", "sim"], required=True)
    args = parser.parse_args()
    convert(args.manifest, args.raw_dir, args.repo_id, args.domain)


if __name__ == "__main__":
    main()
