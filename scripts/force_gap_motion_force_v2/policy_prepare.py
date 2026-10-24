"""Export frozen motion-only effort and shared 50:50 statistics for TAVLA."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from force_gap_forecast_v1.common import finite_array, write_json
from force_gap_forecast_v1.data import TRAIN_IDS, fit_stats, load_pairs
from . import FORMAT, GAP_S, LAYOUT
from .metrics import predict_episode
from .policy_contract import (CONFIG_NAME, CONTRACT_FILE, DEFAULT_ENCODER_MANIFEST, POLICY_FORMAT, POLICY_STATUS,
                              read_encoder_manifest, read_json, repo_ids, sha256, stable_digest,
                              validate_dataset, validate_policy_manifest)
from .runtime import JointMotionForceEncoder


def feature_sources():
    scripts = Path(__file__).resolve().parent.parent
    paths = [Path(__file__), Path(__file__).parent / "policy_contract.py",
             Path(__file__).parent / "runtime.py", Path(__file__).parent / "model.py",
             Path(__file__).parent / "metrics.py", Path(__file__).parent / "__init__.py",
             scripts / "force_gap_forecast_v1/model.py", scripts / "force_gap_forecast_v1/data.py",
             scripts / "force_gap_forecast_v1/common.py", scripts / "force_gap_forecast_v1/__init__.py",
             scripts / "force_gap_forecast_v1/prepare_policy.py", scripts / "convert_user_hdf5_to_lerobot.py"]
    return {str(path): sha256(path) for path in paths}


def video_sources(manifest):
    result = {}
    for domain in ("real", "sim"):
        root = Path(manifest[f"{domain}_root"])
        result[domain] = {str(tid): {name: sha256(root / f"traj_{tid}" / name)
                                    for name in ("front_camera.mp4", "wrist_camera.mp4")} for tid in range(40)}
    return result


def build_identity(manifest, audit, videos, cache_tag):
    return dict(encoder_sha256=manifest["encoder_sha256"], layout=LAYOUT, encoder_format=FORMAT,
                source_audit_sha256=manifest["audit_sha256"], source_video_sha256=videos,
                actual_sim_state_verified=audit["actual_sim_state_verified"], fps=10, encoder_frozen=True,
                policy_history_frames=10, segment_gap_s=GAP_S, cache_tag=cache_tag,
                feature_source_sha256=feature_sources())


def episode_arrays(path, ep):
    n = len(ep.force)
    with h5py.File(path, "r") as h:
        grip = finite_array(h["obs/state/gripper_pos"][:], (n, 1), "gripper_pos")
        arm = finite_array(h["action/actual/arm"][:], (n, 7), "action_arm")
        action_grip = finite_array(h["action/actual/gripper"][:], (n, 1), "action_gripper")
    if np.any((grip < 0) | (grip > 1)) or np.any((action_grip < 0) | (action_grip > 1)):
        raise ValueError(f"Normalized gripper values in [0,1] required: {path}")
    return np.concatenate((ep.q, grip), -1), np.concatenate((arm, action_grip), -1)


@torch.inference_mode()
def export_features(encoder, ep, domain):
    encoder.model.eval()
    encoder.model.requires_grad_(False)
    effort = predict_episode(encoder.model, ep, encoder.stats, domain)["z"].astype(np.float32)
    return finite_array(effort, (len(ep.force), 32), "encoded effort")


def convert(manifest_path, manifest, audit, episodes, identity, feature_id, domain, repo):
    from convert_user_hdf5_to_lerobot import _create_dataset, _read_video
    from lerobot.common.datasets.compute_stats import compute_stats
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    from lerobot.common.datasets.utils import serialize_dict

    root = Path(manifest[f"{domain}_root"])
    dest = Path(LEROBOT_HOME) / repo
    domain_id = ("real", "sim").index(domain)
    sources = {str(tid): audit["trajectories"][str(tid)]["sources"][domain_id]["sha256"] for tid in range(40)}
    expected = dict(encoder_sha256=manifest["encoder_sha256"], feature_id=feature_id, domain=domain,
                    layout=LAYOUT, source_h5_sha256=sources, source_video_sha256=identity["source_video_sha256"][domain],
                    actual_sim_state_verified=audit["actual_sim_state_verified"], encoder_frozen=True)
    contract_path = dest / "meta" / CONTRACT_FILE
    if dest.exists():
        if contract_path.is_file():
            saved = read_json(contract_path)
            if all(saved.get(key) == value for key, value in expected.items()):
                validate_dataset(dest, saved)
                print(f"Reuse complete feature export: {dest}", flush=True)
                return dest
        raise FileExistsError(f"Incomplete/different cache {dest}; choose a new CACHE_TAG and POLICY_DIR. No overwrite.")
    encoder = JointMotionForceEncoder(manifest_path, domain, device="cpu", threads=1)
    if encoder.model.domain_gains.tolist() != [0., 0.]:
        raise ValueError("Checkpoint weights do not match the motion-only gains declared by the manifest")
    if encoder.stats != fit_stats(episodes, TRAIN_IDS):
        raise ValueError("Encoder normalization does not match the recorded train data")
    # Validate arrays and reference targets before creating an export.
    arrays = {}
    for tid in range(40):
        path = root / f"traj_{tid}/data.h5"
        if sha256(path) != sources[str(tid)]:
            raise ValueError(f"HDF changed since encoder training: {path}")
        ep = episodes[(tid, domain_id)]
        arrays[tid] = episode_arrays(path, ep)
        period = np.diff(ep.time)
        regular = period[period <= GAP_S]
        if not len(regular) or not np.isclose(np.median(regular), .1, rtol=.1, atol=1e-4):
            raise ValueError(f"Expected approximately 10 Hz input: {path}")
        # Match the existing TAVLA converter: each domain keeps its own action/actual labels.
        # Pairing for force supervision does not imply identical policy action labels.
    dataset = _create_dataset(repo, fps=10, effort_dim=32, effort_names=tuple(LAYOUT), overwrite=False,
                              use_videos=False, image_writer_processes=0, image_writer_threads=0)
    ranges, frame_count = [], 0
    for tid in range(40):
        ep = episodes[(tid, domain_id)]
        state, action = arrays[tid]
        effort = export_features(encoder, ep, domain_id)
        directory = root / f"traj_{tid}"
        for name, digest in expected["source_video_sha256"][str(tid)].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Camera source changed during preparation: {directory / name}")
        front = _read_video(directory / "front_camera.mp4", len(state))
        wrist = _read_video(directory / "wrist_camera.mp4", len(state))
        # Episode splits make both action chunks and 10-frame effort queries respect resets.
        boundaries = np.r_[np.unique(ep.starts), len(state)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
            for t in range(int(lo), int(hi)):
                dataset.add_frame({"observation.state": state[t], "observation.effort": effort[t], "action": action[t],
                                   "observation.images.cam_high": front[t], "observation.images.cam_left_wrist": wrist[t],
                                   "observation.images.cam_right_wrist": wrist[t]})
            dataset.save_episode(task="peg-in-hole")
            ranges.append(dict(trajectory=tid, start_frame=int(lo), end_frame_exclusive=int(hi)))
        frame_count += len(state)
        print(f"Export {domain}: traj_{tid}, frames={len(state)}", flush=True)
    dataset.consolidate(run_compute_stats=False)
    dataset.meta.stats = compute_stats(dataset, batch_size=8, num_workers=0)
    write_json(dataset.root / "meta/stats.json", serialize_dict(dataset.meta.stats))
    contract = dict(expected, completed=True, total_frames=frame_count, episode_ranges=ranges,
                    info_sha256=sha256(dest / "meta/info.json"))
    write_json(contract_path, contract)
    validate_dataset(dest, contract)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_ENCODER_MANIFEST))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-tag", default="default")
    args = parser.parse_args()
    args.manifest, output = args.manifest.resolve(), args.output_dir.resolve()
    manifest, checkpoint, audit = read_encoder_manifest(args.manifest)
    episodes, current_audit = load_pairs(manifest["real_root"], manifest["sim_root"])
    if current_audit != audit:
        raise ValueError("Paired HDF sources/provenance differ from the chosen encoder")
    torch.set_num_threads(1)
    videos = video_sources(manifest)
    identity = build_identity(manifest, audit, videos, args.cache_tag)
    feature_id = stable_digest(identity)
    request = dict(encoder_manifest=str(args.manifest), encoder_sha256=manifest["encoder_sha256"],
                   feature_id=feature_id, dataset_identity=identity)
    request_path = output / "preparation_request.json"
    if output.exists() and any(output.iterdir()):
        if not request_path.is_file() or read_json(request_path) != request:
            raise ValueError("Preparation inputs/code/cache tag changed; choose a new POLICY_DIR")
    output.mkdir(parents=True, exist_ok=True)
    write_json(request_path, request)
    policy_path = output / "policy_manifest.json"
    if policy_path.is_file():
        from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
        validate_policy_manifest(policy_path, LEROBOT_HOME)
        print(f"Reuse policy preparation: {policy_path}", flush=True)
        return
    print(f"Frozen seed={manifest['selected_seed']}, gains={manifest['domains']}; "
          f"actual Sim={audit['actual_sim_state_verified']}; Real/Sim=50:50", flush=True)
    repos = repo_ids(manifest["encoder_sha256"], feature_id)
    roots = [convert(args.manifest, manifest, audit, episodes, identity, feature_id, domain, repo)
             for domain, repo in zip(("real", "sim"), repos, strict=True)]
    # This helper computes moments after action delta conversion and 32-D padding,
    # includes every frame and uses equal domain weights rather than mixing quantiles.
    from force_gap_forecast_v1.prepare_policy import shared_stats
    assets = output / "assets"
    shared_stats(repos, assets)
    policy = dict(artifact_format=POLICY_FORMAT, status=POLICY_STATUS, encoder_artifact_format=FORMAT,
                  encoder_manifest=str(args.manifest), checkpoint=str(checkpoint), encoder_sha256=manifest["encoder_sha256"],
                  encoder_status=manifest["status"], selected_seed=manifest["selected_seed"], domains=manifest["domains"],
                  encoder_frozen=True, layout=LAYOUT, config_name=CONFIG_NAME, repo_ids=repos,
                  feature_id=feature_id, dataset_identity=identity, dataset_roots=[str(root) for root in roots],
                  dataset_contract_sha256=[sha256(root / "meta" / CONTRACT_FILE) for root in roots],
                  sampling_weights=[.5, .5], norm_stats_dir=str(assets), norm_asset_id="shared",
                  norm_stats_sha256=sha256(assets / "shared/norm_stats.json"), use_quantile_norm=False,
                  effort_shape=[10, 32], action_shape=[50, 8], encoder_history_frames=32, policy_history_frames=10,
                  actions_are_absolute=True, execute_action_index=1, sample_period_s=.1,
                  actual_sim_state_verified=audit["actual_sim_state_verified"],
                  waveform_acceptance_passed=read_json(args.manifest.parent / manifest["acceptance"])["passed"])
    write_json(policy_path, policy)
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    validate_policy_manifest(policy_path, LEROBOT_HOME)
    print(f"Prepared: {policy_path}", flush=True)


if __name__ == "__main__":
    main()
