"""Contracts for an explicitly experimental TAVLA run using frozen motion-only features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import FORMAT, LAYOUT

CONFIG_NAME = "pi0_lora_user_single_arm_motion_only_v2_cotrain_50_50"
POLICY_FORMAT = "tavla_motion_only_v2_policy_experiment_v1"
POLICY_STATUS = "offline_replay_experiment"
CONTRACT_FILE = "motion_only_v2_contract.json"
DEFAULT_ENCODER_MANIFEST = "eval_outputs/force_gap_motion_force_v2/20261012_215515_838437/manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_encoder_manifest(path):
    path = Path(path).resolve()
    value = read_json(path)
    if (value.get("artifact_format") != FORMAT or value.get("layout") != LAYOUT or
            value.get("history") != 32 or value.get("future") != 5 or value.get("policy_history") != 10 or
            value.get("force_contract") != "aligned_base_xyz_n_v1"):
        raise ValueError("Expected the motion-force-v2 encoder and its 32-D feature contract")
    if set(value.get("domains", {})) != {"0", "1"} or any(
            value["domains"][str(domain)].get("gain") != 0. for domain in (0, 1)):
        raise ValueError("This TAVLA recipe selects the motion-only baseline: both domain gains must be zero")
    checkpoint = path.parent / value["checkpoint"]
    if sha256(checkpoint) != value["encoder_sha256"]:
        raise ValueError("Frozen encoder SHA mismatch")
    for key in ("audit", "acceptance"):
        if sha256(path.parent / value[key]) != value[f"{key}_sha256"]:
            raise ValueError(f"Encoder {key} evidence changed")
    audit = read_json(path.parent / value["audit"])
    if audit.get("actual_sim_state_verified") != value.get("actual_sim_state_verified"):
        raise ValueError("Encoder and data provenance flags differ")
    return value, checkpoint, audit


def repo_ids(encoder_sha, feature_id):
    return [f"local/tavla_motion_only_v2_{encoder_sha[:12]}_{feature_id[:12]}_{domain}"
            for domain in ("real", "sim")]


def validate_dataset(root, contract):
    root = Path(root)
    if contract.get("completed") is not True or contract.get("layout") != LAYOUT:
        raise ValueError(f"Incomplete or incompatible feature export: {root}")
    info_path = root / "meta/info.json"
    if sha256(info_path) != contract["info_sha256"]:
        raise ValueError(f"Dataset schema/episode counts changed: {root}")
    info = read_json(info_path)
    expected = {"observation.state": [8], "observation.effort": [32], "action": [8]}
    for key, shape in expected.items():
        if info["features"].get(key, {}).get("shape") != shape:
            raise ValueError(f"Invalid {key} shape: {root}")
    for key in ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"):
        if info["features"].get(key, {}).get("dtype") != "image":
            raise ValueError(f"Expected embedded RGB images: {root}, {key}")
    if (info["fps"] != 10 or info["total_frames"] != contract["total_frames"] or
            info["total_episodes"] != len(contract["episode_ranges"])):
        raise ValueError(f"Dataset frame/episode contract mismatch: {root}")


def validate_policy_manifest(path, lerobot_home=None):
    value = read_json(path)
    if (value.get("artifact_format") != POLICY_FORMAT or value.get("status") != POLICY_STATUS or
            value.get("config_name") != CONFIG_NAME or value.get("encoder_artifact_format") != FORMAT or
            value.get("layout") != LAYOUT or value.get("effort_shape") != [10, 32] or
            value.get("action_shape") != [50, 8] or value.get("sampling_weights") != [.5, .5] or
            value.get("use_quantile_norm") is not False or value.get("encoder_frozen") is not True or
            value.get("encoder_history_frames") != 32 or value.get("policy_history_frames") != 10 or
            value.get("actions_are_absolute") is not True or value.get("norm_asset_id") != "shared"):
        raise ValueError("Invalid experimental motion-only TAVLA contract")
    encoder, _, _ = read_encoder_manifest(value["encoder_manifest"])
    if (encoder["encoder_sha256"] != value["encoder_sha256"] or
            encoder["selected_seed"] != value["selected_seed"] or
            encoder["domains"] != value["domains"] or
            encoder.get("actual_sim_state_verified") != value["actual_sim_state_verified"]):
        raise ValueError("Policy points to a different encoder/provenance")
    if stable_digest(value["dataset_identity"]) != value["feature_id"]:
        raise ValueError("Dataset identity changed")
    if (value["dataset_identity"].get("encoder_sha256") != value["encoder_sha256"] or
            value["repo_ids"] != repo_ids(value["encoder_sha256"], value["feature_id"])):
        raise ValueError("Feature-cache identity does not match the selected encoder")
    if len(value["dataset_roots"]) != 2 or len(value["dataset_contract_sha256"]) != 2:
        raise ValueError("Two Real/Sim exports are required")
    for domain, repo, root, digest in zip(("real", "sim"), value["repo_ids"], value["dataset_roots"],
                                         value["dataset_contract_sha256"], strict=True):
        root = Path(root).resolve()
        if lerobot_home is not None and root != (Path(lerobot_home) / repo).resolve():
            raise ValueError("LEROBOT_HOME differs from the prepared feature-cache location")
        path = root / "meta" / CONTRACT_FILE
        if sha256(path) != digest:
            raise ValueError("Feature export contract changed")
        contract = read_json(path)
        if (contract.get("domain") != domain or contract.get("encoder_sha256") != value["encoder_sha256"] or
                contract.get("feature_id") != value["feature_id"]):
            raise ValueError("Feature export encoder/domain mismatch")
        validate_dataset(root, contract)
    stats = Path(value["norm_stats_dir"]) / value["norm_asset_id"] / "norm_stats.json"
    if sha256(stats) != value["norm_stats_sha256"]:
        raise ValueError("Shared policy normalization changed")
    return value
