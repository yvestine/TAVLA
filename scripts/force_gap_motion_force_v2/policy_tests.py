"""CPU checks of feature parity, isolated cache contracts and the four-GPU command."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from force_gap_forecast_v1.common import write_json
from force_gap_forecast_v1.data import Episode, fit_stats, segment_limits
from . import FORMAT, LAYOUT
from .model import JointMotionForceForecast
from .policy_contract import (CONFIG_NAME, CONTRACT_FILE, POLICY_FORMAT, POLICY_STATUS, read_encoder_manifest,
                              repo_ids, sha256, stable_digest, validate_policy_manifest)
from .policy_prepare import export_features
from .policy_train import command
from .runtime import JointMotionForceEncoder


class MotionOnlyPolicyContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        n = 18
        time = np.arange(n, dtype=np.float64) * .1
        time[9:] += 1
        starts, ends = segment_limits(time)
        q = rng.normal(0, .1, (n, 7)).astype(np.float32)
        dq = rng.normal(0, .1, (n, 7)).astype(np.float32)
        force = rng.normal(0, 2, (n, 3)).astype(np.float32)
        self.episodes = {(0, domain): Episode(0, force.copy() * (1. if domain == 0 else .7),
                         q.copy(), dq.copy(), time.copy(), np.full(n, "unannotated", object), starts.copy(), ends.copy())
                         for domain in (0, 1)}
        self.stats = fit_stats(self.episodes, [0])
        model = JointMotionForceForecast(self.stats).eval()
        model.domain_gains.zero_()
        checkpoint = self.root / "encoder.pt"
        torch.save(dict(artifact_format=FORMAT, layout=LAYOUT, stats=self.stats, state_dict=model.state_dict(),
                        use_force=True), checkpoint)
        write_json(self.root / "audit.json", dict(actual_sim_state_verified=False))
        write_json(self.root / "acceptance.json", dict(passed=False))
        self.encoder_manifest = self.root / "encoder_manifest.json"
        encoder = dict(artifact_format=FORMAT, layout=LAYOUT, checkpoint=checkpoint.name,
                       encoder_sha256=sha256(checkpoint), domains={str(d): dict(epoch=0, gain=0.) for d in (0, 1)},
                       selected_seed=44, status="diagnostic", history=32, future=5, policy_history=10,
                       actual_sim_state_verified=False, force_contract="aligned_base_xyz_n_v1",
                       real_root=str(self.root / "real"), sim_root=str(self.root / "sim"), audit="audit.json",
                       audit_sha256=sha256(self.root / "audit.json"), acceptance="acceptance.json",
                       acceptance_sha256=sha256(self.root / "acceptance.json"))
        write_json(self.encoder_manifest, encoder)
        identity = dict(encoder_sha256=encoder["encoder_sha256"], cache_tag="synthetic")
        feature_id = stable_digest(identity)
        repos = repo_ids(encoder["encoder_sha256"], feature_id)
        self.cache = self.root / "cache"
        roots, digests = [], []
        for domain, repo in zip(("real", "sim"), repos, strict=True):
            root = self.cache / repo
            features = {key: dict(shape=shape, dtype="float32") for key, shape in
                        (("observation.state", [8]), ("observation.effort", [32]), ("action", [8]))}
            for key in ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"):
                features[key] = dict(dtype="image", shape=[3, 480, 640])
            write_json(root / "meta/info.json", dict(features=features, fps=10, total_frames=n, total_episodes=2))
            contract = dict(encoder_sha256=encoder["encoder_sha256"], feature_id=feature_id, domain=domain, layout=LAYOUT,
                            completed=True, total_frames=n, episode_ranges=[dict(start_frame=0, end_frame_exclusive=9),
                            dict(start_frame=9, end_frame_exclusive=n)], info_sha256=sha256(root / "meta/info.json"))
            write_json(root / "meta" / CONTRACT_FILE, contract)
            roots.append(str(root))
            digests.append(sha256(root / "meta" / CONTRACT_FILE))
        stats = self.root / "assets/shared/norm_stats.json"
        write_json(stats, {"norm_stats": {key: dict(mean=[0.] * 32, std=[1.] * 32)
                                        for key in ("state", "actions", "effort")}})
        self.policy = dict(artifact_format=POLICY_FORMAT, status=POLICY_STATUS, config_name=CONFIG_NAME,
                           encoder_artifact_format=FORMAT, encoder_manifest=str(self.encoder_manifest),
                           encoder_sha256=encoder["encoder_sha256"], selected_seed=44, domains=encoder["domains"],
                           actual_sim_state_verified=False, layout=LAYOUT, effort_shape=[10, 32], action_shape=[50, 8],
                           sampling_weights=[.5, .5], use_quantile_norm=False, encoder_frozen=True,
                           encoder_history_frames=32, policy_history_frames=10, actions_are_absolute=True,
                           feature_id=feature_id, dataset_identity=identity, repo_ids=repos, dataset_roots=roots,
                           dataset_contract_sha256=digests, norm_stats_dir=str(self.root / "assets"),
                           norm_asset_id="shared", norm_stats_sha256=sha256(stats))
        self.policy_path = self.root / "policy_manifest.json"
        write_json(self.policy_path, self.policy)

    def tearDown(self):
        self.temp.cleanup()

    def test_exported_rows_match_online_history_and_gap_resets(self):
        for domain, name in enumerate(("real", "sim")):
            ep = self.episodes[(0, domain)]
            offline = JointMotionForceEncoder(self.encoder_manifest, name)
            effort = export_features(offline, ep, domain)
            self.assertTrue(all(not p.requires_grad for p in offline.model.parameters()))
            online = JointMotionForceEncoder(self.encoder_manifest, name)
            for t in range(len(ep.force)):
                z = online.push(ep.force[t], ep.q[t], ep.dq[t], ep.time[t])
                np.testing.assert_allclose(effort[t], z, atol=3e-5, rtol=1e-5)
                indices = np.maximum(np.arange(t - 9, t + 1), ep.starts[t])
                np.testing.assert_allclose(effort[indices], online.effort_history(), atol=3e-5, rtol=1e-5)
            np.testing.assert_array_equal(effort[9, 3:6], np.zeros(3))
            if domain == 0:
                np.testing.assert_array_equal(effort[:, :3], ep.force)

    def test_sim_motion_only_export_is_independent_of_sim_force(self):
        ep = self.episodes[(0, 1)]
        changed = copy.deepcopy(ep)
        changed.force *= -3
        encoder = JointMotionForceEncoder(self.encoder_manifest, "sim")
        np.testing.assert_allclose(export_features(encoder, ep, 1), export_features(encoder, changed, 1), atol=1e-6)

    def test_experimental_contract_retains_the_original_provenance(self):
        value = validate_policy_manifest(self.policy_path, self.cache)
        self.assertEqual(value["status"], "offline_replay_experiment")
        self.assertFalse(value["actual_sim_state_verified"])
        encoder, _, _ = read_encoder_manifest(self.encoder_manifest)
        self.assertEqual(encoder["status"], "diagnostic")
        with self.assertRaises(ValueError):
            validate_policy_manifest(self.policy_path, self.root / "other_cache")

    def test_contract_rejects_mixed_or_altered_features_and_normalization(self):
        for field, value in (("encoder_sha256", "a" * 64), ("effort_shape", [10, 6]),
                             ("sampling_weights", [.9, .1]), ("actual_sim_state_verified", True)):
            modified = dict(self.policy, **{field: value})
            write_json(self.policy_path, modified)
            with self.assertRaises(ValueError, msg=field):
                validate_policy_manifest(self.policy_path, self.cache)
        write_json(self.policy_path, self.policy)
        info = Path(self.policy["dataset_roots"][0]) / "meta/info.json"
        with info.open("a") as handle:
            handle.write(" ")
        with self.assertRaises(ValueError):
            validate_policy_manifest(self.policy_path, self.cache)

    def test_changed_shared_stats_are_rejected(self):
        stats = self.root / "assets/shared/norm_stats.json"
        with stats.open("a") as handle:
            handle.write(" ")
        with self.assertRaises(ValueError):
            validate_policy_manifest(self.policy_path, self.cache)

    def test_train_command_uses_four_gpus_global_batch_eight_without_overwrite(self):
        argv = command(self.root, "python", "synthetic_run")
        for key, expected in (("--batch-size", "8"), ("--fsdp-devices", "4"), ("--num-train-steps", "30000")):
            self.assertEqual(argv[argv.index(key) + 1], expected)
        self.assertNotIn("--overwrite", argv)
        self.assertNotIn("--resume", argv)
        self.assertIn("--resume", command(self.root, "python", "synthetic_run", resume=True))

    def test_existing_converter_keeps_each_domains_native_action_labels(self):
        from insertion_alignment_encoder.convert_force_alignment_latent import _load_episode
        for domain, name in enumerate(("real", "sim")):
            ep = self.episodes[(0, domain)]
            path = self.root / name / "traj_0/data.h5"
            path.parent.mkdir(parents=True)
            arm = np.full((len(ep.force), 7), domain + .25, np.float32)
            grip = np.full((len(ep.force), 1), .5, np.float32)
            with h5py.File(path, "w") as h:
                h["obs/state/joint_pos"] = ep.q
                h["obs/state/joint_vel"] = ep.dq
                h["obs/state/gripper_pos"] = grip
                h["obs/state/ee_force_base"] = ep.force
                h["action/actual/arm"] = arm
                h["action/actual/gripper"] = grip
                h["timestamps"] = ep.time
            encoder = JointMotionForceEncoder(self.encoder_manifest, name)
            images = np.zeros((len(ep.force), 3, 2, 2), np.uint8)
            with patch("insertion_alignment_encoder.convert_force_alignment_latent._read_video", return_value=images):
                state, actions, effort, _, _ = _load_episode(path, "obs/state/ee_force_base", encoder,
                                                            "motion_only_v2", name)
            np.testing.assert_array_equal(actions[:, :7], arm)
            np.testing.assert_array_equal(state[:, :7], ep.q)
            np.testing.assert_allclose(effort, export_features(encoder, ep, domain), atol=3e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
