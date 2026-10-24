"""Synthetic CPU checks only; no experiment training or running-server requests."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from . import FORMAT, LAYOUT
from .common import read_manifest, sha256, write_json
from .data import Blocks, Episode, fit_stats, segment_limits
from .losses import objective
from .model import ForceForecast
from .runtime import ForceGapEncoder
from .sim_adapter import TavlaAdapter
from .train import state_digest
from .validate import validate


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        self.episodes = {}
        for i in range(3):
            n = 40
            force = rng.normal(0, 2, (n, 3)).astype(np.float32)
            force[20, 2] += 25
            q = rng.normal(0, .1, (n, 7)).astype(np.float32)
            dq = rng.normal(0, .2, (n, 7)).astype(np.float32)
            t = np.arange(n, dtype=np.float64) * .1
            for d in range(2):
                starts, ends = segment_limits(t)
                self.episodes[(i, d)] = Episode(i, force.copy() * (1 if d == 0 else .8), q.copy(), dq.copy(),
                                               t.copy(), np.full(n, "unannotated", object), starts, ends)
        self.stats = fit_stats(self.episodes, [0, 1])
        self.model = ForceForecast(self.stats)
        path = self.root / "encoder.pt"
        torch.save({"artifact_format": FORMAT, "layout": LAYOUT, "stats": self.stats,
                    "state_dict": self.model.state_dict()}, path)
        self.manifest = self.root / "manifest.json"
        write_json(self.manifest, {"artifact_format": FORMAT, "layout": LAYOUT, "checkpoint": path.name,
                                  "encoder_sha256": sha256(path), "status": "diagnostic"})

    def tearDown(self):
        self.temp.cleanup()

    def test_streaming_and_causality(self):
        result = validate(self.manifest)
        for domain in ("real", "sim"):
            self.assertLessEqual(result[domain]["batch_stream_max_abs"], 1e-5)
            self.assertLessEqual(result[domain]["train_runtime_max_abs"], 1e-5)
        # Latency is reported separately; unit correctness must not depend on CI load.

    def test_frozen_gradients_and_masks(self):
        self.model.adapt()
        before = state_digest(self.model, True)
        dataset = Blocks(self.episodes, [0, 1], self.stats)
        from torch.utils.data._utils.collate import default_collate
        batch = default_collate([dataset[20], dataset[60]])
        self.model.train()
        with torch.no_grad():
            teacher = self.model.block(batch, 0)
        output = self.model.block(batch, 1)
        loss, _ = objective(self.model, output, batch, teacher)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.sim_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.real_encoder.parameters()))
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=3e-4)
        opt.step()
        self.assertEqual(before, state_digest(self.model, True))
        batch = default_collate([dataset[0], dataset[39]])
        self.model.eval()
        out = self.model.block(batch, 1)
        first, _ = objective(self.model, out, batch)
        batch["past"][~batch["past_valid"]] += 10000
        batch["future"][~batch["future_valid"]] -= 10000
        second, _ = objective(self.model, out, batch)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_train_only_stats(self):
        original = fit_stats(self.episodes, [0, 1])
        self.episodes[(2, 0)].force[:] = 1e6
        self.assertEqual(original, fit_stats(self.episodes, [0, 1]))

    def test_boundaries_and_terminal_masks(self):
        dataset = Blocks(self.episodes, [0], self.stats)
        row = dataset[0]
        self.assertEqual(row["x"].shape, (2, 9, 32, 21))
        self.assertEqual(row["past_valid"].sum(), 1)
        self.assertFalse(dataset[39]["future_valid"][-1].any())
        ep = self.episodes[(0, 0)]
        ep.time[20:] += 1
        ep.starts, ep.ends = segment_limits(ep.time)
        gap = Blocks(self.episodes, [0], self.stats)[20]
        self.assertFalse(gap["delta_valid"][-1])
        self.assertTrue(np.all(gap["x"][0, -1] == gap["x"][0, -1, -1]))

    def test_invalid_inputs_and_sha(self):
        runtime = ForceGapEncoder(self.manifest, "sim")
        with self.assertRaises(ValueError):
            runtime.push(np.array([0, 0, np.nan]), np.zeros(7), np.zeros(7), 0)
        with self.assertRaises(ValueError):
            ForceGapEncoder(self.manifest, "unknown")
        with self.assertRaises(ValueError):
            read_manifest(self.manifest, require_ready=True)
        (self.root / "encoder.pt").write_bytes(b"wrong")
        with self.assertRaises(ValueError):
            ForceGapEncoder(self.manifest, "sim")

    def test_adapter_contract(self):
        adapter = TavlaAdapter(self.manifest, "real")
        adapter.update(np.ones(3), np.zeros(7), np.zeros(7), 0)
        image = np.zeros((8, 8, 3), np.uint8)
        obs = adapter.observation(image, image, .5)
        self.assertEqual(obs["effort"].shape, (10, 32))
        self.assertEqual(obs["state"].shape, (8,))
        from .serve import ValidatedPolicy
        class Stub:
            def infer(self, request):
                return {"actions": np.zeros((50, 8), np.float32)}
        ValidatedPolicy(Stub(), obs["encoder_sha256"]).infer(obs)
        with self.assertRaises(ValueError):
            ValidatedPolicy(Stub(), "other").infer(obs)

    def test_stage_selection_refit_and_cache(self):
        from .train import run_stage
        args = SimpleNamespace(resume=False, device="cpu", batch_size=2, lr=3e-4, max_epochs=1, patience=1)
        teacher, _, selection = run_stage(self.episodes, [0], [1], args, 42, self.root / "teacher_select")
        self.assertEqual(selection["selected_epoch"], 1)
        self.assertEqual(selection["optimizer_steps"], 3)
        student, _, summary = run_stage(self.episodes, [0], [1], args, 1042, self.root / "student_select", teacher=teacher)
        self.assertTrue(summary["completed"])
        self.assertEqual(state_digest(student, True), summary["frozen_sha256"])
        before = state_digest(student)
        cached, _, _ = run_stage(self.episodes, [0], [1], args, 1042, self.root / "student_select", teacher=teacher)
        self.assertEqual(before, state_digest(cached))
        final_teacher, _, _ = run_stage(self.episodes, [0, 1], [], args, 42, self.root / "teacher_refit", epochs=1)
        final_student, _, record = run_stage(self.episodes, [0, 1], [], args, 1042, self.root / "student_refit", teacher=final_teacher, epochs=1)
        self.assertEqual(record["selected_epoch"], 1)
        self.assertTrue((self.root / "student_refit/last.pt").is_file())

    def test_portable_export(self):
        from unittest.mock import patch
        from .export import main
        destination = self.root / "bundle"
        with patch("sys.argv", ["export", "--manifest", str(self.manifest), "--output-dir", str(destination)]):
            main()
        portable = destination / "manifest.json"
        original = ForceGapEncoder(self.manifest, "sim")
        exported = ForceGapEncoder(portable, "sim")
        ep = self.episodes[(0, 1)]
        self.assertTrue(np.array_equal(original.batch(ep.force, ep.q, ep.dq, ep.time),
                                       exported.batch(ep.force, ep.q, ep.dq, ep.time)))

    def test_baseline_fit_and_metrics(self):
        from .baselines import fit, evaluate
        for kind in ("affine", "arx"):
            baseline = fit(self.episodes, [0, 1], self.stats, kind)
            result = evaluate(baseline, self.episodes, [2], self.stats)
            self.assertTrue(np.isfinite(result["2"]["all"]["score"]))

    def test_motion_only_has_no_force_input(self):
        dataset = Blocks(self.episodes, [0], self.stats, motion_only=True)
        self.assertTrue(np.all(dataset[20]["x"][..., :6] == 0))

    def test_refuse_manual_promotion(self):
        value = json.loads(self.manifest.read_text())
        write_json(self.root / "acceptance.json", {"passed": True})
        write_json(self.root / "data_audit.json", {"actual_sim_state_verified": False})
        value.update(status="policy_train_ready", acceptance="acceptance.json", audit="data_audit.json",
                     acceptance_sha256=sha256(self.root / "acceptance.json"), audit_sha256=sha256(self.root / "data_audit.json"))
        write_json(self.manifest, value)
        with self.assertRaises(ValueError):
            read_manifest(self.manifest, require_ready=True)

    def test_capture_observed_state_and_reference_actions(self):
        import h5py
        import cv2
        from .capture import PairedReplayRecorder
        real = self.root / "real.h5"
        with h5py.File(real, "w") as h:
            h["action/actual/arm"] = np.ones((2, 7), np.float32)
            h["action/actual/gripper"] = np.full((2, 1), .5, np.float32)
            h["timestamps"] = np.array([0., .1])
        folder = self.root / "capture"
        recorder = PairedReplayRecorder(folder, real, "calibrated_link7")
        rgb = np.zeros((480, 640, 3), np.uint8)
        with self.assertRaises(ValueError):
            recorder.append(0, np.ones(6), np.zeros(7), np.zeros(7), 0, .5, rgb, np.zeros((2, 2, 3), np.uint8))
        for i in range(2):
            recorder.append(i, np.ones(6), np.zeros(7), np.zeros(7), i * .1, .5, rgb, rgb)
        recorder.close()
        with h5py.File(folder / "data.h5") as h:
            self.assertEqual(h.attrs["state_source"], "sim_observed")
            self.assertEqual(h.attrs["paired_real_sha256"], sha256(real))
            self.assertTrue(np.all(h["obs/state/joint_pos"][:] == 0))
            self.assertTrue(np.all(h["action/actual/arm"][:] == 1))
            self.assertTrue(np.all(h["obs/state/ee_force_base"][:] == -1))
        for name in ("front_camera", "wrist_camera"):
            video = cv2.VideoCapture(str(folder / f"{name}.mp4"))
            frames = 0
            while video.read()[0]:
                frames += 1
            video.release()
            self.assertEqual(frames, 2)
        with self.assertRaises(RuntimeError):
            recorder.close()

    def test_new_policy_config_shared_norm_and_legacy_unchanged(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}):
            from openpi.training import config
            from openpi.shared import normalize
            from openpi import transforms
        sha = json.loads(self.manifest.read_text())["encoder_sha256"]
        repos = [f"local/tavla_force_gap_forecast_{sha[:12]}_{d}" for d in ("real", "sim")]
        cache = self.root / "lerobot"
        contracts = []
        for repo in repos:
            p = cache / repo / "meta/force_gap_contract.json"
            write_json(p, {"encoder_sha256": sha, "layout": LAYOUT})
            contracts.append(sha256(p))
        stats_root = self.root / "assets"
        stats = {key: normalize.NormStats(mean=np.arange(32, dtype=float), std=np.full(32, 2.))
                 for key in ("state", "actions", "effort")}
        normalize.save(stats_root / "shared", stats)
        name = "pi0_lora_user_single_arm_force_gap_forecast_cotrain_50_50"
        policy_manifest = self.root / "policy_manifest.json"
        write_json(policy_manifest, {"artifact_format": FORMAT, "layout": LAYOUT, "status": "policy_train_ready",
                                    "config_name": name, "encoder_sha256": sha, "repo_ids": repos,
                                    "sampling_weights": [.5, .5], "norm_stats_dir": str(stats_root), "norm_asset_id": "shared",
                                    "norm_stats_sha256": sha256(stats_root / "shared/norm_stats.json"),
                                    "dataset_contract_sha256": contracts, "use_quantile_norm": False})
        legacy = config.get_config("pi0_lora_user_single_arm_force_align_latent_cotrain_50_50")
        with patch.dict(os.environ, {"FORCE_GAP_FORECAST_POLICY_MANIFEST": str(policy_manifest)}), patch.object(config, "LEROBOT_HOME", cache):
            cfg = config.get_config(name)
            dc = cfg.data.create(cfg.assets_dirs, cfg.model)
            self.assertFalse(dc.use_quantile_norm)
            self.assertFalse(cfg.data.padding_stat)
            self.assertEqual(cfg.batch_size, 8)
            self.assertEqual(cfg.num_train_steps, 30000)
            self.assertEqual(dc.repo_id, repos)
            value = np.ones((10, 32), np.float32)
            transform = transforms.Normalize(dc.norm_stats, use_quantiles=False)
            real = transform({"effort": value.copy()})["effort"]
            sim = transform({"effort": value.copy()})["effort"]
            self.assertTrue(np.array_equal(real, sim))
        self.assertEqual(legacy.data.repo_id, ["local/tavla_single_arm_force_align_latent", "local/tavla_single_arm_force_align_latent_sim"])
        self.assertTrue(legacy.data.padding_stat)


if __name__ == "__main__":
    unittest.main()
