"""CPU contract checks for preservation, causality, masks and portable inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from force_gap_forecast_v1.common import sha256, write_json
from force_gap_forecast_v1.data import Blocks, Episode, fit_stats, segment_limits
from force_gap_forecast_v1.losses import objective
from force_gap_forecast_v1.metrics import predict_episode as predict_motion
from force_gap_forecast_v1.model import ForceForecast
from . import FORMAT, LAYOUT
from .metrics import predict_episode
from .model import MotionForceForecast
from .runtime import MotionForceEncoder
from .train import fit_stage, frozen_digest, load_motion


class ResidualContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        self.episodes = {}
        for tid in range(3):
            n = 40
            force = rng.normal(0, 2, (n, 3)).astype(np.float32)
            force[18, 2] += 12
            q = rng.normal(0, .2, (n, 7)).astype(np.float32)
            dq = rng.normal(0, .1, (n, 7)).astype(np.float32)
            time = np.arange(n, dtype=np.float64) * .1
            time[23:] += 1.
            for domain in (0, 1):
                starts, ends = segment_limits(time)
                self.episodes[(tid, domain)] = Episode(tid, force.copy() * (1 if domain == 0 else .7),
                    q.copy(), dq.copy(), time.copy(), np.full(n, "unannotated", object), starts, ends)
        self.stats = fit_stats(self.episodes, [0])
        self.motion = ForceForecast(self.stats).eval()
        self.model = MotionForceForecast(self.stats, self.motion.state_dict()).eval()
        self.base = self.root / "motion.pt"
        torch.save({"stats": self.stats, "state_dict": self.motion.state_dict(),
                    "stage": {"motion_only": True, "train_ids": [0]}}, self.base)

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self, directory, model=None):
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "encoder.pt"
        torch.save({"artifact_format": FORMAT, "layout": LAYOUT, "stats": self.stats,
                    "state_dict": (model or self.model).state_dict()}, checkpoint)
        path = directory / "manifest.json"
        write_json(path, {"artifact_format": FORMAT, "layout": LAYOUT, "checkpoint": checkpoint.name,
                          "encoder_sha256": sha256(checkpoint), "status": "diagnostic"})
        return path

    def test_initial_sim_predictions_equal_motion_only(self):
        ep = self.episodes[(0, 1)]
        actual = predict_episode(self.model, ep, self.stats, 1)
        reference = predict_motion(self.motion, ep, self.stats, 1, motion_only=True)
        for key in ("h", "history", "current", "future", "z"):
            np.testing.assert_allclose(actual[key], reference[key], atol=1e-6, rtol=1e-6)
        batch = default_collate([Blocks(self.episodes, [0], self.stats)[18]])
        motion_batch = default_collate([Blocks(self.episodes, [0], self.stats, motion_only=True)[18]])
        with torch.no_grad():
            actual_block = self.model.block(batch, 1)
            base_block = self.motion.block(motion_batch, 1, motion_only=True)
        for key in ("h", "past", "future", "z"):
            torch.testing.assert_close(actual_block[key], base_block[key])

    def test_optimizer_updates_force_branch_and_keeps_motion_frozen(self):
        batch = default_collate([Blocks(self.episodes, [0], self.stats)[18],
                                 Blocks(self.episodes, [0], self.stats)[30]])
        before = frozen_digest(self.model)
        self.model.train()
        self.assertFalse(self.model.motion.training)
        self.assertTrue(self.model.force_residual.training)
        optimizer = torch.optim.AdamW(self.model.force_residual.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            output = self.model.block(batch, 1)
            loss, _ = objective(self.model, output, batch, alignment=False)
            loss.backward()
            optimizer.step()
        self.assertEqual(before, frozen_digest(self.model))
        self.assertTrue(all(p.grad is None for p in self.model.motion.parameters()))
        self.assertGreater(self.model.force_residual.gru.weight_ih_l0.grad.abs().sum().item(), 0)
        ep = self.episodes[(0, 1)]
        changed = copy.deepcopy(ep)
        changed.force *= -2
        first = predict_episode(self.model, ep, self.stats, 1)
        second = predict_episode(self.model, changed, self.stats, 1)
        self.assertGreater(np.max(np.abs(first["future"] - second["future"])), 1e-6)
        np.testing.assert_allclose(first["h"], second["h"], atol=1e-6)

    def test_gain_zero_restores_baseline_even_with_nonzero_residual(self):
        with torch.no_grad():
            self.model.force_residual.residual[-1].weight.normal_(0, .1)
            self.model.force_residual.residual[-1].bias.fill_(.5)
            self.model.residual_gain.zero_()
        ep = self.episodes[(0, 1)]
        actual = predict_episode(self.model, ep, self.stats, 1)
        expected = predict_motion(self.motion, ep, self.stats, 1, motion_only=True)
        for key in ("current", "history", "future", "z"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, rtol=1e-6)

    def test_predictions_do_not_read_future_inputs(self):
        with torch.no_grad():
            self.model.force_residual.residual[-1].weight.normal_(0, .1)
        ep = self.episodes[(0, 1)]
        changed = copy.deepcopy(ep)
        changed.force[11:] += 100
        changed.q[11:] -= 100
        changed.dq[11:] += 100
        original = predict_episode(self.model, ep, self.stats, 1)
        modified = predict_episode(self.model, changed, self.stats, 1)
        for key in ("current", "history", "future", "z"):
            np.testing.assert_allclose(original[key][:11], modified[key][:11], atol=1e-6, rtol=1e-6)

    def test_streaming_batch_and_training_windows_agree_after_gap(self):
        with torch.no_grad():
            self.model.force_residual.residual[-1].weight.normal_(0, .03)
            self.model.residual_gain.fill_(.5)
        manifest = self.save_manifest(self.root / "exported")
        self.base.unlink()  # Inference must not require the source motion checkpoint.
        for domain, name in enumerate(("real", "sim")):
            ep = self.episodes[(0, domain)]
            expected = predict_episode(self.model, ep, self.stats, domain)
            runtime = MotionForceEncoder(manifest, name)
            zs, futures = [], []
            for force, q, dq, timestamp in zip(ep.force, ep.q, ep.dq, ep.time):
                zs.append(runtime.push(force, q, dq, timestamp))
                futures.append(runtime.predict_future())
            np.testing.assert_allclose(zs, expected["z"], atol=3e-5, rtol=1e-5)
            np.testing.assert_allclose(futures, expected["future"], atol=3e-5, rtol=1e-5)
            self.assertEqual(runtime.effort_history().shape, (10, 32))
            self.assertTrue(np.array_equal(np.asarray(zs)[23, 3:6], np.zeros(3)))
            dataset = Blocks(self.episodes, [0], self.stats)
            for anchor in (18, 23, 30):
                with torch.no_grad():
                    output = self.model.block(default_collate([dataset[anchor]]), domain)
                np.testing.assert_allclose(output["future"][0, -1], expected["future"][anchor], atol=3e-5, rtol=1e-5)
                np.testing.assert_allclose(output["z"][0, -1], expected["z"][anchor], atol=3e-5, rtol=1e-5)

    def test_invalid_targets_outside_segments_do_not_change_loss(self):
        dataset = Blocks(self.episodes, [0], self.stats)
        batch = default_collate([dataset[0], dataset[22], dataset[23], dataset[39]])
        self.assertFalse(batch["future_valid"][1, -1].any())
        self.assertFalse(batch["delta_valid"][2, -1])
        first, _ = objective(self.model, self.model.block(batch, 1), batch, alignment=False)
        batch["past"][~batch["past_valid"]] += 10000
        batch["future"][~batch["future_valid"]] -= 10000
        second, _ = objective(self.model, self.model.block(batch, 1), batch, alignment=False)
        torch.testing.assert_close(first, second)

    def test_wrong_fold_normalization_or_motion_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            load_motion(self.base, self.episodes, [0, 1])
        self.episodes[(0, 1)].force += 100
        with self.assertRaises(ValueError):
            load_motion(self.base, self.episodes, [0])

    def test_fit_refit_and_cache(self):
        args = SimpleNamespace(device="cpu", resume=False, max_epochs=2, patience=2,
            batch_size=2, lr=1e-3, real_loss_weight=.25, residual_l2=.001, gains=[0., .5, 1.])
        selected, _, summary = fit_stage(self.base, self.episodes, [0], [1], args, 2042, self.root / "select")
        self.assertTrue(summary["completed"])
        self.assertIn(summary["selected_gain"], args.gains)
        self.assertEqual(summary["optimizer_steps"], 6)
        self.assertEqual(frozen_digest(selected), summary["frozen_sha256"])
        args.resume = True
        restored, _, cached = fit_stage(self.base, self.episodes, [0], [1], args, 2042, self.root / "select")
        self.assertEqual(summary, cached)
        for key, value in selected.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        args.resume = False
        _, _, refit = fit_stage(self.base, self.episodes, [0], [], args, 2042,
                                self.root / "refit", epochs=1, gain=.5)
        self.assertEqual(refit["optimizer_steps"], 3)
        self.assertEqual(refit["selected_gain"], .5)
        inactive, _, bypass = fit_stage(self.base, self.episodes, [0], [], args, 2042,
                                        self.root / "bypass", epochs=0, gain=0.)
        self.assertEqual(bypass["optimizer_steps"], 0)
        self.assertEqual(inactive.residual_gain.item(), 0.)

    def test_manifest_integrity_and_stream_input_validation(self):
        manifest = self.save_manifest(self.root / "exported")
        runtime = MotionForceEncoder(manifest)
        with self.assertRaises(ValueError):
            runtime.push(np.array([0, 0, np.nan]), np.zeros(7), np.zeros(7), 0)
        runtime.push(np.zeros(3), np.zeros(7), np.zeros(7), 0)
        with self.assertRaises(ValueError):
            runtime.push(np.zeros(3), np.zeros(7), np.zeros(7), 0)
        (manifest.parent / "encoder.pt").write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            MotionForceEncoder(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
