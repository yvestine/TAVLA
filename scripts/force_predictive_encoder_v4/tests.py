"""Numerical, causal, freezing, checkpoint and streaming checks on synthetic CPU data."""

# ruff: noqa: PT009, PT027 -- These tests intentionally run with standard-library unittest.

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from force_gap_forecast_v1.common import sha256
from force_gap_forecast_v1.common import write_json
from force_gap_forecast_v1.data import Episode
from force_gap_forecast_v1.data import segment_limits
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from . import FORMAT
from . import FUTURE
from . import LAYOUT
from . import baselines
from .data import Blocks
from .data import episode_inputs
from .data import make_stats
from .data import targets
from .losses import objective
from .metrics import evaluate
from .metrics import predict_episode
from .metrics import score_prediction
from .model import PredictiveForce
from .runtime import ForcePredictiveEncoder
from .train import digest
from .train import load_model
from .train import run_stage
from .train import save_model


def episodes(count=4, n=20, *, gap=True):
    result = {}
    for tid in range(count):
        t = np.arange(n, dtype=np.float64) * 0.1
        if gap:
            t[n // 2 :] += 1.0
        force = np.column_stack((np.sin(t * 3), np.cos(t * 2), np.sin(t * 5))) + tid * 0.1
        force[n // 3, 2] += 2
        starts, ends = segment_limits(t)
        for domain in (0, 1):
            result[(tid, domain)] = Episode(
                tid,
                (force * (1 if domain == 0 else 0.6)).astype(np.float32),
                np.zeros((n, 7), np.float32),
                np.zeros((n, 7), np.float32),
                t.copy(),
                np.full(n, "contact", object),
                starts.copy(),
                ends.copy(),
            )
    return result


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episodes = episodes()
        self.stats = make_stats(self.episodes, [0, 1])
        self.arrays = {
            (tid, domain): torch.from_numpy(episode_inputs(ep, self.stats, ("real", "sim")[domain]))
            for (tid, domain), ep in self.episodes.items()
        }
        self.dataset = Blocks(self.episodes, [0, 1], self.arrays, self.stats)
        self.model = PredictiveForce(self.stats)

    def tearDown(self):
        self.temp.cleanup()

    def test_loss_includes_current_to_first_future_transition(self):
        target = torch.zeros(1, 1, FUTURE + 1, 3)
        pred = target.clone().requires_grad_()
        pred.data[..., 0, :] = 1
        scales = SimpleNamespace(target_scale=torch.ones(3), increment_scale=torch.ones(3))
        batch = {"target": target, "mask": torch.ones(1, 1, FUTURE + 1, dtype=torch.bool)}
        loss, terms = objective(scales, {"force": pred}, batch)
        self.assertEqual(terms["future"], 0)
        self.assertGreater(terms["current"], 0)
        self.assertGreater(terms["delta"], 0)
        self.assertEqual(float(torch.diff(pred[..., 1:, :], dim=-2).square().sum()), 0)
        loss.backward()
        self.assertGreater(float(pred.grad[..., 1, :].abs().sum()), 0)

    def test_increment_scale_is_independent_of_force_scale(self):
        target = torch.zeros(1, 1, FUTURE + 1, 3)
        pred = target.clone()
        pred[..., 1:, :] = 0.2
        batch = {"target": target, "mask": torch.ones(1, 1, FUTURE + 1, dtype=torch.bool)}
        a = SimpleNamespace(target_scale=torch.ones(3), increment_scale=torch.ones(3))
        b = SimpleNamespace(target_scale=torch.ones(3), increment_scale=torch.full((3,), 0.5))
        _, ta = objective(a, {"force": pred}, batch)
        _, tb = objective(b, {"force": pred}, batch)
        self.assertAlmostEqual(tb["delta"], 4 * ta["delta"], places=6)
        self.assertEqual(ta["future"], tb["future"])

    def test_empty_and_partial_masks_have_zero_invalid_gradients(self):
        pred = torch.ones(1, 2, FUTURE + 1, 3, requires_grad=True)
        batch = {"target": torch.zeros_like(pred), "mask": torch.zeros(1, 2, FUTURE + 1, dtype=torch.bool)}
        loss, _ = objective(self.model, {"force": pred}, batch)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(pred.grad.abs().sum()), 0)
        pred = torch.zeros(1, 2, FUTURE + 1, 3, requires_grad=True)
        pred.data[..., -1, :] = 1000
        batch["mask"].fill_(1)
        batch["mask"][..., -1] = False
        loss, _ = objective(self.model, {"force": pred}, batch)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(pred.grad.abs().sum()), 0)

    def test_statistics_use_train_only_and_exclude_reset_jumps(self):
        changed = copy.deepcopy(self.episodes)
        for domain in (0, 1):
            changed[(2, domain)].force *= 1000
        self.assertEqual(make_stats(changed, [0, 1]), self.stats)
        differences = []
        for tid in (0, 1):
            ep = self.episodes[(tid, 0)]
            differences.append(np.diff(ep.force, axis=0)[ep.starts[1:] == ep.starts[:-1]])
        expected = np.maximum(np.concatenate(differences).std(0), 0.05)
        np.testing.assert_allclose(self.stats["target"]["increment_scale_n"], expected)

    def test_force_inputs_are_causal_and_ignore_motion(self):
        changed = copy.deepcopy(self.episodes[(0, 1)])
        changed.force[6:] += 100
        changed.q += 200
        changed.dq -= 500
        after = episode_inputs(changed, self.stats, "sim")
        np.testing.assert_array_equal(after[:6], self.arrays[(0, 1)].numpy()[:6])
        changed.force = self.episodes[(0, 1)].force.copy()
        np.testing.assert_array_equal(episode_inputs(changed, self.stats, "sim"), self.arrays[(0, 1)])

    def test_labels_are_real_and_masks_exclude_padding_and_gaps(self):
        ep = self.episodes[(0, 0)]
        at_reset = self.dataset[len(ep.force) // 2]
        self.assertEqual(int(at_reset["mask"][:, 0].sum()), 1)
        self.assertTrue(bool(at_reset["mask"][-1].all()))
        before_reset = self.dataset[len(ep.force) // 2 - 1]
        self.assertFalse(bool(before_reset["mask"][-1, 1:].any()))
        np.testing.assert_array_equal(at_reset["target"][-1, 0], ep.force[len(ep.force) // 2])
        self.assertFalse(np.array_equal(at_reset["target"][-1, 0], self.episodes[(0, 1)].force[len(ep.force) // 2]))

    def test_frozen_decoder_passes_gradient_to_student_only(self):
        self.model.initialize_student()
        self.model.configure("student")
        frozen_before = digest(self.model, frozen_only=True)
        student_before = self.model.encoders[1].trend.out.weight.detach().clone()
        batch = default_collate([self.dataset[6], self.dataset[7]])
        optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=0.01)
        loss, _ = objective(self.model, self.model.block(batch["x"], 1), batch)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.forecaster.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.encoders[0].parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.encoders[1].parameters()), 0)
        optimizer.step()
        self.assertEqual(digest(self.model, frozen_only=True), frozen_before)
        self.assertFalse(torch.equal(student_before, self.model.encoders[1].trend.out.weight))

    def test_metrics_include_first_increment_and_ignore_last_truncation(self):
        truth, _ = targets(self.episodes, 0, np.arange(len(self.episodes[(0, 0)].force)))
        pred = truth.copy()
        pred[:, 0] += 1
        ep = self.episodes[(0, 0)]
        row = score_prediction(pred, ep, self.episodes[(0, 1)], 0.1)
        self.assertEqual(row["forecast_score_n"], 0)
        self.assertAlmostEqual(row["first_increment_rmse_n"], 1, places=6)
        self.assertGreater(row["increment_rmse_n"], 0)
        valid = evaluate(self.model, self.episodes, [2], self.arrays)
        self.assertIn("effective_rank", valid["sim"]["latent"])
        self.assertTrue(np.isfinite(valid["sim"]["summary"]["forecast_score_n"]))

    def test_streaming_matches_batch_and_validates_checkpoint(self):
        checkpoint = self.root / "encoder.pt"
        save_model(checkpoint, self.model, {})
        manifest = self.root / "manifest.json"
        write_json(
            manifest,
            {
                "artifact_format": FORMAT,
                "layout": LAYOUT,
                "target_domain": "real",
                "horizons": list(range(FUTURE + 1)),
                "checkpoint": checkpoint.name,
                "encoder_sha256": sha256(checkpoint),
            },
        )
        for domain, name in enumerate(("real", "sim")):
            stream = ForcePredictiveEncoder(manifest, domain=name)
            offline = predict_episode(self.model, self.arrays[(0, domain)], domain)
            force, z = [], []
            ep = self.episodes[(0, domain)]
            for f, t in zip(ep.force, ep.time, strict=True):
                z.append(stream.push(f, t))
                force.append(np.concatenate([stream.predict_current()[None], stream.predict_future()]))
            np.testing.assert_allclose(z, offline["z"], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(force, offline["force"], atol=2e-5, rtol=2e-5)
            self.assertEqual(stream.effort_history().shape, (10, 32))
            with self.assertRaises(ValueError):
                stream.push(ep.force[-1], ep.time[-1])
            stream.push(ep.force[-1], ep.time[-1] + 1)
            self.assertEqual(len(stream.time), 1)
            with self.assertRaises(ValueError):
                stream.push([np.nan, 0, 0], ep.time[-1] + 2)
        meta = json.loads(manifest.read_text())
        meta["encoder_sha256"] = "changed"
        write_json(manifest, meta)
        with self.assertRaisesRegex(ValueError, "SHA"):
            ForcePredictiveEncoder(manifest)

    def test_force_only_baselines_and_stage_resume(self):
        coef = baselines.fit(self.episodes, [0, 1], self.stats, 1, "ar")
        result = baselines.evaluate(coef, self.episodes, [2], self.stats, 1, "ar")
        self.assertTrue(np.isfinite(result["summary"]["forecast_score_n"]))
        args = SimpleNamespace(
            seed=42, device="cpu", resume=False, lr=3e-4, batch_size=2, max_epochs=1, patience=1, delta_weight=0.5
        )
        initial = PredictiveForce(self.stats)
        teacher, tc = run_stage(
            initial, self.dataset, self.episodes, [2], self.arrays, args, "teacher", self.root / "teacher"
        )
        model, sc = run_stage(
            teacher, self.dataset, self.episodes, [2], self.arrays, args, "student", self.root / "student"
        )
        self.assertEqual(tc["selected_epoch"], 1)
        self.assertEqual(sc["selected_epoch"], 1)
        args.resume = True
        restored, _ = run_stage(
            teacher, self.dataset, self.episodes, [2], self.arrays, args, "student", self.root / "student"
        )
        self.assertEqual(digest(model), digest(restored))
        changed = copy.deepcopy(teacher)
        with torch.no_grad():
            changed.forecaster[0].weight.add_(1)
        with self.assertRaisesRegex(ValueError, "parent"):
            run_stage(changed, self.dataset, self.episodes, [2], self.arrays, args, "student", self.root / "student")
        loaded, _ = load_model(self.root / "student/model.pt")
        self.assertEqual(digest(loaded), digest(model))


if __name__ == "__main__":
    unittest.main()
