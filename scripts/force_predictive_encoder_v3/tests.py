"""Synthetic CPU tests; no real-data fitting, policy work or remote GPU jobs."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from force_gap_forecast_v1.common import sha256, write_json
from force_gap_forecast_v1.data import Episode, segment_limits
from . import FORMAT, HISTORY, LAYOUT, PAST
from .data import Blocks, FOLDS, force_window, make_stats
from .losses import objective, spectral_loss, trend_alignment
from .metrics import predict_episode, reconstruction_metrics
from .model import PredictiveForce
from .report import acceptance, comparison, write_report
from .runtime import ForcePredictiveEncoder
from .train import (choose_alignment, digest, final_choices, run_stage, save_model,
                    train_fold, train_final, train_single)


def episodes(count=3, n=40, gap=True):
    rng = np.random.default_rng(300)
    result = {}
    for tid in range(count):
        time = np.arange(n, dtype=np.float64) * .1
        if gap:
            time[n // 2:] += 1.
        force = np.column_stack((np.sin(np.arange(n) * .13), np.cos(np.arange(n) * .3), np.sin(np.arange(n) * .5)))
        force = (force + rng.normal(0, .05, (n, 3)) + tid * .015).astype(np.float32)
        force[n // 3, 2] += 3
        q = rng.normal(0, .2, (n, 7)).astype(np.float32)
        dq = rng.normal(0, .1, (n, 7)).astype(np.float32)
        starts, ends = segment_limits(time)
        for d in (0, 1):
            result[(tid, d)] = Episode(tid, force.copy() * (1 if not d else .6), q.copy(), dq.copy(),
                                       time.copy(), np.full(n, "misaligned_contact", object), starts.copy(), ends.copy())
    return result


def options(**updates):
    values = dict(device="cpu", resume=False, max_epochs=1, patience=1, batch_size=2, lr=3e-4,
                  align_weights=[0., .1], old_base_run=None)
    values.update(updates)
    return SimpleNamespace(**values)


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episodes = episodes()
        self.stats = make_stats(self.episodes, [0, 1])
        self.model = PredictiveForce(self.stats).eval()

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, model=None, stats=None):
        model, stats = model or self.model, stats or self.stats
        checkpoint = self.root / "encoder.pt"
        save_model(checkpoint, model, stats, {}, True)
        path = self.root / "manifest.json"
        write_json(path, dict(artifact_format=FORMAT, layout=LAYOUT, checkpoint=checkpoint.name,
                             encoder_sha256=sha256(checkpoint)))
        return path

    def test_reversible_preprocessing_keeps_pulse_and_alternation(self):
        force = np.zeros((HISTORY, 3), np.float32)
        force[:, 0] = np.where(np.arange(HISTORY) % 2, -3., 3.)
        force[25, 2] = 20
        x, components = force_window(force, np.arange(HISTORY) * .1, self.stats, "real")
        mean, scale = np.asarray(self.stats["real"]["mean"][:3]), np.asarray(self.stats["real"]["scale"][:3])
        np.testing.assert_allclose(components.sum(0) * scale + mean, force[-PAST:], atol=3e-6)
        np.testing.assert_allclose(x[0, :, :3] + x[1, :, :3], (force - mean) / scale, atol=3e-6)

    def test_force_encoding_is_causal_and_independent_of_motion(self):
        ep = self.episodes[(0, 0)]
        changed = copy.deepcopy(ep)
        changed.q *= 10
        changed.dq *= -5
        changed.force[12:] += 10
        before = predict_episode(self.model, ep, self.stats, 0)
        after = predict_episode(self.model, changed, self.stats, 0)
        np.testing.assert_allclose(before["z"][:12], after["z"][:12], atol=1e-6)
        self.assertGreater(np.max(np.abs(before["z"][12:20] - after["z"][12:20])), 1e-5)
        changed.force = ep.force.copy()
        after = predict_episode(self.model, changed, self.stats, 0)
        np.testing.assert_array_equal(before["z"], after["z"])

    def test_training_windows_do_not_read_future_or_cross_gaps(self):
        data = Blocks(self.episodes, [0], self.stats)
        sample = data[12]
        changed = copy.deepcopy(self.episodes)
        changed[(0, 0)].force[13:] += 100
        changed[(0, 1)].force[13:] += 200
        other = Blocks(changed, [0], self.stats)[12]
        np.testing.assert_array_equal(sample["x"], other["x"])
        at_reset = data[20]
        self.assertEqual(int(at_reset["valid"].sum()), 1)
        self.assertEqual(int(at_reset["past_mask"][-1].sum()), 1)
        self.assertFalse(data[19]["future_mask"][-1].any())

    def test_alignment_does_not_force_transient_dimensions_to_match(self):
        z = torch.randn(2, 17, 32)
        teacher = z.clone()
        teacher[..., 16:] += 100
        valid = torch.ones(2, 17, dtype=torch.bool)
        self.assertEqual(float(trend_alignment(z, teacher, valid)), 0.)
        teacher[..., :16] += 1
        self.assertGreater(float(trend_alignment(z, teacher, valid)), 0.)

    def test_empty_masks_have_finite_zero_loss(self):
        batch = default_collate([Blocks(self.episodes, [0], self.stats)[12]])
        for key in ("valid", "past_mask", "future_mask"):
            batch[key].zero_()
        output = self.model.block(batch, 0)
        loss, _ = objective(self.model, output, batch, 0)
        self.assertEqual(float(loss), 0.)
        loss.backward()

    def test_spectral_loss_detects_phase_and_spikes(self):
        target = torch.zeros(2, 16, 3)
        target[:, 8, 2] = 10
        mask = torch.ones(2, 16, dtype=torch.bool)
        self.assertEqual(float(spectral_loss(target, target, mask)), 0.)
        self.assertGreater(float(spectral_loss(target.roll(1, 1), target, mask)), 0.)

    def test_runtime_matches_batch_for_both_domains_and_resets(self):
        path = self.manifest()
        for d, domain in enumerate(("real", "sim")):
            ep = self.episodes[(0, d)]
            offline = predict_episode(self.model, ep, self.stats, d, True)
            stream = ForcePredictiveEncoder(path, domain)
            latent, pure, joint = [], [], []
            for t in range(len(ep.force)):
                latent.append(stream.push(ep.force[t], ep.time[t]))
                pure.append(stream.predict_future())
                lo = max(int(ep.starts[t]), t - HISTORY + 1)
                joint.append(stream.predict_future(ep.q[lo:t + 1], ep.dq[lo:t + 1]))
            np.testing.assert_allclose(latent, offline["z"], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(pure, offline["future"], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(joint, offline["joint"], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(stream.reconstruct_history(), offline["history"][-1], atol=1e-5)
            self.assertEqual(stream.effort_history().shape, (10, 32))
            with self.assertRaises(ValueError):
                stream.push(ep.force[-1], ep.time[-1])
            with self.assertRaises(ValueError):
                stream.predict_future(ep.q, ep.dq)
            stream.reset()
            first = stream.push(ep.force[0], 0.)
            np.testing.assert_array_equal(stream.effort_history(), np.repeat(first[None], 10, axis=0))

    def test_metrics_use_reconstruction_and_actual_peak_timestamps(self):
        ep = self.episodes[(0, 0)]
        indices = np.maximum(np.arange(len(ep.force))[:, None] + np.arange(-PAST + 1, 1), ep.starts[:, None])
        exact = ep.force[indices]
        row = reconstruction_metrics(exact, ep, self.stats, "real")
        for key in ("reconstruction_rmse_n", "high_band_mse", "stream_high_band_mse", "peak_amplitude_mae_n", "peak_timing_mae_s"):
            self.assertEqual(row[key], 0.)
        damaged = exact.copy()
        damaged[:, :, 2] *= .1
        row = reconstruction_metrics(damaged, ep, self.stats, "real")
        self.assertGreater(row["high_band_mse"], 0.)
        self.assertGreater(row["peak_amplitude_mae_n"], 0.)

    def test_student_and_motion_training_do_not_change_frozen_weights(self):
        args = options()
        model, _ = run_stage(self.episodes, [0, 1], [], self.stats, args, 42, self.root / "teacher", "teacher", epochs=1)
        teacher_before = copy.deepcopy(model.encoders[0].state_dict())
        decoder_before = copy.deepcopy(model.forecaster.state_dict())
        student, _ = run_stage(self.episodes, [0, 1], [], self.stats, args, 1042, self.root / "student", "student",
                                model, align_weight=.1, epochs=1)
        for name, value in teacher_before.items():
            torch.testing.assert_close(student.encoders[0].state_dict()[name], value, rtol=0, atol=0)
        for name, value in decoder_before.items():
            torch.testing.assert_close(student.forecaster.state_dict()[name], value, rtol=0, atol=0)
        self.assertFalse(torch.equal(student.encoders[0].trend.out.weight, student.encoders[1].trend.out.weight))
        before = copy.deepcopy(student.encoders.state_dict())
        motion, _ = run_stage(self.episodes, [0, 1], [], self.stats, args, 2042, self.root / "motion", "motion", student, epochs=1)
        for name, value in before.items():
            torch.testing.assert_close(motion.encoders.state_dict()[name], value, rtol=0, atol=0)
        resumed, _ = run_stage(self.episodes, [0, 1], [], self.stats, options(resume=True), 2042,
                               self.root / "motion", "motion", student, epochs=1)
        self.assertEqual(digest(motion), digest(resumed))
        with self.assertRaises(ValueError):
            run_stage(self.episodes, [0, 1], [], self.stats, options(resume=True, lr=.001), 2042,
                      self.root / "motion", "motion", student, epochs=1)

    def test_selection_rejects_frequency_regression(self):
        choices = {"0.0": dict(selected_metrics=dict(forecast_score_n=1., high_band_mse=.1)),
                   "0.1": dict(selected_metrics=dict(forecast_score_n=.8, high_band_mse=.2))}
        self.assertEqual(choose_alignment(choices), 0.)
        choices["0.1"]["selected_metrics"]["high_band_mse"] = .1
        self.assertEqual(choose_alignment(choices), .1)

    def test_acceptance_guards_actual_annotation_phase_names(self):
        models = {}
        for name in ("full_force", "full_joint", "motion_control", "hold", "ar", "no_alignment", "no_fast"):
            metric = dict(forecast_score_n=.8 if name.startswith("full") else 1.,
                          high_band_mse=.1 if name != "no_fast" else .2)
            models[name] = {d: {"1": {phase: metric.copy() for phase in ("all", "aligned_progress", "terminal")}}
                            for d in ("real", "sim")}
        for d in ("real", "sim"):
            for phase in ("aligned_progress", "terminal"):
                models["full_joint"][d]["1"][phase]["forecast_score_n"] = 2.
        result = dict(seed=42, models=models, alignment=dict(
            full=dict(trend_distance=.5, teacher_train_std=[1.] * 16), no_alignment=dict(trend_distance=1.)))
        gate = acceptance([result], dict(options=dict(seeds=[42], folds=[0])))
        self.assertTrue(any("aligned_progress" in reason for reason in gate["reasons"]))
        self.assertTrue(any("terminal" in reason for reason in gate["reasons"]))

    def test_final_refit_exports_and_nonzero_alignment_refit_reuses_teacher(self):
        data = episodes(40, 18, gap=False)
        choices = dict(teacher_epoch=1, student_epoch=1, motion_epoch=1, control_epoch=1, align_weight=.1,
                       candidates={"0.0": dict(selected_epoch=1), "0.1": dict(selected_epoch=1)})
        for fold in range(4):
            write_json(self.root / "jobs" / f"seed42_fold{fold}" / "result.json", dict(selections=dict(full=choices)))
        train_final(data, options(batch_size=16), 42, self.root)
        checkpoint = self.root / "encoder_seed42.pt"
        self.assertTrue(checkpoint.is_file())
        write_json(self.root / "manifest.json", dict(artifact_format=FORMAT, layout=LAYOUT,
                   checkpoint=checkpoint.name, encoder_sha256=sha256(checkpoint)))
        runtime = ForcePredictiveEncoder(self.root / "manifest.json")
        runtime.push(np.zeros(3, np.float32), 0.)
        self.assertEqual(runtime.predict_future(np.zeros((1, 7)), np.zeros((1, 7))).shape, (5, 3))
        from .train import refit_force
        from .data import TRAIN_IDS
        stats = make_stats(data, TRAIN_IDS)
        alternative = refit_force(data, TRAIN_IDS, stats, options(batch_size=16), 42,
                                  self.root / "final" / "seed42" / "full", choices, weight=0.)
        self.assertIsInstance(alternative, PredictiveForce)

    def test_full_synthetic_fold_and_report(self):
        # Exercise teacher selection, every lambda, outer refits, motion/control, ablation and baselines.
        data = episodes(40, 20, gap=False)
        output = self.root / "jobs" / "seed42_fold0"
        train_fold(data, options(batch_size=8), 42, 0, output)
        import json
        result = json.loads((output / "result.json").read_text())
        self.assertTrue(result["completed"])
        self.assertEqual(len(result["models"]["full_force"]["sim"]), 8)
        self.assertFalse(set(result["outer_valid"]) & set(result["outer_train"]))
        recipe = dict(options=dict(seeds=[42], folds=[0]), data=dict(actual_sim_state_verified=False))
        gate = write_report(self.root, [result], recipe)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["complete_protocol"])
        compared = comparison([result], "full_joint", "motion_control", "sim")
        self.assertEqual(compared["trajectories"], 8)
        choice = final_choices([result], [0., .1])
        self.assertGreater(choice["teacher_epoch"], 0)
        write_json(self.root / "recipe.json", recipe)
        from .visualize import visualize
        visualize(self.root, limit=1)
        self.assertTrue((self.root / "visualizations" / "index.html").is_file())

    def test_single_run_uses_one_holdout_and_exports_encoder(self):
        data = episodes(40, 18, gap=False)
        args = options(batch_size=16, validation_fold=0, single_align_weight=.1)
        result = train_single(data, args, 42, self.root)
        self.assertTrue(result["completed"])
        self.assertEqual(result["mode"], "single_fixed_holdout")
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["valid_ids"], sorted(FOLDS[0]))
        self.assertFalse(set(result["train_ids"]) & set(result["valid_ids"]))
        self.assertEqual(result["align_weight"], .1)
        checkpoint = self.root / "encoder_seed42.pt"
        self.assertTrue(checkpoint.is_file())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertTrue(payload["motion_ready"])
        self.assertTrue((self.root / "single" / "single_report.md").is_file())

    def test_single_run_is_default_and_full_cv_is_opt_in(self):
        from .run import parse_args
        original = sys.argv
        try:
            sys.argv = ["run.py", "--output-dir", str(self.root / "default")]
            default = parse_args()
            self.assertTrue(default.single_run)
            self.assertEqual(default.seeds, [42])
            self.assertEqual(default.folds, [0])
            self.assertEqual(default.align_weights, [.1])
            sys.argv = ["run.py", "--output-dir", str(self.root / "full"), "--full-cv"]
            full = parse_args()
            self.assertFalse(full.single_run)
            self.assertEqual(full.seeds, [42, 43, 44])
            self.assertEqual(full.folds, [0, 1, 2, 3])
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
