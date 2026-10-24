"""CPU contract checks. Run explicitly; importing training does not start a run."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from force_gap_forecast_v1.common import sha256, write_json
from force_gap_forecast_v1.data import Blocks, Episode, fit_stats, segment_limits
from force_gap_forecast_v1.metrics import predict_episode as predict_motion
from force_gap_forecast_v1.model import ForceForecast
from force_gap_forecast_v1.train import cpu_state
from . import FORMAT, LAYOUT
from .losses import forecast_loss
from .metrics import predict_episode, score_prediction
from .model import JointMotionForceForecast
from .runtime import JointMotionForceEncoder
from .train import acceptance, final_choice, fit_stage, frozen_digest, load_motion, motion_fold


def options(**updates):
    value = dict(device="cpu", resume=False, max_epochs=2, patience=2, warmup_epochs=0, batch_size=4,
                 lr=1e-3, decoder_lr=1e-4, fz_weight=2., current_weight=.1, peak_weight=.1,
                 timing_weight=.05, adapter_l2=1e-3, gains=[0., .5, 1.], guard_tolerance=.02,
                 inner_min_improvement=.01)
    value.update(updates)
    return SimpleNamespace(**value)


class JointForecastContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        self.episodes = {}
        for tid in range(3):
            n = 38
            time = np.arange(n, dtype=np.float64) * .1
            time[23:] += 1
            force = rng.normal(0, 2, (n, 3)).astype(np.float32)
            force[18, 2] += 12
            q = rng.normal(0, .2, (n, 7)).astype(np.float32)
            dq = rng.normal(0, .1, (n, 7)).astype(np.float32)
            starts, ends = segment_limits(time)
            for domain in (0, 1):
                self.episodes[(tid, domain)] = Episode(tid, force.copy() * (1 if domain == 0 else .7),
                    q.copy(), dq.copy(), time.copy(), np.full(n, "misaligned_contact", object), starts.copy(), ends.copy())
        self.stats = fit_stats(self.episodes, [0])
        self.motion = ForceForecast(self.stats).eval()
        self.model = JointMotionForceForecast(self.stats, self.motion.state_dict()).eval()
        self.base = self.root / "motion.pt"
        torch.save(dict(stats=self.stats, state_dict=self.motion.state_dict(),
                        stage=dict(motion_only=True, train_ids=[0])), self.base)

    def tearDown(self):
        self.temp.cleanup()

    def activate(self, model=None):
        model = model or self.model
        with torch.no_grad():
            for adapter in model.adapters:
                adapter.latent.weight.normal_(0, .03)
                adapter.history.weight.normal_(0, .03)
            for head in model.future_heads:
                head[-1].bias.add_(.15)

    def manifest(self, model=None):
        model = model or self.model
        checkpoint = self.root / "encoder.pt"
        torch.save(dict(artifact_format=FORMAT, layout=LAYOUT, stats=self.stats, use_force=model.use_force,
                        state_dict=cpu_state(model)), checkpoint)
        path = self.root / "manifest.json"
        write_json(path, dict(artifact_format=FORMAT, layout=LAYOUT, checkpoint=checkpoint.name,
                             encoder_sha256=sha256(checkpoint), status="diagnostic"))
        return path

    def test_sim_initial_and_zero_gain_preserve_motion_prior(self):
        ep = self.episodes[(0, 1)]
        expected = predict_motion(self.motion, ep, self.stats, 1, motion_only=True)
        actual = predict_episode(self.model, ep, self.stats, 1)
        for key in ("h", "history", "current", "future", "z"):
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, rtol=1e-6)
        self.activate()
        self.model.domain_gains[1] = 0
        disabled = predict_episode(self.model, ep, self.stats, 1)
        for key in ("h", "history", "current", "future", "z"):
            np.testing.assert_allclose(disabled[key], expected[key], atol=2e-6, rtol=1e-6)

    def test_force_changes_joint_latent_and_equal_capacity_control_is_invariant(self):
        self.activate()
        ep = self.episodes[(0, 1)]
        changed = copy.deepcopy(ep)
        changed.force *= -2
        first = predict_episode(self.model, ep, self.stats, 1)
        second = predict_episode(self.model, changed, self.stats, 1)
        self.assertGreater(np.max(np.abs(first["h"] - second["h"])), 1e-6)
        self.assertGreater(np.max(np.abs(first["future"] - second["future"])), 1e-6)
        control = copy.deepcopy(self.model)
        control.use_force = False
        self.assertEqual(sum(p.numel() for p in control.parameters()), sum(p.numel() for p in self.model.parameters()))
        first = predict_episode(control, ep, self.stats, 1)
        second = predict_episode(control, changed, self.stats, 1)
        for key in ("h", "current", "future", "z"):
            np.testing.assert_allclose(first[key], second[key], atol=1e-6)

    def test_real_sensor_conditions_future_when_sim_gain_is_zero(self):
        self.model.domain_gains.zero_()
        ep = self.episodes[(0, 0)]
        changed = copy.deepcopy(ep)
        changed.force += 3
        first = predict_episode(self.model, ep, self.stats, 0)
        second = predict_episode(self.model, changed, self.stats, 0)
        np.testing.assert_array_equal(first["current"], ep.force)
        self.assertGreater(np.max(np.abs(first["future"] - second["future"])), 1e-5)
        self.model.domain_gains[0] = 1
        np.testing.assert_array_equal(predict_episode(self.model, ep, self.stats, 0)["current"], ep.force)
        self.assertEqual(self.model.domain_gains[1].item(), 0)

    def test_corrected_current_has_a_gradient_path_to_future_decoder(self):
        batch = default_collate([Blocks(self.episodes, [0], self.stats)[18]])
        x = batch["x"][:, 1]
        b, k = x.shape[:2]
        parts = self.model.encode(x.reshape(b * k, *x.shape[2:]), 1)
        parts = {name: value.reshape(b, k, *value.shape[1:]).detach() for name, value in parts.items()}
        parts["dp"].requires_grad_()
        output = self.model.compose(parts, batch["sim_reset"], batch["current"], 1, gain=1.)
        output["future"][:, 1:].square().sum().backward()
        self.assertGreater(parts["dp"].grad[..., -1, :].abs().sum().item(), 0)

    def test_gradients_warmup_and_immutable_motion(self):
        batch = default_collate([Blocks(self.episodes, [0], self.stats)[18]])
        before = frozen_digest(self.model)
        optimizer = torch.optim.AdamW(list(self.model.adapters.parameters()) + list(self.model.future_heads.parameters()), lr=1e-3)
        self.model.train()
        self.model.set_decoder_trainable(False)
        output = self.model.block(batch, 1)
        loss, _ = forecast_loss(self.model, output, batch, 1, options())
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.future_heads.parameters()))
        self.assertFalse(self.model.motion.training)
        optimizer.step()
        self.model.set_decoder_trainable(True)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = forecast_loss(self.model, self.model.block(batch, 1), batch, 1, options())
            loss.backward()
            optimizer.step()
        self.assertGreater(self.model.adapters[1].gru.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None for p in self.model.future_heads[1].parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.motion.parameters()))
        self.assertEqual(before, frozen_digest(self.model))

    def test_no_future_input_leakage(self):
        self.activate()
        ep = self.episodes[(0, 1)]
        changed = copy.deepcopy(ep)
        changed.force[11:] += 100
        changed.q[11:] -= 100
        changed.dq[11:] += 100
        first = predict_episode(self.model, ep, self.stats, 1)
        second = predict_episode(self.model, changed, self.stats, 1)
        for key in ("h", "history", "current", "future", "z"):
            np.testing.assert_allclose(first[key][:11], second[key][:11], atol=1e-6, rtol=1e-6)

    def test_streaming_training_batch_and_gap_resets_agree(self):
        self.activate()
        self.model.domain_gains.copy_(torch.tensor([.5, .25]))
        manifest = self.manifest()
        self.base.unlink()  # Portable inference does not depend on the old checkpoint path.
        for domain, name in enumerate(("real", "sim")):
            ep = self.episodes[(0, domain)]
            expected = predict_episode(self.model, ep, self.stats, domain)
            runtime = JointMotionForceEncoder(manifest, name)
            latents, futures = [], []
            for force, q, dq, timestamp in zip(ep.force, ep.q, ep.dq, ep.time):
                latents.append(runtime.push(force, q, dq, timestamp))
                futures.append(runtime.predict_future())
            np.testing.assert_allclose(latents, expected["z"], atol=3e-5, rtol=1e-5)
            np.testing.assert_allclose(futures, expected["future"], atol=3e-5, rtol=1e-5)
            np.testing.assert_array_equal(np.asarray(latents)[23, 3:6], np.zeros(3))
            self.assertEqual(runtime.effort_history().shape, (10, 32))
            np.testing.assert_allclose(runtime.batch(ep.force, ep.q, ep.dq, ep.time), expected["z"], atol=3e-5, rtol=1e-5)
            for anchor in (18, 23, 30):
                batch = default_collate([Blocks(self.episodes, [0], self.stats)[anchor]])
                with torch.no_grad():
                    output = self.model.block(batch, domain)
                np.testing.assert_allclose(output["z"][0, -1], expected["z"][anchor], atol=3e-5, rtol=1e-5)
                np.testing.assert_allclose(output["future"][0, -1], expected["future"][anchor], atol=3e-5, rtol=1e-5)

    def test_masked_loss_ignores_invalid_future_targets(self):
        batch = default_collate([Blocks(self.episodes, [0], self.stats)[22]])
        output = self.model.block(batch, 1)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["future"][~changed["future_valid"]] += 10000
        first, _ = forecast_loss(self.model, output, batch, 1, options())
        second, _ = forecast_loss(self.model, output, changed, 1, options())
        torch.testing.assert_close(first, second)

    def test_future_score_does_not_reward_real_current_bypass(self):
        ep = self.episodes[(0, 0)]
        prediction = predict_episode(self.model, ep, self.stats, 0)
        first = score_prediction(prediction, ep, self.stats["target"]["scale"], ep)
        altered = {key: value.copy() for key, value in prediction.items()}
        altered["current"] += 100
        second = score_prediction(altered, ep, self.stats["target"]["scale"], ep)
        self.assertEqual(first["all"]["forecast_score_n"], second["all"]["forecast_score_n"])
        self.assertEqual(first["all"]["current_rmse_n"], 0.)
        self.assertIn("future_5_z_rmse_n", first["misaligned_contact"])

    def test_independent_refit_epochs_and_resume_identity(self):
        args = options()
        refit = {"0": dict(epoch=1, gain=.5), "1": dict(epoch=0, gain=0.)}
        model, _, summary = fit_stage(self.base, self.episodes, [0], [], args, 42,
                                      self.root / "stage", refit=refit)
        self.assertEqual(summary["domains"]["0"]["epoch"], 1)
        self.assertEqual(summary["domains"]["1"]["epoch"], 0)
        self.assertEqual(model.domain_gains.tolist(), [.5, 0.])
        expected = predict_motion(self.motion, self.episodes[(0, 1)], self.stats, 1, motion_only=True)
        np.testing.assert_allclose(predict_episode(model, self.episodes[(0, 1)], self.stats, 1)["future"],
                                   expected["future"], atol=3e-6, rtol=1e-6)
        reloaded, _, _ = fit_stage(self.base, self.episodes, [0], [], options(resume=True), 42,
                                   self.root / "stage", refit=refit)
        self.assertEqual(reloaded.domain_gains.tolist(), [.5, 0.])
        with self.assertRaises(ValueError):
            fit_stage(self.base, self.episodes, [0], [], options(resume=True), 42,
                      self.root / "stage", use_force=False, refit=refit)
        with self.assertRaises(ValueError):
            load_motion(self.base, self.episodes, [1])
        modified = copy.deepcopy(self.episodes)
        modified[(0, 1)].force += 1
        with self.assertRaises(ValueError):
            load_motion(self.base, modified, [0])

    def test_inner_selection_smoke(self):
        _, _, summary = fit_stage(self.base, self.episodes, [0], [1], options(max_epochs=1), 42,
                                  self.root / "selection")
        self.assertTrue(summary["completed"])
        self.assertEqual(set(summary["domains"]), {"0", "1"})
        self.assertEqual(len(summary["records"]), 1)

    def test_fresh_motion_fold_uses_the_new_data_train_split(self):
        args = options(reuse_base=False, output_dir=self.root)
        recipe = dict(lr=1e-3, batch_size=4, max_epochs=1, patience=1)
        selected, refitted, choice = motion_fold(self.episodes, args, recipe, {}, 42, 0, [0, 1], [0], [1])
        self.assertEqual(load_motion(selected, self.episodes, [0])["stage"]["train_ids"], [0])
        self.assertEqual(load_motion(refitted, self.episodes, [0, 1])["stage"]["train_ids"], [0, 1])
        self.assertEqual(choice["teacher_epoch"], 1)
        self.assertEqual(choice["student_epoch"], 1)

    def test_final_domain_gains_are_independent_and_ties_choose_smaller(self):
        rows = [{"domains": {"0": dict(epoch=8, gain=1.), "1": dict(epoch=0, gain=0.)}},
                {"domains": {"0": dict(epoch=12, gain=1.), "1": dict(epoch=4, gain=.5)}}]
        choice = final_choice(rows, [0., .5, 1.])
        self.assertEqual(choice["0"], dict(epoch=10, gain=1.))
        self.assertEqual(choice["1"], dict(epoch=0, gain=0.))

    def test_acceptance_requires_a_practical_gain_over_capacity_control(self):
        def row(score):
            return {"all": dict(forecast_score_n=score, score=score, current_rmse_n=1.,
                                future_1_rmse_n=score, future_3_rmse_n=score, future_5_rmse_n=score)}
        seeds = [42, 43, 44]
        outer = {name: {str(seed): {str(tid): row(score) for tid in range(4)} for seed in seeds}
                 for name, score in (("full", .999), ("motion_only", 1.), ("motion_control", .99))}
        gate = acceptance(outer, seeds, True, options(min_improvement=.02))
        self.assertFalse(gate["passed"])
        self.assertTrue(any("motion_control" in reason for reason in gate["reasons"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
