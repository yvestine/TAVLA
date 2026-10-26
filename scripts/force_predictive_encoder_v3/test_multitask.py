"""CPU integration tests for real-only training and paired multi-task continuation."""

# ruff: noqa: PT009, PT027 -- Keep the existing v3 stdlib unittest interface.

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import force_predictive_encoder_v3.multitask as multitask
from force_predictive_encoder_v3.multitask import main
from force_predictive_encoder_v3.multitask import parse_args
from force_predictive_encoder_v3.multitask import prepare
from force_predictive_encoder_v3.multitask_data import MultiTaskBlocks
from force_predictive_encoder_v3.multitask_data import load_real
from force_predictive_encoder_v3.multitask_data import load_sim
from force_predictive_encoder_v3.multitask_data import prepare_arrays
from force_predictive_encoder_v3.multitask_data import task_config
from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
import h5py
import numpy as np
import torch


class MultiTaskTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "tasks.json"
        tasks = []
        for task, count in (("insert", 2), ("slide", 4)):
            tasks.append({"name": task, "real_root": f"real/{task}", "sim_root": f"sim/{task}"})
            for tid in range(count):
                n = 40 + tid
                t = np.arange(n, dtype=np.float64) * 0.1
                t[20:] += 1.0
                force = np.column_stack((np.sin(t), np.cos(t), np.sin(3 * t) + tid * 0.1)).astype(np.float32)
                for domain in ("real", "sim"):
                    path = self.root / domain / task / f"traj_{tid}" / "data.h5"
                    path.parent.mkdir(parents=True)
                    with h5py.File(path, "w") as h:
                        h["obs/state/ee_force_base"] = force if domain == "real" else 0.7 * force
                        h["timestamps"] = t
                        if domain == "sim":
                            h["pairing/real_frame_id"] = np.arange(n)
                            h["pairing/real_timestamp_s"] = t
        self.config.write_text(json.dumps({"version": 1, "tasks": tasks}))

    def tearDown(self):
        self.temp.cleanup()

    def args(self, stage, output, *extra):
        return [
            "--stage",
            stage,
            "--config",
            str(self.config),
            "--data-root",
            str(self.root),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--threads",
            "1",
            *extra,
        ]

    def test_task_sampling_normalization_and_causal_gap_masks(self):
        output = self.root / "audit"
        _, real, _, stats, split, _ = prepare(parse_args(self.args("teacher", output)))
        arrays = prepare_arrays(real, stats, "real")
        data = MultiTaskBlocks(real, None, split["train"], [arrays], stats["event_threshold_n"]["real"])
        probabilities = {}
        for (key, _), weight in zip(data.keys, data.weights, strict=True):
            probabilities[real[key].task] = probabilities.get(real[key].task, 0.0) + float(weight)
        self.assertAlmostEqual(probabilities["insert"], 0.5)
        self.assertAlmostEqual(probabilities["slide"], 0.5)
        expected = np.concatenate([real[key].force for key in split["train"]])
        np.testing.assert_allclose(stats["real"]["mean"], expected.mean(0))
        self.assertNotIn("sim", stats)
        key = split["train"][0]
        before = data[data.keys.index((key, 19))]
        after = data[data.keys.index((key, 20))]
        self.assertFalse(before["future_mask"][-1].any())
        self.assertEqual(int(after["past_mask"][-1].sum()), 1)
        self.assertEqual(int(after["valid"].sum()), 1)
        changed = copy.deepcopy(real)
        changed[key].force[13:] += 100
        changed_arrays = prepare_arrays(changed, stats, "real")
        altered = MultiTaskBlocks(changed, None, split["train"], [changed_arrays], stats["event_threshold_n"]["real"])
        index = data.keys.index((key, 12))
        np.testing.assert_array_equal(data[index]["x"], altered[index]["x"])

    def test_reference_pairing_and_real_data_changes_are_rejected(self):
        tasks = task_config(self.config, self.root)
        real = load_real(tasks)
        sim, _ = load_sim(tasks, real)
        self.assertEqual(set(sim), set(real))
        path = self.root / "sim/insert/traj_0/data.h5"
        with h5py.File(path, "r+") as h:
            h["pairing/real_frame_id"][1] = 0
        with self.assertRaisesRegex(ValueError, "real_frame_id"):
            load_sim(tasks, real)
        with h5py.File(path, "r+") as h:
            h["pairing/real_frame_id"][1] = 1
            del h["pairing/real_frame_id"]
        with self.assertRaisesRegex(ValueError, "missing pairing"):
            load_sim(tasks, real)
        config = json.loads(self.config.read_text())
        config["tasks"][0]["pairing"] = "frame_index"
        self.config.write_text(json.dumps(config))
        _, pairing = load_sim(task_config(self.config, self.root), real)
        self.assertEqual(pairing["insert/traj_0"], "declared_frame_index")

    def test_teacher_then_student_freezing_export_and_resume(self):
        teacher = self.root / "teacher"
        student = self.root / "student"
        # The Real stage must work even when the entire Sim dataset is unavailable.
        sim_root = self.root / "sim"
        sim_hidden = self.root / "sim_hidden"
        sim_root.rename(sim_hidden)
        main(self.args("teacher", teacher))
        sim_hidden.rename(sim_root)
        teacher_manifest = json.loads((teacher / "manifest.json").read_text())
        teacher_model = ForcePredictiveEncoder(teacher / "manifest.json", "real")
        with self.assertRaisesRegex(ValueError, "not trained"):
            ForcePredictiveEncoder(teacher / "manifest.json", "sim")
        future_before = teacher_model.push(np.ones(3, np.float32), 0.0)
        main(self.args("student", student, "--teacher-run", str(teacher)))
        encoder = ForcePredictiveEncoder(student / "manifest.json", "sim")
        self.assertEqual(encoder.push(np.ones(3, np.float32), 0.0).shape, (32,))
        self.assertEqual(encoder.predict_future().shape, (5, 3))
        self.assertEqual(encoder.effort_history().shape, (10, 32))
        self.assertFalse(encoder.motion_ready)
        for prefix in ("encoders.0.", "reconstructors.0.", "forecaster.", "target_"):
            for name, value in teacher_model.model.state_dict().items():
                if name.startswith(prefix):
                    torch.testing.assert_close(value, encoder.model.state_dict()[name], atol=0, rtol=0)
        self.assertNotEqual(teacher_model.stats.keys(), encoder.stats.keys())
        self.assertEqual(teacher_model.stats["real"], encoder.stats["real"])
        self.assertEqual(
            json.loads((teacher / "split.json").read_text()), json.loads((student / "split.json").read_text())
        )
        np.testing.assert_array_equal(
            ForcePredictiveEncoder(student / "manifest.json", "real").push(np.ones(3, np.float32), 0.0), future_before
        )
        # Re-export from an epoch checkpoint after an interrupted final export.
        (teacher / "manifest.json").unlink()
        main(self.args("teacher", teacher, "--resume"))
        self.assertEqual(
            json.loads((teacher / "manifest.json").read_text())["encoder_sha256"], teacher_manifest["encoder_sha256"]
        )
        main(self.args("student", student, "--teacher-run", str(teacher), "--resume"))
        with self.assertRaises(FileExistsError):
            main(self.args("teacher", teacher))
        with h5py.File(self.root / "real/insert/traj_0/data.h5", "r+") as h:
            h["obs/state/ee_force_base"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "changed since teacher"):
            main(self.args("student", self.root / "bad_student", "--teacher-run", str(teacher)))

    def test_interrupted_epoch_resume_matches_uninterrupted_training(self):
        baseline = self.root / "uninterrupted"
        resumed = self.root / "resumed"
        baseline_args = [*self.args("teacher", baseline), "--max-epochs", "2"]
        resumed_args = [*self.args("teacher", resumed), "--max-epochs", "2"]
        main(baseline_args)
        save = multitask.atomic_torch_save

        def interrupt_after_save(path, payload):
            save(path, payload)
            raise RuntimeError("simulated interruption")

        with (
            mock.patch.object(multitask, "atomic_torch_save", side_effect=interrupt_after_save),
            self.assertRaisesRegex(RuntimeError, "simulated interruption"),
        ):
            main(resumed_args)
        main([*resumed_args, "--resume"])
        expected = torch.load(baseline / "last.pt", map_location="cpu", weights_only=True)
        actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        for name, value in expected["state_dict"].items():
            torch.testing.assert_close(actual["state_dict"][name], value, atol=0, rtol=0)
        self.assertEqual(expected["best_epoch"], actual["best_epoch"])
        self.assertEqual(expected["best_score"], actual["best_score"])


if __name__ == "__main__":
    unittest.main()
