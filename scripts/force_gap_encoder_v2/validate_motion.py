"""User-run software contract checks; no training or policy/server execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .motion_infer import MotionForceLatentEncoder
from .motion_features import numpy_features, torch_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/motion_encoder.pt"))
    parser.add_argument("--report", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/motion_encoder.report.json"))
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--trajectories", type=int, nargs="+", default=(0, 2, 21, 22, 23, 33, 34, 39))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if report.get("checkpoint_sha256") != digest:
        raise ValueError("Report/checkpoint hash mismatch")
    rows = []
    try:
        if report.get("status") != "complete":
            raise ValueError("Training report is not complete")
        expected = report["data"]["diagnostic_trajectories"]
        if sorted(args.trajectories) != expected:
            raise ValueError("Software qualification requires all eight recorded diagnostic trajectories")
        if not 0 < args.tolerance <= 1e-5 or args.force_key != report["config"]["force_key"]:
            raise ValueError("Use the trained force key and a tolerance no larger than 1e-5")
        for trajectory in args.trajectories:
            for source, root in (("real", args.real_root), ("sim_contact_isolated", args.sim_root)):
                with h5py.File(root / f"traj_{trajectory}" / "data.h5", "r") as h5:
                    force = np.asarray(h5[args.force_key], np.float32)[:, :3]
                    q = np.asarray(h5["obs/state/joint_pos"], np.float32)
                    velocity = np.asarray(h5["obs/state/joint_vel"], np.float32)
                    time = np.asarray(h5["timestamps"], np.float64)
                arrays_hash = hashlib.sha256()
                for value in (force, q, velocity, time):
                    arrays_hash.update(np.ascontiguousarray(value).tobytes())
                provenance = report["data"]["provenance"][str(trajectory)]["sources"][0 if source == "real" else 1]
                if arrays_hash.hexdigest() != provenance["selected_arrays_sha256"]:
                    raise ValueError("Software-check data differs from training provenance")
                encoder = MotionForceLatentEncoder(args.checkpoint, source=source, device=args.device)
                batch = encoder.batch(force, q, velocity, time)
                streamed = []
                for frame, (f, joints, speed, timestamp) in enumerate(zip(force, q, velocity, time, strict=True)):
                    z = encoder.push(f, joints, speed, timestamp)
                    if frame < encoder.calibration_frames - 1:
                        assert z is None and not encoder.ready, "Premature startup output"
                    else:
                        assert z is not None
                        streamed.append(z)
                streamed = np.stack(streamed)
                error = float(np.max(np.abs(batch - streamed)))
                assert error <= args.tolerance and np.isfinite(streamed).all(), f"Batch/stream error {error}"
                assert encoder.effort_history().shape == (10, 32)
                prefix_end = max(encoder.calibration_frames + 1, len(force) // 2)
                prefix = encoder.batch(force[:prefix_end], q[:prefix_end], velocity[:prefix_end], time[:prefix_end])
                prefix_error = float(np.max(np.abs(prefix - batch[:len(prefix)])))
                assert prefix_error <= args.tolerance, "Future input changed a past output"
                future_force, future_q, future_velocity = force.copy(), q.copy(), velocity.copy()
                future_force[prefix_end:] += 3.
                future_q[prefix_end:] += .2
                future_velocity[prefix_end:] += .3
                changed_future = encoder.batch(future_force, future_q, future_velocity, time)
                future_error = float(np.max(np.abs(changed_future[:len(prefix)] - batch[:len(prefix)])))
                assert future_error <= args.tolerance, "Changing future observations changed prior outputs"
                original = encoder.features(force, q, velocity, time)
                augmentation_error = 0.
                for gain in (.5, 1., 2.):
                    expected_features = numpy_features(original["vectors"] * gain, original["scales"], original["elapsed"])
                    gained = torch_features(torch.from_numpy(original["vectors"])[None] * gain,
                        torch.from_numpy(original["scales"])[None], torch.from_numpy(original["elapsed"])[None], (0., 0., -1.))[0].numpy()
                    augmentation_error = max(augmentation_error, float(np.max(np.abs(gained - expected_features))))
                assert augmentation_error <= args.tolerance, "Train/inference feature implementations disagree"
                if source == "sim_contact_isolated":
                    other_state = encoder.features(force, q + .5, velocity + .3, time)
                    assert np.array_equal(original["features"], other_state["features"]), "Sim encoder received copied joint-state features"
                injected_force = force.copy()
                injected_force[encoder.calibration_frames:] += np.asarray([0., 0., 5.], np.float32)
                injected = encoder.features(injected_force, q, velocity, time)
                assert np.array_equal(original["background"], injected["background"]), "Contact force changed background prediction"
                expected = np.zeros_like(force)
                expected[encoder.calibration_frames:, 2] = 5.
                assert np.max(np.abs(injected["residual"] - original["residual"] - expected)) <= 1e-4
                injected_latent = encoder.batch(injected_force, q, velocity, time)
                latent_response = float(np.max(np.abs(injected_latent - batch)))
                assert latent_response > 1e-6, "Encoder ignores a sustained 5 N perturbation"
                encoder.reset()
                injected_stream, anchors = [], None
                for frame, (f, joints, speed, timestamp) in enumerate(zip(injected_force, q, velocity, time, strict=True)):
                    z = encoder.push(f, joints, speed, timestamp)
                    if z is None:
                        continue
                    injected_stream.append(z)
                    if anchors is None:
                        anchors = (encoder._baseline.copy(), encoder._phi0.copy(), encoder._scales.copy())
                    else:
                        assert all(np.array_equal(before, after) for before, after in
                                   zip(anchors, (encoder._baseline, encoder._phi0, encoder._scales), strict=True)), "Streaming reference adapted to contact"
                    assert np.max(np.abs(encoder._residual_history[-1] - injected["residual"][frame])) <= 1e-5, "Streaming background absorbed injected force"
                injected_stream_error = float(np.max(np.abs(np.stack(injected_stream) - injected_latent)))
                assert injected_stream_error <= args.tolerance, "Injected-force batch/stream mismatch"
                encoder.reset()
                assert not encoder.ready and not encoder._forces and not encoder._times
                assert not encoder._feature_history and not encoder._residual_history and not encoder._smooth_history and not encoder._time_history
                assert encoder._baseline is None and encoder._phi0 is None and encoder._scales is None and encoder._last_timestamp is None
                try:
                    encoder.effort_history()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("reset retained output history")
                replayed = None
                for frame in range(encoder.calibration_frames):
                    replayed = encoder.push(force[frame], q[frame], velocity[frame], time[frame])
                assert np.max(np.abs(replayed - batch[0])) <= args.tolerance, "Reset changed the startup output"
                encoder.reset()
                for bad in (np.zeros(2), np.asarray([float("nan"), 0., 0.])):
                    try:
                        encoder.push(bad, q[0], velocity[0], time[0])
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Malformed force accepted")
                assert encoder.push(force[0], q[0], velocity[0], time[0]) is None
                try:
                    encoder.push(force[1], q[1], velocity[1], time[0])
                except ValueError:
                    pass
                else:
                    raise AssertionError("Non-increasing timestamp accepted")
                row = {"trajectory": trajectory, "source": source, "batch_stream_max_abs": error,
                       "future_prefix_max_abs": prefix_error, "future_perturbation_max_abs": future_error,
                       "train_infer_gain_features_max_abs": augmentation_error,
                       "sustained_force_batch_stream_max_abs": injected_stream_error,
                       "sustained_force_latent_max_response": latent_response}
                rows.append(row)
                print(json.dumps(row), flush=True)
        report["software_checks"] = {"passed": True, "checkpoint_sha256": digest, "rows": rows,
            "checks": ["startup", "batch_stream", "future_prefix", "future_perturbation", "gain_features", "sim_state_independence",
                       "sustained_force", "reset", "invalid_force", "timestamp"]}
    except Exception as error:
        report["software_checks"] = {"passed": False, "checkpoint_sha256": digest, "rows": rows, "error": str(error)}
        report["qualified"] = False
        temporary = args.report.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.report)
        raise
    report["qualified"] = bool(report["acceptance"]["passed"])
    temporary = args.report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(f"SOFTWARE CHECKS: PASS; OFFLINE QUALIFIED: {report['qualified']}")


if __name__ == "__main__":
    main()
