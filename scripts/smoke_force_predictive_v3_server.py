#!/usr/bin/env python3
"""Replay recorded simulator observations through the raw-wrench websocket server."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import h5py
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trajectory = root / "sim-data-aligned/traj_0"
    frame = 44
    with h5py.File(trajectory / "data.h5", "r") as handle:
        wrench = handle["obs/state/ee_wrench_base"][: frame + 1].astype(np.float32)
        times = handle["timestamps"][: frame + 1].astype(np.float64)
        state = np.r_[
            handle["obs/state/joint_pos"][frame], np.asarray(handle["obs/state/gripper_pos"][frame]).reshape(-1)
        ].astype(np.float32)
    images = {}
    for key, filename in (("cam_high", "front_camera.mp4"), ("cam_left_wrist", "wrist_camera.mp4")):
        video = cv2.VideoCapture(str(trajectory / filename))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = video.read()
        video.release()
        if not ok:
            raise RuntimeError(f"Unable to read {filename} frame {frame}")
        images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    observation = dict(images=images, state=state, effort=wrench, timestamps_s=times, prompt="peg-in-hole")
    packer = msgpack_numpy.Packer()
    uri = f"ws://{args.host}:{args.port}"
    report = dict(
        tested_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), uri=uri, trajectory=str(trajectory), frame=frame
    )

    def receive(ws):
        payload = ws.recv(timeout=180)
        if isinstance(payload, str):
            raise RuntimeError(payload)
        return msgpack_numpy.unpackb(payload)

    def request(ws, obs):
        start = time.perf_counter()
        ws.send(packer.pack(obs))
        result = receive(ws)
        return result, time.perf_counter() - start

    def reset(ws):
        result, _ = request(ws, {"reset": True})
        assert result == {"reset": True}

    def validate(result):
        assert result["actions"].shape == (50, 8)
        assert result["predicted_force"].shape == (5, 3)
        assert np.isfinite(result["actions"]).all()
        assert np.isfinite(result["predicted_force"]).all()

    with connect(uri, compression=None, max_size=None, open_timeout=10) as ws:
        metadata = receive(ws)
        assert metadata["policy_variant"] == "force_predictive_v3_cotrain_50_50_server_encoder"
        assert metadata["checkpoint"].endswith("_thpfix/29999")
        assert metadata["encoder_sha256"] == "da9117c7091b910e70a6efa6fa09fd68dd77dfa5997353e06e8b5ea11d176edb"
        assert metadata["encoder_domain"] == "sim"
        assert metadata["history_isolated_per_connection"]
        report["metadata"] = metadata
        reset(ws)
        first, cold = request(ws, observation)
        validate(first)
        print(json.dumps({"stage": "first_inference", "seconds": cold}), flush=True)
        reset(ws)
        repeated, warm = request(ws, observation)
        validate(repeated)
        np.testing.assert_allclose(first["actions"], repeated["actions"], atol=1e-6)
        np.testing.assert_array_equal(first["predicted_force"], repeated["predicted_force"])
        reset(ws)
        raw = wrench.copy()
        raw[:, :3] *= -1
        raw[:, 3:] = 123
        calibrated, calibrated_time = request(
            ws, dict(observation, effort=raw, force_source="panda_link7_wrench_calibrated")
        )
        validate(calibrated)
        np.testing.assert_allclose(first["actions"], calibrated["actions"], atol=1e-6)
        np.testing.assert_array_equal(first["predicted_force"], calibrated["predicted_force"])
        reset(ws)
        updated, update_time = request(ws, dict(effort=wrench[:-1], timestamps_s=times[:-1], update_only=True))
        assert updated["updated"] == len(wrench) - 1
        streamed, streamed_time = request(ws, dict(observation, effort=wrench[-1:], timestamps_s=times[-1:]))
        validate(streamed)
        np.testing.assert_allclose(first["actions"], streamed["actions"], atol=1e-6)
        np.testing.assert_array_equal(first["predicted_force"], streamed["predicted_force"])
        # Opening another connection permits timestamps from zero without affecting
        # the first connection's encoder history.
        with connect(uri, compression=None, max_size=None, open_timeout=10) as other:
            receive(other)
            independent, _ = request(other, dict(effort=wrench[0], timestamp_s=float(times[0]), update_only=True))
            assert independent["updated"] == 1
            reset(other)
        continued, _ = request(ws, dict(effort=wrench[-1], timestamp_s=float(times[-1] + 0.1), update_only=True))
        assert continued["updated"] == 1
        reset(ws)
        report.update(
            passed=True,
            cold_inference_s=cold,
            warm_inference_45_samples_s=warm,
            calibrated_link7_inference_s=calibrated_time,
            force_only_update_44_samples_s=update_time,
            warm_inference_1_sample_s=streamed_time,
            reset_deterministic=True,
            calibrated_link7_parity=True,
            streaming_batch_parity=True,
            per_connection_history_passed=True,
            action_shape=list(first["actions"].shape),
            predicted_force_shape=list(first["predicted_force"].shape),
            action_min=float(first["actions"].min()),
            action_max=float(first["actions"].max()),
        )
        np.savez(
            root / "logs/server_8006_force_predictive_v3_smoke_outputs.npz",
            actions=first["actions"],
            predicted_force=first["predicted_force"],
            state=state,
        )
    output = root / "logs/server_8006_force_predictive_v3_smoke.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
