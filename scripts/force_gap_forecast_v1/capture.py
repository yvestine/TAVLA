"""Simulator-side recorder. Call append from the SAME replay's observation step."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .common import aligned_force_from_calibrated_link7, finite_array, sha256


class PairedReplayRecorder:
    def __init__(self, destination, paired_real_h5, force_source="already_aligned", fps=10):
        self.destination, self.real_path = Path(destination), Path(paired_real_h5)
        if self.destination.exists():
            raise FileExistsError("Capture requires a new trajectory directory")
        if force_source not in ("already_aligned", "calibrated_link7") or fps != 10:
            raise ValueError("Declare the original calibrated force convention; capture at 10 Hz")
        self.force_source, self.fps = force_source, fps
        with h5py.File(self.real_path, "r") as h:
            self.arm_target = h["action/actual/arm"][:].astype(np.float32)
            self.gripper_target = h["action/actual/gripper"][:].astype(np.float32)
            self.real_times = h["timestamps"][:].astype(np.float64)
        self.real_sha = sha256(self.real_path)
        self.destination.mkdir(parents=True)
        self.rows, self.writers = [], {}
        self.closed = False

    def append(self, real_frame_id, force, observed_q, observed_dq, timestamp_s,
               observed_gripper_pos, front_rgb, wrist_rgb):
        import cv2
        if self.closed:
            raise RuntimeError("Recorder is closed")
        if real_frame_id != len(self.rows) or real_frame_id >= len(self.real_times):
            raise ValueError("Log one SAME-STEP actual observation per reference frame, in order")
        if self.force_source == "calibrated_link7":
            force = aligned_force_from_calibrated_link7(force)
        force = finite_array(force, (3,), "aligned force")
        q = finite_array(observed_q, (7,), "actual Sim joint positions")
        dq = finite_array(observed_dq, (7,), "actual Sim joint velocities")
        timestamp, grip = float(timestamp_s), float(observed_gripper_pos)
        if not np.isfinite([timestamp, grip]).all() or not 0 <= grip <= 1:
            raise ValueError("Finite timestamp and normalized gripper 0..1 required")
        if self.rows and timestamp <= self.rows[-1][3]:
            raise ValueError("Simulation observation timestamps must increase")
        validated_images = []
        for key, rgb in (("front_camera", front_rgb), ("wrist_camera", wrist_rgb)):
            rgb = np.asarray(rgb)
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError("Synchronized cameras must be RGB uint8 (480,640,3)")
            validated_images.append((key, rgb))
        for key, rgb in validated_images:
            if key not in self.writers:
                writer = cv2.VideoWriter(str(self.destination / f"{key}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (640, 480))
                if not writer.isOpened():
                    raise RuntimeError("Cannot open capture video writer")
                self.writers[key] = writer
            self.writers[key].write(np.ascontiguousarray(rgb[..., ::-1]))
        self.rows.append((force.copy(), q.copy(), dq.copy(), timestamp, grip))

    def close(self):
        if self.closed:
            raise RuntimeError("Recorder already closed")
        self.closed = True
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
        if len(self.rows) != len(self.real_times):
            raise ValueError("Incomplete replay: retained diagnostics, no completed data.h5 produced")
        force = np.stack([r[0] for r in self.rows])
        q, dq = np.stack([r[1] for r in self.rows]), np.stack([r[2] for r in self.rows])
        time = np.array([r[3] for r in self.rows], np.float64)
        grip = np.array([r[4] for r in self.rows], np.float32)[:, None]
        tmp = self.destination / "data.h5.partial"
        with h5py.File(tmp, "w") as h:
            h.attrs.update(state_source="sim_observed", synchronized_same_replay=True,
                           cameras_synchronized_same_replay=True, force_contract="aligned_base_xyz_n_v1",
                           paired_real_sha256=self.real_sha, capture_force_source=self.force_source,
                           action_source="paired_real_reference_targets", data_source="isaac_sim_replay")
            for key, value in {"obs/state/ee_force_base": force, "obs/state/joint_pos": q,
                               "obs/state/joint_vel": dq, "obs/state/gripper_pos": grip,
                               "timestamps": time, "pairing/real_frame_id": np.arange(len(time)),
                               "pairing/real_timestamp_s": self.real_times,
                               "action/actual/arm": self.arm_target, "action/actual/gripper": self.gripper_target}.items():
                h.create_dataset(key, data=value)
        tmp.replace(self.destination / "data.h5")
