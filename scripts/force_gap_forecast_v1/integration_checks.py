"""Complete orchestration on tiny synthetic CPU data, never the user's dataset."""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import h5py
import numpy as np

from .common import read_manifest
from .runtime import ForceGapEncoder
from .train import main


def check():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory(prefix="force_forecast_synthetic_") as directory:
        root = Path(directory)
        labels = root / "labels.csv"
        with labels.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["trajectory", "start_frame", "end_frame", "label"])
            writer.writeheader()
            for i in range(40):
                for lo, hi, phase in ((0, 7, "free"), (8, 11, "aligned_progress"), (12, 15, "terminal_hold")):
                    writer.writerow(dict(trajectory=f"traj_{i}", start_frame=lo, end_frame=hi, label=phase))
                n = 16
                force = rng.normal(0, 2, (n, 3)).astype(np.float32)
                force[10, 2] += 10
                q = rng.normal(0, .1, (n, 7)).astype(np.float32)
                dq = rng.normal(0, .1, (n, 7)).astype(np.float32)
                for domain in ("real", "sim"):
                    path = root / domain / f"traj_{i}/data.h5"
                    path.parent.mkdir(parents=True)
                    with h5py.File(path, "w") as h:
                        h["obs/state/ee_force_base"] = force if domain == "real" else force * .8
                        h["obs/state/joint_pos"] = q
                        h["obs/state/joint_vel"] = dq
                        h["timestamps"] = np.arange(n, dtype=np.float64) * .1
        output = root / "run"
        args = ["train", "--real-root", str(root / "real"), "--sim-root", str(root / "sim"),
                "--labels", str(labels), "--output-dir", str(output), "--seeds", "42",
                "--max-epochs", "1", "--device", "cpu", "--threads", "1"]
        log = io.StringIO()
        with patch("sys.argv", args), contextlib.redirect_stdout(log):
            main()
        value, _ = read_manifest(output / "manifest.json")
        if value["status"] != "diagnostic":
            raise AssertionError("Reduced synthetic protocol was incorrectly promoted")
        gate = json.loads((output / "acceptance.json").read_text())
        if "full three-seed ablation protocol incomplete" not in gate["reasons"]:
            raise AssertionError("Missing protocol gate")
        with contextlib.redirect_stdout(log), patch("sys.argv", args + ["--resume"]):
            main()
        encoder = ForceGapEncoder(output / "manifest.json", "sim")
        z = encoder.push(np.zeros(3), np.zeros(7), np.zeros(7), 0)
        if z.shape != (32,):
            raise AssertionError("Final artifact not loadable")
        if not list(output.glob("stages/*/curves.png")) or not list(output.glob("diagnostic_predictions/*/*.png")):
            raise AssertionError("Missing curve/waveform plots")
        return {"passed": True, "fixture": "40 tiny synthetic pairs, CPU, one epoch / one seed only",
                "checked": ["nested CV orchestration", "four model variants", "baseline selection",
                            "diagnostic checkpoint preservation", "resume", "reports/plots", "runtime reload"],
                "stage_count": len(list(output.glob("stages/*/model.pt")))}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
