"""Exercise two CPU workers, HDF contracts, aggregation and strict resume in a temporary directory."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
from .tests import episodes
from .train import read_json


def main():
    project = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="force_v3_integration_") as temporary:
        root = Path(temporary)
        data = episodes(40, 18, gap=False)
        for (tid, d), ep in data.items():
            path = root / ("real" if d == 0 else "sim") / f"traj_{tid}" / "data.h5"
            path.parent.mkdir(parents=True)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("obs/state/ee_force_base", data=ep.force)
                handle.create_dataset("obs/state/joint_pos", data=ep.q)
                handle.create_dataset("obs/state/joint_vel", data=ep.dq)
                handle.create_dataset("timestamps", data=ep.time)
        labels = root / "labels.csv"
        labels.write_text("trajectory,start_frame,end_frame,label\n" + "".join(
            f"traj_{i},0,17,misaligned_contact\n" for i in range(40)))
        output = root / "run"
        command = [sys.executable, "-m", "force_predictive_encoder_v3.run", "--output-dir", str(output),
                   "--real-root", str(root / "real"), "--sim-root", str(root / "sim"), "--labels", str(labels),
                   "--gpu-ids", "0,1", "--device", "cpu", "--seeds", "42", "43", "--folds", "0",
                   "--max-epochs", "1", "--patience", "1", "--batch-size", "16", "--threads", "1",
                   "--align-weights", "0", "--no-old-baseline", "--no-plots", "--full-cv"]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project / "scripts")
        env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"

        def execute(extra=(), success=True):
            result = subprocess.run(command + list(extra), env=env, cwd=project, capture_output=True, text=True, timeout=240)
            if (result.returncode == 0) != success:
                logs = "\n".join(p.read_text()[-5000:] for p in (output / "logs").glob("*.log"))
                raise AssertionError(result.stdout + result.stderr + logs)
            return result

        execute()
        assert read_json(output / "run_summary.json")["completed"]
        assert not read_json(output / "run_summary.json")["final_encoder_created"]
        assert len(read_json(output / "outer_metrics.json")["jobs"]) == 2
        paths = list((output / "jobs").glob("*/full/teacher_refit/model.pt"))
        times = {str(p): p.stat().st_mtime_ns for p in paths}
        resumed = execute(["--resume"])
        assert "Reusing completed seed42_fold0" in resumed.stdout
        assert {str(p): p.stat().st_mtime_ns for p in paths} == times
        rejected = execute(["--resume", "--lr", "0.001"], success=False)
        assert "identical code, data, labels and training parameters" in rejected.stderr

        single_output = root / "single_run"
        single_command = [sys.executable, "-m", "force_predictive_encoder_v3.run", "--output-dir", str(single_output),
                          "--real-root", str(root / "real"), "--sim-root", str(root / "sim"), "--labels", str(labels),
                          "--gpu-ids", "0", "--device", "cpu", "--max-epochs", "1",
                          "--patience", "1", "--batch-size", "16", "--threads", "1", "--no-plots"]
        single = subprocess.run(single_command, env=env, cwd=project, capture_output=True, text=True, timeout=240)
        if single.returncode:
            logs = "\n".join(p.read_text()[-5000:] for p in (single_output / "logs").glob("*.log"))
            raise AssertionError(single.stdout + single.stderr + logs)
        assert read_json(single_output / "recipe.json")["options"]["seeds"] == [42]
        assert read_json(single_output / "recipe.json")["options"]["folds"] == [0]
        assert read_json(single_output / "run_summary.json")["final_encoder_created"]
        assert (single_output / "single" / "single_report.md").is_file()
        from .runtime import ForcePredictiveEncoder
        encoder = ForcePredictiveEncoder(single_output / "manifest.json", device="cpu")
        encoder.push([0., 0., 0.], 0.)
        assert encoder.predict_future().shape == (5, 3)
        print("Integration checks passed: multi-worker full-CV protocol, HDF loading, report, resume/rejection, and default single-run export/inference.")


if __name__ == "__main__":
    main()
