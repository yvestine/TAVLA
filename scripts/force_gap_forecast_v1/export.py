"""Export the exact runtime/adapter to the other simulation host."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil

from .common import read_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, checkpoint = read_manifest(args.manifest)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Choose a new export directory")
    package = output / "force_gap_forecast_v1"
    package.mkdir(parents=True)
    for name in ("__init__.py", "common.py", "model.py", "data.py", "metrics.py", "runtime.py", "sim_adapter.py", "capture.py", "serial_eval.py", "validate.py"):
        shutil.copy2(Path(__file__).parent / name, package / name)
    shutil.copy2(checkpoint, output / "encoder.pt")
    portable = copy.deepcopy(manifest)
    portable["checkpoint"] = "encoder.pt"
    for key in ("acceptance", "audit"):
        if key in manifest:
            shutil.copy2(args.manifest.resolve().parent / manifest[key], output / manifest[key])
    write_json(output / "manifest.json", portable)
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "packages/openpi-client/src/openpi_client", output / "openpi_client", ignore=shutil.ignore_patterns("__pycache__"))
    (output / "requirements.txt").write_text("numpy>=1.24\ntorch>=2.6,<3\nh5py>=3.8\nopencv-python>=4.8\nwebsockets>=13\nmsgpack>=1\ntyping_extensions>=4\n")
    shutil.copy2(Path(__file__).parent / "README.md", output / "README.md")
    print(output)


if __name__ == "__main__":
    main()
