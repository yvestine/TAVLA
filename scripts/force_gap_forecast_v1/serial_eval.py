"""Serial closed-loop evaluation. Simulator integration is an explicit callback.

runner(variant, host, port, seed, trace_path, encoder_manifest) returns
{depth_error_m, remaining_depth_m, xy_error_m, failure_reason}. remaining_depth_m
is signed (positive = insertion is too shallow). New-model runner uses TavlaAdapter
and EpisodeTrace; legacy variants retain their existing sensor/effort adapters.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

from .common import read_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Simulator callback module:function")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--existing-seeds", type=int, nargs="+", default=list(range(40)))
    parser.add_argument("--fresh-seeds", type=int, default=100)
    parser.add_argument("--fresh-start", type=int, default=10000)
    args = parser.parse_args()
    manifest, _ = read_manifest(args.manifest, require_ready=True)
    if not 100 <= args.fresh_seeds <= 200:
        parser.error("fresh-seeds must be 100..200")
    new_seeds = list(range(args.fresh_start, args.fresh_start + args.fresh_seeds))
    if set(new_seeds) & set(args.existing_seeds) or len(set(args.existing_seeds)) != len(args.existing_seeds):
        parser.error("Repeated / overlapping seeds")
    module, function = args.runner.split(":")
    runner = getattr(importlib.import_module(module), function)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for cohort, seeds in (("existing", args.existing_seeds), ("fresh", new_seeds)):
        for variant, port in (("raw", 8005), ("legacy_encoder", 8001), ("force_forecast", 8006)):
            for seed in seeds:
                trace = output / cohort / variant / f"seed{seed}.npz"
                trace.parent.mkdir(parents=True, exist_ok=True)
                row = runner(variant=variant, host=args.host, port=port, seed=seed, trace_path=trace,
                             encoder_manifest=args.manifest)
                depth, xy = float(row["depth_error_m"]), float(row["xy_error_m"])
                remaining = float(row["remaining_depth_m"])
                if not np.isfinite([depth, xy, remaining]).all() or min(depth, xy) < 0:
                    raise ValueError("Final depth/XY errors must be finite nonnegative metres")
                if not np.isclose(abs(remaining), depth, rtol=1e-5, atol=1e-6) or "failure_reason" not in row:
                    raise ValueError("Report signed remaining depth and failure reason; distinguish undershoot from overshoot")
                if not trace.is_file():
                    raise ValueError("Simulator must write the executed-action/force trace")
                if variant == "force_forecast":
                    with np.load(trace, allow_pickle=False) as data:
                        required = {"timestamp_s", "input_force_n", "q_rad", "dq_rad_s", "z", "future_real_force_n", "executed_action", "encoder_sha256"}
                        if not required.issubset(data.files) or not np.all(data["encoder_sha256"] == manifest["encoder_sha256"]):
                            raise ValueError("New-policy trace missing required signals / encoder SHA")
                        n = len(data["timestamp_s"])
                        if n < 1 or not np.isfinite(data["timestamp_s"]).all() or np.any(np.diff(data["timestamp_s"]) <= 0):
                            raise ValueError("Nonempty increasing trace timestamps required")
                        for key, shape in (("input_force_n", (n, 3)), ("q_rad", (n, 7)), ("dq_rad_s", (n, 7)),
                                           ("z", (n, 32)), ("future_real_force_n", (n, 5, 3)), ("executed_action", (n, 8))):
                            if data[key].shape != shape or not np.isfinite(data[key]).all():
                                raise ValueError(f"Invalid trace {key}: expected finite {shape}")
                row = {**row, "cohort": cohort, "variant": variant, "port": port, "seed": seed,
                       "success_3mm": depth <= .003 and xy < .003,
                       "depth_shortfall": remaining > .003, "trace": str(trace)}
                results.append(row)
                write_json(output / "episodes.json", results)
    summary = {}
    for cohort in ("existing", "fresh"):
        summary[cohort] = {}
        for variant in ("raw", "legacy_encoder", "force_forecast"):
            rows = [row for row in results if row["cohort"] == cohort and row["variant"] == variant]
            summary[cohort][variant] = {"episodes": len(rows), "success_rate_3mm": float(np.mean([r["success_3mm"] for r in rows])),
                                       "depth_shortfalls": sum(r["depth_shortfall"] for r in rows)}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
