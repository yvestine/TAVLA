"""CPU artifact acceptance: causality, streaming equality, pulse bypass, latency."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .common import write_json
from .data import Episode, segment_limits
from .metrics import predict_episode
from .runtime import ForceGapEncoder


def validate(manifest, count=64):
    rng = np.random.default_rng(123)
    force = rng.normal(0, 2, (count, 3)).astype(np.float32)
    force[count // 2, 2] += 20
    q = rng.normal(0, .05, (count, 7)).astype(np.float32)
    dq = rng.normal(0, .1, (count, 7)).astype(np.float32)
    timestamps = np.arange(count, dtype=np.float64) * .1
    starts, ends = segment_limits(timestamps)
    ep = Episode(0, force, q, dq, timestamps, np.full(count, "unannotated", object), starts, ends)
    result = {}
    for domain in ("real", "sim"):
        runtime = ForceGapEncoder(manifest, domain)
        batch = runtime.batch(force, q, dq, timestamps)
        runtime.reset()
        elapsed, stream = [], []
        for i in range(count):
            begin = time.perf_counter()
            stream.append(runtime.push(force[i], q[i], dq[i], timestamps[i]))
            elapsed.append((time.perf_counter() - begin) * 1000)
        error = float(np.max(np.abs(batch - np.stack(stream))))
        if error > 1e-5:
            raise AssertionError("Batch/stream parity failed")
        training_prediction = predict_episode(runtime.model, ep, runtime.stats, domain == "sim", batch_size=1)["z"]
        training_error = float(np.max(np.abs(batch - training_prediction)))
        if training_error > 1e-5:
            raise AssertionError(f"Train/runtime preprocessing parity failed: {training_error}")
        future = runtime.predict_future()
        if future.shape != (5, 3) or future.dtype != np.float32 or not np.isfinite(future).all():
            raise AssertionError("Future force contract failed")
        prefix = runtime.batch(force[:count // 2], q[:count // 2], dq[:count // 2], timestamps[:count // 2])
        if not np.array_equal(prefix, batch[:count // 2]):
            raise AssertionError("Future observations changed earlier features")
        if domain == "real":
            expected_delta = np.diff(force, axis=0, prepend=force[:1])
            if not np.array_equal(batch[:, :3], force) or not np.array_equal(batch[:, 3:6], expected_delta):
                raise AssertionError("Raw Real force/delta bypass lost information")
        runtime.batch(force, q, dq, timestamps)
        try:
            runtime.push(force[-1], q[-1], dq[-1], timestamps[-1])
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate timestamp accepted")
        after_gap = runtime.push(force[-1], q[-1], dq[-1], timestamps[-1] + 1.)
        if not np.array_equal(after_gap[3:6], np.zeros(3, np.float32)):
            raise AssertionError("Gap failed to reset canonical delta")
        p95 = float(np.percentile(elapsed[4:], 95))
        result[domain] = {"batch_stream_max_abs": error, "train_runtime_max_abs": training_error,
                          "cpu_threads": 1, "cpu_p95_ms": p95, "latency_passed": p95 <= 20,
                          "causality_passed": True, "gap_reset_passed": True}
    result["passed"] = all(row["latency_passed"] for row in result.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.manifest)
    write_json(args.output, result)
    print(result)
    if not result["passed"]:
        raise SystemExit("Latency gate failed on this host; report retained")


if __name__ == "__main__":
    main()
