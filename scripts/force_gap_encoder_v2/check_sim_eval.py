#!/usr/bin/env python3
"""Apply the shared closed-loop rollout gate to an external simulator JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=40)
    parser.add_argument("--minimum-successes", type=int, default=36)
    parser.add_argument("--xy-p95-limit-mm", type=float, default=3.0)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema") != "tavla_force_gap_rollout_v1":
        raise ValueError("report schema must be tavla_force_gap_rollout_v1")
    if args.expected_episodes < 1 or args.minimum_successes < 0:
        raise ValueError("episode and success thresholds must be nonnegative, with at least one episode")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} episode records, got {len(episodes) if isinstance(episodes, list) else 'invalid'}")

    ids = [int(item["trajectory_id"]) for item in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("trajectory_id values must be unique")
    missing = [item["trajectory_id"] for item in episodes if "success" not in item or "force_safety_ok" not in item or "xy_error_mm" not in item]
    if missing:
        raise ValueError(f"episode records are missing required fields: {missing}")

    xy_errors = []
    for item in episodes:
        if not isinstance(item["success"], bool) or not isinstance(item["force_safety_ok"], bool):
            raise ValueError(f"trajectory {item['trajectory_id']}: success and force_safety_ok must be JSON booleans")
        values = np.asarray(item["xy_error_mm"], dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"trajectory {item['trajectory_id']}: xy_error_mm must be a non-empty finite nonnegative list")
        xy_errors.append(values)
    pooled_xy = np.concatenate(xy_errors)
    success_count = sum(bool(item["success"]) for item in episodes)
    safety_violations = [int(item["trajectory_id"]) for item in episodes if not bool(item["force_safety_ok"])]
    xy_p95 = float(np.quantile(pooled_xy, 0.95))

    passed = (
        success_count >= args.minimum_successes
        and xy_p95 < args.xy_p95_limit_mm
        and not safety_violations
    )
    print(f"model: {report.get('model', 'unspecified')}")
    print(f"successes: {success_count}/{len(episodes)} (minimum {args.minimum_successes})")
    print(f"pooled XY error P95: {xy_p95:.3f} mm (limit < {args.xy_p95_limit_mm:.3f} mm)")
    print(f"force-protection violations: {safety_violations}")
    print("GATE: PASS" if passed else "GATE: FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
