#!/usr/bin/env python3
"""Plot loss curves from scripts/train.py terminal logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


STEP_RE = re.compile(r"Step\s+(\d+):\s+(.*)")
METRIC_RE = re.compile(r"([A-Za-z0-9_./-]+)=([-+0-9.eE]+)")


def parse_log(path: Path) -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {"step": []}
    for line in path.read_text(errors="ignore").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        metrics = dict(METRIC_RE.findall(match.group(2)))
        if not metrics:
            continue
        curves["step"].append(step)
        for name, value in metrics.items():
            curves.setdefault(name, []).append(float(value))
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, default=Path("train_curves.png"))
    args = parser.parse_args()

    curves = parse_log(args.log)
    steps = curves.pop("step", [])
    if not steps:
        raise SystemExit(f"No 'Step ... loss=...' lines found in {args.log}")

    import matplotlib.pyplot as plt

    names = [name for name, values in curves.items() if len(values) == len(steps)]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 3 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names, strict=True):
        ax.plot(steps, curves[name], linewidth=1.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("step")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
