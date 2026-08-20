"""Visualize unpaired wrench alignment methods.

This standalone diagnostic imports the comparison module but does not alter
any adapter, dataset, training config, or server.  The source and target
streams are unpaired; all plots therefore compare distributions, not
frame-to-frame trajectories.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.shared.wrench_adapter import load_adapter
from compare_unpaired_wrench_alignment import _apply_copula
from compare_unpaired_wrench_alignment import _apply_quantile
from compare_unpaired_wrench_alignment import _fit_copula
from compare_unpaired_wrench_alignment import _fit_quantile
from compare_unpaired_wrench_alignment import _fit_affine
from compare_unpaired_wrench_alignment import _apply_affine
from compare_unpaired_wrench_alignment import concat
from compare_unpaired_wrench_alignment import load_episodes
from compare_unpaired_wrench_alignment import split_episodes


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
COLORS = {
    "sim_raw": "#e67e22",
    "affine": "#8e44ad",
    "copula_block": "#16a085",
    "real": "#2471a3",
}


def empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(values), grid, side="right") / max(len(values), 1)


def corr(values: np.ndarray) -> np.ndarray:
    return np.corrcoef(values.T)


def plot_distribution(ax, arrays: dict[str, np.ndarray], channel: int) -> None:
    all_values = np.concatenate([value[:, channel] for value in arrays.values()])
    lo = float(np.quantile(all_values, 0.005))
    hi = float(np.quantile(all_values, 0.995))
    if hi <= lo:
        hi = lo + 1.0
    grid = np.linspace(lo, hi, 400)
    for name, value in arrays.items():
        ax.plot(grid, empirical_cdf(value[:, channel], grid), label=name, color=COLORS[name], linewidth=2)
    ax.set_title(CHANNELS[channel])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_xlabel("N / N·m")
    if channel == 0:
        ax.set_ylabel("empirical CDF")


def plot_norm_cdf(ax, arrays: dict[str, np.ndarray], first: int, last: int, title: str) -> None:
    all_values = np.concatenate([np.linalg.norm(value[:, first:last], axis=1) for value in arrays.values()])
    grid = np.linspace(0, float(np.quantile(all_values, 0.995)), 400)
    for name, value in arrays.items():
        norm = np.linalg.norm(value[:, first:last], axis=1)
        ax.plot(grid, empirical_cdf(norm, grid), label=name, color=COLORS[name], linewidth=2)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_xlabel("norm (N / N·m)")
    ax.set_ylabel("empirical CDF")


def plot_heatmap(ax, value: np.ndarray, title: str) -> None:
    image = ax.imshow(value, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(range(6), CHANNELS, rotation=45, ha="right")
    ax.set_yticks(range(6), CHANNELS)
    for row in range(6):
        for col in range(6):
            ax.text(col, row, f"{value[row, col]:.2f}", ha="center", va="center", fontsize=7)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data-sim-wrench-final-hdf5"))
    parser.add_argument("--source-pattern", default="episode_*/data.h5")
    parser.add_argument("--source-h5-key", default="decision/obs/state/wrench_final")
    parser.add_argument("--target-dir", type=Path, default=Path("data"))
    parser.add_argument("--target-pattern", default="traj_*/data.h5")
    parser.add_argument("--target-h5-key", default="obs/state/ee_wrench_base")
    parser.add_argument("--mlp-adapter", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/wrench_alignment/copula_block_comparison.png"))
    args = parser.parse_args()

    source = load_episodes(args.source_dir, args.source_pattern, args.source_h5_key)
    target = load_episodes(args.target_dir, args.target_pattern, args.target_h5_key)
    source_train, source_test = split_episodes(source, 0.20, args.seed)
    target_train, target_test = split_episodes(target, 0.20, args.seed + 1)
    source_train_values = concat(source_train)
    target_train_values = concat(target_train)
    source_test_values = concat(source_test)
    target_test_values = concat(target_test)

    affine_fit = _fit_affine(source_train_values, target_train_values)
    copula_fit = _fit_copula(source_train_values, target_train_values, shrinkage=0.10)
    arrays = {
        "sim_raw": source_test_values,
        "affine": _apply_affine(source_test_values, affine_fit),
        "copula_block": _apply_copula(source_test_values, copula_fit),
        "real": target_test_values,
    }
    if args.mlp_adapter:
        mlp = load_adapter(args.mlp_adapter)
        arrays["polyfit_mlp"] = mlp.transform_numpy(source_test_values)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=(1.0, 1.0, 1.0, 1.15))
    fig.suptitle(
        "Unpaired sim-real wrench alignment: held-out episodes\n"
        "orange=raw sim, purple=robust affine, green=copula-block, blue=real; no frame-wise pairing",
        fontsize=15,
    )

    for channel in range(6):
        row = channel // 3
        col = channel % 3
        plot_distribution(fig.add_subplot(grid[row, col]), arrays, channel)
    legend_ax = fig.add_subplot(grid[0:2, 3])
    legend_ax.axis("off")
    handles = [plt.Line2D([], [], color=COLORS[name], linewidth=3, label=name) for name in arrays]
    legend_ax.legend(handles=handles, loc="center", frameon=True, title="held-out streams")
    legend_ax.text(
        0.02,
        0.08,
        "The curves should overlap if marginal distributions match.\n"
        "This does not prove that a particular simulated frame\n"
        "corresponds to a particular real frame.",
        transform=legend_ax.transAxes,
        va="bottom",
        wrap=True,
    )

    plot_norm_cdf(fig.add_subplot(grid[2, 0]), arrays, 0, 3, "force norm CDF")
    plot_norm_cdf(fig.add_subplot(grid[2, 1]), arrays, 3, 6, "torque norm CDF")
    ax = fig.add_subplot(grid[2, 2])
    names = list(arrays)
    x = np.arange(len(names))
    width = 0.18
    for index, channel in enumerate((0, 1, 2, 3, 4, 5)):
        values = [np.quantile(np.abs(arrays[name][:, channel]), 0.95) for name in names]
        ax.bar(x + (index - 2.5) * width, values, width, label=CHANNELS[channel])
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_title("absolute P95 by channel")
    ax.set_ylabel("N / N·m")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    plot_heatmap(fig.add_subplot(grid[2, 3]), corr(arrays["real"]), "real correlation")

    plot_heatmap(fig.add_subplot(grid[3, 0]), corr(arrays["sim_raw"]), "raw sim correlation")
    plot_heatmap(fig.add_subplot(grid[3, 1]), corr(arrays["affine"]), "affine correlation")
    plot_heatmap(fig.add_subplot(grid[3, 2]), corr(arrays["copula_block"]), "copula-block correlation")
    ax = fig.add_subplot(grid[3, 3])
    ax.axis("off")
    lines = [
        f"held-out sim frames: {len(source_test_values)}",
        f"held-out real frames: {len(target_test_values)}",
        "",
        "copula-block operation:",
        "1. empirical marginal quantiles",
        "2. Gaussian-rank transform",
        "3. force/torque block covariance alignment",
        "4. map back to real marginal units",
        "",
        "Use this as a distribution diagnostic,",
        "not a physical force calibration proof.",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
