"""Dependency-light Pillow visualization for unpaired wrench methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openpi.shared.wrench_adapter import load_adapter
from compare_unpaired_wrench_alignment import _apply_affine
from compare_unpaired_wrench_alignment import _apply_copula
from compare_unpaired_wrench_alignment import _fit_affine
from compare_unpaired_wrench_alignment import _fit_copula
from compare_unpaired_wrench_alignment import concat
from compare_unpaired_wrench_alignment import load_episodes
from compare_unpaired_wrench_alignment import split_episodes


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
COLORS = {
    "sim_raw": (230, 126, 34),
    "affine": (142, 68, 173),
    "copula_block": (22, 160, 133),
    "real": (36, 113, 163),
}
BG = (244, 246, 249)
PANEL = (255, 255, 255)
TEXT = (30, 30, 30)
MUTED = (90, 90, 90)


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


TITLE = font(28)
SUBTITLE = font(18)
BODY = font(14)
SMALL = font(11)


def cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(values), grid, side="right") / max(len(values), 1)


def line_plot(draw, box, series, title, xlabel, ylabel, nonnegative=False):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, outline=(190, 190, 190), fill=PANEL, width=2)
    draw.text((x0 + 12, y0 + 10), title, fill=TEXT, font=SUBTITLE)
    px0, py0, px1, py1 = x0 + 48, y0 + 43, x1 - 16, y1 - 32
    draw.line((px0, py1, px1, py1), fill=(80, 80, 80), width=2)
    draw.line((px0, py0, px0, py1), fill=(80, 80, 80), width=2)
    all_values = np.concatenate(list(series.values()))
    if nonnegative:
        lo = 0.0
    else:
        lo = float(np.quantile(all_values, 0.005))
    hi = float(np.quantile(all_values, 0.995))
    if hi <= lo:
        hi = lo + 1.0
    grid = np.linspace(lo, hi, 300)
    for name, values in series.items():
        yy = cdf(values, grid)
        points = []
        for xv, yv in zip(grid, yy, strict=True):
            x = int(px0 + (xv - lo) / (hi - lo) * (px1 - px0))
            y = int(py1 - yv * (py1 - py0))
            points.append((x, y))
        draw.line(points, fill=COLORS[name], width=3)
    draw.text((px0, py1 + 8), f"{lo:.2g}", fill=MUTED, font=SMALL)
    draw.text((px1 - 38, py1 + 8), f"{hi:.2g}", fill=MUTED, font=SMALL)
    draw.text((x0 + 5, py0 - 4), "1", fill=MUTED, font=SMALL)
    draw.text((x0 + 10, py1 - 4), "0", fill=MUTED, font=SMALL)
    draw.text((px1 - 100, py1 + 8), xlabel, fill=MUTED, font=SMALL)
    draw.text((x0 + 7, py0 + 10), ylabel, fill=MUTED, font=SMALL)


def heatmap(draw, box, values, title):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, outline=(190, 190, 190), fill=PANEL, width=2)
    draw.text((x0 + 12, y0 + 10), title, fill=TEXT, font=SUBTITLE)
    left, top = x0 + 60, y0 + 50
    size = min((x1 - left - 20) // 6, (y1 - top - 20) // 6)
    for row in range(6):
        for col in range(6):
            value = float(np.clip(values[row, col], -1, 1))
            # blue (-1) -> white (0) -> red (+1)
            if value >= 0:
                color = (255, int(245 - 150 * value), int(245 - 150 * value))
            else:
                color = (int(245 - 150 * -value), int(245 - 150 * -value), 255)
            xx = left + col * size
            yy = top + row * size
            draw.rectangle((xx, yy, xx + size, yy + size), fill=color, outline=(220, 220, 220))
            draw.text((xx + size // 2, yy + size // 2), f"{value:.1f}", fill=TEXT, font=font(9), anchor="mm")
    for i, name in enumerate(CHANNELS):
        draw.text((left + i * size + size // 2, top - 8), name, fill=MUTED, font=font(9), anchor="ms")
        draw.text((left - 8, top + i * size + size // 2), name, fill=MUTED, font=font(9), anchor="rs")


def correlation(values: np.ndarray) -> np.ndarray:
    return np.corrcoef(values.T)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data-sim-wrench-final-hdf5"))
    parser.add_argument("--source-pattern", default="episode_*/data.h5")
    parser.add_argument("--source-h5-key", default="decision/obs/state/wrench_final")
    parser.add_argument("--target-dir", type=Path, default=Path("data"))
    parser.add_argument("--target-pattern", default="traj_*/data.h5")
    parser.add_argument("--target-h5-key", default="obs/state/ee_wrench_base")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/wrench_alignment/copula_block_comparison.png"))
    args = parser.parse_args()

    source = load_episodes(args.source_dir, args.source_pattern, args.source_h5_key)
    target = load_episodes(args.target_dir, args.target_pattern, args.target_h5_key)
    source_train, source_test = split_episodes(source, 0.20, 7)
    target_train, target_test = split_episodes(target, 0.20, 8)
    source_train_values = concat(source_train)
    target_train_values = concat(target_train)
    source_test_values = concat(source_test)
    target_test_values = concat(target_test)
    affine = _fit_affine(source_train_values, target_train_values)
    copula = _fit_copula(source_train_values, target_train_values, shrinkage=0.10)
    arrays = {
        "sim_raw": source_test_values,
        "affine": _apply_affine(source_test_values, affine),
        "copula_block": _apply_copula(source_test_values, copula),
        "real": target_test_values,
    }

    width, height = 1900, 1700
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((35, 22), "Unpaired sim-real wrench alignment", fill=TEXT, font=TITLE)
    draw.text(
        (38, 62),
        f"Held-out comparison: sim {len(source_test_values)} frames / real {len(target_test_values)} frames; no frame-wise pairing",
        fill=MUTED,
        font=BODY,
    )

    # Six marginal CDFs.
    for channel in range(6):
        row, col = divmod(channel, 3)
        x0 = 30 + col * 420
        y0 = 105 + row * 260
        series = {name: value[:, channel] for name, value in arrays.items()}
        line_plot(draw, (x0, y0, x0 + 395, y0 + 230), series, CHANNELS[channel], "N / N·m", "CDF")

    # Legend / interpretation.
    box = (1300, 105, 1870, 335)
    draw.rounded_rectangle(box, radius=10, outline=(190, 190, 190), fill=PANEL, width=2)
    draw.text((box[0] + 15, box[1] + 15), "Interpretation", fill=TEXT, font=SUBTITLE)
    legend_y = box[1] + 55
    for name, color in COLORS.items():
        draw.line((box[0] + 18, legend_y + 8, box[0] + 55, legend_y + 8), fill=color, width=4)
        draw.text((box[0] + 70, legend_y), name, fill=TEXT, font=BODY)
        legend_y += 30
    draw.text(
        (box[0] + 18, legend_y + 8),
        "Curves closer together = better marginal alignment.\nThis is not proof of physical contact correspondence.",
        fill=MUTED,
        font=BODY,
    )

    # Norm CDFs and P95 bars.
    force_series = {name: np.linalg.norm(value[:, :3], axis=1) for name, value in arrays.items()}
    torque_series = {name: np.linalg.norm(value[:, 3:], axis=1) for name, value in arrays.items()}
    line_plot(draw, (30, 645, 475, 900), force_series, "force norm CDF", "N", "CDF", nonnegative=True)
    line_plot(draw, (500, 645, 945, 900), torque_series, "torque norm CDF", "N·m", "CDF", nonnegative=True)

    bar_box = (970, 645, 1300, 900)
    draw.rounded_rectangle(bar_box, radius=10, outline=(190, 190, 190), fill=PANEL, width=2)
    draw.text((bar_box[0] + 12, bar_box[1] + 10), "absolute P95", fill=TEXT, font=SUBTITLE)
    channels = range(6)
    max_value = max(float(np.quantile(np.abs(value[:, i]), 0.95)) for value in arrays.values() for i in channels)
    for index, name in enumerate(arrays):
        xx = bar_box[0] + 20 + index * 68
        draw.text((xx + 18, bar_box[3] - 29), name[:6], fill=COLORS[name], font=font(10), anchor="ms")
        for channel in channels:
            value = float(np.quantile(np.abs(arrays[name][:, channel]), 0.95))
            bar_h = int((value / max_value) * 145)
            yy = bar_box[3] - 55 - bar_h
            draw.rectangle((xx + channel * 7, yy, xx + channel * 7 + 5, bar_box[3] - 55), fill=COLORS[name])
    draw.text((bar_box[0] + 10, bar_box[1] + 48), "channels: Fx Fy Fz Tx Ty Tz", fill=MUTED, font=SMALL)

    # Correlation heatmaps.
    heatmap(draw, (1330, 645, 1610, 900), correlation(arrays["sim_raw"]), "raw sim correlation")
    heatmap(draw, (1630, 645, 1880, 900), correlation(arrays["copula_block"]), "copula correlation")

    # Summary panel.
    summary_box = (30, 950, 1880, 1640)
    draw.rounded_rectangle(summary_box, radius=10, outline=(190, 190, 190), fill=PANEL, width=2)
    draw.text((summary_box[0] + 18, summary_box[1] + 15), "Numerical summary on the same held-out split", fill=TEXT, font=SUBTITLE)
    names = list(arrays)
    metrics = [
        ("force norm median", lambda v: np.median(np.linalg.norm(v[:, :3], axis=1))),
        ("force norm P95", lambda v: np.quantile(np.linalg.norm(v[:, :3], axis=1), 0.95)),
        ("torque norm median", lambda v: np.median(np.linalg.norm(v[:, 3:], axis=1))),
        ("torque norm P95", lambda v: np.quantile(np.linalg.norm(v[:, 3:], axis=1), 0.95)),
        ("mean Fx/Fy/Fz", lambda v: np.mean(v[:, :3])),
        ("mean Tx/Ty/Tz", lambda v: np.mean(v[:, 3:])),
    ]
    table_x = [summary_box[0] + 20, summary_box[0] + 300, summary_box[0] + 590, summary_box[0] + 880, summary_box[0] + 1170]
    draw.text((table_x[0], summary_box[1] + 58), "metric", fill=MUTED, font=BODY)
    for x, name in zip(table_x[1:], names, strict=True):
        draw.text((x, summary_box[1] + 58), name, fill=COLORS[name], font=BODY)
    for row, (label, fn) in enumerate(metrics):
        yy = summary_box[1] + 92 + row * 38
        draw.text((table_x[0], yy), label, fill=TEXT, font=BODY)
        for x, name in zip(table_x[1:], names, strict=True):
            draw.text((x, yy), f"{fn(arrays[name]):.3f}", fill=TEXT, font=BODY)
    draw.text((summary_box[0] + 20, summary_box[1] + 350), "copula-block transform", fill=TEXT, font=SUBTITLE)
    explanation = [
        "1. Fit empirical marginal quantiles on training episodes.",
        "2. Convert each channel to Gaussian-rank coordinates.",
        "3. Align force covariance and torque covariance separately.",
        "4. Map back to real marginal units.",
        "",
        "The green curves/values are the candidate final method.",
        "It improves distribution alignment but is not a paired physical calibration.",
    ]
    for row, line in enumerate(explanation):
        draw.text((summary_box[0] + 35, summary_box[1] + 390 + row * 27), line, fill=MUTED if row >= 5 else TEXT, font=BODY)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
