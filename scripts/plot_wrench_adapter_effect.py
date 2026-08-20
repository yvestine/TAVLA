"""Plot source/target/adapted wrench distributions for an adapter audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from openpi.shared.wrench_adapter import load_adapter


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load(path: Path, pattern: str, h5_key: str | None) -> np.ndarray:
    values = []
    for item in sorted(path.glob(pattern)):
        if item.suffix.lower() in {".h5", ".hdf5"}:
            if h5_key is None:
                raise ValueError(f"--h5-key is required for {item}")
            with h5py.File(item, "r") as h5:
                value = np.asarray(h5[h5_key][:], dtype=np.float32)
        else:
            value = np.asarray(np.genfromtxt(item, delimiter=",", skip_header=1), dtype=np.float32)
            if value.ndim == 1:
                value = value.reshape(1, -1)
        if value.ndim != 2 or value.shape[1] != 6 or not np.isfinite(value).all():
            raise ValueError(f"Invalid wrench array in {item}: {value.shape}")
        values.append(value)
    if not values:
        raise FileNotFoundError(f"No files matched {path / pattern}")
    return np.concatenate(values)


def _norm(values: np.ndarray, first: int, last: int) -> np.ndarray:
    return np.linalg.norm(values[:, first:last], axis=1)


def _cdf(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(values), bins, side="right") / max(len(values), 1)


def _summary(values: np.ndarray) -> dict[str, object]:
    return {
        "frames": int(len(values)),
        "mean": values.mean(axis=0).tolist(),
        "p95_abs": np.quantile(np.abs(values), 0.95, axis=0).tolist(),
        "force_norm_median": float(np.median(_norm(values, 0, 3))),
        "force_norm_p95": float(np.quantile(_norm(values, 0, 3), 0.95)),
        "torque_norm_median": float(np.median(_norm(values, 3, 6))),
        "torque_norm_p95": float(np.quantile(_norm(values, 3, 6), 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-pattern", required=True)
    parser.add_argument("--source-h5-key")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--target-pattern", required=True)
    parser.add_argument("--target-h5-key")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = _load(args.source_dir, args.source_pattern, args.source_h5_key)
    target = _load(args.target_dir, args.target_pattern, args.target_h5_key)
    adapter = load_adapter(args.adapter)
    adapted = adapter.transform_numpy(source)

    width, height = 1800, 1150
    image = Image.new("RGB", (width, height), (244, 246, 249))
    draw = ImageDraw.Draw(image)
    title = _font(30)
    subtitle = _font(18)
    text = _font(16)
    draw.text((35, 25), "Wrench adapter effect audit", fill=(15, 15, 15), font=title)
    draw.text(
        (38, 65),
        f"source={len(source)} frames   target={len(target)} frames   adapter={args.adapter.name}",
        fill=(80, 80, 80),
        font=subtitle,
    )

    # Per-channel P95 bars.
    x0, y0, x1, y1 = 50, 125, 870, 555
    draw.rounded_rectangle((30, 105, 890, 590), radius=12, outline=(180, 180, 180), fill=(252, 252, 252), width=2)
    draw.text((52, 120), "Absolute P95 by channel", fill=(25, 25, 25), font=subtitle)
    draw.line((x0, y1, x1, y1), fill=(80, 80, 80), width=2)
    draw.line((x0, y0, x0, y1), fill=(80, 80, 80), width=2)
    p95_values = np.stack((np.quantile(np.abs(source), 0.95, axis=0), np.quantile(np.abs(adapted), 0.95, axis=0), np.quantile(np.abs(target), 0.95, axis=0)))
    for i, channel in enumerate(CHANNELS):
        maximum = max(float(p95_values[:, i].max()), 1e-6)
        center = x0 + (i + 0.5) * (x1 - x0) / 6
        bar_width = 24
        for j, color in enumerate(((232, 126, 4), (114, 73, 168), (33, 104, 180))):
            value = float(p95_values[j, i])
            top = y1 - (y1 - y0) * value / maximum
            left = center + (j - 1) * (bar_width + 3)
            draw.rectangle((left, top, left + bar_width, y1), fill=color)
        draw.text((center - 18, y1 + 10), channel, fill=(30, 30, 30), font=text)
        draw.text((center - 35, y0 - 24), f"{maximum:.2g}", fill=(70, 70, 70), font=text)
    draw.text((x0 + 5, y0 + 8), "orange source / purple adapted / blue target", fill=(70, 70, 70), font=text)

    # Norm CDFs.
    x0, y0, x1, y1 = 930, 125, 1745, 555
    draw.rounded_rectangle((910, 105, 1770, 590), radius=12, outline=(180, 180, 180), fill=(252, 252, 252), width=2)
    draw.text((932, 120), "Force and torque norm CDF", fill=(25, 25, 25), font=subtitle)
    for top, bottom, first, last, label in ((165, 335, 0, 3, "force norm"), (380, 550, 3, 6, "torque norm")):
        draw.line((x0, bottom, x1, bottom), fill=(80, 80, 80), width=2)
        draw.line((x0, top, x0, bottom), fill=(80, 80, 80), width=2)
        arrays = (_norm(source, first, last), _norm(adapted, first, last), _norm(target, first, last))
        high = max(float(np.quantile(np.concatenate(arrays), 0.995)), 1e-6)
        bins = np.linspace(0, high, 100)
        for values, color in zip(arrays, ((232, 126, 4), (114, 73, 168), (33, 104, 180)), strict=True):
            points = []
            cdf = _cdf(np.clip(values, 0, high), bins)
            for x_value, y_value in zip(bins, cdf, strict=True):
                points.append((x0 + int((x_value / high) * (x1 - x0)), bottom - int(y_value * (bottom - top))))
            draw.line(points, fill=color, width=3)
        draw.text((x0 + 8, top + 8), label, fill=(70, 70, 70), font=text)
        draw.text((x1 - 70, bottom + 8), f"{high:.2g}", fill=(70, 70, 70), font=text)
    draw.text((x0 + 10, y0 + 8), "orange source / purple adapted / blue target", fill=(70, 70, 70), font=text)

    # Summary table.
    draw.rounded_rectangle((30, 625, 1770, 1095), radius=12, outline=(180, 180, 180), fill=(252, 252, 252), width=2)
    draw.text((52, 645), "Numerical summary", fill=(25, 25, 25), font=subtitle)
    summaries = {"source": _summary(source), "adapted": _summary(adapted), "target": _summary(target)}
    rows = (
        ("force norm median", "force_norm_median"),
        ("force norm P95", "force_norm_p95"),
        ("torque norm median", "torque_norm_median"),
        ("torque norm P95", "torque_norm_p95"),
    )
    draw.text((70, 700), "metric", fill=(60, 60, 60), font=text)
    for index, name in enumerate(summaries):
        draw.text((420 + index * 320, 700), name, fill=(60, 60, 60), font=text)
    for row_index, (label, key) in enumerate(rows):
        yy = 745 + row_index * 48
        draw.text((70, yy), label, fill=(45, 45, 45), font=text)
        for index, name in enumerate(summaries):
            draw.text((420 + index * 320, yy), f"{summaries[name][key]:.3f}", fill=(15, 15, 15), font=text)
    draw.text((70, 970), "This plot compares marginal distributions only; it does not validate frame-level contact correspondence.", fill=(150, 65, 0), font=text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
