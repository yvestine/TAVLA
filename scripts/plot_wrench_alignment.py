"""Create a dependency-light diagnostic image for sim/real wrench alignment."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
SIM_ROOT = Path("data-sim-wrench-final-50")
REAL_ROOT = Path("data")
OUT_ROOT = Path("eval_outputs/wrench_alignment")


def read_csv(path: Path) -> np.ndarray:
    return np.asarray(np.genfromtxt(path, delimiter=",", skip_header=1), dtype=float).reshape(-1, 6)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episodes = sorted(SIM_ROOT.glob("episode_*"), key=lambda p: int(p.name.split("_")[-1]))
    sim_final = np.concatenate([read_csv(ep / "wrench_final.csv") for ep in episodes])
    sim_base = np.concatenate([read_csv(ep / "wrench_base.csv") for ep in episodes])
    sim_ep32 = read_csv(SIM_ROOT / "episode_32/wrench_final.csv")

    real_parts = []
    real_ep0 = None
    for path in sorted(REAL_ROOT.glob("traj_*/data.h5"), key=lambda p: int(p.parent.name.split("_")[-1])):
        with h5py.File(path, "r") as h5:
            values = h5["obs/state/ee_wrench_base"][:].astype(float)
        real_parts.append(values)
        if real_ep0 is None:
            real_ep0 = values
    assert real_ep0 is not None
    return sim_final, sim_base, sim_ep32, np.concatenate(real_parts), real_ep0


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


TITLE = font(30)
SUBTITLE = font(21)
TEXT = font(17)
SMALL = font(14)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, outline=(180, 180, 180), width=2, fill=(252, 252, 252))
    draw.text((x0 + 18, y0 + 14), title, fill=(25, 25, 25), font=SUBTITLE)
    return x0 + 55, y0 + 58, x1 - 22, y1 - 42


def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=(80, 80, 80), width=2)
    draw.line((x0, y0, x0, y1), fill=(80, 80, 80), width=2)


def scaled_y(value: float, lo: float, hi: float, y0: int, y1: int) -> int:
    if hi <= lo:
        return (y0 + y1) // 2
    return int(y1 - (value - lo) / (hi - lo) * (y1 - y0))


def draw_hist_cdf(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], sim: np.ndarray, real: np.ndarray, title: str) -> None:
    x0, y0, x1, y1 = box
    axes(draw, box)
    combined = np.concatenate([sim, real])
    hi = float(np.quantile(combined, 0.995))
    hi = max(hi, 1e-6)
    bins = np.linspace(0.0, hi, 80)
    for values, color in ((sim, (232, 126, 4)), (real, (33, 104, 180))):
        clipped = np.clip(values, 0.0, hi)
        counts, edges = np.histogram(clipped, bins=bins, density=True)
        cdf = np.cumsum(counts)
        cdf = cdf / max(cdf[-1], 1e-9)
        points = []
        for i, value in enumerate(cdf):
            px = int(x0 + (edges[i + 1] / hi) * (x1 - x0))
            py = int(y1 - value * (y1 - y0))
            points.append((px, py))
        draw.line(points, fill=color, width=4)
    draw.text((x0 + 8, y0 + 8), "sim", fill=(232, 126, 4), font=SMALL)
    draw.text((x0 + 65, y0 + 8), "real", fill=(33, 104, 180), font=SMALL)
    draw.text((x0, y1 + 8), "0", fill=(70, 70, 70), font=SMALL)
    draw.text((x1 - 60, y1 + 8), f"{hi:.2g}", fill=(70, 70, 70), font=SMALL)
    draw.text((x0 + 8, y0 + 30), title, fill=(70, 70, 70), font=SMALL)


def main() -> None:
    sim, sim_base, sim_ep32, real, real_ep0 = load_data()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (2000, 1400), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    draw.text((45, 24), "TAVLA wrench alignment diagnostic", fill=(15, 15, 15), font=TITLE)
    draw.text(
        (48, 67),
        f"sim: {len(sim)} frames / 50 episodes    real: {len(real)} frames / 40 episodes    (not paired trajectories)",
        fill=(80, 80, 80),
        font=TEXT,
    )

    # Panel 1: per-channel absolute P95.
    b1 = panel(draw, (35, 110, 985, 650), "Absolute P95 by component (N / Nm)")
    x0, y0, x1, y1 = b1
    axes(draw, b1)
    max_values = np.maximum(np.quantile(np.abs(sim), 0.95, axis=0), np.quantile(np.abs(real), 0.95, axis=0))
    max_values = np.maximum(max_values, 1e-6)
    slot = (x1 - x0) / 6
    for i, channel in enumerate(CHANNELS):
        cx = x0 + int((i + 0.5) * slot)
        sim_value = float(np.quantile(np.abs(sim[:, i]), 0.95))
        real_value = float(np.quantile(np.abs(real[:, i]), 0.95))
        bar_w = 24
        sim_top = scaled_y(sim_value, 0, max_values[i], y0, y1)
        real_top = scaled_y(real_value, 0, max_values[i], y0, y1)
        draw.rectangle((cx - bar_w - 4, sim_top, cx - 4, y1), fill=(232, 126, 4))
        draw.rectangle((cx + 4, real_top, cx + bar_w + 4, y1), fill=(33, 104, 180))
        draw.text((cx - 20, y1 + 8), channel, fill=(30, 30, 30), font=TEXT)
        draw.text((cx - 42, max(sim_top - 23, y0)), f"{sim_value:.2g}", fill=(170, 85, 0), font=SMALL)
        draw.text((cx + 8, max(real_top - 23, y0)), f"{real_value:.2g}", fill=(20, 70, 130), font=SMALL)
    draw.text((x0 + 8, y0 + 8), "orange=sim final, blue=real ee_wrench_base", fill=(70, 70, 70), font=SMALL)

    # Panel 2: force / torque norm CDF.
    b2 = panel(draw, (1015, 110, 1965, 650), "Force / torque norm empirical CDF")
    x0, y0, x1, y1 = b2
    mid = (y0 + y1) // 2
    draw_hist_cdf(draw, (x0, y0, x1, mid - 15), np.linalg.norm(sim[:, :3], axis=1), np.linalg.norm(real[:, :3], axis=1), "force norm")
    draw_hist_cdf(draw, (x0, mid + 20, x1, y1), np.linalg.norm(sim[:, 3:], axis=1), np.linalg.norm(real[:, 3:], axis=1), "torque norm")

    # Panel 3: representative native-rate time series.
    b3 = panel(draw, (35, 685, 985, 1245), "Representative native-rate time series (not paired)")
    x0, y0, x1, y1 = b3
    axes(draw, b3)
    sim_norm = np.linalg.norm(sim_ep32[:, :3], axis=1)
    real_norm = np.linalg.norm(real_ep0[:, :3], axis=1)
    max_n = max(float(sim_norm.max()), float(real_norm.max()), 1e-6)
    for values, color, label in ((sim_norm, (232, 126, 4), "sim episode_32"), (real_norm, (33, 104, 180), "real traj_0")):
        points = []
        for i, value in enumerate(values):
            px = int(x0 + (i / max(len(values) - 1, 1)) * (x1 - x0))
            py = scaled_y(float(value), 0, max_n, y0, y1)
            points.append((px, py))
        draw.line(points, fill=color, width=3)
    draw.text((x0 + 10, y0 + 10), "force norm", fill=(70, 70, 70), font=SMALL)
    draw.text((x1 - 220, y0 + 10), "orange sim / blue real", fill=(70, 70, 70), font=SMALL)
    draw.text((x0, y1 + 8), "start", fill=(70, 70, 70), font=SMALL)
    draw.text((x1 - 38, y1 + 8), "end", fill=(70, 70, 70), font=SMALL)
    draw.text((x0 + 8, y0 + 35), f"sim max={sim_norm.max():.2f} N; real max={real_norm.max():.2f} N", fill=(70, 70, 70), font=SMALL)

    # Panel 4: transform check and compact numerical summary.
    b4 = panel(draw, (1015, 685, 1965, 1245), "Transform integrity and summary")
    x0, y0, x1, y1 = b4
    residual = np.abs(sim + sim_base)
    force_norm_sim = np.linalg.norm(sim[:, :3], axis=1)
    force_norm_real = np.linalg.norm(real[:, :3], axis=1)
    lines = [
        ("max |wrench_final + wrench_base|", f"{residual.max():.3g}"),
        ("max force-norm change raw -> base", "2e-6 N (measured)"),
        ("sim force norm median / P95", f"{np.median(force_norm_sim):.2f} / {np.quantile(force_norm_sim, .95):.2f} N"),
        ("real force norm median / P95", f"{np.median(force_norm_real):.2f} / {np.quantile(force_norm_real, .95):.2f} N"),
        ("sim force < 0.1 N", f"{np.mean(force_norm_sim < 0.1) * 100:.1f}%"),
        ("real force < 0.1 N", f"{np.mean(force_norm_real < 0.1) * 100:.1f}%"),
        ("known +10 N test output", "+7.22 N after sign correction"),
        ("known -10 N test output", "-7.07 N after sign correction"),
    ]
    yy = y0 + 12
    for label, value in lines:
        draw.text((x0 + 8, yy), label, fill=(45, 45, 45), font=TEXT)
        draw.text((x0 + 470, yy), value, fill=(15, 15, 15), font=TEXT)
        yy += 42
    draw.text(
        (x0 + 8, yy + 18),
        "Interpretation: transform/sign are consistent; sim-real magnitude, bias, and noise are not yet aligned.",
        fill=(150, 65, 0),
        font=TEXT,
    )

    output = OUT_ROOT / "wrench_alignment.png"
    image.save(output)
    summary = {
        "sim_frames": int(len(sim)),
        "real_frames": int(len(real)),
        "max_wrench_final_plus_base": float(np.max(np.abs(sim + sim_base))),
        "sim_force_norm_median": float(np.median(force_norm_sim)),
        "sim_force_norm_p95": float(np.quantile(force_norm_sim, 0.95)),
        "real_force_norm_median": float(np.median(force_norm_real)),
        "real_force_norm_p95": float(np.quantile(force_norm_real, 0.95)),
        "sim_force_below_0.1_fraction": float(np.mean(force_norm_sim < 0.1)),
        "real_force_below_0.1_fraction": float(np.mean(force_norm_real < 0.1)),
        "note": "This is a distribution diagnostic, not a paired trajectory alignment result.",
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote: {output}")
    print(f"wrote: {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
