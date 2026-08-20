"""Analyze force changes around simulated insertion phases and real episode endpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIM_ROOT = Path("data-sim-wrench-final-50")
REAL_ROOT = Path("data")
OUT_ROOT = Path("eval_outputs/wrench_alignment")


def sim_wrench(path: Path) -> np.ndarray:
    return np.asarray(np.genfromtxt(path / "wrench_final.csv", delimiter=",", skip_header=1), dtype=float).reshape(-1, 6)


def real_wrench(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return h5["obs/state/ee_wrench_base"][:].astype(float)


def norm_profile(values: list[np.ndarray], bins: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks = []
    for value in values:
        force = np.linalg.norm(value[:, :3], axis=1)
        row = []
        for index in range(bins):
            start = int(np.floor(index / bins * len(force)))
            end = max(start + 1, int(np.floor((index + 1) / bins * len(force))))
            row.append(float(np.mean(force[start:end])))
        chunks.append(row)
    array = np.asarray(chunks)
    return np.mean(array, axis=0), np.quantile(array, 0.1, axis=0), np.quantile(array, 0.9, axis=0)


def event_rows(kind: str, paths: list[Path]) -> list[dict[str, float | str | int]]:
    rows = []
    for path in paths:
        if kind == "sim":
            values = sim_wrench(path)
            reward_terms = np.genfromtxt(path / "reward_terms.csv", delimiter=",", names=True)
            n = min(len(values), len(reward_terms))
            values = values[:n]
            reward_terms = reward_terms[:n]
        else:
            values = real_wrench(path)
            reward_terms = None

        force_norm = np.linalg.norm(values[:, :3], axis=1)
        torque_norm = np.linalg.norm(values[:, 3:], axis=1)
        delta_norm = np.r_[0.0, np.linalg.norm(np.diff(values, axis=0), axis=1)]
        force_peak = int(np.argmax(force_norm))
        delta_peak = int(np.argmax(delta_norm))
        row: dict[str, float | str | int] = {
            "kind": kind,
            "episode": path.name,
            "frames": int(len(values)),
            "force_mean": float(np.mean(force_norm)),
            "force_median": float(np.median(force_norm)),
            "force_p95": float(np.quantile(force_norm, 0.95)),
            "force_max": float(force_norm[force_peak]),
            "force_peak_t": float(force_peak / max(len(values) - 1, 1)),
            "delta_p95": float(np.quantile(delta_norm, 0.95)),
            "delta_max": float(delta_norm[delta_peak]),
            "delta_peak_t": float(delta_peak / max(len(values) - 1, 1)),
            "torque_p95": float(np.quantile(torque_norm, 0.95)),
        }
        if reward_terms is not None:
            engaged = reward_terms["curr_engaged"] > 0.5
            insertion = reward_terms["insertion_progress"] > 1e-6
            engaged_index = np.where(engaged)[0]
            insertion_index = np.where(insertion)[0]
            row["first_engaged_t"] = float(engaged_index[0] / max(len(values) - 1, 1)) if len(engaged_index) else 1.0
            row["first_insertion_t"] = float(insertion_index[0] / max(len(values) - 1, 1)) if len(insertion_index) else 1.0
            row["max_insertion_progress"] = float(np.max(reward_terms["insertion_progress"]))
            row["force_mean_pre_engaged"] = float(np.mean(force_norm[~engaged])) if np.any(~engaged) else float("nan")
            row["force_mean_engaged"] = float(np.mean(force_norm[engaged])) if np.any(engaged) else float("nan")
            row["force_change_engaged_minus_pre"] = row["force_mean_engaged"] - row["force_mean_pre_engaged"]
        rows.append(row)
    return rows


def load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


TITLE = load_font(28)
SUBTITLE = load_font(20)
TEXT = load_font(16)
SMALL = load_font(13)


def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=(75, 75, 75), width=2)
    draw.line((x0, y0, x0, y1), fill=(75, 75, 75), width=2)


def ycoord(value: float, lo: float, hi: float, y0: int, y1: int) -> int:
    return int(y1 - (value - lo) / max(hi - lo, 1e-9) * (y1 - y0))


def draw_profile(draw, box, sim_values, real_values):
    x0, y0, x1, y1 = box
    axes(draw, box)
    sim_mean, sim_lo, sim_hi = norm_profile(sim_values)
    real_mean, real_lo, real_hi = norm_profile(real_values)
    hi = max(float(np.max(sim_hi)), float(np.max(real_hi)), 1e-6)
    for mean, lo, upper, color in ((sim_mean, sim_lo, sim_hi, (232, 126, 4)), (real_mean, real_lo, real_hi, (33, 104, 180))):
        points = []
        for i, value in enumerate(mean):
            px = int(x0 + i / 19 * (x1 - x0))
            points.append((px, ycoord(float(value), 0, hi, y0, y1)))
        draw.line(points, fill=color, width=4)
        for i, (lower, upper) in enumerate(zip(lo, upper, strict=True)):
            px = int(x0 + i / 19 * (x1 - x0))
            draw.line((px, ycoord(float(lower), 0, hi, y0, y1), px, ycoord(float(upper), 0, hi, y0, y1)), fill=color, width=2)
    draw.text((x0 + 8, y0 + 8), "orange sim / blue real; whisker=p10-p90", fill=(70, 70, 70), font=SMALL)
    draw.text((x0, y1 + 7), "episode start", fill=(70, 70, 70), font=SMALL)
    draw.text((x1 - 80, y1 + 7), "end", fill=(70, 70, 70), font=SMALL)


def draw_hist(draw, box, sim_values, real_values):
    x0, y0, x1, y1 = box
    axes(draw, box)
    bins = np.linspace(0, 1, 11)
    for values, color in ((sim_values, (232, 126, 4)), (real_values, (33, 104, 180))):
        counts, _ = np.histogram(values, bins=bins)
        max_count = max(int(np.max(counts)), 1)
        for i, count in enumerate(counts):
            left = int(x0 + i / 10 * (x1 - x0))
            right = int(x0 + (i + 1) / 10 * (x1 - x0))
            top = int(y1 - count / max_count * (y1 - y0))
            if color[0] > color[2]:
                draw.rectangle((left + 5, top, (left + right) // 2, y1), fill=color)
            else:
                draw.rectangle(((left + right) // 2, top, right - 5, y1), fill=color)
    draw.text((x0 + 8, y0 + 8), "orange sim / blue real", fill=(70, 70, 70), font=SMALL)
    draw.text((x0, y1 + 7), "0.0", fill=(70, 70, 70), font=SMALL)
    draw.text((x1 - 30, y1 + 7), "1.0", fill=(70, 70, 70), font=SMALL)


def draw_phase_example(draw, box, path: Path):
    x0, y0, x1, y1 = box
    axes(draw, box)
    values = sim_wrench(path)
    terms = np.genfromtxt(path / "reward_terms.csv", delimiter=",", names=True)[: len(values)]
    force = np.linalg.norm(values[:, :3], axis=1)
    max_force = max(float(np.quantile(force, 0.99)), 1e-6)
    for series, color, scale in ((force, (232, 126, 4), max_force), (terms["insertion_progress"], (33, 104, 180), 1.0)):
        points = []
        for i, value in enumerate(series):
            px = int(x0 + i / max(len(series) - 1, 1) * (x1 - x0))
            py = ycoord(float(value), 0, scale, y0, y1)
            points.append((px, py))
        draw.line(points, fill=color, width=3)
    engaged = np.where(terms["curr_engaged"] > 0.5)[0]
    if len(engaged):
        px = int(x0 + engaged[0] / max(len(values) - 1, 1) * (x1 - x0))
        draw.line((px, y0, px, y1), fill=(110, 30, 150), width=2)
    draw.text((x0 + 8, y0 + 8), "orange force norm / blue insertion_progress", fill=(70, 70, 70), font=SMALL)
    draw.text((x0 + 8, y0 + 28), "purple line = first curr_engaged", fill=(110, 30, 150), font=SMALL)


def main() -> None:
    sim_paths = sorted(SIM_ROOT.glob("episode_*"), key=lambda p: int(p.name.split("_")[-1]))
    real_paths = sorted(REAL_ROOT.glob("traj_*/data.h5"), key=lambda p: int(p.parent.name.split("_")[-1]))
    sim_values = [sim_wrench(p) for p in sim_paths]
    real_values = [real_wrench(p) for p in real_paths]
    sim_rows = event_rows("sim", sim_paths)
    real_rows = event_rows("real", real_paths)
    rows = sim_rows + real_rows
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row})
    with (OUT_ROOT / "episode_event_statistics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sim_peak_t = np.asarray([row["force_peak_t"] for row in sim_rows], dtype=float)
    real_peak_t = np.asarray([row["force_peak_t"] for row in real_rows], dtype=float)
    sim_delta_t = np.asarray([row["delta_peak_t"] for row in sim_rows], dtype=float)
    real_delta_t = np.asarray([row["delta_peak_t"] for row in real_rows], dtype=float)
    sim_change = np.asarray([row["force_change_engaged_minus_pre"] for row in sim_rows], dtype=float)
    summary = {
        "sim_episodes": len(sim_rows),
        "real_episodes": len(real_rows),
        "sim_force_peak_t_median": float(np.median(sim_peak_t)),
        "real_force_peak_t_median": float(np.median(real_peak_t)),
        "sim_force_peak_in_last_10pct": float(np.mean(sim_peak_t > 0.9)),
        "real_force_peak_in_last_10pct": float(np.mean(real_peak_t > 0.9)),
        "sim_delta_peak_in_last_10pct": float(np.mean(sim_delta_t > 0.9)),
        "real_delta_peak_in_last_10pct": float(np.mean(real_delta_t > 0.9)),
        "sim_first_insertion_t_median": float(np.median([row["first_insertion_t"] for row in sim_rows])),
        "sim_first_engaged_t_median": float(np.median([row["first_engaged_t"] for row in sim_rows])),
        "sim_force_change_after_engaged_median": float(np.median(sim_change)),
        "sim_force_change_after_engaged_mean": float(np.mean(sim_change)),
        "sim_force_insertion_corr_median": float(np.median([
            np.corrcoef(
                np.genfromtxt(p / "reward_terms.csv", delimiter=",", names=True)["insertion_progress"][: len(sim_wrench(p))],
                np.linalg.norm(sim_wrench(p)[:, :3], axis=1),
            )[0, 1]
            for p in sim_paths
        ])),
        "interpretation": "Sim force peaks do not temporally match the real episode-end force events; sim reward insertion phases show little force increase.",
        "caveat": "Real data has no explicit contact_start/insertion_start label; real peak location is only a contact-event candidate.",
    }
    (OUT_ROOT / "contact_event_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    image = Image.new("RGB", (1900, 1300), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    draw.text((42, 24), "Contact / insertion event analysis", fill=(15, 15, 15), font=TITLE)
    draw.text((44, 65), "Sim insertion_progress/curr_engaged are available; real event labels are unavailable", fill=(80, 80, 80), font=TEXT)

    boxes = [(30, 105, 930, 560), (970, 105, 1870, 560), (30, 595, 930, 1170), (970, 595, 1870, 1170)]
    for box, title in zip(boxes, ("Force norm over normalized episode time", "Location of maximum force event", "Sim episode_0: force vs insertion progress", "Interpretation"), strict=True):
        draw.rounded_rectangle(box, radius=12, outline=(180, 180, 180), width=2, fill=(252, 252, 252))
        draw.text((box[0] + 18, box[1] + 14), title, fill=(25, 25, 25), font=SUBTITLE)

    draw_profile(draw, (90, 180, 870, 500), sim_values, real_values)
    draw_hist(draw, (1030, 180, 1810, 500), sim_peak_t, real_peak_t)
    draw_phase_example(draw, (90, 685, 870, 1110), SIM_ROOT / "episode_0")

    x0, y0 = 1030, 690
    text_lines = [
        f"Sim force peak median time: {np.median(sim_peak_t):.3f}",
        f"Real force peak median time: {np.median(real_peak_t):.3f}",
        f"Sim peaks in final 10%: {np.mean(sim_peak_t > 0.9) * 100:.1f}%",
        f"Real peaks in final 10%: {np.mean(real_peak_t > 0.9) * 100:.1f}%",
        "",
        f"Sim first insertion median: {summary['sim_first_insertion_t_median']:.3f}",
        f"Sim first engaged median: {summary['sim_first_engaged_t_median']:.3f}",
        f"Force change after engaged: {summary['sim_force_change_after_engaged_median']:.3f} N median",
        "",
        "Conclusion:",
        "The sim force does change and has spikes, but its",
        "largest events are not aligned with the sim insertion",
        "phase, and the real force events are concentrated near",
        "episode end. This is evidence of phase/distribution",
        "mismatch, not proof that the sensor is numerically broken.",
    ]
    for line in text_lines:
        draw.text((x0, y0), line, fill=(150, 65, 0) if line == "Conclusion:" else (40, 40, 40), font=TEXT)
        y0 += 32

    output = OUT_ROOT / "contact_event_analysis.png"
    image.save(output)
    print(f"wrote: {output}")
    print(f"wrote: {OUT_ROOT / 'episode_event_statistics.csv'}")
    print(f"wrote: {OUT_ROOT / 'contact_event_summary.json'}")


if __name__ == "__main__":
    main()
