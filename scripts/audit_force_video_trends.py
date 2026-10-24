#!/usr/bin/env python3
"""Inspect current paired force/video data; no encoder or policy training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

PHASES = ("misaligned_contact", "aligned_progress")
COLORS = {"free": "#888888", "uncertain": "#e49b22", "misaligned_contact": "#da5e60", "aligned_progress": "#55a571", "terminal": "#9476bc"}


def mean3(x, width=3):
    sums = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)))
    end = np.arange(len(x)) + 1
    start = np.maximum(end - width, 0)
    return (sums[end] - sums[start]) / (end - start)[:, None]


def corr(a, b, rank=False):
    if len(a) < 8 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return None
    if rank:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def auc(negative, positive):
    if min(len(negative), len(positive)) < 8:
        return None
    ranks = rankdata(np.r_[negative, positive])
    return float((ranks[len(negative):].sum() - len(positive) * (len(positive) + 1) / 2) / (len(negative) * len(positive)))


def stats(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(a):
        return {"count": 0}
    rng = np.random.default_rng(42)
    medians = np.median(a[rng.integers(0, len(a), size=(2000, len(a)))], axis=1)
    return {"count": len(a), "median": float(np.median(a)), "q25": float(np.quantile(a, .25)),
            "q75": float(np.quantile(a, .75)), "median_ci95": np.quantile(medians, [.025, .975]).tolist(),
            "above_03": int((a > .3).sum()), "above_05": int((a > .5).sum()), "below_zero": int((a < 0).sum())}


def frame_images(path, frames):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {path}")
    result = {}
    for frame in sorted(set(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, im = cap.read()
        if not ok:
            raise ValueError(f"Missing video frame {frame}: {path}")
        result[frame] = im
    cap.release()
    return result


def keyframes(spans, n):
    free = next(s for s in spans if s["label"] == "free")
    wrong = next((s for s in spans if s["label"] == PHASES[0]), None)
    aligned = next((s for s in spans if s["label"] == PHASES[1]), None)
    first = wrong or aligned or next(s for s in spans if s["label"] != "free")
    candidates = [free["end"], first["start"], (first["start"] + first["end"]) // 2,
                  aligned["start"] if aligned else first["end"], aligned["end"] if aligned else n - 2, n - 1]
    return candidates


def tile(im, text, width=240, height=180, crop=True):
    if crop:
        # Fixed crop includes peg/hole in these wrist views; no outcome-dependent crop.
        im = im[140:480, 100:580]
    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (width, 24), (0, 0, 0), -1)
    cv2.putText(im, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .47, (0, 255, 255), 1)
    return im


def make_sheets(repo, output, episodes):
    for group in range(10):
        rows = []
        for trajectory in range(group * 4, group * 4 + 4):
            e = episodes[trajectory]
            frames = keyframes(e["spans"], len(e["real"]))
            for root, tag in (("data", "R"), ("sim-data-aligned", "S")):
                images = frame_images(repo / root / f"traj_{trajectory}" / "wrist_camera.mp4", frames)
                rows.append(np.hstack([tile(images[f], f"{tag} traj{trajectory} f{f}") for f in frames]))
        cv2.imwrite(str(output / f"force_video_sheet_{group:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"wrist review sheet {group + 1}/10", flush=True)


def detail(repo, output, episodes, trajectory):
    e = episodes[trajectory]
    first = next(s["end"] for s in e["spans"] if s["label"] == "free")
    frames = sorted(set(range(max(0, first - 5), len(e["real"]), 5)) | set(keyframes(e["spans"], len(e["real"]))))
    paths = [(root, camera) for root in ("data", "sim-data-aligned") for camera in ("front_camera.mp4", "wrist_camera.mp4")]
    images = {p: frame_images(repo / p[0] / f"traj_{trajectory}" / p[1], frames) for p in paths}
    cells = []
    for f in frames:
        cams = [tile(images[p][f], f"{'R' if p[0]=='data' else 'S'} {'front' if p[1].startswith('front') else 'wrist'} f{f}", width=280, height=210, crop=p[1].startswith('wrist')) for p in paths]
        cells.append(np.vstack((np.hstack(cams[:2]), np.hstack(cams[2:]))))
    rows = []
    for start in range(0, len(cells), 3):
        chunk = cells[start:start + 3]
        chunk += [np.zeros_like(cells[0])] * (3 - len(chunk))
        rows.append(np.hstack(chunk))
    cv2.imwrite(str(output / f"force_video_detail_{trajectory:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])


def plot_episode(output, trajectory, e):
    x = np.arange(len(e["real"]))
    fig, ax = plt.subplots(6, 1, figsize=(13, 10), sharex=True)
    for domain, panel in (("real", 0), ("sim", 1)):
        force = e[domain]
        smooth = e[domain + "_smooth"]
        ax[panel].plot(x, force[:, 2], alpha=.3, color="black", label="raw Fz")
        ax[panel].plot(x, smooth[:, 0], label="Fx (first20 centered)")
        ax[panel].plot(x, smooth[:, 1], label="Fy (first20 centered)")
        ax[panel].plot(x, smooth[:, 2], label="Fz (first20 centered)")
        ax[panel].set_ylabel(f"{domain.title()} N")
        ax[panel].legend(loc="upper left", ncol=4, fontsize=8)
    for s in e["spans"]:
        keep = slice(s["start"], s["end"] + 1)
        for domain, color in (("real", "#2c6ea6"), ("sim", "#d4672d")):
            fz = e[domain + "_smooth"][keep, 2]
            z = (fz - np.mean(fz)) / max(np.std(fz), .005)
            ax[2].plot(x[keep], z, color=color, label=domain if s is e["spans"][0] else None)
    ax[2].set_ylabel("Within-phase\nFz z-score")
    ax[2].legend(loc="upper left")
    for domain, color in (("real", "#2c6ea6"), ("sim", "#d4672d")):
        ax[3].plot(x, e[domain + "_signals"]["delta5_fz"], color=color, label=domain)
        ax[4].plot(x, e[domain + "_signals"]["lateral_ratio"], color=color, label=domain)
    ax[3].set_ylabel("Fz(t)-Fz(t-5)\nN")
    ax[4].set_ylabel("Lateral fraction")
    ax[5].plot(x, (e["pose"][0, 2] - e["pose"][:, 2]) * 1000, label="Real recorded downward motion", color="black")
    ax[5].set_ylabel("Downward mm")
    ax[5].legend(loc="upper left", fontsize=8)
    for axis in ax:
        for s in e["spans"]:
            axis.axvspan(s["start"] - .5, s["end"] + .5, color=COLORS[s["label"]], alpha=.12)
        axis.axvline(e["cutoff"], color="black", ls="--", lw=.8)
        axis.grid(alpha=.2)
    ax[-1].set_xlabel("Frame index (10 fps video; exact HDF5 time saved in report)")
    labels = "; ".join(f"{s['label']} {s['start']}-{s['end']}" for s in e["spans"])
    ax[0].set_title(f"traj_{trajectory}: {labels}\nDashed = motion_v1 training cutoff. Real and Sim axes use separate N ranges.", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / f"force_video_trace_{trajectory:02d}.png", dpi=140)
    plt.close(fig)


def run(args):
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((repo / "annotations/insertion_quality_labels_v2.csv").open(encoding="utf-8-sig")))
    motion = json.loads((output / "motion_encoder.report.json").read_text())
    episodes, records, video_meta, phase_comparison = {}, [], [], []
    for trajectory in range(40):
        spans = [{"start": int(r["start_frame"]), "end": int(r["end_frame"]), "label": r["label"], "confidence": r["confidence"]} for r in rows if r["trajectory"] == f"traj_{trajectory}"]
        e = {"spans": spans, "cutoff": int(motion["data"]["cutoffs"][f"traj_{trajectory}"])}
        for domain, root in (("real", "data"), ("sim", "sim-data-aligned")):
            with h5py.File(repo / root / f"traj_{trajectory}" / "data.h5", "r") as h:
                force = np.asarray(h["obs/state/ee_force_base"], float)
                timestamps = np.asarray(h["timestamps"], float)
                if domain == "real":
                    e["pose"] = np.asarray(h["obs/state/ee_pose"], float)[:, :3]
                    e["timestamps"] = timestamps
                elif not np.allclose(e["timestamps"], timestamps, atol=1e-6, rtol=0):
                    raise ValueError("Paired timebase mismatch")
            e[domain] = force
            smooth = mean3(force - np.median(force[:20], axis=0))
            e[domain + "_smooth"] = smooth
            n = np.linalg.norm(smooth, axis=1)
            lateral = np.linalg.norm(smooth[:, :2], axis=1)
            indices = np.maximum(np.arange(len(force)) - 5, 0)
            signals = {"fz": smooth[:, 2], "fx": smooth[:, 0], "fy": smooth[:, 1], "norm": n, "lateral": lateral,
                       "axial_fraction": np.abs(smooth[:, 2]) / np.maximum(n, .01),
                       "lateral_ratio": lateral / np.maximum(n, .01),
                       "delta5_fz": smooth[:, 2] - smooth[indices, 2],
                       "delta5_norm": n - n[indices], "trend_energy": np.linalg.norm(smooth - smooth[indices], axis=1)}
            e[domain + "_signals"] = signals
            free = spans[0]
            free_idx = np.arange(max(20, free["start"]), free["end"] + 1)
            e[domain + "_free_p95"] = float(np.quantile(n[free_idx], .95))
            for camera in ("front_camera.mp4", "wrist_camera.mp4"):
                path = repo / root / f"traj_{trajectory}" / camera
                cap = cv2.VideoCapture(str(path))
                frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                if frames != len(force) or abs(fps - 10.) > .01:
                    raise ValueError(f"Unmatched frame count/fps: {path}")
                video_meta.append({"trajectory": trajectory, "domain": domain, "camera": camera, "frames": frames, "fps": fps,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for s in spans:
            if s["label"] not in PHASES:
                continue
            a, b = s["start"] + 2, s["end"] - 2
            idx = np.arange(a, b + 1)
            if len(idx) < 8:
                continue
            record = {"trajectory": trajectory, "phase": s["label"], "confidence": s["confidence"], "start": a, "end": b,
                      "frames": len(idx), "precutoff_fraction": float((idx < e["cutoff"]).mean())}
            for name in ("fz", "fx", "fy", "norm", "lateral"):
                real, sim = e["real_signals"][name][idx], e["sim_signals"][name][idx]
                record[name + "_r"] = corr(real, sim)
                record[name + "_spearman"] = corr(real, sim, rank=True)
                lagged = [(lag, corr(real[max(0, -lag):len(real) - max(0, lag)], sim[max(0, lag):len(sim) - max(0, -lag)])) for lag in range(-5, 6)]
                lagged = [(lag, v) for lag, v in lagged if v is not None]
                lag, value = max(lagged, key=lambda pair: (pair[1], -abs(pair[0]))) if lagged else (None, None)
                record[name + "_best_lag_r"], record[name + "_best_lag_frames"] = value, lag
            trend_idx = np.arange(s["start"] + 5, b + 1)
            record["delta5_fz_r"] = corr(e["real_signals"]["delta5_fz"][trend_idx], e["sim_signals"]["delta5_fz"][trend_idx])
            record["delta5_norm_r"] = corr(e["real_signals"]["delta5_norm"][trend_idx], e["sim_signals"]["delta5_norm"][trend_idx])
            for domain in ("real", "sim"):
                signals = e[domain + "_signals"]
                for name in ("norm", "lateral", "axial_fraction", "lateral_ratio", "trend_energy"):
                    record[domain + "_" + name + "_median"] = float(np.median(signals[name][idx]))
                record[domain + "_norm_p95"] = float(np.quantile(signals["norm"][idx], .95))
                record[domain + "_free_p95"] = e[domain + "_free_p95"]
                record[domain + "_phase_to_free_p95_ratio"] = record[domain + "_norm_p95"] / max(e[domain + "_free_p95"], .01)
                record[domain + "_fz_variation_p90p10"] = float(np.quantile(signals["fz"][idx], .9) - np.quantile(signals["fz"][idx], .1))
            records.append(record)
        neg = next((s for s in spans if s["label"] == PHASES[0]), None)
        pos = next((s for s in spans if s["label"] == PHASES[1]), None)
        if neg and pos:
            for domain in ("real", "sim"):
                for name, sign in (("axial_fraction", 1), ("lateral_ratio", -1), ("norm", 1), ("trend_energy", -1)):
                    values = e[domain + "_signals"][name] * sign
                    value = auc(values[neg["start"] + 2:neg["end"] - 1], values[pos["start"] + 2:pos["end"] - 1])
                    if value is not None:
                        phase_comparison.append({"trajectory": trajectory, "domain": domain, "feature": name, "auc": value})
        episodes[trajectory] = e
        plot_episode(output, trajectory, e)
        print(f"force/phase analysis {trajectory + 1}/40", flush=True)
    summary = {}
    for phase in PHASES:
        subset = [r for r in records if r["phase"] == phase]
        metrics = [k for k in subset[0] if k.endswith(("_r", "_spearman", "_median", "_ratio", "_variation_p90p10"))]
        summary[phase] = {"trajectories": len(subset), "metrics": {k: stats([r[k] for r in subset]) for k in metrics}}
    feature_summary = {domain: {name: stats([r["auc"] for r in phase_comparison if r["domain"] == domain and r["feature"] == name]) for name in ("axial_fraction", "lateral_ratio", "norm", "trend_energy")} for domain in ("real", "sim")}
    report = {"source": "data + CURRENT sim-data-aligned", "trajectories": 40, "videos": video_meta,
              "method": {"force": "ee_force_base xyz in N, native recorded sign", "baseline": "fixed first20 median; signed-axis correlations are offset invariant",
                         "smoothing": "causal3", "phase_guard": 2, "minimum_frames": 8, "lag_search_frames": [-5, 5],
                         "lag_note": "Best-lag correlation is exploratory and optimistically selected; primary metric is same-frame correlation.",
                         "scope": "All reviewed contact frames, including post motion_v1 cutoff; terminal excluded from phase comparisons.",
                         "labels": "Transferred annotations used force/pose/video; AUC is descriptive association, not independent validation.",
                         "video_provenance": "Aligned builder reuses videos from sim-data; same frame count and fps are checked, co-rendering with calibrated force cannot be verified here."},
              "summary": summary, "within_trajectory_phase_auc": feature_summary, "phase_records": records, "phase_auc_records": phase_comparison}
    (output / "force_video_trends.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (output / "force_video_trends.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    make_sheets(repo, output, episodes)
    for trajectory in args.detail:
        detail(repo, output, episodes, trajectory)
    cards = []
    for trajectory, e in episodes.items():
        phase_buttons = " ".join(f'<button onclick="seek({trajectory},{s["start"]})">{s["label"]} f{s["start"]}</button>' for s in e["spans"])
        videos = "".join(f'<div>{root}/{cam}<video data-t="{trajectory}" src="../../{root}/traj_{trajectory}/{cam}" controls preload="none"></video></div>' for root in ("data", "sim-data-aligned") for cam in ("front_camera.mp4", "wrist_camera.mp4"))
        cards.append(f'<details><summary>traj_{trajectory} — {len(e["real"])} frames; encoder cutoff f{e["cutoff"]}</summary><p>{phase_buttons}</p><input type="range" min="0" max="{len(e["real"])-1}" value="0" oninput="seek({trajectory},Number(this.value))"><span id="frame{trajectory}">f0</span><div class="videos">{videos}</div><img src="force_video_trace_{trajectory:02d}.png"></details>')
    html = '<!doctype html><meta charset="utf-8"><title>Real/Sim force and video review</title><style>body{max-width:1400px;margin:24px auto;font-family:sans-serif}details{margin:12px;padding:12px;border:1px solid #ccc}img{width:100%}.videos{display:grid;grid-template-columns:1fr 1fr}video{width:100%}button{margin:3px}input{width:70%}</style><h1>当前 Real/Sim 力趋势与视频对照</h1><p>按帧同步四个视角，视频名义10 fps；曲线保留各域牛顿值。标签来自旧审阅并转移，不能假定两域实际接触相同。末端事件与插入段分开。</p>' + ''.join(cards) + '<script>function seek(t,f){document.querySelectorAll(`video[data-t="${t}"]`).forEach(v=>{v.currentTime=f/10;v.pause()});document.getElementById(`frame${t}`).textContent=`f${f} / ${(f/10).toFixed(1)}s`}</script>'
    (output / "force_video_trends.html").write_text(html, encoding="utf-8")
    print(json.dumps({"phase_summary": summary, "phase_auc": feature_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/force_gap_encoder_v2"))
    parser.add_argument("--detail", type=int, nargs="*", default=(0, 9, 12, 22))
    run(parser.parse_args())
