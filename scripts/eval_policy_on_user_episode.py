#!/usr/bin/env python3
"""Replay one recorded HDF5 episode through a trained policy and plot action errors."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openpi.policies import policy_config
from openpi.training import config as train_config


ACTION_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]


def _read_video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(indices)
    frames = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_idx = 0
    while len(frames) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_idx += 1
    cap.release()
    missing = sorted(wanted - set(frames))
    if missing:
        raise RuntimeError(f"Missing video frames in {path}: {missing[:10]}")
    return frames


def _load_episode(episode_dir: Path, indices: list[int], effort_key: str, action_mode: str):
    h5_path = episode_dir / "data.h5"
    with h5py.File(h5_path, "r") as f:
        joint_pos = f["obs/state/joint_pos"][:].astype(np.float32)
        gripper_pos = f["obs/state/gripper_pos"][:].astype(np.float32)
        effort = f[effort_key][:].astype(np.float32)
        action_arm = f[f"action/{action_mode}/arm"][:].astype(np.float32)
        action_gripper = f[f"action/{action_mode}/gripper"][:].astype(np.float32)
        timestamps = f["timestamps"][:].astype(np.float64)

    state = np.concatenate([joint_pos, gripper_pos], axis=-1)
    action = np.concatenate([action_arm, action_gripper], axis=-1)
    front = _read_video_frames(episode_dir / "front_camera.mp4", indices)
    wrist = _read_video_frames(episode_dir / "wrist_camera.mp4", indices)
    return state, effort, action, timestamps, front, wrist


def _make_obs(state, effort, front, wrist, prompt: str) -> dict:
    return {
        "images": {
            "cam_high": front,
            "cam_left_wrist": wrist,
            "cam_right_wrist": wrist,
        },
        "state": state,
        "effort": effort[None, :],
        "prompt": prompt,
    }


def _draw_line_plot(series: list[tuple[str, np.ndarray, tuple[int, int, int]]], title: str, out: Path) -> None:
    width = 1200
    panel_h = 210
    margin_l = 82
    margin_r = 24
    margin_t = 54
    margin_b = 34
    h = margin_t + panel_h * len(ACTION_NAMES) + margin_b
    img = Image.new("RGB", (width, h), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((20, 16), title, fill=(0, 0, 0), font=font)

    x_count = len(series[0][1])
    x0, x1 = margin_l, width - margin_r
    for dim, name in enumerate(ACTION_NAMES):
        y0 = margin_t + dim * panel_h + 20
        y1 = margin_t + (dim + 1) * panel_h - 24
        values = np.concatenate([arr[:, dim] for _, arr, _ in series])
        lo, hi = float(np.min(values)), float(np.max(values))
        if abs(hi - lo) < 1e-8:
            lo -= 1.0
            hi += 1.0
        pad = (hi - lo) * 0.08
        lo -= pad
        hi += pad

        draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210))
        draw.text((18, (y0 + y1) // 2 - 6), name, fill=(0, 0, 0), font=font)
        draw.text((x0, y0 - 14), f"{hi:.3f}", fill=(80, 80, 80), font=font)
        draw.text((x0, y1 + 2), f"{lo:.3f}", fill=(80, 80, 80), font=font)

        for label, arr, color in series:
            pts = []
            for i, v in enumerate(arr[:, dim]):
                x = x0 + int((x1 - x0) * i / max(x_count - 1, 1))
                y = y1 - int((y1 - y0) * (float(v) - lo) / (hi - lo))
                pts.append((x, y))
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)

    legend_x = width - 250
    for i, (label, _, color) in enumerate(series):
        y = 18 + i * 18
        draw.line((legend_x, y + 6, legend_x + 28, y + 6), fill=color, width=3)
        draw.text((legend_x + 36, y), label, fill=(0, 0, 0), font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def _draw_error_plot(errors: np.ndarray, out: Path) -> None:
    width, height = 1100, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((20, 16), "Mean absolute error per evaluated frame", fill=(0, 0, 0), font=font)
    x0, x1 = 70, width - 30
    y0, y1 = 60, height - 55
    draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210))
    y_max = float(max(np.max(errors), 1e-6))
    pts = []
    for i, v in enumerate(errors):
        x = x0 + int((x1 - x0) * i / max(len(errors) - 1, 1))
        y = y1 - int((y1 - y0) * float(v) / y_max)
        pts.append((x, y))
    if len(pts) >= 2:
        draw.line(pts, fill=(210, 60, 60), width=3)
    draw.text((x0, y0 - 16), f"{y_max:.4f}", fill=(80, 80, 80), font=font)
    draw.text((x0, y1 + 8), "evaluated frames", fill=(80, 80, 80), font=font)
    draw.text((20, y0), "MAE", fill=(0, 0, 0), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, default=Path("data/traj_0"))
    parser.add_argument("--config-name", default="pi0_lora_user_single_arm_effort")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pi0_lora_user_single_arm_effort/first_lora/29999"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_outputs/traj_0_first_lora_29999"))
    parser.add_argument("--effort-key", default="obs/state/joint_torque_external")
    parser.add_argument("--action-mode", choices=["actual", "policy"], default="actual")
    parser.add_argument("--prompt", default="peg-in-hole")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--chunk-index", type=int, default=1)
    parser.add_argument("--num-sample-steps", type=int, default=10)
    args = parser.parse_args()

    print(f"loading config: {args.config_name}", flush=True)
    cfg = train_config.get_config(args.config_name)
    print(f"loading policy checkpoint: {args.checkpoint_dir}", flush=True)
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint_dir,
        sample_kwargs={"num_steps": args.num_sample_steps},
        default_prompt=args.prompt,
    )
    print("policy loaded", flush=True)

    with h5py.File(args.episode_dir / "data.h5", "r") as f:
        total = len(f["timestamps"])
    last_start = total - args.chunk_index - 1
    indices = list(range(0, max(last_start, 0), args.stride))[: args.max_frames]
    if not indices:
        raise SystemExit("No frames selected for evaluation.")

    print(f"loading episode frames: {args.episode_dir}, selected={len(indices)}", flush=True)
    state, effort, true_action, timestamps, front, wrist = _load_episode(
        args.episode_dir, indices, args.effort_key, args.action_mode
    )
    print("running policy inference", flush=True)

    pred = []
    truth = []
    rows = []
    for i, t in enumerate(indices):
        obs = _make_obs(state[t], effort[t], front[t], wrist[t], args.prompt)
        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        pred_action = actions[args.chunk_index]
        true = true_action[t + args.chunk_index]
        pred.append(pred_action)
        truth.append(true)
        mae = float(np.mean(np.abs(pred_action - true)))
        rows.append([t, timestamps[t], mae, *pred_action.tolist(), *true.tolist()])
        print(f"[{i + 1:03d}/{len(indices):03d}] frame={t} mae={mae:.6f}", flush=True)

    pred_arr = np.asarray(pred, dtype=np.float32)
    true_arr = np.asarray(truth, dtype=np.float32)
    err = np.abs(pred_arr - true_arr)
    mae_per_frame = err.mean(axis=1)
    mae_per_dim = err.mean(axis=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "pred_vs_true.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", "timestamp", "mae", *[f"pred_{n}" for n in ACTION_NAMES], *[f"true_{n}" for n in ACTION_NAMES]]
        )
        writer.writerows(rows)

    _draw_line_plot(
        [
            ("pred", pred_arr, (220, 50, 47)),
            ("true", true_arr, (40, 90, 210)),
        ],
        f"{args.episode_dir}  chunk_index={args.chunk_index}",
        args.out_dir / "actions_pred_vs_true.png",
    )
    _draw_error_plot(mae_per_frame, args.out_dir / "mae_per_frame.png")

    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"episode_dir: {args.episode_dir}",
                f"checkpoint_dir: {args.checkpoint_dir}",
                f"frames: {len(indices)}",
                f"chunk_index: {args.chunk_index}",
                f"mean_mae: {float(mae_per_frame.mean()):.6f}",
                f"max_frame_mae: {float(mae_per_frame.max()):.6f}",
                "mae_per_dim:",
                *[f"  {name}: {value:.6f}" for name, value in zip(ACTION_NAMES, mae_per_dim, strict=True)],
            ]
        )
        + "\n"
    )
    print(f"saved: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
