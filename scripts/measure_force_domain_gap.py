#!/usr/bin/env python3
"""Quantify the Real-vs-Sim force gap for the insertion-quality encoder.

This is a read-only diagnostic.  It does NOT train, and it does not touch any
TAVLA/pi0 checkpoint.  It answers one question with numbers: *how far apart are
Real and Sim in the space the encoder actually consumes*, and *which parts of
that space transfer*?

Two levels of gap are measured on the paired Real/Sim aligned trajectories:

1. Feature-level gap on the 10-D causal force-trend features (the encoder input).
   Direction/ratio channels (force_dir, trend_dir, lateral_ratio, axial_fraction)
   are expected to transfer; magnitude channels (force_norm, trend_norm) are not.
   This directly tests the "direction transfers, magnitude does not" finding.
2. Latent-level gap, optionally, given a trained cross-validation run directory:
   for each held-out trajectory the fold model encodes that trajectory's Real and
   Sim windows, and the Real-vs-Sim separation of the latents is measured.  A
   domain-invariant encoder should shrink this relative to the feature-level gap.

Gap metrics per group: standardized mean difference, proxy-A-distance (from a
logistic Real-vs-Sim classifier, 2*(1-2*err)), and linear MMD.  Larger =
more separable = larger domain gap.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.force_features import DEFAULT_CALIBRATION_FRAMES
from openpi.force_features import REAL_FORCE_TREND_FEATURE_NAMES
from openpi.force_features import build_real_force_trend_features

from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config


# Which of the 10 channels are direction/ratio (scale-free, expected to transfer)
# versus magnitude (Newton-valued, expected to carry the raw sim-real gap).
DIRECTION_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7)  # force_dir xyz, lateral, axial, trend_dir xyz
MAGNITUDE_CHANNELS = (8, 9)  # force_norm, trend_norm
def _fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.5) -> np.ndarray:
    """Minimal L2-regularized logistic regression (full-batch gradient descent).

    Returns weights including a bias column.  Kept dependency-free on purpose so
    this diagnostic needs nothing beyond numpy.
    """
    x = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        logits = np.clip(x @ w, -40.0, 40.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ (prob - y) / len(x) + 1e-3 * w
        w -= lr * gradient
    return w


def proxy_a_distance(real: np.ndarray, sim: np.ndarray, seed: int) -> float:
    """Proxy-A-distance from a logistic Real-vs-Sim classifier.

    2 * (1 - 2 * generalization_error).  0 means the two domains are
    indistinguishable in this space (gap closed); ~2 means trivially separable.
    """
    rng = np.random.default_rng(seed)
    x = np.vstack([real, sim]).astype(np.float64)
    y = np.concatenate([np.zeros(len(real)), np.ones(len(sim))]).astype(np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    center = np.median(x, axis=0)
    q25, q75 = np.quantile(x, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    x = (x - center) / scale
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    split = len(x) // 2
    train_x, train_y = x[:split], y[:split]
    test_x, test_y = x[split:], y[split:]
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return float("nan")
    w = _fit_logistic(train_x, train_y)
    test_aug = np.hstack([test_x, np.ones((len(test_x), 1))])
    prediction = (test_aug @ w >= 0.0).astype(np.float64)
    error = float(np.mean(prediction != test_y))
    return float(2.0 * (1.0 - 2.0 * error))


def linear_mmd(real: np.ndarray, sim: np.ndarray) -> float:
    """Squared MMD with a linear kernel = squared L2 distance of the means."""
    return float(np.sum((real.mean(axis=0) - sim.mean(axis=0)) ** 2))


def standardized_mean_diff(real: np.ndarray, sim: np.ndarray) -> float:
    """Mean |Real-Sim| per channel, standardized by the pooled std."""
    pooled = np.std(np.vstack([real, sim]), axis=0) + 1e-9
    return float(np.mean(np.abs(real.mean(axis=0) - sim.mean(axis=0)) / pooled))


def gap_metrics(real: np.ndarray, sim: np.ndarray, seed: int) -> dict[str, float]:
    return {
        "standardized_mean_diff": standardized_mean_diff(real, sim),
        "proxy_a_distance": proxy_a_distance(real, sim, seed),
        "linear_mmd": linear_mmd(real, sim),
        "n_real": int(len(real)),
        "n_sim": int(len(sim)),
    }
def build_paired_features(
    data: dict[int, object],
    trajectories: list[int],
    axis: np.ndarray,
    rep: dict[str, object],
    calibration_frames: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Return per-trajectory Real and Sim 10-D force features (post-contact only).

    The leading ``calibration_frames`` are free-space calibration and carry no
    contact information, so the gap is measured on frames after the window.
    """
    features: dict[int, dict[str, np.ndarray]] = {}
    for trajectory in trajectories:
        entry: dict[str, np.ndarray] = {}
        for domain in ("real", "sim"):
            force = data[trajectory].real_force if domain == "real" else data[trajectory].sim_force
            values, _ = build_real_force_trend_features(
                force,
                free_end=calibration_frames - 1,
                insertion_axis=axis,
                smoothing_frames=int(rep["causal_smoothing_frames"]),
                trend_lag_frames=int(rep["trend_lag_frames"]),
                epsilon=float(rep["epsilon"]),
                active_noise_mad_multiplier=float(rep["active_noise_mad_multiplier"]),
                active_noise_floor=float(rep["active_noise_floor_newton"]),
            )
            entry[domain] = values[calibration_frames:]
        features[trajectory] = entry
    return features
def latent_gap_from_run(
    run_dir: Path,
    variant: str,
    config: dict[str, object],
    repo: Path,
    device_name: str,
    seed: int,
) -> dict[str, object] | None:
    """Encode each held-out trajectory's Real and Sim windows with its fold model.

    Uses the leave-one-trajectory-out checkpoints already written under
    ``run_dir/folds/traj_<id>/<variant>/model.pt``.  Each trajectory is encoded
    by the fold that held it out, so the latent is always out-of-sample.  Read
    only; no training, no checkpoint writes.
    """
    import torch

    import train_insertion_quality_v3_transformer as v3
    from insertion_quality_v1_lib import fit_representation_scaler

    folds_dir = run_dir / "folds"
    if not folds_dir.exists():
        print(f"  (no folds dir under {run_dir}; skipping latent gap)", flush=True)
        return None
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    all_ids = [int(v) for v in config.get("all_trajectories", sorted(int(k) for k in config["free_end_frames"]))]
    folds = [int(v) for v in config["evaluation_trajectories"]]
    data = load_trajectories(repo, config, all_ids)
    real_latents: list[np.ndarray] = []
    sim_latents: list[np.ndarray] = []
    domains = list(config.get("input_domains", ["real", "sim"]))
    # For encoding we need both domains regardless of the training config.
    config_both = {**config, "input_domains": ["real", "sim"]}
    for held_out in folds:
        model_path = folds_dir / f"traj_{held_out}" / variant / "model.pt"
        if not model_path.exists():
            continue
        train_ids = [t for t in all_ids if t != held_out]
        variant_cfg = config["variants"][variant]
        use_pose = variant_cfg["use_pose"]
        scaler = fit_representation_scaler(data, train_ids, config_both) if config.get("force_feature_mode", "v3_bounded") == "v3_bounded" else None
        # Build features for all trajectories (the normalizer is fit on train_ids,
        # matching how the fold model was trained), then keep only the held-out
        # windows so every latent is out-of-sample for this fold's checkpoint.
        features, _ = v3.build_features(data, all_ids, train_ids, config_both, scaler, use_pose)
        arrays = v3.build_windows(data, [held_out], set(all_ids), read_annotations(repo / "annotations/insertion_quality_labels_v2.csv"), config_both, None, features)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        input_dim = arrays.x.shape[-1]
        model = v3.TemporalTransformer(input_dim, config, use_domain=float(variant_cfg["domain_lambda"]) > 0).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            for domain_id, domain in enumerate(["real", "sim"]):
                mask = arrays.domain == domain_id
                if not np.any(mask):
                    continue
                x = arrays.x[mask]
                parts: list[np.ndarray] = []
                for start in range(0, len(x), 512):
                    _, latent = model.encode_sequence(torch.from_numpy(x[start : start + 512]).to(device))
                    parts.append(latent.cpu().numpy())
                latent = np.concatenate(parts)
                (real_latents if domain == "real" else sim_latents).append(latent)
    if not real_latents or not sim_latents:
        return None
    real = np.concatenate(real_latents)
    sim = np.concatenate(sim_latents)
    return {"variant": variant, "latent_gap": gap_metrics(real, sim, seed)}
def run(
    repo: Path,
    config_path: Path,
    output_dir: Path,
    latent_run_dir: Path | None,
    latent_variant: str | None,
    device_name: str,
) -> None:
    config = read_config(config_path)
    if "force_config" in config:
        force_config_path = Path(config["force_config"])
        if not force_config_path.is_absolute():
            force_config_path = repo / force_config_path
        force_config = read_config(force_config_path)
        config["free_end_frames"] = force_config["free_end_frames"]
        config["representation"] = {**force_config["representation"], **config.get("representation", {})}
    seed = int(config.get("seed", 17))
    rep = config["representation"]
    calibration_frames = int(rep.get("calibration_frames", DEFAULT_CALIBRATION_FRAMES))
    axis = np.asarray(rep["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    trajectories = [int(v) for v in config["evaluation_trajectories"]]
    all_ids = [int(v) for v in config.get("all_trajectories", sorted(int(k) for k in config["free_end_frames"]))]
    data = load_trajectories(repo, config, all_ids)

    features = build_paired_features(data, trajectories, axis, rep, calibration_frames)
    real_all = np.concatenate([features[t]["real"] for t in trajectories])
    sim_all = np.concatenate([features[t]["sim"] for t in trajectories])

    names = list(REAL_FORCE_TREND_FEATURE_NAMES)
    per_channel: list[dict[str, object]] = []
    for channel, name in enumerate(names):
        real_c = real_all[:, channel : channel + 1]
        sim_c = sim_all[:, channel : channel + 1]
        kind = "magnitude" if channel in MAGNITUDE_CHANNELS else "direction"
        per_channel.append({"channel": name, "kind": kind, **gap_metrics(real_c, sim_c, seed)})

    groups = {
        "all_10d": (real_all, sim_all),
        "direction_channels": (real_all[:, list(DIRECTION_CHANNELS)], sim_all[:, list(DIRECTION_CHANNELS)]),
        "magnitude_channels": (real_all[:, list(MAGNITUDE_CHANNELS)], sim_all[:, list(MAGNITUDE_CHANNELS)]),
    }
    feature_gap = {name: gap_metrics(real, sim, seed) for name, (real, sim) in groups.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    latent_gap = None
    if latent_run_dir is not None and latent_variant is not None:
        latent_gap = latent_gap_from_run(latent_run_dir, latent_variant, config, repo, device_name, seed)

    result = {
        "calibration_frames": calibration_frames,
        "trajectories": trajectories,
        "raw_force_amplitude": {
            "real_absmax_N": float(np.max([np.abs(data[t].real_force).max() for t in trajectories])),
            "sim_absmax_N": float(np.max([np.abs(data[t].sim_force).max() for t in trajectories])),
        },
        "feature_gap": feature_gap,
        "per_channel_gap": per_channel,
        "latent_gap": latent_gap,
    }
    (output_dir / "force_domain_gap.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_channel_gap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["channel", "kind", "standardized_mean_diff", "proxy_a_distance", "linear_mmd", "n_real", "n_sim"])
        writer.writeheader()
        writer.writerows(per_channel)
    write_gap_report(output_dir, result)
    print(json.dumps({"feature_gap": feature_gap, "latent_gap": latent_gap}, indent=2))
    print(f"\nWrote force-domain-gap report to {output_dir}")
def write_gap_report(output_dir: Path, result: dict[str, object]) -> None:
    fg = result["feature_gap"]
    raw = result["raw_force_amplitude"]
    lines = [
        "# Real-vs-Sim force domain gap (read-only diagnostic)",
        "",
        f"- Trajectories: `{len(result['trajectories'])}` (encoder evaluation set).",
        f"- Calibration window: first `{result['calibration_frames']}` frames (dropped; gap measured on contact phase).",
        f"- Raw force amplitude: Real absmax `{raw['real_absmax_N']:.2f} N` vs Sim `{raw['sim_absmax_N']:.2f} N` "
        f"(raw ratio ~`{raw['real_absmax_N'] / max(raw['sim_absmax_N'], 1e-6):.0f}x`).",
        "",
        "Gap metrics (larger = more separable = larger domain gap). Proxy-A-distance",
        "in `[0, 2]`: 0 = Real/Sim indistinguishable, 2 = trivially separable.",
        "",
        "## Feature-level gap (10-D encoder input)",
        "",
        "| group | proxy-A-dist | std-mean-diff | linear MMD |",
        "|---|---|---|---|",
    ]
    for name in ("all_10d", "direction_channels", "magnitude_channels"):
        m = fg[name]
        lines.append(
            f"| {name} | {m['proxy_a_distance']:.3f} | {m['standardized_mean_diff']:.3f} | {m['linear_mmd']:.4f} |"
        )
    lines.extend([
        "",
        "The comparison that matters: **direction channels vs magnitude channels**.",
        "A smaller direction gap confirms the literature finding that force *direction*",
        "transfers across the sim-real boundary while *magnitude* does not.",
        "",
    ])
    if result.get("latent_gap"):
        lg = result["latent_gap"]["latent_gap"]
        lines.extend([
            "## Latent-level gap (encoder output)",
            "",
            f"- Variant: `{result['latent_gap']['variant']}`.",
            f"- Proxy-A-distance: `{lg['proxy_a_distance']:.3f}`; std-mean-diff `{lg['standardized_mean_diff']:.3f}`; linear MMD `{lg['linear_mmd']:.4f}`.",
            "- Encoded out-of-sample (each trajectory by the fold that held it out).",
            "- The current encoder is trained sim-only, so this latent gap is a *baseline*:",
            "  a domain-invariant (GRL + paired) encoder should push it below the feature-level gap.",
            "",
        ])
    lines.append("This diagnostic does not modify labels, checkpoints, or run training.\n")
    (output_dir / "force_domain_gap_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_sim_aligned_force_pose.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/force_domain_gap"))
    parser.add_argument("--latent-run-dir", type=Path, default=None, help="A CV run dir with folds/traj_*/<variant>/model.pt to measure latent gap")
    parser.add_argument("--latent-variant", default=None, help="Variant name whose fold checkpoints to encode with")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    latent_run_dir = None
    if args.latent_run_dir is not None:
        latent_run_dir = args.latent_run_dir if args.latent_run_dir.is_absolute() else repo / args.latent_run_dir
    run(repo, config, output, latent_run_dir, args.latent_variant, args.device)


if __name__ == "__main__":
    main()
