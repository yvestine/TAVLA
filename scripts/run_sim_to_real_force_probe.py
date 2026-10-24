#!/usr/bin/env python3
"""Step 1 + Step 2: sim->real transfer probe for control-relevant force targets.

This is the decisive offline test of the encoder's actual purpose: does a
representation of force carry control-relevant information *across* the sim-real
boundary?  We train a small probe on SIM windows only and evaluate it on
held-out REAL windows, with NO real fine-tuning.  The targets are deterministic
control quantities (contact / lateral corrective direction / axial resistance),
not human phase labels, so the probe measures transfer of usable force
information rather than agreement with transferred annotations.

Representations compared (Step 2 lives here as the direction-only vs full ablation):
  * ``raw_force``       -- the raw 3-axis force history (magnitude differs ~5x
    across domains; expected to transfer poorly).
  * ``direction_only``  -- the 8 scale-free direction/ratio channels of the
    10-D causal feature (the Direction-Matters representation: drop magnitude).
  * ``force_feats_10d`` -- all 10 causal features (direction + magnitude).
  * ``latent_baseline`` -- frozen latent of the paired-but-not-aligned encoder.
  * ``latent_dinv``     -- frozen latent of the domain-invariant encoder.

The two ``latent_*`` representations reuse the leave-one-trajectory-out fold
checkpoints already written under the domain-invariant run dir, so each Real
trajectory is encoded by the fold that held it out (out-of-sample).  Nothing
here trains, reads, or modifies any TAVLA/pi0 checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openpi.force_features import DEFAULT_CALIBRATION_FRAMES
from openpi.force_features import build_real_force_trend_features

from insertion_control_targets import build_control_targets
from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config
from insertion_quality_v1_lib import write_csv


DIRECTION_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]  # force_dir xyz, lateral_ratio, axial_frac, trend_dir xyz


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


@dataclass
class DomainWindows:
    """Per-(trajectory, domain) windowed inputs plus aligned control targets."""

    x: dict[str, np.ndarray]           # representation name -> [n, history, dim]
    trajectory: np.ndarray             # [n]
    frame: np.ndarray                  # [n] end frame of each window
    domain: np.ndarray                 # [n] 0=real, 1=sim
    contact: np.ndarray                # [n]
    axial_resist: np.ndarray           # [n] in {-1,0,1}
    lateral_dir: np.ndarray            # [n, 2]
    lateral_valid: np.ndarray          # [n] bool


def causal_force_features(force: np.ndarray, free_end: int, rep: dict, axis: np.ndarray) -> np.ndarray:
    values, _ = build_real_force_trend_features(
        force,
        free_end=free_end,
        insertion_axis=axis,
        smoothing_frames=int(rep["causal_smoothing_frames"]),
        trend_lag_frames=int(rep["trend_lag_frames"]),
        epsilon=float(rep["epsilon"]),
        active_noise_mad_multiplier=float(rep["active_noise_mad_multiplier"]),
        active_noise_floor=float(rep["active_noise_floor_newton"]),
    )
    return values


def build_domain_windows(data, ids, config, rep, axis, calibration_frames, history) -> DomainWindows:
    """Build force representations and control targets for every (traj, domain).

    ``raw_force`` uses the baseline-subtracted raw force so the probe sees the
    magnitude that differs across domains; the causal features and their
    direction-only subset are computed once and sliced.
    """
    per_rep = {name: [] for name in ("raw_force", "direction_only", "force_feats_10d")}
    traj_list, frame_list, domain_list = [], [], []
    contact_list, axial_list, lateral_list, lvalid_list = [], [], [], []
    for trajectory in ids:
        for domain_id, domain in enumerate(("real", "sim")):
            force = data[trajectory].real_force if domain == "real" else data[trajectory].sim_force
            baseline = np.median(force[:calibration_frames], axis=0)
            raw = (force - baseline).astype(np.float32)
            feats = causal_force_features(force, calibration_frames - 1, rep, axis).astype(np.float32)
            targets = build_control_targets(
                force,
                free_end=calibration_frames - 1,
                insertion_axis=axis,
                smoothing_frames=int(rep["causal_smoothing_frames"]),
                active_noise_mad_multiplier=float(rep["active_noise_mad_multiplier"]),
                active_noise_floor=float(rep["active_noise_floor_newton"]),
            )
            n_frames = len(force)
            for end in range(history - 1, n_frames):
                sl = slice(end - history + 1, end + 1)
                per_rep["raw_force"].append(raw[sl])
                per_rep["force_feats_10d"].append(feats[sl])
                per_rep["direction_only"].append(feats[sl][:, DIRECTION_CHANNELS])
                traj_list.append(trajectory)
                frame_list.append(end)
                domain_list.append(domain_id)
                contact_list.append(int(targets.contact_active[end]))
                axial_list.append(int(targets.axial_resist[end]))
                lateral_list.append(targets.lateral_dir[end])
                lvalid_list.append(bool(targets.lateral_valid[end]))
    return DomainWindows(
        x={name: np.asarray(rows, dtype=np.float32) for name, rows in per_rep.items()},
        trajectory=np.asarray(traj_list, dtype=np.int64),
        frame=np.asarray(frame_list, dtype=np.int64),
        domain=np.asarray(domain_list, dtype=np.int64),
        contact=np.asarray(contact_list, dtype=np.int64),
        axial_resist=np.asarray(axial_list, dtype=np.int64),
        lateral_dir=np.asarray(lateral_list, dtype=np.float32),
        lateral_valid=np.asarray(lvalid_list, dtype=bool),
    )


def add_encoder_latents(
    windows: DomainWindows,
    data,
    config: dict,
    run_dir: Path,
    variant: str,
    rep_name: str,
    device_name: str,
    strict: bool = True,
) -> dict[str, object]:
    """Per-fold sim->real probe *inside a single encoder's latent space*.

    A subtlety that would otherwise poison the comparison: every leave-one-out
    fold trains an INDEPENDENT encoder, so fold A's 32-D latent axes have no fixed
    relationship to fold B's.  We therefore never mix latents across folds.  For
    each held-out trajectory we load that fold's model, encode BOTH the Sim
    training windows and the held-out Real windows with it (one common basis), fit
    the probe on Sim, and score on Real.  The held-out Real trajectory was absent
    from the encoder's training, so the Real latent is out-of-sample; pooling the
    per-fold Real predictions gives the reported numbers.  Read-only.
    """
    import torch

    import train_insertion_quality_v3_transformer as v3
    from train_insertion_quality_domain_invariant import DomainInvariantEncoder

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    all_ids = [int(v) for v in config.get("all_trajectories", sorted(int(k) for k in config["free_end_frames"]))]
    folds = [int(v) for v in config["evaluation_trajectories"]]
    variant_cfg = config["variants"][variant]
    use_pose = variant_cfg["use_pose"]
    use_domain = float(variant_cfg["domain_lambda"]) > 0 or float(variant_cfg.get("domain_ce_weight", 0.0)) > 0
    config_both = {**config, "input_domains": ["real", "sim"]}
    annotations = read_annotations(Path(config["_repo"]) / "annotations/insertion_quality_labels_v2.csv")

    def encode(model, x):
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 512):
                out.append(model.encode(torch.from_numpy(x[start : start + 512]).to(device)).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, int(config["model"]["latent_dim"])), np.float32)

    pooled = {t: {"y": [], "p": []} for t in ("contact", "axial_resist")}
    pooled_lat = {"true": [], "pred": []}

    for held_out in folds:
        model_path = run_dir / "folds" / f"traj_{held_out}" / variant / "model.pt"
        if not model_path.exists():
            continue
        train_ids = [t for t in all_ids if t != held_out]
        features, _ = v3.build_features(data, all_ids, train_ids, config_both, None, use_pose)
        # Encode every window once, with THIS fold's model, so Sim(train) and
        # Real(test) latents share one basis.
        arrays = v3.build_windows(data, all_ids, set(all_ids), annotations, config_both, None, features)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = DomainInvariantEncoder(arrays.x.shape[-1], config, use_domain=use_domain).to(device)
        # Control-encoder checkpoints add contact/axial/lateral heads on top of the
        # same encode() path; strict=False drops those extra keys so the shared
        # latent is loaded identically to the domain-invariant encoder.
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        model.eval()
        latent = encode(model, arrays.x)

        # Align these fold latents back to the shared window table via (traj,frame,domain).
        key_to_latent = {
            (int(arrays.trajectory[i]), int(arrays.frame[i]), int(arrays.domain[i])): latent[i]
            for i in range(len(arrays.trajectory))
        }
        lat = np.full((len(windows.trajectory), latent.shape[1]), np.nan, dtype=np.float32)
        for i in range(len(windows.trajectory)):
            v = key_to_latent.get((int(windows.trajectory[i]), int(windows.frame[i]), int(windows.domain[i])))
            if v is not None:
                lat[i] = v
        finite = np.all(np.isfinite(lat), axis=1)

        sim_mask = (windows.domain == 1) & (windows.trajectory != held_out) & finite
        real_mask = (windows.domain == 0) & (windows.trajectory == held_out) & finite
        if not np.any(real_mask) or not np.any(sim_mask):
            continue
        tr_x, te_x = standardize(lat[sim_mask], lat[real_mask])

        yc = windows.contact[sim_mask]
        if len(np.unique(yc)) == 2:
            w = fit_logistic(tr_x, yc.astype(np.float64))
            pooled["contact"]["y"].append(windows.contact[real_mask])
            pooled["contact"]["p"].append(predict_logistic(w, te_x))

        tr_ax = sim_mask & (windows.axial_resist >= 0)
        te_ax = real_mask & (windows.axial_resist >= 0)
        if np.any(tr_ax) and np.any(te_ax) and len(np.unique(windows.axial_resist[tr_ax])) == 2:
            a_tr, a_te = standardize(lat[tr_ax], lat[te_ax])
            w = fit_logistic(a_tr, windows.axial_resist[tr_ax].astype(np.float64))
            pooled["axial_resist"]["y"].append(windows.axial_resist[te_ax])
            pooled["axial_resist"]["p"].append(predict_logistic(w, a_te))

        tr_lat = sim_mask & windows.lateral_valid
        te_lat = real_mask & windows.lateral_valid
        if np.sum(tr_lat) > 10 and np.any(te_lat):
            l_tr, l_te = standardize(lat[tr_lat], lat[te_lat])
            beta = fit_linear(l_tr, windows.lateral_dir[tr_lat])
            pred = np.hstack([l_te, np.ones((len(l_te), 1))]) @ beta
            pooled_lat["true"].append(windows.lateral_dir[te_lat])
            pooled_lat["pred"].append(pred)

    result: dict[str, object] = {}
    for name in ("contact", "axial_resist"):
        if pooled[name]["y"]:
            y = np.concatenate(pooled[name]["y"])
            p = np.concatenate(pooled[name]["p"])
            result[name] = {**binary_scores(y, p), "n": int(len(y))}
        else:
            result[name] = {"macro_f1": float("nan"), "n": 0}
    if pooled_lat["true"]:
        yt = np.concatenate(pooled_lat["true"])
        yp = np.concatenate(pooled_lat["pred"])
        result["lateral_dir"] = {**lateral_scores(yt, yp), "n": int(len(yt))}
    else:
        result["lateral_dir"] = {"mean_cosine": float("nan"), "n": 0}
    return result


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_train = train.reshape(len(train), -1)
    flat_test = test.reshape(len(test), -1)
    center = np.nanmedian(flat_train, axis=0)
    q25, q75 = np.nanquantile(flat_train, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return (flat_train - center) / scale, (flat_test - center) / scale


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 600, lr: float = 0.3, l2: float = 1e-3) -> np.ndarray:
    x = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        logits = np.clip(x @ w, -40, 40)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = x.T @ (prob - y) / len(x) + l2 * w
        w -= lr * grad
    return w


def predict_logistic(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.hstack([x, np.ones((len(x), 1))])
    return (1.0 / (1.0 + np.exp(-np.clip(x @ w, -40, 40))))


def binary_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(np.int64)
    f1s = []
    for label in (0, 1):
        tp = float(np.sum((y == label) & (pred == label)))
        fp = float(np.sum((y != label) & (pred == label)))
        fn = float(np.sum((y == label) & (pred != label)))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"))
    pos, neg = p[y == 1], p[y == 0]
    auroc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")
    return {"macro_f1": float(np.nanmean(f1s)), "accuracy": float(np.mean(y == pred)), "auroc": auroc}


def fit_linear(x: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    """Ridge-regress each target column on x (with bias). Returns [dim+1, out]."""
    x = np.hstack([x, np.ones((len(x), 1))])
    reg = l2 * np.eye(x.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def lateral_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Cosine alignment of predicted vs true unit lateral direction."""
    pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-9)
    cos = np.sum(pred_norm * y_true, axis=1)
    return {
        "mean_cosine": float(np.mean(cos)),
        "median_cosine": float(np.median(cos)),
        "frac_within_90deg": float(np.mean(cos > 0.0)),
    }


def evaluate_representation(
    windows: DomainWindows,
    rep_name: str,
    folds: list[int],
) -> dict[str, object]:
    """Leave-one-Real-trajectory-out: train probe on ALL sim, test on held-out real.

    For each held-out Real trajectory we fit the probe on every Sim window whose
    trajectory is not the held-out one, then score on that Real trajectory's
    windows.  No Real labels are ever used for fitting -- this is strict
    sim->real transfer.  Pooling over folds gives the reported numbers.
    """
    x_all = windows.x[rep_name]
    finite_rows = np.all(np.isfinite(x_all.reshape(len(x_all), -1)), axis=1)

    pooled = {t: {"y": [], "p": []} for t in ("contact", "axial_resist")}
    pooled_lat = {"true": [], "pred": []}

    for held_out in folds:
        sim_mask = (windows.domain == 1) & (windows.trajectory != held_out) & finite_rows
        real_mask = (windows.domain == 0) & (windows.trajectory == held_out) & finite_rows
        if not np.any(real_mask) or not np.any(sim_mask):
            continue
        train_x_raw, test_x_raw = x_all[sim_mask], x_all[real_mask]
        train_x, test_x = standardize(train_x_raw, test_x_raw)

        # contact_active: defined on every frame
        yc = windows.contact[sim_mask]
        if len(np.unique(yc)) == 2:
            w = fit_logistic(train_x, yc.astype(np.float64))
            pooled["contact"]["y"].append(windows.contact[real_mask])
            pooled["contact"]["p"].append(predict_logistic(w, test_x))

        # axial_resist: defined only under contact (label in {0,1})
        train_ax = sim_mask & (windows.axial_resist >= 0)
        test_ax = real_mask & (windows.axial_resist >= 0)
        if np.any(train_ax) and np.any(test_ax) and len(np.unique(windows.axial_resist[train_ax])) == 2:
            tx, ex = standardize(x_all[train_ax], x_all[test_ax])
            w = fit_logistic(tx, windows.axial_resist[train_ax].astype(np.float64))
            pooled["axial_resist"]["y"].append(windows.axial_resist[test_ax])
            pooled["axial_resist"]["p"].append(predict_logistic(w, ex))

        # lateral_dir: 2-D regression on frames with a resolvable lateral force
        train_lat = sim_mask & windows.lateral_valid
        test_lat = real_mask & windows.lateral_valid
        if np.sum(train_lat) > 10 and np.any(test_lat):
            tx, ex = standardize(x_all[train_lat], x_all[test_lat])
            beta = fit_linear(tx, windows.lateral_dir[train_lat])
            pred = np.hstack([ex, np.ones((len(ex), 1))]) @ beta
            pooled_lat["true"].append(windows.lateral_dir[test_lat])
            pooled_lat["pred"].append(pred)

    result: dict[str, object] = {}
    for name in ("contact", "axial_resist"):
        if pooled[name]["y"]:
            y = np.concatenate(pooled[name]["y"])
            p = np.concatenate(pooled[name]["p"])
            result[name] = {**binary_scores(y, p), "n": int(len(y))}
        else:
            result[name] = {"macro_f1": float("nan"), "n": 0}
    if pooled_lat["true"]:
        yt = np.concatenate(pooled_lat["true"])
        yp = np.concatenate(pooled_lat["pred"])
        result["lateral_dir"] = {**lateral_scores(yt, yp), "n": int(len(yt))}
    else:
        result["lateral_dir"] = {"mean_cosine": float("nan"), "n": 0}
    return result


def write_report(output_dir: Path, results: dict[str, dict], config: dict, order: list[str]) -> None:
    lines = [
        "# Sim->Real Force Transfer Probe (control-relevant targets)",
        "",
        "## Setup",
        "",
        "- Probe trained on **Sim windows only**, evaluated on **held-out Real** "
        "(leave-one-trajectory-out); no Real labels used for fitting.",
        "- Targets are deterministic control quantities from each domain's own force, "
        "not transferred human phase labels:",
        "  - `contact_active` (binary): tool in contact vs free, above the free-space noise gate.",
        "  - `axial_resist` (binary, contact frames only): axial force opposes insertion (jam/backdrive).",
        "  - `lateral_dir` (unit 2-vector): corrective lateral push direction; scored by cosine to truth.",
        "- Representations: `raw_force` (magnitude-bearing), `direction_only` (8 scale-free channels = "
        "the Direction-Matters representation), `force_feats_10d` (direction+magnitude), and the frozen "
        "encoder latents `latent_baseline` / `latent_dinv`.",
        "- No TAVLA/pi0 checkpoint or policy code was read or modified.",
        "",
        "## Results (higher = better transfer)",
        "",
        "| representation | contact F1 | axial-resist F1 | axial AUROC | lateral cos | lateral <90° |",
        "|---|---|---|---|---|---|",
    ]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        c = r.get("contact", {})
        a = r.get("axial_resist", {})
        l = r.get("lateral_dir", {})
        lines.append(
            f"| {name} | {c.get('macro_f1', float('nan')):.3f} | {a.get('macro_f1', float('nan')):.3f} | "
            f"{a.get('auroc', float('nan')):.3f} | {l.get('mean_cosine', float('nan')):.3f} | "
            f"{l.get('frac_within_90deg', float('nan')):.3f} |"
        )
    lines.extend([
        "",
        "## How to read this",
        "",
        "- If `raw_force` transfers poorly but a representation transfers well, that representation "
        "carries **control-relevant force information that survives the sim->real gap** -- the precondition "
        "for a residual policy learned in sim to run on the real robot.",
        "- `direction_only` vs `force_feats_10d` / `latent_dinv` is the Direction-Matters ablation: if "
        "keeping magnitude (and aligning it) beats dropping it, the encoder does more than re-derive "
        "'direction transfers, magnitude does not'.",
        "- `latent_dinv` vs `latent_baseline` isolates the effect of the domain-adversary + paired-InfoNCE "
        "alignment on downstream transfer, beyond the raw latent-gap number.",
        "",
        "This probe is offline and standalone; it does not by itself integrate the encoder into TAVLA.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(repo: Path, config_path: Path, latent_run_dir: Path, control_run_dir: Path, output_dir: Path, device_name: str) -> None:
    config = read_config(config_path)
    if "force_config" in config:
        fc_path = Path(config["force_config"])
        fc_path = fc_path if fc_path.is_absolute() else repo / fc_path
        force_config = read_config(fc_path)
        config["free_end_frames"] = force_config["free_end_frames"]
        config["representation"] = {**force_config["representation"], **config.get("representation", {})}
    config["_repo"] = str(repo)
    rep = config["representation"]
    calibration_frames = int(rep.get("calibration_frames", DEFAULT_CALIBRATION_FRAMES))
    history = int(rep["history_frames"])
    axis = np.asarray(rep["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    all_ids = [int(v) for v in config.get("all_trajectories", sorted(int(k) for k in config["free_end_frames"]))]
    folds = [int(v) for v in config["evaluation_trajectories"]]
    data = load_trajectories(repo, config, all_ids)

    windows = build_domain_windows(data, all_ids, config, rep, axis, calibration_frames, history)

    feature_reps = ["raw_force", "direction_only", "force_feats_10d"]
    results = {name: evaluate_representation(windows, name, folds) for name in feature_reps if name in windows.x}

    order = list(feature_reps)
    if latent_run_dir is not None and (latent_run_dir / "folds").exists():
        results["latent_baseline"] = add_encoder_latents(
            windows, data, config, latent_run_dir, "force_pose_paired_baseline", "latent_baseline", device_name
        )
        results["latent_dinv"] = add_encoder_latents(
            windows, data, config, latent_run_dir, "force_pose_domain_invariant", "latent_dinv", device_name
        )
        order += ["latent_baseline", "latent_dinv"]

    # Step-3 control-supervised encoder: same encode() path, extra heads dropped.
    if control_run_dir is not None and (control_run_dir / "folds").exists():
        control_config = {**config, "variants": {
            "force_pose_control_no_adv": {"use_pose": True, "domain_lambda": 0.0, "domain_ce_weight": 0.0},
            "force_pose_control_domain_invariant": {"use_pose": True, "domain_lambda": 0.5, "domain_ce_weight": 0.1},
        }}
        results["latent_control"] = add_encoder_latents(
            windows, data, control_config, control_run_dir, "force_pose_control_domain_invariant", "latent_control", device_name, strict=False
        )
        results["latent_control_no_adv"] = add_encoder_latents(
            windows, data, control_config, control_run_dir, "force_pose_control_no_adv", "latent_control_no_adv", device_name, strict=False
        )
        order += ["latent_control_no_adv", "latent_control"]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "probe_results.json", results)
    rows = []
    for name in order:
        if name not in results:
            continue
        r = results[name]
        rows.append({
            "representation": name,
            "contact_f1": r["contact"].get("macro_f1"),
            "contact_n": r["contact"].get("n"),
            "axial_resist_f1": r["axial_resist"].get("macro_f1"),
            "axial_resist_auroc": r["axial_resist"].get("auroc"),
            "axial_resist_n": r["axial_resist"].get("n"),
            "lateral_mean_cosine": r["lateral_dir"].get("mean_cosine"),
            "lateral_frac_within_90deg": r["lateral_dir"].get("frac_within_90deg"),
            "lateral_n": r["lateral_dir"].get("n"),
        })
    write_csv(output_dir / "probe_results.csv", rows)
    write_report(output_dir, results, config, order)
    report_path = repo / "sim_to_real_force_probe_report.md"
    report_path.write_text((output_dir / "report.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote probe results to {output_dir} and {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_sim_real_domain_invariant.json"))
    parser.add_argument("--latent-run-dir", type=Path, default=Path("eval_outputs/insertion_quality_domain_invariant"))
    parser.add_argument("--control-run-dir", type=Path, default=Path("eval_outputs/insertion_quality_control_encoder"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/sim_to_real_force_probe"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-latents", action="store_true", help="Skip encoder latents (feature-only probe)")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    latent_run_dir = None if args.no_latents else (args.latent_run_dir if args.latent_run_dir.is_absolute() else repo / args.latent_run_dir)
    control_run_dir = None if args.no_latents else (args.control_run_dir if args.control_run_dir.is_absolute() else repo / args.control_run_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, config, latent_run_dir, control_run_dir, output_dir, args.device)


if __name__ == "__main__":
    main()
