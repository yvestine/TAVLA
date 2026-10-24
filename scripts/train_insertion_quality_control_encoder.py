#!/usr/bin/env python3
"""Step 3: control-target-supervised domain-invariant force(+pose) encoder.

The sim->real transfer probe (``run_sim_to_real_force_probe.py``) showed a
concrete failure: the domain-invariant encoder's frozen latent has a tiny
Real-vs-Sim gap (proxy-A-distance ~0.07) yet transfers *control-relevant* force
information WORSE than the raw causal features -- in particular it loses the
lateral corrective direction (probe cosine ~0.11 vs ~0.95 for the direction
features).  The reason is structural: that encoder's latent is shaped only by a
binary quality head + a domain adversary, so it has no reason to keep the
axial-resistance sign or the lateral push direction a residual controller needs,
and the adversary actively erases some of it.

This trainer keeps every mechanism of the domain-invariant encoder (quality head,
GRL domain adversary, paired InfoNCE) and ADDS auxiliary heads that regress the
deterministic control targets -- ``contact_active``, ``axial_resist``,
``lateral_dir`` -- from the latent, supervised on BOTH domains with each domain's
own force-derived targets.  This forces the shared latent to preserve the
transferable force geometry instead of discarding it, which is exactly the
property the probe measures.  Unlike the domain-invariant trainer it does NOT
restrict to human-labeled windows: the control targets are defined on every
frame, so it trains on all windows and uses human quality labels only where they
exist.

Standalone and additive: reuses the V3 feature builder and the domain-invariant
encoder, writes only under ``--output-dir`` (plus fold ``model.pt`` files the
probe reloads), and never touches any TAVLA/pi0 checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import train_insertion_quality_v3_transformer as v3
from train_insertion_quality_domain_invariant import DomainInvariantEncoder
from train_insertion_quality_domain_invariant import latent_gap
from train_insertion_quality_domain_invariant import paired_infonce
from insertion_control_targets import build_control_targets
from insertion_quality_v1_lib import CONFIDENCE_WEIGHTS
from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config
from insertion_quality_v1_lib import write_csv


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


class ControlEncoder(DomainInvariantEncoder):
    """Domain-invariant encoder + auxiliary control-target heads on the latent.

    ``encode`` is inherited unchanged, so the latent this produces is directly
    comparable to the domain-invariant encoder's and the transfer probe can load
    it with the same ``.encode`` path.  The extra heads only add training signal.
    """

    def __init__(self, input_dim: int, config: dict, use_domain: bool):
        super().__init__(input_dim, config, use_domain=use_domain)
        latent_dim = int(config["model"]["latent_dim"])
        self.contact_head = nn.Linear(latent_dim, 1)
        self.axial_head = nn.Linear(latent_dim, 1)
        self.lateral_head = nn.Linear(latent_dim, 2)

    def control_predictions(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "contact": self.contact_head(latent).squeeze(-1),
            "axial": self.axial_head(latent).squeeze(-1),
            "lateral": self.lateral_head(latent),
        }


def build_control_arrays(
    data,
    all_ids: list[int],
    arrays: v3.WindowArrays,
    config: dict,
) -> dict[str, np.ndarray]:
    """Control targets aligned row-for-row to ``arrays`` (each window's end frame).

    Targets are computed per (trajectory, domain) from that domain's own force,
    with the identical functional form used by the transfer probe, then indexed by
    the window's end frame.  Real windows get Real-force targets, Sim windows get
    Sim-force targets -- the point is that the *definition* is shared while the
    force differs, so a latent that predicts both is domain-invariant in a
    control-relevant sense.
    """
    rep = config["representation"]
    axis = np.asarray(rep["insertion_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    calibration_frames = int(rep["calibration_frames"])
    domains = list(config.get("input_domains", ["real", "sim"]))
    per_domain: dict[tuple[int, int], object] = {}
    for trajectory in all_ids:
        for domain_id, domain in enumerate(domains):
            force = data[trajectory].real_force if domain == "real" else data[trajectory].sim_force
            per_domain[(trajectory, domain_id)] = build_control_targets(
                force,
                free_end=calibration_frames - 1,
                insertion_axis=axis,
                smoothing_frames=int(rep["causal_smoothing_frames"]),
                active_noise_mad_multiplier=float(rep["active_noise_mad_multiplier"]),
                active_noise_floor=float(rep["active_noise_floor_newton"]),
            )
    n = len(arrays.frame)
    contact = np.zeros(n, dtype=np.float32)
    axial = np.full(n, -1, dtype=np.int64)
    lateral = np.zeros((n, 2), dtype=np.float32)
    lateral_valid = np.zeros(n, dtype=bool)
    for row in range(n):
        targets = per_domain[(int(arrays.trajectory[row]), int(arrays.domain[row]))]
        frame = int(arrays.frame[row])
        contact[row] = float(targets.contact_active[frame])
        axial[row] = int(targets.axial_resist[frame])
        lateral[row] = targets.lateral_dir[frame]
        lateral_valid[row] = bool(targets.lateral_valid[frame])
    return {"contact": contact, "axial": axial, "lateral": lateral, "lateral_valid": lateral_valid}


def dataset(arrays: v3.WindowArrays, control: dict[str, np.ndarray]) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.x),
        torch.from_numpy(arrays.human_y),
        torch.from_numpy(arrays.human_weight),
        torch.from_numpy(arrays.domain),
        torch.from_numpy(arrays.trajectory),
        torch.from_numpy(arrays.frame),
        torch.from_numpy(control["contact"]),
        torch.from_numpy(control["axial"]),
        torch.from_numpy(control["lateral"]),
        torch.from_numpy(control["lateral_valid"]),
    )


def control_loss(
    predictions: dict[str, torch.Tensor],
    contact_y: torch.Tensor,
    axial_y: torch.Tensor,
    lateral_y: torch.Tensor,
    lateral_valid: torch.Tensor,
) -> torch.Tensor:
    """Sum of the three control-target losses on the frames where each is defined.

    ``contact`` is defined on every frame (BCE); ``axial_resist`` only under
    contact (BCE on the >=0 rows); ``lateral_dir`` only on lateral-valid frames
    (negative cosine so the latent must point the right way, not just correlate).
    """
    loss = F.binary_cross_entropy_with_logits(predictions["contact"], contact_y)
    axial_keep = axial_y >= 0
    if torch.any(axial_keep):
        loss = loss + F.binary_cross_entropy_with_logits(
            predictions["axial"][axial_keep], axial_y[axial_keep].float()
        )
    if torch.any(lateral_valid):
        pred_dir = F.normalize(predictions["lateral"][lateral_valid], dim=1)
        cosine = torch.sum(pred_dir * lateral_y[lateral_valid], dim=1)
        loss = loss + (1.0 - cosine).mean()
    return loss


def encode_all(model: ControlEncoder, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            parts.append(model.encode(torch.from_numpy(x[start : start + 512]).to(device)).cpu().numpy())
    return np.concatenate(parts)


def predict_quality(model: ControlEncoder, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            quality, _, _ = model(torch.from_numpy(x[start : start + 512]).to(device))
            parts.append(torch.softmax(quality, dim=-1).cpu().numpy())
    return np.concatenate(parts)


def train_model(
    train: v3.WindowArrays,
    control: dict[str, np.ndarray],
    input_dim: int,
    config: dict,
    variant: dict,
    device: torch.device,
) -> tuple[ControlEncoder, list[dict[str, float]]]:
    domain_lambda = float(variant["domain_lambda"])
    infonce_weight = float(variant["infonce_weight"])
    domain_ce_weight = float(variant["domain_ce_weight"])
    control_weight = float(variant.get("control_weight", 1.0))
    use_domain = domain_lambda > 0 or domain_ce_weight > 0
    model = ControlEncoder(input_dim, config, use_domain=use_domain).to(device)
    cfg = config["model"]
    temperature = float(config.get("infonce_temperature", 0.1))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loader = DataLoader(dataset(train, control), batch_size=int(cfg["batch_size"]), shuffle=True)
    human_labels = train.human_y[train.human_y >= 0]
    counts = np.bincount(human_labels, minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / np.maximum(2 * counts, 1), dtype=torch.float32, device=device)
    weights = cfg["loss_weights"]
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        sums = {name: 0.0 for name in ("total", "human", "domain", "infonce", "control")}
        batches = 0
        for x, human_y, human_weight, domain_y, trajectory_id, frame_id, contact_y, axial_y, lateral_y, lateral_valid in loader:
            x, human_y, human_weight = x.to(device), human_y.to(device), human_weight.to(device)
            domain_y, trajectory_id, frame_id = domain_y.to(device), trajectory_id.to(device), frame_id.to(device)
            contact_y, axial_y = contact_y.to(device), axial_y.to(device)
            lateral_y, lateral_valid = lateral_y.to(device), lateral_valid.to(device)
            quality_logits, latent, domain_logits = model(x, domain_lambda)
            human_keep = human_y >= 0
            human_loss = (
                (F.cross_entropy(quality_logits[human_keep], human_y[human_keep], weight=class_weights, reduction="none") * human_weight[human_keep]).sum()
                / human_weight[human_keep].sum().clamp_min(1.0)
                if torch.any(human_keep) else quality_logits.sum() * 0.0
            )
            domain_loss = F.cross_entropy(domain_logits, domain_y) if domain_logits is not None else quality_logits.sum() * 0.0
            infonce_loss = (
                paired_infonce(latent, domain_y, trajectory_id, frame_id, temperature)
                if infonce_weight > 0 else latent.sum() * 0.0
            )
            ctrl_loss = control_loss(model.control_predictions(latent), contact_y, axial_y, lateral_y, lateral_valid)
            loss = (
                float(weights["human_quality"]) * human_loss
                + domain_ce_weight * domain_loss
                + infonce_weight * infonce_loss
                + control_weight * ctrl_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in (("total", loss), ("human", human_loss), ("domain", domain_loss), ("infonce", infonce_loss), ("control", ctrl_loss)):
                sums[name] += float(value.detach().cpu())
            batches += 1
        row = {"epoch": float(epoch), **{name: value / max(batches, 1) for name, value in sums.items()}}
        history.append(row)
        if row["total"] < best_loss - 1e-4:
            best_loss = row["total"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == int(cfg["epochs"]):
            print(f"    epoch {epoch:02d}/{cfg['epochs']} total={row['total']:.4f} human={row['human']:.4f} domain={row['domain']:.4f} infonce={row['infonce']:.4f} control={row['control']:.4f}", flush=True)
        if stale >= int(cfg["patience"]):
            break
    model.load_state_dict(best_state)
    return model, history


def run(
    repo: Path,
    config_path: Path,
    annotations_path: Path,
    output_dir: Path,
    device_name: str,
    max_folds: int | None,
    epochs_override: int | None,
) -> None:
    config = read_config(config_path)
    if "force_config" in config:
        force_config_path = Path(config["force_config"])
        if not force_config_path.is_absolute():
            force_config_path = repo / force_config_path
        force_config = read_config(force_config_path)
        config["free_end_frames"] = force_config["free_end_frames"]
        config["representation"] = {**force_config["representation"], **config.get("representation", {})}
    if epochs_override is not None:
        config["model"]["epochs"] = epochs_override
    annotations = read_annotations(annotations_path)
    all_ids = [int(v) for v in config.get("all_trajectories", sorted(int(k) for k in config["free_end_frames"]))]
    folds = [int(v) for v in config["evaluation_trajectories"]]
    if max_folds is not None:
        folds = folds[:max_folds]
    domains = list(config["input_domains"])
    evaluation_domain = config.get("evaluation_domain", "real")
    seed = int(config["seed"])
    data = load_trajectories(repo, config, all_ids)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_name} unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_config.json", config)
    started = time.time()

    all_metrics: dict[str, list[dict[str, object]]] = {name: [] for name in config["variants"]}
    all_predictions: dict[str, list[dict[str, object]]] = {name: [] for name in config["variants"]}
    gap_rows: dict[str, list[dict[str, object]]] = {name: [] for name in config["variants"]}
    pooled_latents: dict[str, dict[str, list[np.ndarray]]] = {name: {"real": [], "sim": []} for name in config["variants"]}

    for fold_index, held_out in enumerate(folds, start=1):
        print(f"\n[fold {fold_index}/{len(folds)}] held_out=traj_{held_out} device={device}", flush=True)
        train_ids = [t for t in all_ids if t != held_out]
        fold_dir = output_dir / "folds" / f"traj_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        feature_cache: dict[str, tuple[object, object, v3.WindowArrays, dict[str, np.ndarray]]] = {}
        for variant_name, variant in config["variants"].items():
            print(f"  [{variant_name}]", flush=True)
            use_pose = variant["use_pose"]
            feature_key = str(use_pose)
            if feature_key not in feature_cache:
                features, feature_metadata = v3.build_features(data, all_ids, train_ids, config, None, use_pose)
                arrays = v3.build_windows(data, all_ids, set(all_ids), annotations, config, None, features)
                control = build_control_arrays(data, all_ids, arrays, config)
                feature_cache[feature_key] = (features, feature_metadata, arrays, control)
            _, feature_metadata, arrays, control = feature_cache[feature_key]

            train_mask = arrays.trajectory != held_out
            train = v3.select(arrays, train_mask)
            train_control = {k: v[train_mask] for k, v in control.items()}
            test = v3.select(arrays, arrays.trajectory == held_out)
            input_dim = train.x.shape[-1]
            v3.set_seed(seed + held_out)
            model, training_history = train_model(train, train_control, input_dim, config, variant, device)
            variant_dir = fold_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            write_csv(variant_dir / "training_history.csv", training_history)
            torch.save({"state_dict": model.state_dict(), "variant": variant_name, "feature_metadata": feature_metadata}, variant_dir / "model.pt")

            test_q = predict_quality(model, test.x, device)
            eval_mask = (test.domain == domains.index(evaluation_domain)) & (test.human_y >= 0) & (test.human_weight >= CONFIDENCE_WEIGHTS["medium"])
            metrics = v3.binary_metrics(test.human_y[eval_mask], test_q[eval_mask, 1]) if np.any(eval_mask) else {"macro_f1": float("nan")}
            all_metrics[variant_name].append({"held_out_trajectory": f"traj_{held_out}", "quality_windows": int(np.sum(eval_mask)), **{f"quality_{k}": v for k, v in metrics.items()}})
            eval_domain_mask = test.domain == domains.index(evaluation_domain)
            for index in np.flatnonzero(eval_domain_mask):
                all_predictions[variant_name].append({
                    "trajectory": f"traj_{held_out}", "frame": int(test.frame[index]),
                    "p_misaligned": float(test_q[index, 0]), "p_aligned": float(test_q[index, 1]),
                    "human_label": int(test.human_y[index]),
                })

            real_test = v3.select(test, test.domain == domains.index("real"))
            sim_test = v3.select(test, test.domain == domains.index("sim"))
            if len(real_test.x) and len(sim_test.x):
                real_latent = encode_all(model, real_test.x, device)
                sim_latent = encode_all(model, sim_test.x, device)
                pooled_latents[variant_name]["real"].append(real_latent)
                pooled_latents[variant_name]["sim"].append(sim_latent)
                gap_rows[variant_name].append({"held_out_trajectory": f"traj_{held_out}", **latent_gap(real_latent, sim_latent, seed)})
            print(f"    result macro_f1={metrics.get('macro_f1', float('nan')):.3f} aligned_f1={metrics.get('aligned_f1', float('nan')):.3f}", flush=True)

    write_outputs(repo, config, output_dir, folds, all_metrics, all_predictions, gap_rows, pooled_latents, evaluation_domain, seed, started, max_folds, epochs_override)


def write_outputs(repo, config, output_dir, folds, all_metrics, all_predictions, gap_rows, pooled_latents, evaluation_domain, seed, started, max_folds, epochs_override) -> None:
    aggregate: dict[str, object] = {"folds_completed": len(folds), "elapsed_minutes": (time.time() - started) / 60.0, "variants": {}}
    for variant_name in config["variants"]:
        valid = [row for row in all_predictions[variant_name] if row["human_label"] >= 0]
        y = np.asarray([row["human_label"] for row in valid], dtype=np.int64)
        p = np.asarray([row["p_aligned"] for row in valid], dtype=np.float64)
        metrics = v3.binary_metrics(y, p) if len(y) else {"macro_f1": float("nan")}
        pooled_real = np.concatenate(pooled_latents[variant_name]["real"]) if pooled_latents[variant_name]["real"] else np.empty((0, 1))
        pooled_sim = np.concatenate(pooled_latents[variant_name]["sim"]) if pooled_latents[variant_name]["sim"] else np.empty((0, 1))
        pooled_gap = latent_gap(pooled_real, pooled_sim, seed) if len(pooled_real) and len(pooled_sim) else None
        aggregate["variants"][variant_name] = {
            **{f"pooled_{key}": value for key, value in metrics.items()},
            "pooled_latent_gap": pooled_gap,
            "folds": len(all_metrics[variant_name]),
        }
        write_csv(output_dir / f"{variant_name}_fold_metrics.csv", all_metrics[variant_name])
        if all_predictions[variant_name]:
            write_csv(output_dir / f"{variant_name}_predictions.csv", all_predictions[variant_name])
        if gap_rows[variant_name]:
            write_csv(output_dir / f"{variant_name}_latent_gap_per_fold.csv", gap_rows[variant_name])
    save_json(output_dir / "aggregate_metrics.json", aggregate)

    report = [
        f"# {config.get('report_title', 'Control-Supervised Domain-Invariant Encoder')}",
        "",
        "## Setup",
        "",
        f"- History window: `{config['representation']['history_frames']}` frames; force calibration: first `{config['representation'].get('calibration_frames')}` frames.",
        f"- Input domains: `{config['input_domains']}`; evaluation domain: `{evaluation_domain}` (sim->real generalization).",
        "- Adds contact/axial-resist/lateral-direction auxiliary heads supervised on BOTH domains with force-derived control targets.",
        "- Quality metrics on held-out REAL windows; latent gap on held-out Real-vs-Sim windows (out-of-sample).",
        "- No TAVLA/pi0 checkpoint or policy code was modified.",
        "",
        "## Results",
        "",
        "| variant | Real Macro-F1 | Real aligned F1 | Real AUROC | latent proxy-A-dist |",
        "|---|---|---|---|---|",
    ]
    for variant_name, values in aggregate["variants"].items():
        gap = values["pooled_latent_gap"] or {}
        report.append(
            f"| {variant_name} | {values.get('pooled_macro_f1', float('nan')):.3f} | "
            f"{values.get('pooled_aligned_f1', float('nan')):.3f} | {values.get('pooled_auroc', float('nan')):.3f} | "
            f"{gap.get('proxy_a_distance', float('nan')):.3f} |"
        )
    report.extend([
        "",
        "## Interpretation",
        "",
        "- The decisive metric for this encoder is the sim->real transfer probe, not these numbers: run",
        "  `run_sim_to_real_force_probe.py --latent-run-dir <this output dir>` and compare `latent_control` "
        "against `direction_only` / `force_feats_10d`.",
        "- The auxiliary control heads exist only to shape the latent; at deployment the frozen `encode` output is the artifact.",
        "",
        "This encoder is standalone; policy integration is a separate, additive step.",
    ])
    report_text = "\n".join(report) + "\n"
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")
    if max_folds is None and epochs_override is None:
        (repo / config.get("report_filename", "insertion_quality_encoder_control_report.md")).write_text(report_text, encoding="utf-8")
    print("\nAggregate metrics:")
    print(json.dumps(aggregate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_sim_real_control_encoder.json"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/insertion_quality_control_encoder"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    annotations = args.annotations if args.annotations.is_absolute() else repo / args.annotations
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, config, annotations, output_dir, args.device, args.max_folds, args.epochs)


if __name__ == "__main__":
    main()
