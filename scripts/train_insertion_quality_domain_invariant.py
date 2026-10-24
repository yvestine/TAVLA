#!/usr/bin/env python3
"""Domain-invariant force(+pose) insertion-quality encoder (Real+Sim).

This is a NEW standalone trainer.  It does not modify the V3 trainer, the shared
library, or any TAVLA/pi0 checkpoint; it writes only under its ``--output-dir``.

Motivation (see eval_outputs/force_domain_gap): on the 10-D force features the
Real-vs-Sim proxy-A-distance is ~1.7 on magnitude channels but ~0.5 on direction
channels.  The sim-only encoder already leans on direction (latent gap ~0.57),
but it was never trained to be domain-invariant.  This trainer closes the gap on
purpose so the latent is usable as (a) an auxiliary input to TAVLA and (b) an
observation for residual RL learned in sim and deployed on real.

Three mechanisms, all grounded in the surveyed literature:
  * Gradient-reversal domain adversary (Ganin-style DANN) pushes Real and Sim
    latents together, but a companion domain classifier is kept honest so the
    representation stays "close but not collapsed".
  * Paired InfoNCE pulls the SAME (trajectory, frame) Real and Sim windows to the
    same latent (the two domains are frame-paired by construction), which is a
    stronger, supervised-by-pairing signal than distribution matching alone.
  * Phase labels (transferred by paired frame) supervise BOTH domains, so the
    latent keeps the misaligned-vs-aligned structure while becoming domain-free.

Evaluation is leave-one-trajectory-out; quality metrics are reported on the
held-out REAL windows (sim->real generalization), and the held-out Real-vs-Sim
latent gap is measured per fold and pooled.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from openpi.force_features import REAL_FORCE_TREND_FEATURE_NAMES

import train_insertion_quality_v3_transformer as v3
from insertion_quality_v1_lib import CONFIDENCE_WEIGHTS
from insertion_quality_v1_lib import load_trajectories
from insertion_quality_v1_lib import read_annotations
from insertion_quality_v1_lib import read_config
from insertion_quality_v1_lib import write_csv


def save_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def build_paired_index(arrays: v3.WindowArrays, domains: list[str]) -> dict[tuple[int, int], dict[int, int]]:
    """Map (trajectory, frame) -> {domain_id: row index} for InfoNCE pairing.

    Real and Sim share the exact same (trajectory, frame) grid, so a window that
    exists in one domain exists in the other; this index recovers the partner row.
    """
    index: dict[tuple[int, int], dict[int, int]] = {}
    for row in range(len(arrays.frame)):
        key = (int(arrays.trajectory[row]), int(arrays.frame[row]))
        index.setdefault(key, {})[int(arrays.domain[row])] = row
    return index


def paired_infonce(
    latent: torch.Tensor,
    domain_ids: torch.Tensor,
    trajectory_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE over Real/Sim latents of the same (trajectory, frame).

    Positives are the two domains of one paired window; negatives are all other
    windows in the batch.  Returns 0 when the batch has no cross-domain pair.
    """
    real_mask = domain_ids == 0
    sim_mask = domain_ids == 1
    if not torch.any(real_mask) or not torch.any(sim_mask):
        return latent.sum() * 0.0
    real_idx = torch.nonzero(real_mask, as_tuple=False).squeeze(1)
    sim_idx = torch.nonzero(sim_mask, as_tuple=False).squeeze(1)
    real_keys = trajectory_ids[real_idx] * 100000 + frame_ids[real_idx]
    sim_keys = trajectory_ids[sim_idx] * 100000 + frame_ids[sim_idx]
    # Match Real rows to their Sim partner by identical (trajectory, frame) key.
    match = real_keys[:, None] == sim_keys[None, :]
    has_partner = torch.any(match, dim=1)
    if not torch.any(has_partner):
        return latent.sum() * 0.0
    real_sel = real_idx[has_partner]
    partner = torch.argmax(match[has_partner].float(), dim=1)
    sim_sel = sim_idx[partner]
    z_real = F.normalize(latent[real_sel], dim=1)
    z_sim = F.normalize(latent[sim_sel], dim=1)
    logits = z_real @ z_sim.t() / temperature
    target = torch.arange(len(z_real), device=latent.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
class DomainInvariantEncoder(nn.Module):
    """Temporal Transformer with a quality head and a GRL domain head.

    Deliberately self-contained (does not reuse the V3 terminal/reconstruction
    heads) so its loss surface is exactly quality + domain-adversary + InfoNCE.
    """

    def __init__(self, input_dim: int, config: dict[str, object], use_domain: bool):
        super().__init__()
        model_cfg = config["model"]
        d_model = int(model_cfg["d_model"])
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, int(config["representation"]["history_frames"]), d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(model_cfg["nhead"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(model_cfg["num_layers"]))
        latent_dim = int(model_cfg["latent_dim"])
        self.to_latent = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim), nn.Tanh())
        self.quality_head = nn.Linear(latent_dim, 2)
        self.domain_head = (
            nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 2)) if use_domain else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.input_projection(x) + self.position[:, : x.shape[1]]
        sequence = self.temporal(sequence)
        return self.to_latent(sequence[:, -1])

    def forward(self, x: torch.Tensor, domain_lambda: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        latent = self.encode(x)
        domain = self.domain_head(v3.GradientReversal.apply(latent, domain_lambda)) if self.domain_head is not None else None
        return self.quality_head(latent), latent, domain


def dataset(arrays: v3.WindowArrays) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.x),
        torch.from_numpy(arrays.human_y),
        torch.from_numpy(arrays.human_weight),
        torch.from_numpy(arrays.domain),
        torch.from_numpy(arrays.trajectory),
        torch.from_numpy(arrays.frame),
    )


def predict(model: DomainInvariantEncoder, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            quality, _, _ = model(torch.from_numpy(x[start : start + 512]).to(device))
            parts.append(torch.softmax(quality, dim=-1).cpu().numpy())
    return np.concatenate(parts)


def encode_all(model: DomainInvariantEncoder, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            parts.append(model.encode(torch.from_numpy(x[start : start + 512]).to(device)).cpu().numpy())
    return np.concatenate(parts)
def train_model(
    train: v3.WindowArrays,
    input_dim: int,
    config: dict[str, object],
    variant: dict[str, object],
    device: torch.device,
) -> tuple[DomainInvariantEncoder, list[dict[str, float]]]:
    domain_lambda = float(variant["domain_lambda"])
    infonce_weight = float(variant["infonce_weight"])
    domain_ce_weight = float(variant["domain_ce_weight"])
    use_domain = domain_lambda > 0 or domain_ce_weight > 0
    model = DomainInvariantEncoder(input_dim, config, use_domain=use_domain).to(device)
    cfg = config["model"]
    temperature = float(config.get("infonce_temperature", 0.1))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loader = DataLoader(dataset(train), batch_size=int(cfg["batch_size"]), shuffle=True)
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
        sums = {name: 0.0 for name in ("total", "human", "domain", "infonce", "consistency")}
        batches = 0
        for x, human_y, human_weight, domain_y, trajectory_id, frame_id in loader:
            x, human_y, human_weight = x.to(device), human_y.to(device), human_weight.to(device)
            domain_y, trajectory_id, frame_id = domain_y.to(device), trajectory_id.to(device), frame_id.to(device)
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
            noise = torch.randn_like(x) * 0.01
            noisy_quality, _, _ = model(x + noise, domain_lambda)
            consistency_loss = F.mse_loss(torch.softmax(quality_logits.detach(), dim=-1), torch.softmax(noisy_quality, dim=-1))
            loss = (
                float(weights["human_quality"]) * human_loss
                + domain_ce_weight * domain_loss
                + infonce_weight * infonce_loss
                + float(weights.get("temporal_consistency", 0.0)) * consistency_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in (("total", loss), ("human", human_loss), ("domain", domain_loss), ("infonce", infonce_loss), ("consistency", consistency_loss)):
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
            print(f"    epoch {epoch:02d}/{cfg['epochs']} total={row['total']:.4f} human={row['human']:.4f} domain={row['domain']:.4f} infonce={row['infonce']:.4f}", flush=True)
        if stale >= int(cfg["patience"]):
            break
    model.load_state_dict(best_state)
    return model, history
def latent_gap(real: np.ndarray, sim: np.ndarray, seed: int) -> dict[str, float]:
    """Proxy-A-distance + standardized mean diff of Real vs Sim latents."""
    from measure_force_domain_gap import gap_metrics

    return gap_metrics(real, sim, seed)


def run(
    repo: Path,
    config_path: Path,
    annotations_path: Path,
    output_dir: Path,
    device_name: str,
    max_folds: int | None,
    variant_names: list[str] | None,
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
    if variant_names is not None:
        config["variants"] = {name: config["variants"][name] for name in variant_names}
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
    feature_gap_latents: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"real": [], "sim": []} for name in config["variants"]
    }

    for fold_index, held_out in enumerate(folds, start=1):
        print(f"\n[fold {fold_index}/{len(folds)}] held_out=traj_{held_out} device={device}", flush=True)
        train_ids = [t for t in all_ids if t != held_out]
        fold_dir = output_dir / "folds" / f"traj_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        feature_cache: dict[str, tuple[dict[tuple[int, str], np.ndarray], dict[str, object], v3.WindowArrays]] = {}
        for variant_name, variant in config["variants"].items():
            print(f"  [{variant_name}]", flush=True)
            use_pose = variant["use_pose"]
            feature_key = str(use_pose)
            if feature_key not in feature_cache:
                features, feature_metadata = v3.build_features(data, all_ids, train_ids, config, None, use_pose)
                arrays = v3.build_windows(data, all_ids, set(all_ids), annotations, config, None, features)
                feature_cache[feature_key] = (features, feature_metadata, arrays)
            _, feature_metadata, arrays = feature_cache[feature_key]
            train = v3.select(arrays, arrays.trajectory != held_out)
            if config.get("train_labeled_only", False):
                train = v3.select(train, train.human_y >= 0)
            test = v3.select(arrays, arrays.trajectory == held_out)
            input_dim = train.x.shape[-1]
            v3.set_seed(seed + held_out)
            model, training_history = train_model(train, input_dim, config, variant, device)
            variant_dir = fold_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            write_csv(variant_dir / "training_history.csv", training_history)
            torch.save({"state_dict": model.state_dict(), "variant": variant_name, "feature_metadata": feature_metadata}, variant_dir / "model.pt")

            test_q = predict(model, test.x, device)
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

            # Held-out Real-vs-Sim latent gap for this fold (out-of-sample).
            real_test = v3.select(test, test.domain == domains.index("real"))
            sim_test = v3.select(test, test.domain == domains.index("sim"))
            if len(real_test.x) and len(sim_test.x):
                real_latent = encode_all(model, real_test.x, device)
                sim_latent = encode_all(model, sim_test.x, device)
                feature_gap_latents[variant_name]["real"].append(real_latent)
                feature_gap_latents[variant_name]["sim"].append(sim_latent)
                fold_gap = latent_gap(real_latent, sim_latent, seed)
                gap_rows[variant_name].append({"held_out_trajectory": f"traj_{held_out}", **fold_gap})
            print(f"    result macro_f1={metrics.get('macro_f1', float('nan')):.3f} aligned_f1={metrics.get('aligned_f1', float('nan')):.3f}", flush=True)

    write_outputs(repo, config, output_dir, folds, all_metrics, all_predictions, gap_rows, feature_gap_latents, evaluation_domain, seed, started, max_folds, epochs_override)
def write_outputs(
    repo: Path,
    config: dict[str, object],
    output_dir: Path,
    folds: list[int],
    all_metrics: dict[str, list[dict[str, object]]],
    all_predictions: dict[str, list[dict[str, object]]],
    gap_rows: dict[str, list[dict[str, object]]],
    feature_gap_latents: dict[str, dict[str, list[np.ndarray]]],
    evaluation_domain: str,
    seed: int,
    started: float,
    max_folds: int | None,
    epochs_override: int | None,
) -> None:
    aggregate: dict[str, object] = {"folds_completed": len(folds), "elapsed_minutes": (time.time() - started) / 60.0, "variants": {}}
    for variant_name in config["variants"]:
        predictions = all_predictions[variant_name]
        valid = [row for row in predictions if row["human_label"] >= 0]
        y = np.asarray([row["human_label"] for row in valid], dtype=np.int64)
        p = np.asarray([row["p_aligned"] for row in valid], dtype=np.float64)
        metrics = v3.binary_metrics(y, p) if len(y) else {"macro_f1": float("nan")}
        fold_values = np.asarray([float(row.get("quality_macro_f1", float("nan"))) for row in all_metrics[variant_name]])
        pooled_real = np.concatenate(feature_gap_latents[variant_name]["real"]) if feature_gap_latents[variant_name]["real"] else np.empty((0, 1))
        pooled_sim = np.concatenate(feature_gap_latents[variant_name]["sim"]) if feature_gap_latents[variant_name]["sim"] else np.empty((0, 1))
        pooled_gap = latent_gap(pooled_real, pooled_sim, seed) if len(pooled_real) and len(pooled_sim) else None
        aggregate["variants"][variant_name] = {
            **{f"pooled_{key}": value for key, value in metrics.items()},
            "fold_macro_f1_mean": float(np.nanmean(fold_values)) if len(fold_values) else float("nan"),
            "pooled_latent_gap": pooled_gap,
            "folds": len(all_metrics[variant_name]),
        }
        write_csv(output_dir / f"{variant_name}_fold_metrics.csv", all_metrics[variant_name])
        if predictions:
            write_csv(output_dir / f"{variant_name}_predictions.csv", predictions)
        if gap_rows[variant_name]:
            write_csv(output_dir / f"{variant_name}_latent_gap_per_fold.csv", gap_rows[variant_name])
    save_json(output_dir / "aggregate_metrics.json", aggregate)

    report = [
        f"# {config.get('report_title', 'Domain-Invariant Force+Pose Encoder')}",
        "",
        "## Setup",
        "",
        f"- History window: `{config['representation']['history_frames']}` frames; force calibration: first `{config['representation'].get('calibration_frames')}` frames.",
        f"- Input domains: `{config['input_domains']}`; supervised: `{config['supervised_domains']}`; evaluation domain: `{evaluation_domain}` (sim->real generalization).",
        f"- Sim source: `{config['data']['sim_root']}/{config['data']['sim_force_key']}`; Real force: `{config['data']['real_force_key']}`.",
        "- Quality metrics on held-out REAL windows; latent gap on held-out Real-vs-Sim windows (out-of-sample).",
        "- No TAVLA/pi0 checkpoint or policy code was modified.",
        "",
        "## Results",
        "",
        "| variant | Real Macro-F1 | Real aligned F1 | Real AUROC | latent proxy-A-dist | latent std-mean-diff |",
        "|---|---|---|---|---|---|",
    ]
    for variant_name, values in aggregate["variants"].items():
        gap = values["pooled_latent_gap"] or {}
        report.append(
            f"| {variant_name} | {values.get('pooled_macro_f1', float('nan')):.3f} | "
            f"{values.get('pooled_aligned_f1', float('nan')):.3f} | {values.get('pooled_auroc', float('nan')):.3f} | "
            f"{gap.get('proxy_a_distance', float('nan')):.3f} | {gap.get('standardized_mean_diff', float('nan')):.3f} |"
        )
    report.extend([
        "",
        "## Interpretation",
        "",
        "- Lower `latent proxy-A-distance` = Real and Sim latents are less separable = smaller sim-real force gap.",
        "- The baseline variant (domain_lambda=0, infonce=0) is the reference; the domain-invariant variant should lower the latent gap while keeping Real Macro-F1 comparable.",
        "- A gap that collapses to ~0 with a large Macro-F1 drop indicates over-alignment (collapse); the goal is 'close but not collapsed' (surveyed co-training analysis).",
        "- The frozen encoder latent is the artifact to feed TAVLA as an auxiliary input and to expose as a residual-RL observation in sim.",
        "",
        "This encoder is standalone; policy integration is a separate, additive step.",
    ])
    report_text = "\n".join(report) + "\n"
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")
    if max_folds is None and epochs_override is None:
        (repo / config.get("report_filename", "insertion_quality_encoder_domain_invariant_report.md")).write_text(report_text, encoding="utf-8")
    print("\nAggregate metrics:")
    print(json.dumps(aggregate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/insertion_quality_sim_real_domain_invariant.json"))
    parser.add_argument("--annotations", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/insertion_quality_domain_invariant"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    annotations = args.annotations if args.annotations.is_absolute() else repo / args.annotations
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    run(repo, config, annotations, output_dir, args.device, args.max_folds, args.variant, args.epochs)


if __name__ == "__main__":
    main()
