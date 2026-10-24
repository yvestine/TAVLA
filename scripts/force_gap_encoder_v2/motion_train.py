"""Complete motion_v1 experiment; intended to be launched by the user on GPU7."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from .motion_data import build_motion_context, grouped_inner_ids, load_motion_episodes, normalizers
from .motion_eval import (BASELINES, aggregate_phases, aggregate_probes, assess_candidates,
                          evaluate_probe, representation_diagnostics, representations)
from .motion_features import FEATURE_NAMES, disabled_background, select_background
from .motion_fit import fit_legacy_encoder, fit_motion_encoder, infer_encoder, phase_report
from .motion_infer import ARTIFACT_FORMAT
from .train import DEFAULT_HOLDOUT, split_folds


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        return self.log.write(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def context(episodes, fit_ids, valid_ids, background, args):
    windows, signals = build_motion_context(episodes, sorted(set(fit_ids) | set(valid_ids)), background, args)
    center, scale, old_scale = normalizers(episodes, signals, fit_ids, args.baseline_frames)
    return {"train": windows.take(np.isin(windows.trajectory, fit_ids)),
            "valid": windows.take(np.isin(windows.trajectory, valid_ids)),
            "center": center, "scale": scale, "old_scale": old_scale, "background": background,
            "fit_ids": fit_ids, "valid_ids": valid_ids}


def encoder_trial(outer, inner, args, device, seed, weight):
    model, epochs, selection_log = fit_motion_encoder(inner["train"], inner["center"], inner["scale"],
        args, device, seed, weight, validation=inner["valid"])
    del model
    model, _, refit_log = fit_motion_encoder(outer["train"], outer["center"], outer["scale"],
        args, device, seed + 10_000, weight, epochs=epochs)
    z_train, _, _ = infer_encoder(model, outer["train"], outer["center"], outer["scale"], device)
    z_valid, prediction, rec = infer_encoder(model, outer["valid"], outer["center"], outer["scale"], device)
    probe = evaluate_probe(outer["train"], outer["valid"], z_train, z_valid, outer["fit_ids"], args,
                           device, seed + 30_000, shuffle=True)
    row = {"epochs": epochs, "selection_log": selection_log, "refit_log": refit_log,
           "phase": phase_report(outer["valid"], prediction), "validation_reconstruction": rec,
           "probes": {"state_new": probe},
           "structure_diagnostics": representation_diagnostics(outer["train"], outer["valid"], z_train[:, :24], z_valid[:, :24]),
           "full_latent_diagnostics": representation_diagnostics(outer["train"], outer["valid"], z_train, z_valid),
           "background_enabled": outer["background"]["enabled"]}
    del model
    return row


def legacy_trial(outer, inner, legacy_config, args, device, seed):
    model, epochs, _ = fit_legacy_encoder(inner["train"], inner["old_scale"], legacy_config, args,
                                        device, seed, validation=inner["valid"])
    del model
    model, _, _ = fit_legacy_encoder(outer["train"], outer["old_scale"], legacy_config, args,
                                   device, seed + 10_000, epochs=epochs)
    z_train, _, _ = infer_encoder(model, outer["train"], None, outer["old_scale"], device, old=True)
    z_valid, prediction, _ = infer_encoder(model, outer["valid"], None, outer["old_scale"], device, old=True)
    rep_train, rep_valid = representations(outer["train"], legacy=z_train), representations(outer["valid"], legacy=z_valid)
    probes = {}
    for name in BASELINES:
        print(f"  future-motion baseline={name}", flush=True)
        probes[name] = evaluate_probe(outer["train"], outer["valid"], rep_train[name], rep_valid[name],
                                     outer["fit_ids"], args, device, seed + 30_000)
    row = {"epochs": epochs, "phase": phase_report(outer["valid"], prediction), "probes": probes,
           "legacy_diagnostics": representation_diagnostics(outer["train"], outer["valid"], z_train, z_valid),
           "manual_diagnostics": representation_diagnostics(outer["train"], outer["valid"],
                                                            outer["train"].features[:, -1, :15], outer["valid"].features[:, -1, :15]),
           "legacy_recipe": "original phase + 0.15 paired cosine + 0.12 terminal geometry + 0.05 XY direction; historical comparator only"}
    del model
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--cutoff-checkpoint", type=Path, default=Path("eval_outputs/insertion_alignment_encoder/encoder.pt"))
    parser.add_argument("--force-key", default="obs/state/ee_force_base")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/force_gap_encoder_v2/motion_encoder.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--alignment-weights", type=float, nargs="+", default=(0., .01, .1))
    parser.add_argument("--holdout-trajectories", type=int, nargs="+", default=DEFAULT_HOLDOUT)
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--baseline-frames", type=int, default=20)
    parser.add_argument("--boundary-guard", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.history_length != 32 or args.baseline_frames != 20 or args.boundary_guard != 5:
        parser.error("motion_v1 uses history=32, moving pre-contact calibration=20, boundary guard=5")
    if args.batch_size < 128 or args.batch_size % 8 or min(args.epochs, args.probe_epochs, args.patience, args.probe_patience) < 1:
        parser.error("Batch must be >=128 and divisible by 8; epochs/patience must be positive")
    if args.folds != 4 or args.fold_seed != 42 or sorted(args.seeds) != [42, 43, 44] or args.bootstrap_samples < 200:
        parser.error("Use four folds with fold seed 42, initialization seeds 42/43/44 and >=200 bootstrap samples")
    if sorted(args.holdout_trajectories) != sorted(DEFAULT_HOLDOUT):
        parser.error("Keep the eight previously inspected trajectories as the disclosed diagnostic partition")
    if set(args.alignment_weights) != {0., .01, .1}:
        parser.error("The declared comparison requires alignment weights 0, 0.01, 0.1")
    expected_root = Path("eval_outputs/force_gap_encoder_v2").resolve()
    if args.output.parent.resolve() != expected_root or args.output.suffix != ".pt":
        parser.error("Keep the new candidate .pt in eval_outputs/force_gap_encoder_v2")
    if args.output.resolve() in (args.cutoff_checkpoint.resolve(), expected_root / "encoder.pt"):
        parser.error("Output would overwrite an existing encoder")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this environment; launch on the external GPU host")
    return args


def run(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    episodes, old_payload, provenance = load_motion_episodes(args)
    ids = sorted({trajectory for trajectory, _ in episodes})
    holdout = set(args.holdout_trajectories)
    if len(ids) != 40 or len(holdout) != 8 or not holdout <= set(ids):
        raise ValueError("Expected the existing 40 trajectories with 8 disclosed diagnostic trajectories")
    dev = sorted(set(ids) - holdout)
    folds = split_folds(ids, holdout, args.folds, args.fold_seed)
    report_path = args.output.with_suffix(".report.json")
    report = {"artifact_format": ARTIFACT_FORMAT, "recipe": "motion_v1", "status": "running",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "runtime": {"torch_version": str(torch.__version__), "numpy_version": np.__version__,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "data": {"development_trajectories": dev, "diagnostic_trajectories": sorted(holdout), "fixed_folds": folds,
                 "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(), "provenance": provenance,
                 "legacy_cutoff_checkpoint_sha256": hashlib.sha256(args.cutoff_checkpoint.read_bytes()).hexdigest(),
                 "cutoffs": old_payload["geometry"]["first_hit_cutoff_frames"]},
        "interpretation": ["Diagnostic trajectories have already been used in development; they are not untouched final tests.",
            "Sim future targets are paired demonstration motion, not independent measured simulator dynamics.",
            "Metadata consistency is not physical sensor/coordinate verification.",
            "Domain AUC and same-frame cosine are auxiliary; acceptance uses future-motion gains and phase protection.",
            "No TAVLA training, conversion, policy deployment or server startup is part of this experiment."],
        "background_folds": [], "baseline_trials": [],
        "candidate_trials": {str(float(weight)): [] for weight in args.alignment_weights},
        "ablation_trials": [], "software_checks": {"passed": None, "status": "run validate_motion after training"},
        "qualified": False}
    atomic_json(report_path, report)
    contexts = []
    for fold, valid_ids in enumerate(folds):
        train_ids = sorted(set(dev) - set(valid_ids))
        inner_train, inner_valid = grouped_inner_ids(train_ids, args.fold_seed)
        print(f"Preparing fold {fold + 1}/4 background fits (training groups only)", flush=True)
        background = select_background(episodes, train_ids, seed=args.fold_seed)
        inner_background = select_background(episodes, inner_train, seed=args.fold_seed)
        outer = context(episodes, train_ids, valid_ids, background, args)
        inner = context(episodes, inner_train, inner_valid, inner_background, args)
        contexts.append((outer, inner))
        report["background_folds"].append({"fold": fold, "outer": background, "inner": inner_background})
        atomic_json(report_path, report)
    for seed in args.seeds:
        for fold, (outer, inner) in enumerate(contexts):
            fit_seed = seed + fold * 1000
            print(f"Legacy/baselines seed={seed} fold={fold + 1}/4", flush=True)
            row = legacy_trial(outer, inner, old_payload["config"], args, device, fit_seed)
            row.update(seed=seed, fold=fold)
            report["baseline_trials"].append(row)
            atomic_json(report_path, report)
            for weight in args.alignment_weights:
                print(f"motion_v1 seed={seed} fold={fold + 1}/4 lambda={weight:g}", flush=True)
                row = encoder_trial(outer, inner, args, device, fit_seed, weight)
                row.update(seed=seed, fold=fold, alignment_weight=weight)
                report["candidate_trials"][str(float(weight))].append(row)
                atomic_json(report_path, report)
    acceptance = assess_candidates(report["baseline_trials"], report["candidate_trials"], args)
    acceptance["checks"]["full_three_initialization_seeds"] = len(args.seeds) == 3
    acceptance["passed"] = all(acceptance["checks"].values())
    report["acceptance"] = acceptance
    selected_key, selected_weight = acceptance["selected_candidate_key"], acceptance["selected_alignment_weight"]
    report["ablation_no_alignment"] = acceptance["candidates"]["0.0"]
    atomic_json(report_path, report)
    for fold, (outer, inner) in enumerate(contexts):
        if not outer["background"]["enabled"] and not inner["background"]["enabled"]:
            equivalent = [row for row in report["candidate_trials"][selected_key] if row["fold"] == fold]
            report["ablation_trials"].extend([{**row, "background_ablation_identical_to_selected": True} for row in equivalent])
            continue
        no_background = disabled_background("predeclared background-removal ablation")
        ablation_outer = context(episodes, outer["fit_ids"], outer["valid_ids"], no_background, args)
        ablation_inner = context(episodes, inner["fit_ids"], inner["valid_ids"], no_background, args)
        for seed in args.seeds:
            print(f"No-background ablation seed={seed} fold={fold + 1}/4 lambda={selected_weight:g}", flush=True)
            row = encoder_trial(ablation_outer, ablation_inner, args, device, seed + fold * 1000, selected_weight)
            row.update(seed=seed, fold=fold)
            report["ablation_trials"].append(row)
            atomic_json(report_path, report)
    report["ablation_no_background"] = {"motion": aggregate_probes(report["ablation_trials"], "state_new"),
                                        "phase": aggregate_phases(report["ablation_trials"])}
    print("Refitting one final candidate on all 32 development trajectories", flush=True)
    final_background = select_background(episodes, dev, seed=args.fold_seed)
    final_context = context(episodes, dev, sorted(holdout), final_background, args)
    final_epochs = max(1, int(np.median([row["epochs"] for row in report["candidate_trials"][selected_key]])))
    final_seed = args.seeds[0]
    model, _, final_log = fit_motion_encoder(final_context["train"], final_context["center"], final_context["scale"],
        args, device, final_seed + 10_000, selected_weight, epochs=final_epochs)
    z_train, _, _ = infer_encoder(model, final_context["train"], final_context["center"], final_context["scale"], device)
    z_valid, prediction, rec = infer_encoder(model, final_context["valid"], final_context["center"], final_context["scale"], device)
    legacy_epochs = max(1, int(np.median([row["epochs"] for row in report["baseline_trials"]])))
    legacy_model, _, _ = fit_legacy_encoder(final_context["train"], final_context["old_scale"], old_payload["config"],
                                           args, device, final_seed + 10_000, epochs=legacy_epochs)
    legacy_train, _, _ = infer_encoder(legacy_model, final_context["train"], None, final_context["old_scale"], device, old=True)
    legacy_valid, legacy_pred, _ = infer_encoder(legacy_model, final_context["valid"], None, final_context["old_scale"], device, old=True)
    del legacy_model
    rep_train = representations(final_context["train"], legacy=legacy_train, new=z_train)
    rep_valid = representations(final_context["valid"], legacy=legacy_valid, new=z_valid)
    diagnostic_probes = {}
    for name in (acceptance["best_baseline"], "state_new"):
        diagnostic_probes[name] = evaluate_probe(final_context["train"], final_context["valid"], rep_train[name], rep_valid[name],
            dev, args, device, final_seed + 30_000, shuffle=name == "state_new")
    report["diagnostic"] = {"selection_uses_this_partition": False, "phase": phase_report(final_context["valid"], prediction),
        "legacy_phase": phase_report(final_context["valid"], legacy_pred), "reconstruction": rec,
        "probes": diagnostic_probes, "structure_diagnostics": representation_diagnostics(final_context["train"], final_context["valid"], z_train[:, :24], z_valid[:, :24])}
    report["final_training"] = {"seed": final_seed, "epochs": final_epochs, "log": final_log, "background": final_background}
    config = {"recipe": "motion_v1", "history_length": 32, "baseline_frames": 20, "latent_dim": 32,
              "structure_dim": 24, "strength_dim": 8, "alignment_weight": selected_weight, "final_seed": final_seed,
              "epochs": final_epochs, "boundary_guard": 5, "input_force_key": args.force_key,
              "loss_weights": {"phase": 1., "reconstruction": .1, "gain_consistency": .05, "variance": .01},
              "gain_range": [.5, 2.], "gain_distribution": "log_uniform"}
    preprocess = {"feature_order": list(FEATURE_NAMES), "normalizer_center": final_context["center"].tolist(),
                  "normalizer_scale": final_context["scale"].tolist(), "force_units": "newtons", "force_frame": "robot base",
                  "sample_rate_hz_nominal": 10., "insertion_axis_base": [0., 0., -1.], "smoothing_frames": 3,
                  "trend_lags": [1, 5], "background_scale_floor_n": .01,
                  "normalizer_clip": [-10., 10.], "state_basis_order": ["sin(q)[7]", "cos(q)[7]", "joint_vel[7]", "abs(joint_vel)[7]"],
                  "soft_direction_definition": "vector / (vector_norm + startup_background_variation_scale)",
                  "startup_scales_definition": "first20 median norms of smoothed residual/delta1/delta5, each floored at 0.01 N",
                  "time_feature_definition": "actual lag1 seconds / 0.1; actual lag5 seconds / 0.5",
                  "source_treatment": {"real": "fixed startup anchors plus OOF-qualified motion background",
                                       "sim_contact_isolated": "fixed startup force anchor only; no joint-conditioned correction"},
                  "calibration": "20 pre-contact moving samples; constant anchors thereafter"}
    payload = {"artifact_format": ARTIFACT_FORMAT, "version": 1, "config": config, "preprocess": preprocess,
        "background": final_background, "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "data_summary": report["data"], "offline_proxy_acceptance": acceptance,
        "input_contract": {"force": [3], "joint_pos_rad": [7], "joint_vel_rad_s": [7], "timestamp": "seconds strictly increasing",
                           "sources": ["real", "sim_contact_isolated"], "startup_samples": 20,
                           "output_latent": [32], "output_history": [10, 32]}}
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    report.update(status="complete", checkpoint=str(args.output.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), preprocess=preprocess)
    atomic_json(report_path, report)
    print(json.dumps({"selected_alignment_weight": selected_weight, "best_baseline": acceptance["best_baseline"],
                      "mean_relative_improvement": acceptance["mean_relative_improvement"],
                      "checks": acceptance["checks"], "offline_proxy_passed": acceptance["passed"],
                      "software_checks": "pending validate_motion", "checkpoint": str(args.output), "report": str(report_path)}, indent=2), flush=True)


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args)
            except Exception as error:
                traceback.print_exc()
                report_path = args.output.with_suffix(".report.json")
                report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
                report.update(status="failed", qualified=False, error=str(error))
                report.pop("checkpoint_sha256", None)
                atomic_json(report_path, report)
                raise SystemExit(1) from None


if __name__ == "__main__":
    main()
