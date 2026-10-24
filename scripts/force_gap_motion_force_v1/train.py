"""Nested CV of force residuals on immutable, fold-matched motion checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from force_gap_forecast_v1.common import sha256, write_json
from force_gap_forecast_v1.data import Blocks, FOLDS, HOLDOUT, TRAIN_IDS, fit_stats, load_pairs
from force_gap_forecast_v1.losses import objective
from force_gap_forecast_v1.metrics import aggregate
from force_gap_forecast_v1.train import cpu_state, run_stage, seed_all
from . import FORMAT, LAYOUT
from .metrics import evaluate, evaluate_gains
from .model import MotionForceForecast

DEFAULT_BASE = "eval_outputs/force_gap_forecast_v1/20261012_155026_551478"
GUARD_KEYS = ("current_rmse_n", "future_1_rmse_n", "future_3_rmse_n", "future_5_rmse_n",
              "delta_rmse_n", "peak_amplitude_mae_n", "high_band_mse",
              "future_1_peak_event_rmse_n", "future_5_peak_event_rmse_n")


def read_json(path):
    return json.loads(Path(path).read_text())


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in model.motion.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def guard_passed(candidate, baseline, tolerance=.02):
    return all(candidate.get(key) is not None and baseline.get(key) is not None and
               candidate[key] <= baseline[key] * (1 + tolerance) + 1e-8 for key in GUARD_KEYS)


def load_motion(checkpoint, episodes, train_ids):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stage = payload.get("stage", {})
    if not stage.get("motion_only") or stage.get("train_ids") != sorted(train_ids):
        raise ValueError(f"Not a fold-matched motion-only checkpoint: {checkpoint}")
    if payload["stats"] != fit_stats(episodes, train_ids):
        raise ValueError(f"Motion normalization/data changed: {checkpoint}")
    return payload


def save_stage(path, model, stats, summary):
    payload = {"artifact_format": FORMAT, "layout": LAYOUT, "state_dict": cpu_state(model),
               "stats": stats, "stage": summary}
    temporary = path / "model.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(path / "model.pt")
    write_json(path / "history.json", summary)


def fit_stage(checkpoint, episodes, train_ids, valid_ids, args, seed, path, epochs=None, gain=None):
    payload = load_motion(checkpoint, episodes, train_ids)
    stats = payload["stats"]
    seed_all(seed)
    model = MotionForceForecast(stats, payload["state_dict"]).to(args.device)
    base_sha = sha256(checkpoint)
    if (path / "model.pt").is_file() and (path / "history.json").is_file():
        saved = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        summary = read_json(path / "history.json")
        if not args.resume or saved.get("artifact_format") != FORMAT or saved.get("stage") != summary:
            raise ValueError(f"Existing or inconsistent stage; use a matching --resume: {path}")
        if (not summary.get("completed") or summary["base_sha256"] != base_sha or
                summary["train_ids"] != sorted(train_ids) or summary["valid_ids"] != sorted(valid_ids)):
            raise ValueError(f"Stage inputs changed: {path}")
        model.load_state_dict(saved["state_dict"])
        return model, stats, summary
    path.mkdir(parents=True, exist_ok=True)
    before = frozen_digest(model)
    maximum = args.max_epochs if epochs is None else epochs
    if maximum < 0 or (not valid_ids and gain is None):
        raise ValueError("Refit requires nonnegative epochs and an inner-selected gain")
    if maximum == 0 and gain != 0:
        raise ValueError("An untrained residual must use gain=0")
    baseline = None
    best_score, selected_epoch, selected_gain = math.inf, maximum, gain
    best_weights = cpu_state(model)
    if valid_ids:
        baseline = aggregate([row["all"] for row in
                              evaluate_gains(model, episodes, valid_ids, stats, 1, [0])[0.].values()])
        best_score, selected_epoch, selected_gain = baseline["score"], 0, 0.
    parameters = list(model.force_residual.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    dataset = Blocks(episodes, train_ids, stats)
    steps_per_epoch = max(1, math.ceil(len(dataset) / (args.batch_size * 8)))
    records, stale, steps = [], 0, 0
    started = time.monotonic()
    for epoch in range(1, maximum + 1):
        model.train()
        sampler = WeightedRandomSampler(dataset.weights, steps_per_epoch * args.batch_size,
                                        replacement=True, generator=torch.Generator().manual_seed(seed + epoch * 7919))
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        losses = []
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            sim = model.block(batch, 1)
            loss, row = objective(model, sim, batch, alignment=False)
            penalty = sim["correction_norm"]
            if args.real_loss_weight:
                real = model.block(batch, 0)
                real_loss, _ = objective(model, real, batch, alignment=False)
                loss = loss + args.real_loss_weight * real_loss
                penalty = penalty + args.real_loss_weight * real["correction_norm"]
            loss = loss + args.residual_l2 * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite residual loss: {path}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 3., error_if_nonfinite=True)
            optimizer.step()
            row["total"] = float(loss.detach())
            losses.append(row)
            steps += 1
        record = {"epoch": epoch, "optimizer_steps": steps, "elapsed_s": time.monotonic() - started,
                  "train": aggregate(losses)}
        if valid_ids:
            grid = evaluate_gains(model, episodes, valid_ids, stats, 1, args.gains)
            candidates = {str(g): aggregate([row["all"] for row in rows.values()]) for g, rows in grid.items()}
            record["gain_metrics"] = candidates
            improved = False
            for candidate_gain in sorted(grid):
                metric = candidates[str(candidate_gain)]
                if not np.isfinite(metric["score"]):
                    raise FloatingPointError("Non-finite validation metric")
                if (candidate_gain > 0 and guard_passed(metric, baseline) and metric["score"] < best_score):
                    best_score, selected_epoch, selected_gain = metric["score"], epoch, candidate_gain
                    best_weights, improved = cpu_state(model), True
            stale = 0 if improved else stale + 1
            record.update(valid_score=best_score, selected_gain=selected_gain, selected_epoch=selected_epoch)
        records.append(record)
        write_json(path / "history.json", {"completed": False, "records": records})
        if epoch == 1 or epoch % 10 == 0 or epoch == maximum:
            print(f"{path.name}: epoch={epoch}/{maximum} steps={steps} loss={record['train']['total']:.5f} "
                  f"valid={record.get('valid_score')} gain={selected_gain} elapsed={record['elapsed_s']:.1f}s", flush=True)
        if valid_ids and stale >= args.patience:
            break
    torch.save({"state_dict": cpu_state(model), "stats": stats}, path / "last.pt")
    if valid_ids:
        model.load_state_dict(best_weights)
    model.residual_gain.fill_(selected_gain)
    if frozen_digest(model) != before:
        raise RuntimeError("Frozen motion encoder or decoder changed")
    summary = {"completed": True, "records": records, "selected_epoch": selected_epoch,
               "selected_gain": selected_gain, "valid_score": None if baseline is None else best_score,
               "baseline_metrics": baseline, "base_sha256": base_sha, "frozen_sha256": before,
               "train_ids": sorted(train_ids), "valid_ids": sorted(valid_ids),
               "optimizer_steps": steps, "elapsed_s": time.monotonic() - started}
    save_stage(path, model, stats, summary)
    print(f"{path.name}: completed; selected_epoch={selected_epoch}; gain={selected_gain}", flush=True)
    return model, stats, summary


def acceptance(outer, baselines, seeds, complete):
    full = aggregate([row["all"] for trajectories in outer["full"].values() for row in trajectories.values()])
    motion = aggregate([row["all"] for trajectories in outer["motion_only"].values() for row in trajectories.values()])
    tids = sorted(next(iter(outer["full"].values())), key=int)
    base_scores = {name: aggregate([rows[tid]["all"] for tid in tids])["score"]
                   for name, rows in baselines.items()}
    best_name = min(base_scores, key=base_scores.get)
    baseline = aggregate([baselines[best_name][tid]["all"] for tid in tids])
    differences = np.array([np.mean([outer["motion_only"][str(seed)][tid]["all"]["score"] -
                                    outer["full"][str(seed)][tid]["all"]["score"] for seed in seeds]) for tid in tids])
    rng = np.random.default_rng(20261012)
    interval = np.quantile(differences[rng.integers(0, len(tids), (2000, len(tids)))].mean(1), [.025, .975]).tolist()
    reasons = []
    if not complete or len(seeds) != 3:
        reasons.append("full four-fold three-seed protocol incomplete")
    if full["score"] >= motion["score"]:
        reasons.append("force residual does not outperform motion-only")
    if interval[0] <= 0:
        reasons.append("paired trajectory bootstrap improvement over motion-only lower bound <= 0")
    if not guard_passed(full, motion):
        reasons.append("current / future / event / delta / peak / high-band metric worsens >2% vs motion-only")
    if full["score"] > baseline["score"] * .95:
        reasons.append("improvement vs best legacy baseline below 5%")
    seed_scores = {str(seed): aggregate([r["all"] for r in outer["full"][str(seed)].values()])["score"] for seed in seeds}
    for seed in seeds:
        motion_score = aggregate([r["all"] for r in outer["motion_only"][str(seed)].values()])["score"]
        if seed_scores[str(seed)] > motion_score * 1.02:
            reasons.append(f"seed {seed} score worsens >2% vs its motion-only baseline")
    return {"passed": not reasons, "reasons": reasons, "mean_metrics": full, "motion_only_metrics": motion,
            "motion_only_score": motion["score"], "seed_scores": seed_scores,
            "best_baseline": best_name, "baseline_scores": base_scores, "baseline_metrics": baseline,
            "relative_improvement": 1 - full["score"] / baseline["score"],
            "relative_improvement_vs_motion": 1 - full["score"] / motion["score"],
            "paired_bootstrap_95_improvement_vs_motion": interval,
            "thresholds": {"max_regression": .02, "legacy_improvement": .05, "bootstrap_samples": 2000}}


def write_report(output, outer, baselines, gate, old):
    summaries = {name: {seed: aggregate([row["all"] for row in trajectories.values()])
                        for seed, trajectories in seeds.items()} for name, seeds in outer.items()}
    write_json(output / "outer_metrics.json", {"models": outer, "baselines": baselines, "summaries": summaries,
                                              "previous_full_summaries": old["summaries"]["full"]})
    lines = ["# Motion-only + simulation-force residual", "", f"Acceptance: {gate['passed']}", "",
             "full = new frozen-motion + gated-force model; motion_only = identical motion weights, residual disabled.",
             "Eight reused diagnostic trajectories are excluded from all model/gain selection.", "",
             "| Model | Seed | Score | Current RMSE N | +1 RMSE N | +3 RMSE N | +5 RMSE N | Gain |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    selections = read_json(output / "selections.json")
    for name, seeds in summaries.items():
        for seed, row in seeds.items():
            gains = [v["selected_gain"] for v in selections[seed]["full"]] if name == "full" else [0]
            lines.append(f"| {name} | {seed} | {row['score']:.4f} | {row['current_rmse_n']:.4f} | "
                         f"{row['future_1_rmse_n']:.4f} | {row['future_3_rmse_n']:.4f} | {row['future_5_rmse_n']:.4f} | {gains} |")
    lines += ["", f"Score improvement over motion-only: {gate['relative_improvement_vs_motion']:.2%}.",
              "", "Gate reasons: " + "; ".join(gate["reasons"]), "",
              "Final checkpoints require the force_gap_motion_force_v1 runtime; the v1 runtime cannot decode this format."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, default=Path(DEFAULT_BASE))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gains", type=float, nargs="+", default=[0., .25, .5, 1.])
    parser.add_argument("--residual-l2", type=float, default=1e-3)
    parser.add_argument("--real-loss-weight", type=float, default=.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1 or not (math.isfinite(args.lr) and args.lr > 0):
        parser.error("epochs, patience, batch-size, threads and lr must be positive")
    if (len(set(args.seeds)) != len(args.seeds) or len(set(args.folds)) != len(args.folds) or
            not set(args.folds) <= set(range(4))):
        parser.error("distinct seeds and distinct folds in 0..3 required")
    if (0. not in args.gains or len(set(args.gains)) != len(args.gains) or
            any(not 0 <= g <= 1 for g in args.gains) or
            not (math.isfinite(args.residual_l2) and args.residual_l2 >= 0) or
            not (math.isfinite(args.real_loss_weight) and args.real_loss_weight >= 0)):
        parser.error("gains must be distinct in [0,1], include 0; loss weights must be nonnegative")
    args.base_run, args.output_dir = args.base_run.resolve(), args.output_dir.resolve()
    base_recipe = read_json(args.base_run / "recipe.json")
    if not set(args.seeds) <= set(base_recipe["seeds"]) or base_recipe["folds"] != FOLDS:
        parser.error("base run does not contain these seeds or the expected folds")
    # Reusing pretrained normalization/weights requires exactly the original data and implementation.
    dependencies = Path(__file__).resolve().parent.parent / "force_gap_forecast_v1"
    for name in ("__init__.py", "model.py", "data.py", "losses.py", "metrics.py", "train.py", "common.py"):
        if sha256(dependencies / name) != base_recipe["source_sha256"][name]:
            raise ValueError(f"Base implementation changed: {name}; use the original version or retrain the base")
    labels = Path(base_recipe["labels"])
    if sha256(labels) != base_recipe["label_sha256"]:
        raise ValueError("Base trajectory labels changed")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on the external GPU environment or use --device cpu")
    output = args.output_dir
    if output.exists() and any(p.name != "train.log" for p in output.iterdir()) and not args.resume:
        raise FileExistsError("Choose a new output directory or a matching --resume")
    print("Checking paired data and cached motion checkpoints...", flush=True)
    episodes, audit = load_pairs(base_recipe["real_root"], base_recipe["sim_root"], labels)
    if audit != base_recipe["data"]:
        raise ValueError("Data differ from the pretrained motion run; frozen weights cannot be reused")
    checkpoints = {}
    for seed in args.seeds:
        for fold in args.folds:
            for phase in ("select", "refit"):
                path = args.base_run / "stages" / f"seed{seed}_fold{fold}_motion_only_student_{phase}" / "model.pt"
                checkpoints[str(path.relative_to(args.base_run))] = sha256(path)
    source_files = (list(Path(__file__).parent.glob("*.py")) +
                    [dependencies / n for n in base_recipe["source_sha256"]] +
                    [Path(__file__).parent.parent / "visualize_force_gap_forecast_v1.py"])
    recipe = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ("resume", "output_dir", "device", "no_plots")}
    recipe.update(artifact_format=FORMAT, variants=["full", "motion_only"], folds=FOLDS, active_folds=args.folds, train_ids=TRAIN_IDS,
                  diagnostic_ids=HOLDOUT, labels=str(labels), real_root=base_recipe["real_root"], sim_root=base_recipe["sim_root"],
                  label_sha256=base_recipe["label_sha256"], data=audit, base_checkpoints=checkpoints,
                  base_recipe_sha256=sha256(args.base_run / "recipe.json"),
                  base_outer_sha256=sha256(args.base_run / "outer_metrics.json"),
                  base_selections_sha256=sha256(args.base_run / "selections.json"),
                  source_sha256={str(p.parent.name + "/" + p.name): sha256(p) for p in source_files})
    if args.resume and (output / "recipe.json").is_file() and read_json(output / "recipe.json") != recipe:
        raise ValueError("Resume code/data/checkpoints/parameters changed; use a new run")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "recipe.json", recipe)
    write_json(output / "data_audit.json", audit)
    for p in source_files:
        dest = output / "source_snapshot" / p.parent.name / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    old = read_json(args.base_run / "outer_metrics.json")
    old_selections = read_json(args.base_run / "selections.json")
    outer = {name: {str(seed): {} for seed in args.seeds} for name in ("full", "motion_only")}
    selections = {str(seed): {"full": []} for seed in args.seeds}
    started = time.monotonic()
    for fold in args.folds:
        valid_ids = FOLDS[fold]
        train_ids = sorted(set(TRAIN_IDS) - set(valid_ids))
        order = np.random.default_rng(42 + fold).permutation(train_ids)
        inner_valid = sorted(map(int, order[:6]))
        inner_train = sorted(set(train_ids) - set(inner_valid))
        for seed in args.seeds:
            print(f"Fold {fold+1}/4, seed {seed}: fitting force correction", flush=True)
            stage_base = args.base_run / "stages" / f"seed{seed}_fold{fold}_motion_only_student"
            selected, _, choice = fit_stage(Path(str(stage_base) + "_select") / "model.pt", episodes,
                                            inner_train, inner_valid, args, seed + 2000,
                                            output / "stages" / f"seed{seed}_fold{fold}_full_select")
            del selected
            model, stats, _ = fit_stage(Path(str(stage_base) + "_refit") / "model.pt", episodes,
                                        train_ids, [], args, seed + 2000,
                                        output / "stages" / f"seed{seed}_fold{fold}_full_refit",
                                        epochs=choice["selected_epoch"], gain=choice["selected_gain"])
            outer["full"][str(seed)].update(evaluate(model, episodes, valid_ids, stats, 1,
                output / "predictions" / f"seed{seed}_fold{fold}_full"))
            selected_gain = float(model.residual_gain.item())
            model.residual_gain.zero_()
            baseline = evaluate(model, episodes, valid_ids, stats, 1,
                output / "predictions" / f"seed{seed}_fold{fold}_motion_only")
            model.residual_gain.fill_(selected_gain)
            numerical_deltas = []
            for tid, row in baseline.items():
                for key in ("score", "current_rmse_n", "future_1_rmse_n", "future_5_rmse_n"):
                    previous = old["models"]["motion_only"][str(seed)][tid]["all"][key]
                    numerical_deltas.append(abs(row["all"][key] - previous) / max(abs(previous), 1e-8))
                    # CPU and Ampere/cuDNN evaluations can differ slightly. All
                    # fusion comparisons use this same-device recomputed baseline.
                    if not np.isclose(row["all"][key], previous, rtol=2e-3, atol=1e-4):
                        raise RuntimeError(f"Reused baseline mismatch: seed {seed}, trajectory {tid}, {key}")
            outer["motion_only"][str(seed)].update(baseline)
            selections[str(seed)]["full"].append({"fold": fold, "outer_train": train_ids, "outer_valid": valid_ids,
                "inner_train": inner_train, "inner_valid": inner_valid,
                "selected_epoch": choice["selected_epoch"], "selected_gain": choice["selected_gain"],
                "baseline_device": args.device, "baseline_max_relative_delta_vs_saved": max(numerical_deltas)})
            write_json(output / "cv_progress.json", {"outer": outer, "baselines": old["baselines"], "selections": selections})
            del model
    complete = set(args.folds) == set(range(4))
    gate = acceptance(outer, old["baselines"], args.seeds, complete)
    write_json(output / "acceptance.json", gate)
    write_json(output / "selections.json", selections)
    write_report(output, outer, old["baselines"], gate, old)
    if not complete:
        write_json(output / "run_summary.json", {"status": "diagnostic", "final_encoder_created": False,
                   "folds": args.folds, "elapsed_s": time.monotonic() - started})
        print(f"Reduced CV completed: {output / 'report.md'}. Run all four folds for final encoder weights.", flush=True)
        return
    final = {}
    status = ("waveform_validated" if audit["actual_sim_state_verified"] else "replay_validated") if gate["passed"] else "diagnostic"
    for seed in args.seeds:
        motion_choices = old_selections[str(seed)]["motion_only"]
        teacher_epoch = max(1, int(np.median([v["teacher_epoch"] for v in motion_choices])))
        student_epoch = max(1, int(np.median([v["student_epoch"] for v in motion_choices])))
        print(f"Final seed {seed}: training all-trajectory motion foundation", flush=True)
        motion_args = argparse.Namespace(**vars(args))
        for name in ("lr", "batch_size", "patience", "max_epochs"):
            setattr(motion_args, name, base_recipe[name])
        teacher, _, _ = run_stage(episodes, TRAIN_IDS, [], motion_args, seed,
            output / "base_stages" / f"seed{seed}_final_motion_teacher", motion_only=True, epochs=teacher_epoch)
        _, _, _ = run_stage(episodes, TRAIN_IDS, [], motion_args, seed + 1000,
            output / "base_stages" / f"seed{seed}_final_motion_student", teacher=teacher, motion_only=True, epochs=student_epoch)
        del teacher
        choices = selections[str(seed)]["full"]
        median_gain = float(np.median([v["selected_gain"] for v in choices]))
        final_gain = min(args.gains, key=lambda g: (abs(g - median_gain), g))
        active_epochs = [v["selected_epoch"] for v in choices if v["selected_gain"] > 0]
        final_epochs = max(1, int(np.median(active_epochs))) if final_gain and active_epochs else 0
        model, stats, summary = fit_stage(output / "base_stages" / f"seed{seed}_final_motion_student" / "model.pt",
            episodes, TRAIN_IDS, [], args, seed + 2000, output / "stages" / f"seed{seed}_final_full_refit",
            epochs=final_epochs, gain=final_gain)
        checkpoint = output / f"encoder_seed{seed}.pt"
        payload = {"artifact_format": FORMAT, "layout": LAYOUT, "stats": stats, "state_dict": cpu_state(model),
                   "seed": seed, "history": 32, "past": 8, "future": 5, "train_ids": TRAIN_IDS,
                   "diagnostic_ids": HOLDOUT, "status": status, "stage": summary}
        torch.save(payload, checkpoint)
        diagnostic = {domain: evaluate(model, episodes, HOLDOUT, stats, d,
                        output / "diagnostic_predictions" / f"seed{seed}_{domain}")
                      for d, domain in enumerate(("real", "sim"))}
        write_json(output / f"diagnostic_seed{seed}.json", diagnostic)
        final[str(seed)] = {"checkpoint": checkpoint.name, "encoder_sha256": sha256(checkpoint),
                           "teacher_epoch": teacher_epoch, "student_epoch": student_epoch,
                           "residual_epoch": final_epochs, "residual_gain": final_gain}
        del model
    selected_seed = min(args.seeds, key=lambda seed: gate["seed_scores"][str(seed)])
    manifest = {"artifact_format": FORMAT, "layout": LAYOUT, "status": status,
                **final[str(selected_seed)], "selected_seed": selected_seed, "all_seeds": final,
                "selection_source": "outer CV; reused diagnostics excluded", "history": 32, "future": 5,
                "policy_history": 10, "actual_sim_state_verified": audit["actual_sim_state_verified"],
                "real_root": base_recipe["real_root"], "sim_root": base_recipe["sim_root"],
                "force_contract": "aligned_base_xyz_n_v1", "runtime": "force_gap_motion_force_v1.runtime.MotionForceEncoder",
                "base_run": str(args.base_run), "elapsed_s": time.monotonic() - started,
                "acceptance": "acceptance.json", "audit": "data_audit.json",
                "acceptance_sha256": sha256(output / "acceptance.json"), "audit_sha256": sha256(output / "data_audit.json")}
    write_json(output / "manifest.json", manifest)
    if not args.no_plots:
        from visualize_force_gap_forecast_v1 import generate_visualizations
        generate_visualizations(output)
    print(f"Completed: {output / 'manifest.json'}; status={status}; reasons={gate['reasons']}", flush=True)


if __name__ == "__main__":
    main()
