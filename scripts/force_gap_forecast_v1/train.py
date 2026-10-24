"""Nested trajectory CV, immutable diagnostics, waveform-gated final artifacts.

No geometry / phase-classification objective and no simulation-force decoder.
Each stage is restartable only when the recorded inputs and recipe match.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import FORMAT, LAYOUT
from . import baselines
from .common import sha256, write_json
from .data import Blocks, FOLDS, HOLDOUT, TRAIN_IDS, fit_stats, load_pairs
from .losses import objective
from .metrics import aggregate, predict_episode, score_prediction
from .model import ForceForecast


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_digest(model, frozen_only=False):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if frozen_only and name.startswith("sim_encoder."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def evaluate(model, episodes, ids, stats, domain, motion_only=False, save_dir=None):
    result = {}
    for i in ids:
        ep = episodes[(i, domain)]
        pred = predict_episode(model, ep, stats, domain, motion_only)
        result[str(i)] = score_prediction(pred, episodes[(i, 0)], np.asarray(stats["target"]["scale"]), ep)
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_dir / f"traj_{i}.npz", **pred, real_force=episodes[(i, 0)].force,
                                input_force=ep.force, time=ep.time, real_time=episodes[(i, 0)].time)
    return result


def baseline_guard(candidate, baseline):
    return all(candidate.get(key) is not None and baseline.get(key) is not None
               and candidate[key] <= baseline[key] * 1.02 + 1e-8
               for key in ("delta_rmse_n", "peak_amplitude_mae_n", "high_band_mse"))


def run_stage(episodes, train_ids, valid_ids, args, seed, path, teacher=None, epochs=None,
              motion_only=False, alignment=True, frequency=True, guard=None):
    checkpoint, history_path = path / "model.pt", path / "history.json"
    if checkpoint.is_file() and history_path.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = ForceForecast(saved["stats"]).to(args.device)
        model.load_state_dict(saved["state_dict"])
        if teacher is not None:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("sim_encoder."))
        return model, saved["stats"], json.loads(history_path.read_text())
    path.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    stats = fit_stats(episodes, train_ids)
    if teacher is None:
        model = ForceForecast(stats).to(args.device)
        # The unused student is not optimized during teacher training.
        for parameter in model.sim_encoder.parameters():
            parameter.requires_grad_(False)
        domain = 0
    else:
        model = copy.deepcopy(teacher).to(args.device)
        # Teacher and frozen decoder scales were fit on exactly these train IDs.
        if not torch.allclose(model.target_scale.cpu(), torch.tensor(stats["target"]["scale"])):
            raise ValueError("Teacher normalization/train fold mismatch")
        model.adapt()
        domain = 1
    frozen_digest = state_digest(model, True) if domain else None
    dataset = Blocks(episodes, train_ids, stats, motion_only)
    steps_per_epoch = max(1, math.ceil(len(dataset) / (args.batch_size * 8)))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    maximum = epochs if epochs is not None else args.max_epochs
    best_any = best_guarded = None
    best_score, best_guarded_score = math.inf, math.inf
    stale, records, elapsed_start, steps = 0, [], time.monotonic(), 0
    for epoch in range(1, maximum + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch * 7919)
        sampler = WeightedRandomSampler(dataset.weights, steps_per_epoch * args.batch_size, replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        losses = []
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model.block(batch, domain, motion_only)
            teacher_output = None
            if domain:
                with torch.no_grad():
                    teacher_output = model.block(batch, 0, motion_only)
            loss, row = objective(model, output, batch, teacher_output, alignment, frequency)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {path}, epoch={epoch}; diagnostics retained")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3, error_if_nonfinite=True)
            optimizer.step()
            losses.append(row)
            steps += 1
        record = {"epoch": epoch, "steps": steps, "elapsed_s": time.monotonic() - elapsed_start,
                  "train": aggregate(losses)}
        if valid_ids:
            val = evaluate(model, episodes, valid_ids, stats, domain, motion_only)
            metric = aggregate([row["all"] for row in val.values()])
            score = metric["teacher_score" if domain == 0 else "score"]
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite validation score")
            eligible = guard is None or baseline_guard(metric, guard)
            record.update(valid=metric, eligible=eligible)
            if score < best_score:
                best_score, best_any, stale = score, (epoch, cpu_state(model)), 0
            else:
                stale += 1
            if eligible and score < best_guarded_score:
                best_guarded_score, best_guarded = score, (epoch, cpu_state(model))
        records.append(record)
        write_json(history_path, {"records": records, "completed": False})
        if epoch == 1 or epoch % 10 == 0 or epoch == maximum:
            print(f"{path.name}: epoch={epoch}/{maximum} steps={steps} loss={record['train']['total']:.5f} "
                  f"valid={record.get('valid', {}).get('score')} elapsed={record['elapsed_s']:.1f}s", flush=True)
        if valid_ids and stale >= args.patience:
            break
    torch.save({"state_dict": cpu_state(model), "stats": stats, "last_epoch": records[-1]["epoch"]}, path / "last.pt")
    if valid_ids:
        chosen = best_guarded or best_any
        selected, weights = chosen
        model.load_state_dict(weights)
    else:
        selected = maximum
    if domain and state_digest(model, True) != frozen_digest:
        raise RuntimeError("Frozen teacher/decoder changed during student training")
    summary = {"records": records, "selected_epoch": selected, "completed": True,
               "guard_passed": not valid_ids or best_guarded is not None,
               "optimizer_steps": steps, "elapsed_s": time.monotonic() - elapsed_start,
               "train_ids": sorted(train_ids), "valid_ids": sorted(valid_ids), "frozen_sha256": frozen_digest,
               "motion_only": motion_only, "alignment": alignment, "frequency": frequency}
    # Keep the selected and the last diagnostic states; never discard a failed-gate run.
    torch.save({"state_dict": cpu_state(model), "stats": stats, "stage": summary}, checkpoint)
    write_json(history_path, summary)
    return model, stats, summary


def acceptance(outer, baseline_rows, args):
    main = outer["full"]
    ids = TRAIN_IDS
    baseline_scores = {name: np.mean([baseline_rows[name][str(i)]["all"]["score"] for i in ids])
                       for name in baseline_rows}
    best_name = min(baseline_scores, key=baseline_scores.get)
    base = baseline_rows[best_name]
    average = {str(i): aggregate([main[str(seed)][str(i)]["all"] for seed in args.seeds]) for i in ids}
    model_score = np.array([average[str(i)]["score"] for i in ids])
    base_score = np.array([base[str(i)]["all"]["score"] for i in ids])
    relative = 1 - model_score.mean() / max(base_score.mean(), 1e-12)
    rng = np.random.default_rng(20261012)
    indices = rng.integers(0, len(ids), size=(2000, len(ids)))
    improvements = base_score - model_score
    interval = np.quantile(improvements[indices].mean(axis=1), [.025, .975]).tolist()
    averaged = aggregate(list(average.values()))
    baseline = aggregate([base[str(i)]["all"] for i in ids])
    reasons = []
    if relative < .05:
        reasons.append("waveform improvement below 5%")
    if interval[0] <= 0:
        reasons.append("paired trajectory bootstrap improvement lower bound <= 0")
    if not baseline_guard(averaged, baseline):
        reasons.append("delta / peak / high-band error worsens by more than 2%")
    seed_scores = {str(seed): aggregate([main[str(seed)][str(i)]["all"] for i in ids])["score"] for seed in args.seeds}
    if any(score > base_score.mean() * 1.02 for score in seed_scores.values()):
        reasons.append("at least one seed worsens waveform error by more than 2%")
    motion_score = None
    if "motion_only" not in outer:
        reasons.append("motion-only control missing; reduced runs cannot be promoted")
    else:
        motion_score = np.mean([outer["motion_only"][str(seed)][str(i)]["all"]["score"] for seed in args.seeds for i in ids])
        if model_score.mean() >= motion_score:
            reasons.append("force-input model does not outperform motion-only model")
    if set(args.variants) != {"full", "no_alignment", "no_frequency", "motion_only"} or len(args.seeds) != 3:
        reasons.append("full three-seed ablation protocol incomplete")
    return {"passed": not reasons, "reasons": reasons, "best_baseline": best_name,
            "baseline_scores": baseline_scores, "relative_improvement": float(relative),
            "paired_bootstrap_95_improvement": interval, "seed_scores": seed_scores,
            "motion_only_score": None if motion_score is None else float(motion_score),
            "mean_metrics": averaged, "baseline_metrics": baseline,
            "thresholds": {"improvement": .05, "max_error_regression": .02, "bootstrap_samples": 2000}}


def write_reports(output, outer, baselines_rows, gate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries = {name: {seed: aggregate([row["all"] for row in rows.values()]) for seed, rows in seeds.items()}
                 for name, seeds in outer.items()}
    write_json(output / "outer_metrics.json", {"models": outer, "baselines": baselines_rows, "summaries": summaries})
    lines = ["# Force gap forecast v1", "", "Acceptance: " + str(gate["passed"]), "",
             "Existing eight diagnostic trajectories are reused, not a new blind test.", "",
             "| Model | Seed | Score | Current RMSE N | Future 1 RMSE N | Future 5 RMSE N |",
             "|---|---|---|---|---|---|"]
    for name, seeds in summaries.items():
        for seed, row in seeds.items():
            lines.append(f"| {name} | {seed} | {row['score']:.4f} | {row['current_rmse_n']:.4f} | "
                         f"{row['future_1_rmse_n']:.4f} | {row['future_5_rmse_n']:.4f} |")
    lines += ["", "Gate reasons: " + "; ".join(gate["reasons"]), "",
              "Per-axis, per-phase, final-eight-frame and peak-timing metrics: outer_metrics.json."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    for path in output.glob("stages/*/history.json"):
        history = json.loads(path.read_text())
        rows = history["records"]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot([r["epoch"] for r in rows], [r["train"]["total"] for r in rows], label="train loss")
        if "valid" in rows[0]:
            kind = "teacher_score" if "teacher" in path.parent.name else "score"
            ax.plot([r["epoch"] for r in rows], [r["valid"][kind] for r in rows], label="validation score")
        ax.set_xlabel("epoch"); ax.legend(); fig.tight_layout()
        fig.savefig(path.parent / "curves.png", dpi=140); plt.close(fig)
    # Show target/recovered/input and one-step forecast at its TARGET timestamp.
    for path in output.glob("diagnostic_predictions/*/traj_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            t = data["real_time"] - data["real_time"][0]
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            for axis, ax in enumerate(axes):
                ax.plot(t, data["real_force"][:, axis], label="Real target", linewidth=1.5)
                ax.plot(t, data["input_force"][:, axis], label="Input force", alpha=.5)
                ax.plot(t, data["current"][:, axis], label="Current recovered", alpha=.8)
                ax.plot(t[1:], data["future"][:-1, 0, axis], label="Future +1", alpha=.7)
                ax.set_ylabel(f"F{'xyz'[axis]} (N)")
            axes[0].legend(ncol=4); axes[-1].set_xlabel("Real reference time (s)")
            fig.tight_layout(); fig.savefig(path.with_suffix(".png"), dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/insertion_quality_labels_v2.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", default=["full", "no_alignment", "no_frequency", "motion_only"],
                        choices=["full", "no_alignment", "no_frequency", "motion_only"])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1 or not args.lr > 0:
        parser.error("epochs/patience/batch-size/threads/lr must be positive")
    if len(set(args.seeds)) != len(args.seeds) or "full" not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("distinct seeds/variants and full variant required")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the external GPU environment or --device cpu")
    output = args.output_dir.resolve()
    if output.exists() and any(p.name != "train.log" for p in output.iterdir()) and not args.resume:
        raise FileExistsError("Output exists; choose a new run directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    episodes, audit = load_pairs(args.real_root, args.sim_root, args.labels)
    recipe = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in ("resume", "output_dir", "device")}
    recipe.update(artifact_format=FORMAT, folds=FOLDS, train_ids=TRAIN_IDS, diagnostic_ids=HOLDOUT,
                  source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                  label_sha256=sha256(args.labels), data=audit)
    recipe_path = output / "recipe.json"
    if args.resume and recipe_path.is_file() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError("Resume recipe / sources / data changed; start a new immutable run")
    write_json(recipe_path, recipe)
    write_json(output / "data_audit.json", audit)
    source = output / "source_snapshot"
    source.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob("*.py"):
        (source / p.name).write_bytes(p.read_bytes())
    outer = {variant: {str(seed): {} for seed in args.seeds} for variant in args.variants}
    baseline_rows = {name: {} for name in ("raw", "affine", "arx")}
    selections = {}
    started = time.monotonic()
    for fold, valid_ids in enumerate(FOLDS):
        train_ids = sorted(set(TRAIN_IDS) - set(valid_ids))
        order = np.random.default_rng(42 + fold).permutation(train_ids)
        inner_valid = sorted(map(int, order[:6]))
        inner_train = sorted(set(train_ids) - set(inner_valid))
        inner_stats = fit_stats(episodes, inner_train)
        alpha = baselines.choose_alpha(episodes, inner_train, inner_valid, inner_stats)
        baseline_inner = {}
        outer_stats = fit_stats(episodes, train_ids)
        for name in baseline_rows:
            inner_model = {"kind": "raw"} if name == "raw" else baselines.fit(episodes, inner_train, inner_stats, name, alpha if name == "arx" else .1)
            rows = baselines.evaluate(inner_model, episodes, inner_valid, inner_stats)
            baseline_inner[name] = aggregate([row["all"] for row in rows.values()])
            outer_model = {"kind": "raw"} if name == "raw" else baselines.fit(episodes, train_ids, outer_stats, name, alpha if name == "arx" else .1)
            baseline_rows[name].update(baselines.evaluate(outer_model, episodes, valid_ids, outer_stats))
        guard = min(baseline_inner.values(), key=lambda row: row["score"])
        for seed in args.seeds:
            teacher_cache = {}
            for variant in args.variants:
                motion_only, frequency, alignment = variant == "motion_only", variant != "no_frequency", variant != "no_alignment"
                teacher_kind = "motion" if motion_only else "full" if frequency else "no_frequency"
                prefix = f"seed{seed}_fold{fold}_{variant}"
                if teacher_kind not in teacher_cache:
                    inner_teacher, _, teacher_summary = run_stage(episodes, inner_train, inner_valid, args, seed,
                        output / "stages" / f"seed{seed}_fold{fold}_{teacher_kind}_teacher_select", motion_only=motion_only, frequency=frequency)
                    teacher_cache[teacher_kind] = inner_teacher, teacher_summary
                inner_teacher, teacher_summary = teacher_cache[teacher_kind]
                _, _, student_summary = run_stage(episodes, inner_train, inner_valid, args, seed + 1000,
                    output / "stages" / f"{prefix}_student_select", teacher=inner_teacher,
                    motion_only=motion_only, frequency=frequency, alignment=alignment, guard=guard)
                outer_teacher, _, _ = run_stage(episodes, train_ids, [], args, seed,
                    output / "stages" / f"seed{seed}_fold{fold}_{teacher_kind}_teacher_refit", epochs=teacher_summary["selected_epoch"],
                    motion_only=motion_only, frequency=frequency)
                student, stats, _ = run_stage(episodes, train_ids, [], args, seed + 1000,
                    output / "stages" / f"{prefix}_student_refit", teacher=outer_teacher,
                    epochs=student_summary["selected_epoch"], motion_only=motion_only, frequency=frequency, alignment=alignment)
                rows = evaluate(student, episodes, valid_ids, stats, 1, motion_only, output / "predictions" / prefix)
                outer[variant][str(seed)].update(rows)
                selections.setdefault(str(seed), {}).setdefault(variant, []).append({
                    "fold": fold, "outer_train": train_ids, "outer_valid": valid_ids,
                    "inner_train": inner_train, "inner_valid": inner_valid, "arx_alpha": alpha,
                    "teacher_epoch": teacher_summary["selected_epoch"], "student_epoch": student_summary["selected_epoch"],
                    "student_guard_passed": student_summary["guard_passed"]})
                write_json(output / "cv_progress.json", {"outer": outer, "baselines": baseline_rows, "selections": selections})
    gate = acceptance(outer, baseline_rows, args)
    status = ("policy_train_ready" if audit["actual_sim_state_verified"] else "replay_validated") if gate["passed"] else "diagnostic"
    final = {}
    for seed in args.seeds:
        rows = selections[str(seed)]["full"]
        teacher_epoch = max(1, int(np.median([row["teacher_epoch"] for row in rows])))
        student_epoch = max(1, int(np.median([row["student_epoch"] for row in rows])))
        teacher, stats, _ = run_stage(episodes, TRAIN_IDS, [], args, seed,
            output / "stages" / f"seed{seed}_final_teacher", epochs=teacher_epoch)
        student, stats, _ = run_stage(episodes, TRAIN_IDS, [], args, seed + 1000,
            output / "stages" / f"seed{seed}_final_student", teacher=teacher, epochs=student_epoch)
        path = output / f"encoder_seed{seed}.pt"
        payload = {"artifact_format": FORMAT, "layout": LAYOUT, "stats": stats, "state_dict": cpu_state(student),
                   "seed": seed, "history": 32, "past": 8, "future": 5,
                   "train_ids": TRAIN_IDS, "diagnostic_ids": HOLDOUT, "status": status,
                   "teacher_epoch": teacher_epoch, "student_epoch": student_epoch}
        torch.save(payload, path)
        diagnostic = {domain: evaluate(student, episodes, HOLDOUT, stats, d,
                      save_dir=output / "diagnostic_predictions" / f"seed{seed}_{domain}")
                      for d, domain in enumerate(("real", "sim"))}
        write_json(output / f"diagnostic_seed{seed}.json", diagnostic)
        final[str(seed)] = {"checkpoint": path.name, "encoder_sha256": sha256(path),
                           "teacher_epoch": teacher_epoch, "student_epoch": student_epoch}
    selected_seed = min(args.seeds, key=lambda seed: gate["seed_scores"][str(seed)])
    manifest = {"artifact_format": FORMAT, "layout": LAYOUT, "status": status,
                **final[str(selected_seed)], "selected_seed": selected_seed, "all_seeds": final,
                "selection_source": "outer CV only; reused diagnostic eight excluded",
                "history": 32, "future": 5, "policy_history": 10, "baseline_frames": 0,
                "real_root": str(args.real_root.resolve()), "sim_root": str(args.sim_root.resolve()),
                "actual_sim_state_verified": audit["actual_sim_state_verified"], "elapsed_s": time.monotonic() - started,
                "force_contract": "aligned_base_xyz_n_v1", "acceptance": "acceptance.json", "audit": "data_audit.json"}
    write_json(output / "acceptance.json", gate)
    manifest["acceptance_sha256"] = sha256(output / "acceptance.json")
    manifest["audit_sha256"] = sha256(output / "data_audit.json")
    write_json(output / "selections.json", selections)
    write_json(output / "manifest.json", manifest)
    write_reports(output, outer, baseline_rows, gate)
    print(f"Completed: {output / 'manifest.json'}; status={status}; reasons={gate['reasons']}", flush=True)


if __name__ == "__main__":
    main()
