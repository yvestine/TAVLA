"""Nested trajectory CV of joint force fusion, with an equal-capacity no-force control."""
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

from force_gap_forecast_v1 import baselines
from force_gap_forecast_v1.common import sha256, write_json
from force_gap_forecast_v1.data import Blocks, FOLDS, HOLDOUT, TRAIN_IDS, fit_stats, load_pairs
from force_gap_forecast_v1.metrics import aggregate
from force_gap_forecast_v1.train import cpu_state, run_stage, seed_all
from . import FORMAT, LAYOUT
from .losses import forecast_loss
from .metrics import evaluate, evaluate_gains, force_reliance, persistence
from .model import JointMotionForceForecast

DEFAULT_BASE = "eval_outputs/force_gap_forecast_v1/20261012_155026_551478"
VARIANTS = ("full", "motion_control", "motion_only")
GUARDS = ("future_1_rmse_n", "future_3_rmse_n", "future_5_rmse_n", "future_5_z_rmse_n",
          "future_1_peak_event_rmse_n", "future_5_peak_event_rmse_n", "future_z_peak_mae_n",
          "future_z_peak_timing_mae_s")


def read_json(path):
    return json.loads(Path(path).read_text())


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.motion.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def guard_passed(candidate, reference, tolerance, domain=1):
    keys = GUARDS + (("current_rmse_n",) if domain == 1 else ())
    # Empty event sets are reported as unavailable, rather than invented zero errors.
    return all(candidate.get(key) is not None and candidate[key] <= reference[key] * (1 + tolerance) + 1e-8
               for key in keys if reference.get(key) is not None)


def load_motion(checkpoint, episodes, ids):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stage = payload.get("stage", {})
    if not stage.get("motion_only") or stage.get("train_ids") != sorted(ids):
        raise ValueError(f"Motion checkpoint does not match the train fold: {checkpoint}")
    if payload["stats"] != fit_stats(episodes, ids):
        raise ValueError(f"Motion normalization/data differ: {checkpoint}; use --base-mode fresh")
    return payload


def domain_state(model, domain):
    prefixes = (f"adapters.{domain}.", f"future_heads.{domain}.")
    return {key: value for key, value in cpu_state(model).items() if key.startswith(prefixes)}


def save_stage(path, model, stats, summary):
    temporary = path / "model.pt.tmp"
    torch.save(dict(artifact_format=FORMAT, layout=LAYOUT, stats=stats, use_force=model.use_force,
                    state_dict=cpu_state(model), stage=summary), temporary)
    temporary.replace(path / "model.pt")
    write_json(path / "history.json", summary)


def fit_stage(checkpoint, episodes, train_ids, valid_ids, args, seed, path, use_force=True, refit=None):
    payload = load_motion(checkpoint, episodes, train_ids)
    stats = payload["stats"]
    seed_all(seed)
    model = JointMotionForceForecast(stats, payload["state_dict"], use_force).to(args.device)
    identity = dict(base_sha256=sha256(checkpoint), train_ids=sorted(train_ids), valid_ids=sorted(valid_ids),
                    use_force=use_force, refit=refit)
    if (path / "model.pt").is_file() and (path / "history.json").is_file():
        saved = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        summary = read_json(path / "history.json")
        if (not args.resume or saved.get("artifact_format") != FORMAT or saved.get("stage") != summary or
                not summary.get("completed") or any(summary.get(k) != v for k, v in identity.items())):
            raise ValueError(f"Existing/inconsistent stage; use matching --resume or a new run: {path}")
        model.load_state_dict(saved["state_dict"])
        return model, stats, summary
    if not valid_ids and refit is None:
        raise ValueError("Refit must use inner-selected independent domain epochs and gains")
    path.mkdir(parents=True, exist_ok=True)
    before = frozen_digest(model)
    baseline, best, best_weights = {}, {}, {}
    for domain in (0, 1):
        if valid_ids:
            rows = evaluate(model, episodes, valid_ids, stats, domain, gain=0.)
            baseline[domain] = aggregate([row["all"] for row in rows.values()])
            best[domain] = dict(epoch=0, gain=0., forecast_score_n=baseline[domain]["forecast_score_n"])
        else:
            best[domain] = dict(refit[str(domain)])
            if best[domain]["epoch"] < 0 or not 0 <= best[domain]["gain"] <= 1:
                raise ValueError("Invalid refit epoch/gain")
            if best[domain]["epoch"] == 0 and best[domain]["gain"] != 0:
                raise ValueError("An untrained adapter must have gain=0")
        best_weights[domain] = domain_state(model, domain)
    maximum = args.max_epochs if valid_ids else max(best[d]["epoch"] for d in (0, 1))
    # Include decoder copies in the optimizer from the beginning; warmup suppresses their gradients.
    parameters = list(model.adapters.parameters()) + list(model.future_heads.parameters())
    optimizer = torch.optim.AdamW([
        dict(params=model.adapters.parameters(), lr=args.lr),
        dict(params=model.future_heads.parameters(), lr=args.decoder_lr)], weight_decay=1e-4)
    dataset = Blocks(episodes, train_ids, stats)
    steps_per_epoch = max(1, math.ceil(len(dataset) / (args.batch_size * 8)))
    records, stale, steps = [], {0: 0, 1: 0}, 0
    started = time.monotonic()
    for epoch in range(1, maximum + 1):
        model.train()
        model.set_decoder_trainable(epoch > args.warmup_epochs)
        model.domain_gains.fill_(1.)
        sampler = WeightedRandomSampler(dataset.weights, steps_per_epoch * args.batch_size, replacement=True,
                                        generator=torch.Generator().manual_seed(seed + epoch * 7919))
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        losses = []
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            total, row = None, {}
            for domain in (0, 1):
                if not valid_ids and epoch > best[domain]["epoch"]:
                    continue
                loss, detail = forecast_loss(model, model.block(batch, domain), batch, domain, args)
                total = loss if total is None else total + loss
                row.update({f"{('real', 'sim')[domain]}_{k}": v for k, v in detail.items()})
            if total is None:
                continue
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss: {path}, epoch={epoch}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 3., error_if_nonfinite=True)
            optimizer.step()
            losses.append(row)
            steps += 1
        record = dict(epoch=epoch, optimizer_steps=steps, elapsed_s=time.monotonic() - started,
                      decoder_unfrozen=epoch > args.warmup_epochs, train=aggregate(losses))
        if valid_ids:
            record["gain_metrics"] = {}
            for domain in (0, 1):
                grid = evaluate_gains(model, episodes, valid_ids, stats, domain, args.gains)
                improved = False
                for gain in sorted(grid):
                    metric = aggregate([row["all"] for row in grid[gain].values()])
                    record["gain_metrics"][f"{domain}:{gain}"] = metric
                    score = metric["forecast_score_n"]
                    if not np.isfinite(score):
                        raise FloatingPointError("Non-finite validation forecast score")
                    eligible = (gain > 0 and guard_passed(metric, baseline[domain], args.guard_tolerance, domain)
                                and score <= baseline[domain]["forecast_score_n"] * (1 - args.inner_min_improvement))
                    if eligible and score < best[domain]["forecast_score_n"]:
                        best[domain] = dict(epoch=epoch, gain=gain, forecast_score_n=score)
                        best_weights[domain], improved = domain_state(model, domain), True
                stale[domain] = 0 if improved else stale[domain] + 1
            record["best_domains"] = {str(d): dict(best[d]) for d in (0, 1)}
        records.append(record)
        write_json(path / "history.json", dict(completed=False, records=records))
        if epoch == 1 or epoch % 10 == 0 or epoch == maximum:
            print(f"{path.name}: epoch={epoch}/{maximum}; decoder={record['decoder_unfrozen']}; "
                  f"Real={best[0]}; Sim={best[1]}; elapsed={record['elapsed_s']:.1f}s", flush=True)
        # Always give the unfrozen decoder at least patience epochs to improve.
        if valid_ids and epoch >= args.warmup_epochs + args.patience and min(stale.values()) >= args.patience:
            break
    torch.save(dict(state_dict=cpu_state(model), stats=stats, use_force=use_force), path / "last.pt")
    if valid_ids:
        for domain in (0, 1):
            model.load_state_dict(best_weights[domain], strict=False)
    for domain in (0, 1):
        model.domain_gains[domain] = best[domain]["gain"]
    if frozen_digest(model) != before:
        raise RuntimeError("Immutable motion prior changed")
    summary = dict(identity, completed=True, domains={str(d): best[d] for d in (0, 1)}, records=records,
                   baseline_metrics={str(d): baseline[d] for d in baseline}, frozen_sha256=before,
                   optimizer_steps=steps, elapsed_s=time.monotonic() - started)
    save_stage(path, model, stats, summary)
    return model, stats, summary


def motion_args(args, recipe):
    result = argparse.Namespace(**vars(args))
    for key in ("lr", "batch_size", "patience", "max_epochs"):
        setattr(result, key, recipe[key])
    return result


def motion_fold(episodes, args, recipe, old_choices, seed, fold, train_ids, inner_train, inner_valid):
    stem = f"seed{seed}_fold{fold}_motion_only"
    if args.reuse_base:
        stages = args.base_run / "stages"
        return (stages / f"{stem}_student_select" / "model.pt",
                stages / f"{stem}_student_refit" / "model.pt", dict(old_choices[str(seed)]["motion_only"][fold]))
    kwargs = motion_args(args, recipe)
    stages = args.output_dir / "base_stages"
    teacher, _, tc = run_stage(episodes, inner_train, inner_valid, kwargs, seed,
                              stages / f"{stem}_teacher_select", motion_only=True)
    student, _, sc = run_stage(episodes, inner_train, inner_valid, kwargs, seed + 1000,
                              stages / f"{stem}_student_select", teacher=teacher, motion_only=True)
    del teacher, student
    teacher, _, _ = run_stage(episodes, train_ids, [], kwargs, seed,
                             stages / f"{stem}_teacher_refit", motion_only=True, epochs=tc["selected_epoch"])
    student, _, _ = run_stage(episodes, train_ids, [], kwargs, seed + 1000,
                             stages / f"{stem}_student_refit", teacher=teacher, motion_only=True, epochs=sc["selected_epoch"])
    del teacher, student
    return (stages / f"{stem}_student_select" / "model.pt", stages / f"{stem}_student_refit" / "model.pt",
            dict(fold=fold, teacher_epoch=tc["selected_epoch"], student_epoch=sc["selected_epoch"]))


def summarize(outer):
    return {name: {seed: aggregate([row["all"] for row in rows.values()]) for seed, rows in seeds.items()}
            for name, seeds in outer.items()}


def acceptance(outer, seeds, complete, args):
    metrics = {name: aggregate([row["all"] for rows in groups.values() for row in rows.values()])
               for name, groups in outer.items()}
    full = metrics["full"]
    tids = sorted(next(iter(outer["full"].values())), key=int)
    reasons, intervals, improvements = [], {}, {}
    if not complete or len(seeds) != 3:
        reasons.append("full four-fold three-seed protocol incomplete")
    rng = np.random.default_rng(20261012)
    samples = rng.integers(0, len(tids), (2000, len(tids)))
    for name in ("motion_only", "motion_control"):
        reference = metrics[name]
        improvements[name] = ((reference["forecast_score_n"] - full["forecast_score_n"]) /
                              max(reference["forecast_score_n"], 1e-8))
        differences = np.array([np.mean([outer[name][str(seed)][tid]["all"]["forecast_score_n"] -
                                        outer["full"][str(seed)][tid]["all"]["forecast_score_n"]
                                        for seed in seeds]) for tid in tids])
        intervals[name] = np.quantile(differences[samples].mean(1), [.025, .975]).tolist()
        if improvements[name] < args.min_improvement:
            reasons.append(f"future improvement vs {name} below {args.min_improvement:.0%}")
        if intervals[name][0] <= 0:
            reasons.append(f"paired bootstrap future improvement vs {name} lower bound <=0")
        if not guard_passed(full, reference, args.guard_tolerance):
            reasons.append(f"future/current/peak/Fz regressions exceed tolerance vs {name}")
        contact_full = [row["misaligned_contact"] for rows in outer["full"].values()
                        for row in rows.values() if row.get("misaligned_contact", {}).get("forecast_score_n") is not None]
        contact_base = [row["misaligned_contact"] for rows in outer[name].values()
                        for row in rows.values() if row.get("misaligned_contact", {}).get("forecast_score_n") is not None]
        if contact_full and contact_base:
            if aggregate(contact_full)["forecast_score_n"] > aggregate(contact_base)["forecast_score_n"] * (1 + args.guard_tolerance):
                reasons.append(f"misaligned_contact future error worsens vs {name}")
    summaries = summarize(outer)
    seed_scores = {str(seed): summaries["full"][str(seed)]["forecast_score_n"] for seed in seeds}
    for seed in seeds:
        for name in ("motion_only", "motion_control"):
            if seed_scores[str(seed)] > summaries[name][str(seed)]["forecast_score_n"] * (1 + args.guard_tolerance):
                reasons.append(f"seed {seed} future score worsens vs {name}")
    best_name = min(("motion_only", "motion_control"), key=lambda name: metrics[name]["forecast_score_n"])
    reference = metrics[best_name]
    return dict(passed=not reasons, reasons=reasons, mean_metrics=full, seed_scores=seed_scores,
                motion_only_metrics=metrics["motion_only"], motion_control_metrics=metrics["motion_control"],
                best_baseline=best_name, baseline_metrics=reference,
                relative_improvement=(reference["score"] - full["score"]) / max(reference["score"], 1e-8),
                relative_future_improvement=improvements,
                paired_bootstrap_95_future_improvement_n=intervals,
                selection_metric="forecast_score_n = .25 RMSE(+1) + .35 RMSE(+3) + .40 RMSE(+5)",
                thresholds=dict(min_future_improvement=args.min_improvement, guard_tolerance=args.guard_tolerance,
                                bootstrap_samples=2000))


def write_report(output, outer, real_outer, gate, selections, audit):
    lines = ["# Joint motion/force forecast v2", "", f"Sim acceptance: {gate['passed']}", "",
             "Outer trajectories and seeds have equal weight. Selection excludes the eight reused diagnostics.", "",
             "| Domain | Model | Future score N | Current RMSE N | +1 RMSE N | +3 RMSE N | +5 RMSE N | +5 Fz RMSE N |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for domain, groups in (("Sim", outer), ("Real", real_outer)):
        for name, seeds in groups.items():
            m = aggregate([row["all"] for rows in seeds.values() for row in rows.values()])
            keys = ("forecast_score_n", "current_rmse_n", "future_1_rmse_n", "future_3_rmse_n", "future_5_rmse_n", "future_5_z_rmse_n")
            lines.append(f"| {domain} | {name} | " + " | ".join(f"{m[k]:.5f}" for k in keys) + " |")
    lines += ["", "Real current is always the measured sensor force, including gain=0. Only future errors assess Real prediction.",
              "Real motion_only uses the original motion decoder conditioned on measured current/change; it is not Sim motion_only.",
              "motion_control has identical capacity, initialization, optimizer, and auxiliary force-prior supervision; force history/innovation are zeroed.",
              "The auxiliary prior is a separate head on the frozen motion latent, so its targets cannot feed the control forecast.", "",
              f"Actual observed Sim state verified: {audit['actual_sim_state_verified']}",
              "The architecture does not repair copied q/dq, sensor frames, force definitions, or replay synchronization.", "",
              "| Seed | Fold | Variant | Real epoch/gain | Sim epoch/gain |", "|---|---|---|---|---|"]
    for seed, variants in selections.items():
        for name in ("full", "motion_control"):
            for row in variants[name]:
                ds = row["domains"]
                lines.append(f"| {seed} | {row['fold']} | {name} | {ds['0']['epoch']}/{ds['0']['gain']} | {ds['1']['epoch']}/{ds['1']['gain']} |")
    lines += ["", "Reasons: " + "; ".join(gate["reasons"] or ["all thresholds passed"]), "",
              "forecast_score_n uses future physical-N RMSE; legacy score is retained only for comparison/plots.",
              "Peak metrics use five-step Fz absolute peaks; timing metrics use target timestamps. Training timing uses a soft peak position in steps.",
              "Misaligned-contact metrics are in outer_metrics.json and the phase figure.",
              "Final epochs/gains are medians of INNER choices only; outer scores choose the reported seed, diagnostics never select it."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def final_choice(rows, gains):
    result = {}
    for domain in (0, 1):
        choices = [row["domains"][str(domain)] for row in rows]
        median = float(np.median([choice["gain"] for choice in choices]))
        gain = min(gains, key=lambda g: (abs(g - median), g))
        active = [choice["epoch"] for choice in choices if choice["gain"] > 0]
        result[str(domain)] = dict(epoch=max(1, int(np.median(active))) if gain and active else 0, gain=gain)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, default=Path(DEFAULT_BASE))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-root", type=Path)
    parser.add_argument("--sim-root", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--base-mode", choices=("auto", "cached", "fresh"), default="auto")
    parser.add_argument("--require-actual-sim-state", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-5)
    parser.add_argument("--gains", type=float, nargs="+", default=[0., .25, .5, 1.])
    parser.add_argument("--fz-weight", type=float, default=2.)
    parser.add_argument("--current-weight", type=float, default=.1)
    parser.add_argument("--peak-weight", type=float, default=.1)
    parser.add_argument("--timing-weight", type=float, default=.05)
    parser.add_argument("--adapter-l2", type=float, default=1e-3)
    parser.add_argument("--inner-min-improvement", type=float, default=.01)
    parser.add_argument("--min-improvement", type=float, default=.02)
    parser.add_argument("--guard-tolerance", type=float, default=.02)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1 or args.warmup_epochs < 0:
        parser.error("epochs/patience/batch/threads must be positive; warmup nonnegative")
    if args.max_epochs <= args.warmup_epochs:
        parser.error("max-epochs must exceed warmup-epochs")
    if (not args.seeds or not args.folds or len(set(args.seeds)) != len(args.seeds) or
            len(set(args.folds)) != len(args.folds) or not set(args.folds) <= set(range(4))):
        parser.error("distinct seeds and folds in 0..3 required")
    for key in ("lr", "decoder_lr", "fz_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    for key in ("current_weight", "peak_weight", "timing_weight", "adapter_l2", "guard_tolerance"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"{key} must be finite and nonnegative")
    for key in ("inner_min_improvement", "min_improvement"):
        if not 0 <= getattr(args, key) < 1:
            parser.error(f"{key} must be in [0,1)")
    if 0. not in args.gains or len(set(args.gains)) != len(args.gains) or any(not 0 <= g <= 1 for g in args.gains):
        parser.error("gains must be distinct in [0,1] and include 0")
    args.base_run, args.output_dir = args.base_run.resolve(), args.output_dir.resolve()
    return args


def main():
    args = parse_args()
    recipe = read_json(args.base_run / "recipe.json")
    old_choices = read_json(args.base_run / "selections.json")
    args.real_root = (args.real_root or Path(recipe["real_root"])).resolve()
    args.sim_root = (args.sim_root or Path(recipe["sim_root"])).resolve()
    args.labels = (args.labels or Path(recipe["labels"])).resolve()
    episodes, audit = load_pairs(args.real_root, args.sim_root, args.labels)
    for tid in TRAIN_IDS:
        real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
        if not np.any(np.arange(len(real.force)) + 5 < np.minimum(real.ends, sim.ends)):
            raise ValueError(f"traj_{tid}: no paired segment contains a valid five-step forecast target")
    if args.require_actual_sim_state and not audit["actual_sim_state_verified"]:
        raise ValueError("Observed Sim q/dq/frame provenance is not verified; capture actual Sim states first")
    unchanged = audit == recipe["data"] and sha256(args.labels) == recipe["label_sha256"]
    if args.base_mode == "cached" and not unchanged:
        raise ValueError("Cached motion weights require identical data/labels; use --base-mode fresh")
    args.reuse_base = unchanged and args.base_mode != "fresh"
    if args.reuse_base:
        if not set(args.seeds) <= set(recipe["seeds"]) or recipe["folds"] != FOLDS:
            raise ValueError("Cached run lacks the requested seeds/folds")
        deps = Path(__file__).parent.parent / "force_gap_forecast_v1"
        for name in ("__init__.py", "model.py", "data.py", "losses.py", "metrics.py", "train.py", "common.py"):
            if sha256(deps / name) != recipe["source_sha256"][name]:
                raise ValueError(f"Cached motion implementation changed: {name}; use --base-mode fresh")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on your GPU server, or specify --device cpu")
    output = args.output_dir
    if output.exists() and any(path.name != "train.log" for path in output.iterdir()) and not args.resume:
        raise FileExistsError("Choose a new RUN_DIR or a matching --resume")
    if args.resume and not (output / "recipe.json").is_file():
        raise ValueError("Resume requires the original v2 recipe.json; choose a new RUN_DIR if it is missing")
    scripts = Path(__file__).parent.parent
    source_files = (list(Path(__file__).parent.glob("*.py")) + list((scripts / "force_gap_forecast_v1").glob("*.py")) +
                    [scripts / "visualize_force_gap_forecast_v1.py", scripts / "run_force_gap_motion_force_v2.sh"])
    checkpoints = {}
    if args.reuse_base:
        for seed in args.seeds:
            for fold in args.folds:
                for phase in ("select", "refit"):
                    path = args.base_run / "stages" / f"seed{seed}_fold{fold}_motion_only_student_{phase}" / "model.pt"
                    checkpoints[str(path.relative_to(args.base_run))] = sha256(path)
    run_recipe = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                  if key not in ("resume", "output_dir", "device", "no_plots")}
    run_recipe.update(artifact_format=FORMAT, variants=list(VARIANTS), folds=FOLDS, active_folds=args.folds,
                      train_ids=TRAIN_IDS, diagnostic_ids=HOLDOUT, label_sha256=sha256(args.labels), data=audit,
                      base_recipe_sha256=sha256(args.base_run / "recipe.json"),
                      base_selections_sha256=sha256(args.base_run / "selections.json"),
                      base_checkpoints=checkpoints,
                      source_sha256={f"{p.parent.name}/{p.name}": sha256(p) for p in source_files})
    if args.resume and (output / "recipe.json").is_file() and read_json(output / "recipe.json") != run_recipe:
        raise ValueError("Resume code/data/parameters/checkpoints changed; choose a new RUN_DIR")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "recipe.json", run_recipe)
    write_json(output / "data_audit.json", audit)
    for path in source_files:
        dest = output / "source_snapshot" / path.parent.name / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    print(f"Motion foundation={'cached' if args.reuse_base else 'fresh'}; actual Sim={audit['actual_sim_state_verified']}; "
          f"output={output}", flush=True)
    outer = {name: {str(seed): {} for seed in args.seeds} for name in VARIANTS}
    real_outer = {name: {str(seed): {} for seed in args.seeds} for name in (*VARIANTS, "sensor_hold")}
    baseline_rows = {name: {} for name in ("raw", "affine", "arx")}
    selections = {str(seed): {name: [] for name in VARIANTS} for seed in args.seeds}
    reliance = {str(seed): {} for seed in args.seeds}
    started = time.monotonic()
    for fold in args.folds:
        valid_ids = FOLDS[fold]
        train_ids = sorted(set(TRAIN_IDS) - set(valid_ids))
        order = np.random.default_rng(42 + fold).permutation(train_ids)
        inner_valid = sorted(map(int, order[:6]))
        inner_train = sorted(set(train_ids) - set(inner_valid))
        stats = fit_stats(episodes, train_ids)
        alpha = baselines.choose_alpha(episodes, inner_train, inner_valid, fit_stats(episodes, inner_train))
        for name in baseline_rows:
            base = dict(kind="raw") if name == "raw" else baselines.fit(episodes, train_ids, stats, name, alpha)
            baseline_rows[name].update(baselines.evaluate(base, episodes, valid_ids, stats))
        for seed in args.seeds:
            inner_ckpt, outer_ckpt, motion_choice = motion_fold(episodes, args, recipe, old_choices, seed, fold,
                                                               train_ids, inner_train, inner_valid)
            selections[str(seed)]["motion_only"].append(motion_choice)
            for name in ("motion_control", "full"):
                print(f"Fold {fold+1}/4 seed={seed} variant={name}", flush=True)
                selected, _, choice = fit_stage(inner_ckpt, episodes, inner_train, inner_valid, args, seed + 2000,
                                                output / "stages" / f"seed{seed}_fold{fold}_{name}_select",
                                                use_force=name == "full")
                del selected
                refit = {key: dict(epoch=value["epoch"], gain=value["gain"]) for key, value in choice["domains"].items()}
                model, stats, _ = fit_stage(outer_ckpt, episodes, train_ids, [], args, seed + 2000,
                                            output / "stages" / f"seed{seed}_fold{fold}_{name}_refit",
                                            use_force=name == "full", refit=refit)
                outer[name][str(seed)].update(evaluate(model, episodes, valid_ids, stats, 1,
                    output / "predictions" / f"seed{seed}_fold{fold}_{name}"))
                real_outer[name][str(seed)].update(evaluate(model, episodes, valid_ids, stats, 0,
                    output / "real_predictions" / f"seed{seed}_fold{fold}_{name}"))
                if name == "full":
                    reliance[str(seed)].update(force_reliance(model, episodes, valid_ids, stats,
                        output / "predictions" / f"seed{seed}_fold{fold}_full"))
                    outer["motion_only"][str(seed)].update(evaluate(model, episodes, valid_ids, stats, 1,
                        output / "predictions" / f"seed{seed}_fold{fold}_motion_only", gain=0.))
                    real_outer["motion_only"][str(seed)].update(evaluate(model, episodes, valid_ids, stats, 0,
                        output / "real_predictions" / f"seed{seed}_fold{fold}_motion_only", gain=0.))
                    real_outer["sensor_hold"][str(seed)].update(persistence(episodes, valid_ids, stats))
                selections[str(seed)][name].append(dict(fold=fold, inner_train=inner_train, inner_valid=inner_valid,
                                                        outer_train=train_ids, outer_valid=valid_ids, domains=choice["domains"]))
                write_json(output / "cv_progress.json", dict(outer=outer, real_outer=real_outer, selections=selections))
                del model
    complete = set(args.folds) == set(range(4))
    gate = acceptance(outer, args.seeds, complete, args)
    write_json(output / "acceptance.json", gate)
    write_json(output / "selections.json", selections)
    write_json(output / "outer_metrics.json", dict(models=outer, summaries=summarize(outer), baselines=baseline_rows))
    write_json(output / "real_outer_metrics.json", dict(models=real_outer, summaries=summarize(real_outer)))
    write_json(output / "force_input_audit.json", dict(models=reliance,
               mean_response=aggregate([row for rows in reliance.values() for row in rows.values()]),
               meaning="Actual Sim force replaced by train-fold mean, q/dq/time unchanged; sensitivity only, potentially OOD"))
    write_report(output, outer, real_outer, gate, selections, audit)
    if not complete:
        write_json(output / "run_summary.json", dict(status="diagnostic", final_encoder_created=False,
                                                     elapsed_s=time.monotonic() - started))
        print(f"Reduced CV complete: {output / 'report.md'}; no final encoder", flush=True)
        return
    status = ("waveform_validated" if audit["actual_sim_state_verified"] else "replay_validated") if gate["passed"] else "diagnostic"
    final = {}
    for seed in args.seeds:
        choices = selections[str(seed)]["motion_only"]
        teacher_epoch = max(1, int(np.median([row["teacher_epoch"] for row in choices])))
        student_epoch = max(1, int(np.median([row["student_epoch"] for row in choices])))
        base_path = output / "base_stages"
        kwargs = motion_args(args, recipe)
        teacher, _, _ = run_stage(episodes, TRAIN_IDS, [], kwargs, seed,
                                  base_path / f"seed{seed}_final_teacher", motion_only=True, epochs=teacher_epoch)
        student, _, _ = run_stage(episodes, TRAIN_IDS, [], kwargs, seed + 1000,
                                  base_path / f"seed{seed}_final_student", teacher=teacher, motion_only=True, epochs=student_epoch)
        del teacher, student
        refit = final_choice(selections[str(seed)]["full"], args.gains)
        model, stats, summary = fit_stage(base_path / f"seed{seed}_final_student" / "model.pt", episodes,
                                          TRAIN_IDS, [], args, seed + 2000,
                                          output / "stages" / f"seed{seed}_final_full_refit", refit=refit)
        checkpoint = output / f"encoder_seed{seed}.pt"
        torch.save(dict(artifact_format=FORMAT, layout=LAYOUT, stats=stats, state_dict=cpu_state(model),
                        use_force=True, seed=seed, train_ids=TRAIN_IDS, diagnostic_ids=HOLDOUT,
                        history=32, future=5, status=status, stage=summary), checkpoint)
        diagnostic = {domain: evaluate(model, episodes, HOLDOUT, stats, d,
                      output / "diagnostic_predictions" / f"seed{seed}_{domain}")
                      for d, domain in enumerate(("real", "sim"))}
        write_json(output / f"diagnostic_seed{seed}.json", diagnostic)
        final[str(seed)] = dict(checkpoint=checkpoint.name, encoder_sha256=sha256(checkpoint),
                                teacher_epoch=teacher_epoch, student_epoch=student_epoch, domains=refit)
        del model
    selected_seed = min(args.seeds, key=lambda seed: gate["seed_scores"][str(seed)])
    write_json(output / "manifest.json", dict(artifact_format=FORMAT, layout=LAYOUT, status=status,
                **final[str(selected_seed)], selected_seed=selected_seed, all_seeds=final,
                selection_source="outer future-only CV; reused diagnostics excluded", history=32, future=5,
                policy_history=10, actual_sim_state_verified=audit["actual_sim_state_verified"],
                real_root=str(args.real_root), sim_root=str(args.sim_root), force_contract="aligned_base_xyz_n_v1",
                runtime="force_gap_motion_force_v2.runtime.JointMotionForceEncoder", acceptance="acceptance.json",
                acceptance_sha256=sha256(output / "acceptance.json"), audit="data_audit.json",
                audit_sha256=sha256(output / "data_audit.json"), elapsed_s=time.monotonic() - started))
    if not args.no_plots:
        from visualize_force_gap_forecast_v1 import generate_visualizations
        generate_visualizations(output)
    print(f"Completed: {output / 'manifest.json'}; status={status}; reasons={gate['reasons']}", flush=True)


if __name__ == "__main__":
    main()
