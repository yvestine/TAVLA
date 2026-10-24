"""Grouped acceptance: seeds are replicates, trajectories are bootstrap units."""
from __future__ import annotations

import numpy as np

from force_gap_forecast_v1.common import write_json
from .metrics import aggregate
from .train import RECON_GUARDS, guard


def comparison(results, candidate, baseline, domain, metric="forecast_score_n", phase="all"):
    pairs = {}
    for result in results:
        if candidate not in result["models"] or baseline not in result["models"]:
            continue
        a, b = result["models"][candidate][domain], result["models"][baseline][domain]
        for tid in a.keys() & b.keys():
            av, bv = a[tid].get(phase, {}).get(metric), b[tid].get(phase, {}).get(metric)
            if av is not None and bv is not None:
                pairs.setdefault(tid, []).append((av, bv))
    if not pairs:
        return dict(available=False)
    rows = np.array([np.mean(pairs[tid], axis=0) for tid in sorted(pairs, key=int)])
    a, b = rows[:, 0], rows[:, 1]
    improvement = b - a
    rng = np.random.default_rng(7301)
    samples = rng.integers(0, len(rows), size=(2000, len(rows)))
    interval = np.quantile(improvement[samples].mean(1), [.025, .975]).tolist()
    return dict(available=True, candidate=float(a.mean()), baseline=float(b.mean()),
                relative_improvement=float(1 - a.mean() / max(b.mean(), 1e-12)),
                improvement_95_ci=interval, trajectories=len(rows))


def summarize(results):
    output = {domain: {} for domain in ("real", "sim")}
    for domain in output:
        names = sorted(set().union(*(r["models"] for r in results)))
        for name in names:
            phases = {}
            for row in results:
                for values in row["models"].get(name, {}).get(domain, {}).values():
                    for phase, metrics in values.items():
                        phases.setdefault(phase, []).append(metrics)
            output[domain][name] = {phase: aggregate(metrics) for phase, metrics in phases.items()}
    return output


def acceptance(results, recipe):
    summary = summarize(results)
    reasons, comparisons = [], {}
    complete = set(recipe["options"]["folds"]) == set(range(4)) and set(recipe["options"]["seeds"]) == {42, 43, 44}
    if not complete:
        reasons.append("Reduced protocol: acceptance requires folds 0..3 and seeds 42/43/44")
    guards = ("future_1_rmse_n", "future_3_rmse_n", "future_5_rmse_n", "future_5_x_rmse_n",
              "future_5_y_rmse_n", "future_5_z_rmse_n", "future_1_event_rmse_n", "future_5_event_rmse_n",
              "future_z_peak_mae_n", "future_z_peak_timing_mae_s", "future_delta_rmse_n")
    for domain in ("real", "sim"):
        models = summary[domain]
        baseline_pairs = [("full_force", "hold"), ("full_force", "ar"), ("full_joint", "motion_control")]
        if "old_motion_only" in models:
            baseline_pairs.append(("full_joint", "old_motion_only"))
        for a, b in baseline_pairs:
            key = f"{domain}:{a}_vs_{b}"
            row = comparison(results, a, b, domain)
            comparisons[key] = row
            if not row["available"] or row["relative_improvement"] < .05 or row["improvement_95_ci"][0] <= 0:
                reasons.append(f"{key}: future improvement <5% or trajectory bootstrap lower bound <=0")
            if not guard(models[a]["all"], models[b]["all"], guards):
                reasons.append(f"{key}: future/axis/event/peak guard regresses >2%")
            contact_phases = (models[a].keys() & models[b].keys()) - {"all", "free", "uncertain", "unannotated"}
            for phase in sorted(contact_phases):
                phase_compare = comparison(results, a, b, domain, phase=phase)
                if phase_compare["available"] and phase_compare["relative_improvement"] < -.02:
                    reasons.append(f"{key}: {phase} future score regresses >2%")
            for seed in recipe["options"]["seeds"]:
                per_seed = comparison([r for r in results if r["seed"] == seed], a, b, domain)
                if per_seed["available"] and per_seed["relative_improvement"] < -.02:
                    reasons.append(f"{key}: seed {seed} regresses >2%")
        if not guard(models["full_force"]["all"], models["no_alignment"]["all"], RECON_GUARDS):
            reasons.append(f"{domain}: alignment worsens reconstruction/fast changes/peaks >2%")
        fast = comparison(results, "full_force", "no_fast", domain, "high_band_mse")
        comparisons[f"{domain}:transient_ablation"] = fast
        if not fast["available"] or fast["relative_improvement"] <= 0 or fast["improvement_95_ci"][0] <= 0:
            reasons.append(f"{domain}: transient branch has no demonstrated high-band reconstruction benefit")
    distances = {kind: [r["alignment"][kind]["trend_distance"] for r in results] for kind in ("full", "no_alignment")}
    valid_alignment = all(v is not None for values in distances.values() for v in values)
    alignment = dict(available=valid_alignment)
    if valid_alignment:
        full, base = np.mean(distances["full"]), np.mean(distances["no_alignment"])
        alignment.update(relative_improvement=float(1 - full / max(base, 1e-12)), full=float(full), no_alignment=float(base))
        if alignment["relative_improvement"] < .05:
            reasons.append("Paired trend distance does not improve by at least 5%")
    else:
        reasons.append("No valid paired trend segments")
    if any(min(r["alignment"]["full"]["teacher_train_std"]) < .100001 for r in results):
        reasons.append("One or more teacher trend dimensions have train std <0.1")
    return dict(passed=not reasons, reasons=reasons, comparisons=comparisons, alignment=alignment,
                complete_protocol=complete, summaries=summary,
                thresholds=dict(future_improvement=.05, trend_distance_improvement=.05,
                                max_regression=.02, bootstrap_samples=2000))


def write_report(root, results, recipe):
    gate = acceptance(results, recipe)
    write_json(root / "acceptance.json", gate)
    write_json(root / "outer_metrics.json", dict(jobs=results, summaries=gate["summaries"], aggregation="equal trajectory/seed"))
    write_json(root / "selections.json", {f"seed{r['seed']}_fold{r['fold']}": r["selections"] for r in results})
    write_json(root / "force_input_audit.json", {f"seed{r['seed']}_fold{r['fold']}": r["force_input_audit"] for r in results})
    lines = ["# Predictive force encoder v3", "", f"Accepted: **{gate['passed']}**", "",
             "Encoder input: force history and dt only. All 32 dimensions are learned (16 trend + 16 transient).",
             "Force future and joint future are separate tests. Reconstruction is of the input domain, not Sim-to-Real current force.",
             "Frequency units: cycles/frame; high band: 0.25–0.5. Timing metrics use recorded timestamps.",
             f"Actual observed Sim state verified: {recipe['data']['actual_sim_state_verified']}", "",
             "Sim hold is a train-only affine calibration followed by holding the current force; Real hold uses the measured current force.",
             "The historical Real motion_only baseline also conditions its decoder on measured current force/change; its Sim path is motion-only.",
             "The eight reused diagnostics do not select epochs, alignment weights, seeds, or acceptance.", "",
             "| Domain | Model | Future score N | +1 N | +3 N | +5 N | Reconstruction N | High-band error |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "n/a" if v is None else f"{v:.5f}"
    for domain, models in gate["summaries"].items():
        for name, phases in models.items():
            row = phases["all"]
            keys = ("forecast_score_n", "future_1_rmse_n", "future_3_rmse_n", "future_5_rmse_n",
                    "reconstruction_rmse_n", "high_band_mse")
            lines.append(f"| {domain} | {name} | " + " | ".join(fmt(row.get(key)) for key in keys) + " |")
    lines += ["", "## Selection", "", "| Seed | Fold | Alignment | Teacher epoch | Student epoch | Motion epoch | Control epoch |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["selections"]["full"]
        lines.append(f"| {r['seed']} | {r['fold']} | {c['align_weight']} | {c['teacher_epoch']} | {c['student_epoch']} | "
                     f"{c['motion_epoch']} | {c['control_epoch']} |")
    lines += ["", "## Acceptance reasons", ""] + [f"- {reason}" for reason in gate["reasons"] or ["All gates passed."]]
    lines += ["", "Full metrics, paired comparisons, reconstruction guards and confidence intervals: acceptance.json.",
              "Per-trajectory/phase/axis results: outer_metrics.json. Raw arrays: jobs/*/predictions/{full,no_alignment,no_fast}/{real,sim}/traj_*.npz.",
              "Force shuffling is a sensitivity diagnostic, not independent proof of usefulness.",
              "Final seed is fixed before inspecting outer scores. Final weights use only inner-selected schedules.",
              "Copied Sim q/dq limit joint-motion claims to offline replay data; waveform acceptance does not establish closed-loop success."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return gate
