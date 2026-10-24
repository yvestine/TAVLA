"""Grouped future-motion comparisons and predeclared offline acceptance gates."""

from __future__ import annotations

import numpy as np

from .motion_data import grouped_inner_ids
from .motion_fit import fit_probe, probe_predict

DIRECTIONS = ("real_to_sim", "sim_to_real")
BASELINES = ("state_only", "state_background", "state_raw_force", "state_manual", "state_legacy")


def representations(windows, legacy=None, new=None):
    result = {"state_only": np.empty((len(windows), 0), np.float32),
              "state_background": windows.background.reshape(len(windows), -1),
              "state_raw_force": windows.raw_force,
              "state_manual": windows.features.reshape(len(windows), -1)}
    if legacy is not None:
        result["state_legacy"] = legacy
    if new is not None:
        result["state_new"] = new
    return result


def _error_rows(windows, prediction, target_scale):
    normalized = ((prediction - windows.future) / target_scale).reshape(-1, 2, 7) ** 2
    physical = (prediction - windows.future).reshape(-1, 2, 7) ** 2
    result = {}
    for trajectory in np.unique(windows.trajectory):
        keep = windows.trajectory == trajectory
        phases = {}
        for phase in np.unique(windows.phase[keep]):
            phase_keep = keep & (windows.phase == phase)
            phases[str(phase)] = {"frames": int(phase_keep.sum()),
                "normalized_rmse_by_horizon": np.sqrt(normalized[phase_keep].mean(axis=(0, 2))).tolist()}
        result[str(int(trajectory))] = {"frames": int(keep.sum()),
            "normalized_rmse_by_horizon": np.sqrt(normalized[keep].mean(axis=(0, 2))).tolist(),
            "joint_rmse_rad_by_horizon": np.sqrt(physical[keep].mean(axis=(0, 2))).tolist(),
            "median_actual_horizon_seconds": np.median(windows.future_elapsed[keep], axis=0).tolist(),
            "by_annotation_phase": phases}
    return result


def shuffle_same_phase(windows, values, seed):
    rng = np.random.default_rng(seed)
    index = np.arange(len(windows))
    changed = np.zeros(len(windows), bool)
    for phase in np.unique(windows.phase):
        rows = np.flatnonzero(windows.phase == phase)
        if len(rows) < 2:
            continue
        best_score, best_permutation, best_eligible = (-1, -1), None, None
        # Preserve the phase's empirical latent distribution: a permutation,
        # rather than independent draws with replacement. Exclude unchanged
        # or nearby same-trajectory histories from the paired error test.
        for _ in range(32):
            permutation = rng.permutation(rows)
            other = windows.trajectory[rows] != windows.trajectory[permutation]
            eligible = other | (np.abs(windows.frame[rows] - windows.frame[permutation]) >= 32)
            score = (int(eligible.sum()), int(other.sum()))
            if score > best_score:
                best_score, best_permutation, best_eligible = score, permutation, eligible
        index[rows], changed[rows] = best_permutation, best_eligible
    return values[index], changed


def evaluate_probe(train, valid, representation_train, representation_valid, fit_ids, args, device, seed, *, shuffle=False):
    inner_train, inner_valid = grouped_inner_ids(fit_ids, args.fold_seed)
    result = {}
    for source, direction in enumerate(DIRECTIONS):
        training = train.future_valid & (train.domain == source)
        testing = valid.future_valid & (valid.domain != source)
        x_train = np.concatenate((train.state, representation_train), axis=1)
        x_valid = np.concatenate((valid.state, representation_valid), axis=1)
        inner_a = training & np.isin(train.trajectory, inner_train)
        inner_b = training & np.isin(train.trajectory, inner_valid)
        if not inner_a.any() or not inner_b.any() or not testing.any():
            raise ValueError("Empty grouped future-motion probe partition")
        inner_model, _, epochs = fit_probe(x_train[inner_a], train.future[inner_a], args, device,
            seed + source * 1000, validation=(x_train[inner_b], train.future[inner_b]))
        del inner_model
        model, stats, _ = fit_probe(x_train[training], train.future[training], args, device,
                                   seed + source * 1000 + 10_000, epochs=epochs)
        test = valid.take(testing)
        prediction = probe_predict(model, x_valid[testing], stats, device)
        row = {"selected_epochs_on_source_training_groups": epochs,
               "fit_trajectories": sorted(fit_ids), "target_scale_rad": stats["target_scale"].tolist(),
               "by_trajectory": _error_rows(test, prediction, stats["target_scale"])}
        if shuffle:
            shuffled, changed = shuffle_same_phase(test, representation_valid[testing], seed + source * 1000 + 20_000)
            shuffled_x = np.concatenate((test.state, shuffled), axis=1)
            shuffled_prediction = probe_predict(model, shuffled_x, stats, device)
            row["shuffle"] = {"eligible_frames": int(changed.sum()), "total_frames": len(test),
                "original_on_eligible": _error_rows(test.take(changed), prediction[changed], stats["target_scale"]),
                "shuffled_on_eligible": _error_rows(test.take(changed), shuffled_prediction[changed], stats["target_scale"])}
        result[direction] = row
        del model
    return result


def _auc(labels, scores):
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    ranks = np.empty(len(scores), np.float64)
    lo = 0
    while lo < len(scores):
        hi = lo + 1
        while hi < len(scores) and ordered[hi] == ordered[lo]:
            hi += 1
        ranks[order[lo:hi]] = .5 * (lo + 1 + hi)
        lo = hi
    positive, negative = int(labels.sum()), int((labels == 0).sum())
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)) if positive and negative else None


def representation_diagnostics(train, valid, z_train, z_valid):
    """Train-only linear domain probe; matched annotation classes in both domains."""
    a, b = train.y >= 0, valid.y >= 0
    if not a.any() or not b.any():
        return {"domain_auc": None}
    x, y = z_train[a].astype(np.float64), train.domain[a].astype(np.float64)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
    xn = np.clip((x - center) / scale, -10, 10)
    xn = np.concatenate((xn, np.ones((len(xn), 1))), axis=1)
    weights = np.zeros(xn.shape[1])
    sample_weights = np.zeros(len(x))
    labels = train.y[a]
    for domain in (0, 1):
        for phase in (0, 1):
            keep = (y == domain) & (labels == phase)
            sample_weights[keep] = 1.0 / max(int(keep.sum()), 1)
    sample_weights /= sample_weights.sum()
    for _ in range(200):
        probability = 1 / (1 + np.exp(-np.clip(xn @ weights, -35, 35)))
        weights -= .2 * (xn.T @ (sample_weights * (probability - y)) + .001 * weights)
    test = np.clip((z_valid[b] - center) / scale, -10, 10)
    scores = np.concatenate((test, np.ones((len(test), 1))), axis=1) @ weights
    conditional = {}
    for phase in (0, 1):
        groups = [((z_valid[(valid.y == phase) & (valid.domain == d)] - center) / scale) for d in (0, 1)]
        if min(map(len, groups)) >= 2:
            mean_distance = float(np.mean((groups[0].mean(axis=0) - groups[1].mean(axis=0)) ** 2))
            covariance_distance = float(np.mean((np.cov(groups[0], rowvar=False) - np.cov(groups[1], rowvar=False)) ** 2))
            conditional[str(phase)] = {"mean_distance": mean_distance, "covariance_distance": covariance_distance,
                                      "frames_by_domain": list(map(len, groups))}
    lookup = {(int(t), int(f)): i for i, (t, f, d) in enumerate(zip(valid.trajectory, valid.frame, valid.domain, strict=True)) if d == 1}
    pairs = [(i, lookup[(int(valid.trajectory[i]), int(valid.frame[i]))]) for i in np.flatnonzero((valid.y >= 0) & (valid.domain == 0))]
    cosine = None
    if pairs:
        left = z_valid[[real_index for real_index, _ in pairs]]
        right = z_valid[[sim_index for _, sim_index in pairs]]
        cosine = float(np.mean((left * right).sum(axis=1) / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-8)))
    return {"domain_auc": _auc(valid.domain[b], scores),
            "domain_accuracy": float(np.mean((scores >= 0) == valid.domain[b])),
            "conditional_distances": conditional, "same_frame_cosine_descriptive_only": cosine,
            "note": "Same-frame correspondence is a weak teacher pairing, not verified physical contact correspondence."}


def aggregate_probes(trials, name, shuffle_key=None):
    grouped = {direction: {} for direction in DIRECTIONS}
    for trial in trials:
        for direction, record in trial["probes"][name].items():
            rows = record["by_trajectory"] if shuffle_key is None else record["shuffle"][shuffle_key]
            for trajectory, row in rows.items():
                grouped[direction].setdefault(trajectory, []).append(row["normalized_rmse_by_horizon"])
    per_direction = {direction: {key: np.mean(values, axis=0).tolist() for key, values in rows.items()}
                     for direction, rows in grouped.items()}
    direction_errors = {direction: float(np.mean(list(rows.values()))) if rows else None
                        for direction, rows in per_direction.items()}
    trajectories = sorted(set.intersection(*(set(rows) for rows in per_direction.values())), key=int)
    per_trajectory = {key: float(np.mean([per_direction[d][key] for d in DIRECTIONS])) for key in trajectories}
    return {"mean_error": float(np.mean(list(per_trajectory.values()))) if per_trajectory else None,
            "direction_errors": direction_errors, "per_trajectory_error": per_trajectory,
            "per_direction_trajectory_horizon_errors": per_direction,
            "metric": "trajectory-equal, seed-mean normalized joint-delta RMSE, averaged over horizons 3/5 and transfer directions"}


def _f1(matrix):
    matrix = np.asarray(matrix)
    return float(np.mean([2 * matrix[k, k] / max(matrix[k].sum() + matrix[:, k].sum(), 1) for k in (0, 1)]))


def aggregate_phases(trials):
    seeds = sorted({trial["seed"] for trial in trials})
    result = {}
    for domain in ("real", "sim"):
        scores = []
        for seed in seeds:
            matrix = sum((np.asarray(t["phase"][domain]["confusion_matrix"]) for t in trials if t["seed"] == seed), start=np.zeros((2, 2), np.int64))
            scores.append(_f1(matrix))
        result[domain] = {"macro_f1_mean": float(np.mean(scores)), "macro_f1_by_seed": dict(zip(map(str, seeds), scores, strict=True))}
    return result


def paired_bootstrap(left, right, samples=2000, seed=42):
    """CI for left-error minus right-error, resampling whole trajectories."""
    keys = sorted(set(left) & set(right), key=int)
    if len(keys) < 8:
        return {"trajectory_count": len(keys), "mean_difference": None, "ci95": None}
    delta = np.asarray([left[k] - right[k] for k in keys])
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(keys), size=(samples, len(keys)))].mean(axis=1)
    return {"trajectory_count": len(keys), "mean_difference": float(delta.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(), "bootstrap_samples": samples}


def assess_candidates(baseline_trials, candidates, args):
    baselines = {name: aggregate_probes(baseline_trials, name) for name in BASELINES}
    best_baseline = min(BASELINES, key=lambda name: baselines[name]["mean_error"])
    old_phase = aggregate_phases(baseline_trials)
    summaries = {}
    for key, trials in candidates.items():
        motion, phase = aggregate_probes(trials, "state_new"), aggregate_phases(trials)
        phase_pass = all(phase[d]["macro_f1_mean"] >= old_phase[d]["macro_f1_mean"] - .02 for d in ("real", "sim"))
        summaries[key] = {"motion": motion, "phase": phase, "phase_protection_passed": phase_pass}
    eligible = [key for key, row in summaries.items() if row["phase_protection_passed"]]
    selected = min(eligible or list(summaries), key=lambda key: (summaries[key]["motion"]["mean_error"], float(key)))
    chosen, base = summaries[selected], baselines[best_baseline]
    mean_improvement = (base["mean_error"] - chosen["motion"]["mean_error"]) / max(base["mean_error"], 1e-12)
    direction_improvements = {d: (base["direction_errors"][d] - chosen["motion"]["direction_errors"][d]) /
                                max(base["direction_errors"][d], 1e-12) for d in DIRECTIONS}
    bootstrap = paired_bootstrap(base["per_trajectory_error"], chosen["motion"]["per_trajectory_error"], args.bootstrap_samples, args.fold_seed)
    original = aggregate_probes(candidates[selected], "state_new", "original_on_eligible")
    shuffled = aggregate_probes(candidates[selected], "state_new", "shuffled_on_eligible")
    shuffle_bootstrap = paired_bootstrap(shuffled["per_trajectory_error"], original["per_trajectory_error"], args.bootstrap_samples, args.fold_seed)
    checks = {"mean_error_improvement_at_least_5pct": mean_improvement >= .05,
              "no_direction_worsens_more_than_2pct": min(direction_improvements.values()) >= -.02,
              "paired_trajectory_ci_lower_above_zero": bootstrap["ci95"] is not None and bootstrap["ci95"][0] > 0,
              "both_phase_f1_within_002_of_retrained_legacy": chosen["phase_protection_passed"],
              "same_phase_shuffle_significantly_increases_error": shuffle_bootstrap["ci95"] is not None and shuffle_bootstrap["ci95"][0] > 0}
    return {"selected_alignment_weight": float(selected), "selected_candidate_key": selected,
            "best_baseline": best_baseline, "baselines": baselines, "retrained_legacy_phase": old_phase,
            "candidates": summaries, "mean_relative_improvement": mean_improvement,
            "direction_relative_improvements": direction_improvements, "paired_bootstrap": bootstrap,
            "same_phase_shuffle_bootstrap": shuffle_bootstrap, "checks": checks, "passed": all(checks.values()),
            "software_checks_required_separately": True}
