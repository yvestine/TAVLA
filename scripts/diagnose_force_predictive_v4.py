"""CPU diagnostics of existing v4 weights; exploratory probes never replace the trained model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from force_gap_forecast_v1.common import sha256
from force_gap_forecast_v1.common import write_json
from force_gap_forecast_v1.data import load_pairs
from force_predictive_encoder_v4 import FUTURE
from force_predictive_encoder_v4 import HISTORY
from force_predictive_encoder_v4.data import Blocks
from force_predictive_encoder_v4.data import episode_inputs
from force_predictive_encoder_v4.data import targets
from force_predictive_encoder_v4.losses import masked_huber
from force_predictive_encoder_v4.losses import objective
from force_predictive_encoder_v4.metrics import aggregate
from force_predictive_encoder_v4.metrics import predict_episode
from force_predictive_encoder_v4.metrics import score_prediction
from force_predictive_encoder_v4.train import load_model
from force_predictive_encoder_v4.train import seed_all
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate


def correlation(a, b):
    if min(np.std(a), np.std(b)) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(predictions, episodes, ids, domain, threshold):
    rows = {
        str(tid): score_prediction(predictions[tid], episodes[(tid, 0)], episodes[(tid, domain)], threshold)
        for tid in ids
    }
    return {"summary": aggregate(list(rows.values())), "trajectories": rows}


def histories(ep, values):
    return np.stack(
        [values[np.maximum(np.arange(t - HISTORY + 1, t + 1), ep.starts[t])].reshape(-1) for t in range(len(values))]
    ).astype(np.float32)


def signal_audit(episodes, splits, recipe):
    rows, fields_equal, timestamps_equal = {}, True, True
    for tid in sorted({i for ids in splits.values() for i in ids}):
        real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
        for root in (recipe["options"]["real_root"], recipe["options"]["sim_root"]):
            with h5py.File(Path(root) / f"traj_{tid}" / "data.h5") as f:
                fields_equal &= np.array_equal(f["obs/state/ee_force_base"][:], f["obs/state/ee_wrench_base"][:, :3])
        timestamps_equal &= np.array_equal(real.time, sim.time)
        continuous = (real.starts[1:] == real.starts[:-1]) & (sim.starts[1:] == sim.starts[:-1])
        dr, ds = np.diff(real.force, axis=0)[continuous], np.diff(sim.force, axis=0)[continuous]
        rows[str(tid)] = {
            "force_corr_xyz": [correlation(real.force[:, j], sim.force[:, j]) for j in range(3)],
            "increment_corr_xyz": [correlation(dr[:, j], ds[:, j]) for j in range(3)],
            "sim_over_real_std_xyz": (sim.force.std(0) / np.maximum(real.force.std(0), 1e-8)).tolist(),
            "sim_over_real_abs_peak_xyz": (
                np.abs(sim.force).max(0) / np.maximum(np.abs(real.force).max(0), 1e-8)
            ).tolist(),
            "copied_q_dq": bool(np.array_equal(real.q, sim.q) and np.array_equal(real.dq, sim.dq)),
        }
    summary = {}
    for split, ids in splits.items():
        summary[split] = {}
        for key in ("force_corr_xyz", "increment_corr_xyz", "sim_over_real_std_xyz", "sim_over_real_abs_peak_xyz"):
            values = np.array([rows[str(i)][key] for i in ids], dtype=float)
            summary[split][key] = {
                "trajectory_mean": np.nanmean(values, axis=0).tolist(),
                "trajectory_median": np.nanmedian(values, axis=0).tolist(),
            }
    # Offline timing diagnostic: each trajectory chooses its own best correlation.
    # This uses target labels, and is NOT a deployable shift or a generalization test.
    lag_rows = []
    for tid in splits["validation"]:
        real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
        candidates = []
        for lag in range(-10, 11):
            ii = np.arange(len(real.force))
            ii = ii[(ii + lag >= 0) & (ii + lag < len(sim.force))]
            corr = correlation(real.force[ii, 2], sim.force[ii + lag, 2])
            candidates.append((corr if corr is not None else -1, lag))
        value, lag = max(candidates)
        lag_rows.append(
            {
                "trajectory": tid,
                "best_lag_frames": lag,
                "oracle_fz_corr": value,
                "zero_lag_fz_corr": rows[str(tid)]["force_corr_xyz"][2],
            }
        )
    return {
        "trajectories": rows,
        "summaries": summary,
        "force_equals_wrench_xyz": bool(fields_equal),
        "paired_timestamps_equal": bool(timestamps_equal),
        "copied_q_dq_count": sum(v["copied_q_dq"] for v in rows.values()),
        "oracle_per_trajectory_lag_scan": lag_rows,
        "actual_sim_state_verified": recipe["data"]["actual_sim_state_verified"],
    }


def frozen_predictions_audit(cache, episodes, splits, stats):
    result, oracle_offsets = {}, {}
    for split, ids in splits.items():
        result[split], oracle_offsets[split] = {}, {}
        for domain, name in enumerate(("real", "sim")):
            pred = {tid: cache[(tid, domain)]["force"] for tid in ids}
            result[split][name] = evaluate(pred, episodes, ids, domain, stats["target"]["event_threshold_n"])
            centered, fractions = {}, []
            for tid in ids:
                error = pred[tid][:, 0] - episodes[(tid, 0)].force
                offset = error.mean(0)
                fractions.append(float(np.mean(offset**2) / max(np.mean(error**2), 1e-12)))
                centered[tid] = pred[tid] - offset[None, None]
            oracle_offsets[split][name] = evaluate(
                centered, episodes, ids, domain, stats["target"]["event_threshold_n"]
            )
            oracle_offsets[split][name]["trajectory_mean_current_mse_fraction_due_to_offset"] = float(
                np.mean(fractions)
            )
    return {
        "frozen_model": result,
        "oracle_per_trajectory_offset": oracle_offsets,
        "oracle_limitation": "Offset uses the entire target trajectory; it is only an error decomposition, not a usable estimator.",
    }


def ridge_probe(features, episodes, splits, stats, alpha=1.0):
    ids = splits["train"]
    xtrain = np.concatenate([features[tid] for tid in ids]).astype(np.float64)
    mean, scale = xtrain.mean(0), np.maximum(xtrain.std(0), 1e-3)
    matrices = {tid: np.column_stack(((x - mean) / scale, np.ones(len(x)))) for tid, x in features.items()}
    coef = []
    for h in range(FUTURE + 1):
        xs, ys, ws = [], [], []
        for tid in ids:
            truth, mask = targets(episodes, tid, np.arange(len(features[tid])))
            keep = mask[:, h]
            xs.append(matrices[tid][keep])
            ys.append(truth[keep, h])
            ws.append(np.full(keep.sum(), 1.0 / keep.sum()))
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        w *= len(w) / w.sum()
        penalty = alpha * np.eye(x.shape[1])
        penalty[-1, -1] = 0
        coef.append(np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T @ (w[:, None] * y)))
    coef = np.stack(coef)
    pred = {tid: np.einsum("nf,hfc->nhc", x, coef).astype(np.float32) for tid, x in matrices.items()}
    return {
        split: evaluate(pred, episodes, split_ids, 1, stats["target"]["event_threshold_n"])
        for split, split_ids in splits.items()
    }


def head_probe(model, features, episodes, splits, output, name, epochs, *, joint=False, initial_head=None):
    seed_all(42)
    width = next(iter(features.values())).shape[1]
    head = (
        copy.deepcopy(initial_head)
        if initial_head is not None
        else nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 18))
    )
    head.requires_grad_(requires_grad=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    ids = splits["train"]
    domains = (0, 1) if joint else (1,)
    xs = {d: torch.from_numpy(np.concatenate([features[(tid, d)] for tid in ids])) for d in (0, 1)}
    ys, masks, weights = [], [], []
    for tid in ids:
        truth, mask = targets(episodes, tid, np.arange(len(features[(tid, 1)])))
        ys.append(truth)
        masks.append(mask)
        weights.extend(np.full(len(truth), 1.0 / len(truth)))
    ys, masks, weights = (
        torch.from_numpy(np.concatenate(ys)),
        torch.from_numpy(np.concatenate(masks)),
        torch.tensor(weights),
    )
    normalizer = SimpleNamespace(target_scale=model.target_scale, increment_scale=model.increment_scale)
    scales, means = model.target_scale, model.target_mean

    def predict(d, chosen):
        with torch.inference_mode():
            return {
                tid: (head(torch.from_numpy(features[(tid, d)])).reshape(-1, 6, 3) * scales + means).numpy()
                for tid in chosen
            }

    def evaluate_head(split, d):
        chosen = splits[split]
        return evaluate(predict(d, chosen), episodes, chosen, d, model.stats["target"]["event_threshold_n"])

    best, best_state, best_epoch, stale, records = float("inf"), None, 0, 0, []
    for epoch in range(1, epochs + 1):
        head.train()
        g = torch.Generator().manual_seed(42 + epoch * 7919)
        ii = torch.multinomial(weights, int(np.ceil(len(ys) / 512)) * 512, replacement=True, generator=g)
        losses = []
        for indices in ii.split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(
                objective(
                    normalizer,
                    {"force": head(xs[d][indices]).reshape(-1, 6, 3) * scales + means},
                    {"target": ys[indices], "mask": masks[indices]},
                )[0]
                for d in domains
            ) / len(domains)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        validation = evaluate_head("validation", 1)["summary"]
        score = validation["forecast_score_n"]
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
        records.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": validation})
        if epoch == 1 or epoch % 25 == 0:
            print(f"{name}: epoch {epoch}; Sim validation future={score:.4f}", flush=True)
        if stale >= 25:
            break
    head.load_state_dict(best_state)
    result = {
        "selected_epoch": best_epoch,
        "records": records,
        "splits": {
            split: {domain: evaluate_head(split, d) for d, domain in enumerate(("real", "sim"))} for split in splits
        },
        "settings": {
            "max_epochs": epochs,
            "lr": 1e-3,
            "batch": 512,
            "equal_trajectory_sampling": True,
            "joint_real_sim": joint,
            "frozen_encoder": True,
            "delta_weight": 0.5,
        },
        "limitation": "Exploratory cached-feature readout probe; sampling and optimization differ from full encoder training. Validation has already been inspected.",
    }
    torch.save(head.state_dict(), output / f"{name}.pt")
    write_json(output / f"{name}.json", result)
    return result


def gradient_diagnostic(model, episodes, arrays, stats, train_ids):
    data = Blocks(episodes, train_ids, arrays, stats)
    result = {}
    g = torch.Generator().manual_seed(91)
    indices = torch.multinomial(data.weights, 32, replacement=True, generator=g).tolist()
    for domain, name in enumerate(("real", "sim")):
        model.configure("teacher" if domain == 0 else "student")
        model.eval()
        params = list(model.encoders[domain].parameters())
        rows = []
        for lo in range(0, len(indices), 4):
            batch = default_collate([data[i] for i in indices[lo : lo + 4]])
            p = model.block(batch["x"], domain)["force"]
            y = batch["target"]
            mask = batch["mask"]
            err = (p - y) / model.target_scale
            terms = [
                masked_huber(err[..., 0, :], mask[..., 0]),
                masked_huber(err[..., 1:, :], mask[..., 1:]),
                0.5
                * masked_huber(
                    (torch.diff(p, dim=-2) - torch.diff(y, dim=-2)) / model.increment_scale,
                    mask[..., 1:] & mask[..., :-1],
                ),
            ]
            grads = []
            for term in terms:
                grad = torch.autograd.grad(term, params, retain_graph=True)
                grads.append(torch.cat([a.flatten() for a in grad]))

            def cos(a, b):
                return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))

            rows.append(
                {
                    "current_norm": float(grads[0].norm()),
                    "future_norm": float(grads[1].norm()),
                    "weighted_delta_norm": float(grads[2].norm()),
                    "current_future_cosine": cos(grads[0], grads[1]),
                    "future_delta_cosine": cos(grads[1], grads[2]),
                    "current_delta_cosine": cos(grads[0], grads[2]),
                }
            )
        result[name] = {"mean": aggregate(rows), "batches": rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-epochs", type=int, default=150)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = args.run_dir.resolve()
    recipe = json.loads((root / "recipe.json").read_text())
    opts = recipe["options"]
    checkpoint = root / json.loads((root / "manifest.json").read_text())["checkpoint"]
    before = {str(p): sha256(p) for p in (checkpoint, root / "teacher/model.pt", root / "student/model.pt")}
    model, _ = load_model(checkpoint)
    teacher, _ = load_model(root / "teacher/model.pt")
    a, b = teacher.state_dict(), model.state_dict()
    frozen_unchanged = all(torch.equal(a[k], b[k]) for k in a if not k.startswith("encoders.1."))
    episodes, _ = load_pairs(opts["real_root"], opts["sim_root"], opts["labels"])
    splits = {
        "train": recipe["train_ids"],
        "validation": recipe["validation_ids"],
        "diagnostic": recipe["diagnostic_ids"],
    }
    stats = model.stats
    print("Caching CPU features for all trajectories...", flush=True)
    arrays = {
        (i, d): torch.from_numpy(episode_inputs(ep, stats, ("real", "sim")[d])) for (i, d), ep in episodes.items()
    }
    cache = {(i, d): predict_episode(model, arrays[(i, d)], d) for i, d in arrays}
    features_z = {(i, d): row["z"] for (i, d), row in cache.items()}
    features_raw = {}
    for (i, d), ep in episodes.items():
        name = ("real", "sim")[d]
        f = (ep.force - np.asarray(stats[name]["mean"])) / np.asarray(stats[name]["scale"])
        features_raw[(i, d)] = histories(ep, f)
    signals = signal_audit(episodes, splits, recipe)
    write_json(out / "signal_audit.json", signals)
    original = frozen_predictions_audit(cache, episodes, splits, stats)
    write_json(out / "frozen_model_metrics.json", original)
    print(
        "Original metrics:",
        {
            split: {d: v["summary"]["forecast_score_n"] for d, v in data.items()}
            for split, data in original["frozen_model"].items()
        },
        flush=True,
    )
    ridge = {}
    cases = {
        "sim_latent": {i: features_z[(i, 1)] for i in episodes_ids(episodes)},
        "sim_history_xyz": {i: features_raw[(i, 1)] for i in episodes_ids(episodes)},
        "sim_history_z": {i: features_raw[(i, 1)].reshape(-1, HISTORY, 3)[..., 2] for i in episodes_ids(episodes)},
        "motion_history_replay_reference": {
            i: histories(episodes[(i, 1)], np.column_stack((episodes[(i, 1)].q, episodes[(i, 1)].dq)))
            for i in episodes_ids(episodes)
        },
    }
    for name, features in cases.items():
        ridge[name] = ridge_probe(features, episodes, splits, stats)
        print(name, {s: r["summary"]["forecast_score_n"] for s, r in ridge[name].items()}, flush=True)
    write_json(out / "ridge_probes.json", ridge)
    heads = {}
    for name, features, joint, head in [
        ("refit_sim_head", features_z, False, model.forecaster),
        ("refit_joint_head", features_z, True, model.forecaster),
        ("raw_sim_history_mlp", features_raw, False, None),
    ]:
        heads[name] = head_probe(
            model, features, episodes, splits, out, name, args.head_epochs, joint=joint, initial_head=head
        )
        print(name, {s: r["sim"]["summary"]["forecast_score_n"] for s, r in heads[name]["splits"].items()}, flush=True)
    gradient = gradient_diagnostic(model, episodes, arrays, stats, splits["train"])
    write_json(out / "gradient_diagnostic.json", gradient)
    after = {p: sha256(Path(p)) for p in before}
    assert before == after, "Source checkpoint changed during diagnostics"
    write_json(
        out / "summary.json",
        {
            "source_run": str(root),
            "original_checkpoints_unchanged": before == after,
            "real_encoder_and_head_unchanged": frozen_unchanged,
            "original": {s: {d: v["summary"] for d, v in r.items()} for s, r in original["frozen_model"].items()},
            "ridge": {k: {s: v["summary"] for s, v in r.items()} for k, r in ridge.items()},
            "heads": {
                k: {
                    "selected_epoch": r["selected_epoch"],
                    "scores": {s: {d: v["summary"] for d, v in q.items()} for s, q in r["splits"].items()},
                }
                for k, r in heads.items()
            },
            "gradient": gradient,
            "checkpoint_sha256": before,
            "limitation": "Exploratory diagnosis on one existing split. Linear correlations and failed probes do not establish a predictability upper bound.",
        },
    )
    print(f"Complete: {out}", flush=True)


def episodes_ids(episodes):
    return sorted({i for i, d in episodes})


if __name__ == "__main__":
    main()
