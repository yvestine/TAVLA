"""Matched short CPU continuations to distinguish loss tails, delta supervision and head freezing."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from force_gap_forecast_v1.common import sha256
from force_gap_forecast_v1.common import write_json
from force_gap_forecast_v1.data import load_pairs
from force_predictive_encoder_v4.data import Blocks
from force_predictive_encoder_v4.data import episode_inputs
from force_predictive_encoder_v4.losses import masked_huber
from force_predictive_encoder_v4.metrics import evaluate
from force_predictive_encoder_v4.train import cpu_state
from force_predictive_encoder_v4.train import digest
from force_predictive_encoder_v4.train import load_model
from force_predictive_encoder_v4.train import save_model
from force_predictive_encoder_v4.train import seed_all
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler


def loss_for(model, output, batch, delta_weight, quadratic_tail):
    p, y, mask = output["force"], batch["target"], batch["mask"]
    error = (p - y) / model.target_scale
    if quadratic_tail:

        def reduce(e, m):
            weights = m.unsqueeze(-1).expand_as(e)
            return torch.where(weights, 0.5 * e.square(), 0.0).sum() / weights.sum().clamp_min(1)
    else:
        reduce = masked_huber
    current = reduce(error[..., 0, :], mask[..., 0])
    future = reduce(error[..., 1:, :], mask[..., 1:])
    change = masked_huber(
        (torch.diff(p, dim=-2) - torch.diff(y, dim=-2)) / model.increment_scale, mask[..., 1:] & mask[..., :-1]
    )
    return current + future + delta_weight * change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.run_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((root / "recipe.json").read_text())
    opts = recipe["options"]
    path = root / json.loads((root / "manifest.json").read_text())["checkpoint"]
    before = sha256(path)
    initial, _ = load_model(path)
    episodes, _ = load_pairs(opts["real_root"], opts["sim_root"], opts["labels"])
    arrays = {
        (i, d): torch.from_numpy(episode_inputs(ep, initial.stats, ("real", "sim")[d]))
        for (i, d), ep in episodes.items()
    }
    train_ids, valid_ids = recipe["train_ids"], recipe["validation_ids"]
    dataset = Blocks(episodes, train_ids, arrays, initial.stats)
    initial_val = evaluate(initial, episodes, valid_ids, arrays)
    initial_score = initial_val["sim"]["summary"]["forecast_score_n"]
    result = {}
    for name, unfreeze, delta, quadratic in [
        ("continue_control", False, 0.5, False),
        ("unfreeze_sim_head", True, 0.5, False),
        ("remove_delta", False, 0.0, False),
        ("quadratic_force_tail", False, 0.5, True),
    ]:
        seed_all(1042)
        model = copy.deepcopy(initial)
        model.configure("student")
        if unfreeze:
            model.forecaster.requires_grad_(requires_grad=True)
        frozen = digest(model, frozen_only=True)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        best = initial_score
        best_epoch = 0
        best_state = cpu_state(model)
        stale = 0
        records = []
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            g = torch.Generator().manual_seed(1042 + epoch * 7919)
            steps = math.ceil(len(dataset) / (32 * 16))
            sampler = WeightedRandomSampler(dataset.weights, steps * 32, replacement=True, generator=g)
            loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=0)
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(model, model.block(batch["x"], 1), batch, delta, quadratic)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 3.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            row = evaluate(model, episodes, valid_ids, arrays, domains=(1,))["sim"]["summary"]
            score = row["forecast_score_n"]
            if score < best:
                best = score
                best_epoch = epoch
                best_state = cpu_state(model)
                stale = 0
            else:
                stale += 1
            records.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": row})
            if epoch == 1 or epoch % 5 == 0:
                print(f"{name}: epoch={epoch}, validation={score:.4f}, best={best:.4f} at {best_epoch}", flush=True)
            if stale >= 8:
                break
        model.load_state_dict(best_state)
        assert digest(model, frozen_only=True) == frozen, "Frozen parameters changed"
        splits = {}
        for split, ids in [("train", train_ids), ("validation", valid_ids), ("diagnostic", recipe["diagnostic_ids"])]:
            evaluation = evaluate(model, episodes, ids, arrays)
            splits[split] = {d: r["summary"] for d, r in evaluation.items()}
        row = {
            "selected_epoch": best_epoch,
            "records": records,
            "splits": splits,
            "settings": {
                "unfreeze_head": unfreeze,
                "delta_weight": delta,
                "quadratic_force_tail": quadratic,
                "max_epochs": args.max_epochs,
                "lr": 1e-4,
                "patience": 8,
                "batch": 32,
                "initial_checkpoint_sha256": before,
            },
        }
        result[name] = row
        write_json(out / f"{name}.json", row)
        save_model(out / f"{name}.pt", model, {"exploratory_diagnostic": True, "selected_epoch": best_epoch})
        print("SELECTED", name, {s: r["sim"]["forecast_score_n"] for s, r in splits.items()}, flush=True)
    assert sha256(path) == before, "Original checkpoint changed"
    write_json(
        out / "summary.json",
        {
            "initial_validation": {d: r["summary"] for d, r in initial_val.items()},
            "continuations": result,
            "source_checkpoint_unchanged": True,
            "limitation": "Short local continuations from the same checkpoint, with reset optimizer. Epoch zero is an eligible selection. Not from-scratch ablations or an independent test.",
        },
    )


if __name__ == "__main__":
    main()
