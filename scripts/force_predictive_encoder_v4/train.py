"""One fixed trajectory split: train a Real teacher, then a Sim encoder with a frozen decoder."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

from force_gap_forecast_v1.common import sha256
from force_gap_forecast_v1.common import write_json
from force_gap_forecast_v1.data import FOLDS
from force_gap_forecast_v1.data import HOLDOUT
from force_gap_forecast_v1.data import TRAIN_IDS
from force_gap_forecast_v1.data import load_pairs
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from . import BLOCK
from . import FORMAT
from . import FUTURE
from . import HISTORY
from . import LAYOUT
from . import baselines
from .data import Blocks
from .data import episode_inputs
from .data import make_stats
from .losses import objective
from .metrics import aggregate
from .metrics import evaluate
from .model import PredictiveForce


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def digest(model, *, frozen_only=False):
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not frozen_only or name not in trainable:
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def save_model(path, model, stage):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "artifact_format": FORMAT,
            "layout": LAYOUT,
            "target_domain": "real",
            "horizons": list(range(FUTURE + 1)),
            "stats": model.stats,
            "state_dict": cpu_state(model),
            "stage": stage,
        },
        temporary,
    )
    temporary.replace(path)


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("artifact_format") != FORMAT
        or payload.get("layout") != LAYOUT
        or payload.get("target_domain") != "real"
        or payload.get("horizons") != list(range(FUTURE + 1))
    ):
        raise ValueError(f"Not a v4 real-force checkpoint: {path}")
    model = PredictiveForce(payload["stats"]).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, payload


def run_stage(model, dataset, episodes, valid_ids, arrays, args, stage, output):
    domain = 0 if stage == "teacher" else 1
    name = ("real", "sim")[domain]
    seed = args.seed + 1000 * domain
    seed_all(seed)
    model = copy.deepcopy(model).to(args.device)
    if stage == "student":
        model.initialize_student()
    model.configure(stage)
    identity = {"stage": stage, "seed": seed, "initial_state_sha256": digest(model)}
    checkpoint, history_path = output / "model.pt", output / "history.json"
    if args.resume and history_path.is_file():
        previous = json.loads(history_path.read_text())
        if previous.get("identity") != identity:
            raise ValueError(f"Stage parent changed: {output}")
        if previous.get("completed"):
            restored, payload = load_model(checkpoint, args.device)
            if payload["stage"] != previous or digest(restored) != previous["state_sha256"]:
                raise ValueError(f"Completed stage checkpoint changed: {output}")
            restored.configure(stage)
            print(f"Reusing completed {stage}: epoch {previous['selected_epoch']}", flush=True)
            return restored, previous
        print(f"Restarting interrupted {stage} from its fixed parent", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    frozen_sha = digest(model, frozen_only=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    steps = max(1, math.ceil(len(dataset) / (args.batch_size * BLOCK)))
    best_score, best_state, best_epoch, stale = math.inf, None, None, 0
    records = []
    started = time.monotonic()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch * 7919)
        sampler = WeightedRandomSampler(dataset.weights, steps * args.batch_size, replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        losses = []
        for sampled_batch in loader:
            batch = {key: value.to(args.device) for key, value in sampled_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            # Frozen decoder parameters still pass gradients to the Sim encoder.
            loss, detail = objective(model, model.block(batch["x"], domain), batch, args.delta_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {stage} loss at epoch {epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 3.0, error_if_nonfinite=True)
            detail["gradient_norm"] = float(norm.detach())
            optimizer.step()
            losses.append(detail)
        validation = evaluate(model, episodes, valid_ids, arrays, domains=(domain,))[name]
        score = validation["summary"]["forecast_score_n"]
        if score is None or not math.isfinite(score):
            raise ValueError("Validation needs finite +1/+3/+5 forecast metrics")
        if score < best_score:
            best_score, best_state, best_epoch, stale = score, cpu_state(model), epoch, 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train": aggregate(losses),
            "validation": validation,
            "selection_score_n": score,
            "elapsed_s": time.monotonic() - started,
        }
        records.append(record)
        write_json(history_path, {"completed": False, "identity": identity, "records": records})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.max_epochs:
            summary = validation["summary"]
            print(
                f"{stage}: epoch={epoch}/{args.max_epochs} loss={record['train']['total']:.5f} "
                f"current_N={summary['current_rmse_n']:.4f} future_N={score:.4f} "
                f"delta_N={summary['increment_rmse_n']:.4f} "
                f"rank={validation['latent']['effective_rank']:.2f} elapsed={record['elapsed_s']:.0f}s",
                flush=True,
            )
        if stale >= args.patience:
            break
    save_model(output / "last.pt", model, {"epoch": records[-1]["epoch"], "selected": False})
    model.load_state_dict(best_state)
    if digest(model, frozen_only=True) != frozen_sha:
        raise RuntimeError(f"Frozen weights changed during {stage}")
    summary = {
        "completed": True,
        "identity": identity,
        "selected_epoch": best_epoch,
        "selected_future_score_n": best_score,
        "state_sha256": digest(model),
        "frozen_sha256": frozen_sha,
        "records": records,
    }
    save_model(checkpoint, model, summary)
    write_json(history_path, summary)
    return model, summary


def prepare(args):
    episodes, audit = load_pairs(args.real_root, args.sim_root, args.labels)
    scripts = Path(__file__).resolve().parent.parent
    sources = [
        *list(Path(__file__).parent.glob("*.py")),
        Path(__file__).parent / "README.md",
        scripts / "run_force_predictive_encoder_v4.sh",
        *[scripts / "force_predictive_encoder_v3" / name for name in ("__init__.py", "model.py", "data.py")],
        *[scripts / "force_gap_forecast_v1" / name for name in ("__init__.py", "data.py", "common.py")],
    ]
    recipe = {
        "artifact_format": FORMAT,
        "options": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ("output_dir", "resume", "device", "threads")
        },
        "data": audit,
        "label_sha256": sha256(args.labels) if args.labels is not None else None,
        "validation_ids": sorted(FOLDS[args.validation_fold]),
        "train_ids": sorted(set(TRAIN_IDS) - set(FOLDS[args.validation_fold])),
        "diagnostic_ids": HOLDOUT,
        "loss": {
            "current": 1.0,
            "future": 1.0,
            "delta": args.delta_weight,
            "reduction": "masked_element_mean",
            "function": "smooth_l1_beta_1",
            "current_future_scale_floor_n": 0.1,
            "increment_scale_floor_n": 0.05,
            "latent_alignment": 0.0,
            "sim_reconstruction": 0.0,
            "legacy_distillation": 0.0,
        },
        "selection": "minimum trajectory-equal future Real force score (.25*+1 + .35*+3 + .40*+5) in N",
        "source_sha256": {str(p.relative_to(scripts)): sha256(p) for p in sorted(sources)},
    }
    root = args.output_dir
    if args.resume:
        if not (root / "recipe.json").is_file() or json.loads((root / "recipe.json").read_text()) != recipe:
            raise ValueError("Resume requires identical code, data, labels and training parameters; use a new RUN_DIR")
    elif root.exists() and any(p.name != "train.log" for p in root.iterdir()):
        raise FileExistsError("RUN_DIR is not empty; use a new RUN_DIR or matching RESUME=1")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "recipe.json", recipe)
    write_json(root / "data_audit.json", audit)
    for source in sources:
        dest = root / "source_snapshot" / source.relative_to(scripts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    return episodes, recipe


def write_report(root, result):
    lines = [
        "# Force predictive encoder v4",
        "",
        "One fixed trajectory split; validation is used for early stopping. Not a cross-validation estimate.",
        "Diagnostic trajectories were excluded from this run's training/selection, but have been inspected in earlier experiments.",
        "",
        f"Loss: current + future + {result['delta_weight']} * increment (separate train-only scales).",
        "Targets: current and next five REAL force samples, for both input domains.",
        "",
        f"Selected epochs: {result['selected_epochs']}",
        "",
        "| Set | Input | Model | Current RMSE (N) | Future score (N) | First increment RMSE (N) | All increments RMSE (N) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for split in ("validation", "diagnostic"):
        for domain in ("real", "sim"):
            for name, data in [("encoder", result[split][domain]), *result["baselines"][split][domain].items()]:
                r = data["summary"]
                lines.append(
                    f"| {split} | {domain} | {name} | {r['current_rmse_n']:.4f} | "
                    f"{r['forecast_score_n']:.4f} | {r['first_increment_rmse_n']:.4f} | {r['increment_rmse_n']:.4f} |"
                )
    lines += [
        "",
        "Hold uses measured current force for Real input; Sim hold uses a training-fitted affine calibration.",
        "AR uses force history only, fixed ridge alpha=1, fitted only on training trajectories.",
        "All aggregate prediction errors weight trajectories equally. Increment metrics include current -> +1.",
        "",
        "Latent diagnostics:",
    ]
    for domain in ("real", "sim"):
        r = result["validation"][domain]["latent"]
        lines.append(
            f"- {domain}: effective rank {r['effective_rank']:.3f}, first PC fraction {r['first_pc_variance_fraction']:.4f}"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")


def train(episodes, recipe, args):
    root = args.output_dir
    train_ids, valid_ids, diagnostic_ids = [recipe[k] for k in ("train_ids", "validation_ids", "diagnostic_ids")]
    stats = make_stats(episodes, train_ids)
    write_json(root / "normalization.json", stats)
    print(
        f"Output: {root}\n{len(train_ids)} train / {len(valid_ids)} validation / {len(diagnostic_ids)} diagnostic; "
        f"seed={args.seed}; current + future + {args.delta_weight}*delta",
        flush=True,
    )
    arrays = {
        (tid, domain): torch.from_numpy(episode_inputs(ep, stats, ("real", "sim")[domain]))
        for (tid, domain), ep in episodes.items()
    }
    dataset = Blocks(episodes, train_ids, arrays, stats)
    seed_all(args.seed)
    initial = PredictiveForce(stats)
    teacher, tc = run_stage(initial, dataset, episodes, valid_ids, arrays, args, "teacher", root / "teacher")
    model, sc = run_stage(teacher, dataset, episodes, valid_ids, arrays, args, "student", root / "student")
    validation = evaluate(model, episodes, valid_ids, arrays, root / "predictions" / "validation")
    diagnostic = evaluate(model, episodes, diagnostic_ids, arrays, root / "predictions" / "diagnostic")
    result = {
        "completed": True,
        "status": "single_split_diagnostic",
        "seed": args.seed,
        "delta_weight": args.delta_weight,
        "selected_epochs": {"teacher": tc["selected_epoch"], "student": sc["selected_epoch"]},
        "validation": validation,
        "diagnostic": diagnostic,
        "baselines": {"validation": {}, "diagnostic": {}},
        "selection": recipe["selection"],
        "diagnostic_is_blind_test": False,
    }
    coefficients = {}
    for domain, name in enumerate(("real", "sim")):
        for split in result["baselines"].values():
            split[name] = {}
        for kind in ("hold", "ar"):
            coef = baselines.fit(episodes, train_ids, stats, domain, kind)
            if coef is not None:
                coefficients[f"{name}_{kind}"] = coef
            for split, ids in (("validation", valid_ids), ("diagnostic", diagnostic_ids)):
                result["baselines"][split][name][kind] = baselines.evaluate(coef, episodes, ids, stats, domain, kind)
    np.savez_compressed(root / "baseline_coefficients.npz", **coefficients)
    checkpoint = root / f"encoder_seed{args.seed}.pt"
    save_model(checkpoint, model, {"selected_epochs": result["selected_epochs"], "selection": recipe["selection"]})
    write_json(
        root / "manifest.json",
        {
            "artifact_format": FORMAT,
            "layout": LAYOUT,
            "checkpoint": checkpoint.name,
            "encoder_sha256": sha256(checkpoint),
            "history": HISTORY,
            "future": FUTURE,
            "horizons": list(range(FUTURE + 1)),
            "policy_history": 10,
            "target_domain": "real",
            "input": "force_xyz_N_and_timestamps",
            "force_contract": "aligned_base_xyz_n_v1",
            "status": "single_split_diagnostic",
            "selection": recipe["selection"],
            "requires_policy_retraining": True,
            "actual_sim_state_verified": recipe["data"]["actual_sim_state_verified"],
        },
    )
    write_json(root / "result.json", result)
    write_report(root, result)
    print(f"Complete: {root / 'report.md'}\nEncoder: {checkpoint}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, default=Path("data"))
    parser.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    default_labels = Path("annotations/insertion_quality_labels_v2.csv")
    parser.add_argument("--labels", type=Path, default=default_labels if default_labels.is_file() else None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fold", type=int, choices=range(4), default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("Epochs, patience, batch size and threads must be positive")
    if not 0 <= args.seed < 2**32 - 1000:
        parser.error("Seed must be in [0, 2**32-1000)")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.delta_weight) or args.delta_weight < 0:
        parser.error("Learning rate must be positive finite; delta weight must be nonnegative finite")
    for key in ("output_dir", "real_root", "sim_root", "labels"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.labels is not None and not args.labels.is_file():
        parser.error(f"Missing label file: {args.labels}")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run on the external GPU host, or use --device cpu for checks")
    episodes, recipe = prepare(args)
    train(episodes, recipe, args)


if __name__ == "__main__":
    main()
