"""Force-only shared 32-D representations, paired alignment and weak 8001 distillation."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from force_gap_forecast_v1.common import sha256, write_json
from force_gap_forecast_v1.data import FOLDS, HOLDOUT, segment_limits
from insertion_alignment_encoder.model import ForceAlignmentEncoder
from insertion_alignment_encoder.preprocess import force_history_features, initial_baseline, normalize_features

FORMAT = "tavla_force_shared_latent_v1"
HISTORY, BLOCK, FUTURE, GAP = 32, 16, 5, .5
DOMAINS = ("real", "sim")
LAYOUT = [f"shared_{i}" for i in range(32)]


@dataclass
class Episode:
    force: np.ndarray
    time: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def read_pairs(real_root, sim_root):
    files = [{int(p.parent.name.split("_")[-1]): p for p in Path(root).glob("traj_*/data.h5")}
             for root in (real_root, sim_root)]
    if not files[0] or set(files[0]) != set(files[1]):
        raise ValueError("Both directories must contain the same paired traj_*/data.h5 IDs")
    episodes, digest = {}, hashlib.sha256()
    for tid in sorted(files[0]):
        for d in range(2):
            with h5py.File(files[d][tid], "r") as f:
                force = np.asarray(f["obs/state/ee_force_base"], np.float32)
                timestamps = np.asarray(f["timestamps"], np.float64)
            if (force.shape != (len(timestamps), 3) or len(timestamps) < 2 or
                    not np.isfinite(force).all() or not np.isfinite(timestamps).all() or
                    np.any(np.diff(timestamps) <= 0)):
                raise ValueError(f"Invalid force/timestamps: {files[d][tid]}")
            starts, ends = segment_limits(timestamps)
            episodes[(tid, d)] = Episode(force, timestamps, starts, ends)
            digest.update(f"{tid}:{d}".encode())
            digest.update(force.tobytes()); digest.update(timestamps.tobytes())
        real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
        if (len(real.force) != len(sim.force) or not np.allclose(
                real.time - real.time[0], sim.time - sim.time[0], rtol=0, atol=.02)):
            raise ValueError(f"Trajectory {tid}: paired frames/timestamps disagree")
    return episodes, sorted(files[0]), digest.hexdigest()


def derivatives(ep):
    dt = np.clip(np.diff(ep.time, prepend=ep.time[:1]), .02, GAP)
    value = np.diff(ep.force, axis=0, prepend=ep.force[:1]) / dt[:, None]
    value[np.unique(ep.starts)] = 0
    return value.astype(np.float32)


def fit_stats(episodes, train_ids):
    stats = {"train_ids": sorted(train_ids)}
    for d, name in enumerate(DOMAINS):
        values = np.concatenate([np.column_stack((episodes[(i, d)].force, derivatives(episodes[(i, d)])))
                                 for i in train_ids])
        stats[name] = dict(mean=values.mean(0).tolist(), scale=np.maximum(values.std(0), .05).tolist())
    return stats


def window_features(force, timestamps, stats, domain):
    """Accept one 1..32-frame causal force window. No motion or absolute time input."""
    force, timestamps = np.asarray(force, np.float32), np.asarray(timestamps, np.float64)
    if (force.shape != (len(timestamps), 3) or not 1 <= len(force) <= HISTORY or
            not np.isfinite(force).all() or not np.isfinite(timestamps).all() or
            np.any(np.diff(timestamps) <= 0) or np.any(np.diff(timestamps) > GAP)):
        raise ValueError("Expected finite synchronized force [1..32,3] and increasing timestamps without gaps")
    force = np.pad(force, ((HISTORY - len(force), 0), (0, 0)), mode="edge")
    dt = np.r_[.1, np.diff(timestamps)]
    dt = np.r_[np.full(HISTORY - len(timestamps), .1), dt]
    delta = np.diff(force, axis=0, prepend=force[:1]) / np.clip(dt[:, None], .02, GAP)
    values = np.column_stack((force, delta))
    return ((values - np.asarray(stats[domain]["mean"], np.float32)) /
            np.asarray(stats[domain]["scale"], np.float32)).astype(np.float32)


def prepare_arrays(episodes, ids, stats, teacher, legacy_payload, device):
    arrays = {}
    teacher.eval().requires_grad_(False)
    baseline_frames = int(legacy_payload["config"]["baseline_frames"])
    scales = np.asarray(legacy_payload["preprocess"]["feature_scales"], np.float32)
    for tid in ids:
        for d, name in enumerate(DOMAINS):
            ep = episodes[(tid, d)]
            x = np.stack([window_features(ep.force[max(ep.starts[t], t - HISTORY + 1):t + 1],
                                         ep.time[max(ep.starts[t], t - HISTORY + 1):t + 1], stats, name)
                          for t in range(len(ep.force))])
            arrays[(tid, d)] = {"x": torch.from_numpy(x)}
        ep = episodes[(tid, 0)]
        targets = np.zeros((len(ep.force), 32), np.float32)
        ready = np.zeros(len(ep.force), bool)
        for start in np.unique(ep.starts):
            end = ep.ends[start]
            if end - start < baseline_frames:
                continue
            force, timestamps = ep.force[start:end], ep.time[start:end]
            features = normalize_features(force_history_features(force, timestamps,
                                          initial_baseline(force, baseline_frames)), scales)
            ii = np.arange(baseline_frames - 1, len(force))
            indices = np.maximum(ii[:, None] + np.arange(-HISTORY + 1, 1), 0)
            with torch.inference_mode():
                for offset in range(0, len(ii), 256):
                    latent = teacher(torch.from_numpy(features[indices[offset:offset + 256]]).to(device))[0]
                    targets[start + ii[offset:offset + 256]] = latent.cpu().numpy()
            ready[start + ii] = True
        arrays[(tid, 0)].update(old=torch.from_numpy(targets), old_valid=torch.from_numpy(ready))
    return arrays


class PairedBlocks(Dataset):
    def __init__(self, episodes, ids, arrays):
        self.episodes, self.arrays = episodes, arrays
        self.keys = [(i, t) for i in ids for t in range(len(episodes[(i, 0)].force))]
        self.weights = torch.tensor([1 / len(episodes[(i, 0)].force) for i, _ in self.keys], dtype=torch.double)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        tid, anchor = self.keys[index]
        real, sim = self.episodes[(tid, 0)], self.episodes[(tid, 1)]
        start = max(real.starts[anchor], sim.starts[anchor])
        original = np.arange(anchor - BLOCK, anchor + 1)
        ii = np.maximum(original, start)
        return make_batch(self.episodes, self.arrays, tid, ii, original >= start)


def make_batch(episodes, arrays, tid, ii, valid):
    real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
    fi = ii[:, None] + np.arange(1, FUTURE + 1)
    ends = np.minimum(real.ends[ii], sim.ends[ii])
    return dict(x=torch.stack([arrays[(tid, d)]["x"][ii] for d in range(2)]),
                valid=torch.as_tensor(valid), segment=torch.as_tensor(np.maximum(real.starts[ii], sim.starts[ii])),
                future=torch.from_numpy(real.force[np.minimum(fi, len(real.force) - 1)]),
                future_valid=torch.as_tensor((fi < ends[:, None]) & valid[:, None]),
                old=arrays[(tid, 0)]["old"][ii], old_valid=arrays[(tid, 0)]["old_valid"][ii] & torch.as_tensor(valid))


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 24), nn.GELU(), nn.Linear(24, 6))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class SharedForceModel(nn.Module):
    def __init__(self, stats, legacy_state=None):
        super().__init__()
        old = ForceAlignmentEncoder()
        if legacy_state is not None:
            old.load_state_dict(legacy_state)
        self.stem, self.temporal, self.gru = old.stem, old.temporal, old.gru
        self.projection = nn.Sequential(*list(old.projection.children())[:-1])
        for layer in self.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = 0.
        self.adapters = nn.ModuleList([Adapter(), Adapter()])
        self.forecaster = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, FUTURE * 3))
        self.old_head = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.register_buffer("force_mean", torch.tensor(stats["real"]["mean"][:3]))
        self.register_buffer("force_scale", torch.tensor(stats["real"]["scale"][:3]))
        self.register_buffer("old_mean", torch.zeros(32))
        self.register_buffer("old_scale", torch.ones(32))

    def backbone(self, x):
        x = self.temporal(self.stem(x.transpose(1, 2))).transpose(1, 2)
        x, _ = self.gru(x)
        return self.projection(x[:, -1])

    def forward(self, x):
        # [batch, domain, endpoints, history, channels], all32 dimensions shared.
        b, _, t, h, c = x.shape
        x = torch.cat([self.adapters[d](x[:, d].reshape(b * t, h, c)) for d in range(2)])
        z = self.backbone(x).reshape(2, b, t, 32).permute(1, 0, 2, 3)
        normalized = self.forecaster(z).reshape(b, 2, t, FUTURE, 3)
        return dict(z=z, future=normalized * self.force_scale + self.force_mean, old=self.old_head(z))

    def encode(self, x, domain):
        z = self.backbone(self.adapters[domain](x))
        future = self.forecaster(z).reshape(-1, FUTURE, 3) * self.force_scale + self.force_mean
        return z, future


def masked_mean(values, mask):
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def regularization(z):
    if len(z) < 2:
        return z.sum() * 0, z.sum() * 0
    variance = F.relu(1 - torch.sqrt(z.var(0, unbiased=False) + 1e-4)).mean()
    centered = z - z.mean(0)
    cov = centered.T @ centered / (len(z) - 1)
    return variance, (cov - torch.diag_embed(cov.diagonal())).square().sum() / 32


def objective(model, output, batch, epoch):
    z, valid = output["z"], batch["valid"]
    real, sim = z[:, 0], z[:, 1]
    pair = masked_mean((real - sim).square(), valid)
    segments = []
    for size in (4, 8, 16):
        if real.shape[1] < size:
            continue
        r, s = [F.avg_pool1d(a.transpose(1, 2), size, stride=1).transpose(1, 2) for a in (real, sim)]
        parts = batch["segment"].unfold(1, size, 1)
        mask = valid.unfold(1, size, 1).all(-1) & (parts[..., 0] == parts[..., -1])
        value = masked_mean((r - s).square(), mask)
        if r.shape[1] > 1:
            changes = mask[:, 1:] & mask[:, :-1] & (parts[:, 1:, -1] == parts[:, :-1, -1])
            value = value + .5 * masked_mean((torch.diff(r, dim=1) - torch.diff(s, dim=1)).square(), changes)
        segments.append(value)
    segment = torch.stack(segments).mean() if segments else pair * 0
    target = (batch["future"] - model.force_mean) / model.force_scale
    pred = (output["future"] - model.force_mean) / model.force_scale
    future = masked_mean(F.smooth_l1_loss(pred, target[:, None].expand_as(pred), reduction="none"),
                         batch["future_valid"][:, None].expand(-1, 2, -1, -1))
    old_target = (batch["old"] - model.old_mean) / model.old_scale
    old = masked_mean(F.smooth_l1_loss(output["old"], old_target[:, None].expand_as(output["old"]), reduction="none"),
                      batch["old_valid"][:, None].expand(-1, 2, -1))
    # Only independent sampled block endpoints count toward batch variance.
    regs = [regularization(z[:, d, -1][valid[:, -1]]) for d in range(2)]
    variance, covariance = [sum(r[j] for r in regs) / 2 for j in range(2)]
    old_weight = .1 - .08 * min(max(epoch - 10, 0) / 20, 1)
    loss = pair + .5 * segment + .2 * future + old_weight * old + variance + .01 * covariance
    return loss, {k: float(v.detach()) for k, v in dict(total=loss, pair=pair, segment=segment,
                  future=future, old=old, variance=variance, covariance=covariance).items()}


def effective_rank(z):
    eigen = np.maximum(np.linalg.eigvalsh(np.cov(z.T, bias=True)), 0)
    if eigen.sum() < 1e-10:
        return 0.
    p = eigen[eigen > 0] / eigen.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


@torch.inference_mode()
def evaluate(model, episodes, ids, arrays, device, epoch, save_dir=None):
    model.eval()
    rows, latents = {}, [[], []]
    for tid in ids:
        n = len(episodes[(tid, 0)].force)
        outputs = {}
        for lo in range(0, n, 128):
            x = torch.stack([arrays[(tid, d)]["x"][lo:lo + 128] for d in range(2)])[None].to(device)
            for k, value in model(x).items():
                outputs.setdefault(k, []).append(value.cpu())
        outputs = {k: torch.cat(v, dim=2).to(device) for k, v in outputs.items()}
        batch = {k: v[None].to(device) for k, v in make_batch(episodes, arrays, tid, np.arange(n), np.ones(n, bool)).items()}
        _, detail = objective(model, outputs, batch, epoch)
        row = dict(loss=detail, domains={})
        for d, domain in enumerate(DOMAINS):
            z, future = outputs["z"][0, d].cpu().numpy(), outputs["future"][0, d].cpu().numpy()
            latents[d].append(z)
            errors = future - batch["future"][0].cpu().numpy()
            valid = batch["future_valid"][0].cpu().numpy()
            horizon = [float(np.sqrt(np.mean(errors[valid[:, h], h] ** 2))) if valid[:, h].any() else None
                       for h in range(FUTURE)]
            if any(v is None for v in horizon):
                raise ValueError(f"Trajectory {tid}: no valid five-step forecast targets")
            row["domains"][domain] = dict(future_rmse_n=horizon, future_score_n=float(np.dot(np.array(horizon)[[0, 2, 4]], [.25, .35, .4])))
            if save_dir is not None:
                path = Path(save_dir) / domain
                path.mkdir(parents=True, exist_ok=True)
                ep, real = episodes[(tid, d)], episodes[(tid, 0)]
                np.savez_compressed(path / f"traj_{tid}.npz", z=z, future=future, input_force=ep.force,
                                    real_force=real.force, time=ep.time, real_time=real.time, starts=ep.starts,
                                    ends=ep.ends, real_starts=real.starts, real_ends=real.ends)
        rows[str(tid)] = row
    z = [np.concatenate(a) for a in latents]
    spread = float(np.mean(z[0].var(0) + z[1].var(0)))
    pair = float(np.mean([r["loss"]["pair"] for r in rows.values()]))
    segment = float(np.mean([r["loss"]["segment"] for r in rows.values()]))
    std = [float(np.median(a.std(0))) for a in z]
    ranks = [effective_rank(a) for a in z]
    score = (pair + .5 * segment) / max(spread, 1e-8)
    return dict(pair_mse=pair, segment_mse=segment, alignment_score=score, latent_median_std=std,
                latent_effective_rank=ranks, noncollapsed=bool(min(std) >= .1),
                future_score_n={domain: float(np.mean([r["domains"][domain]["future_score_n"] for r in rows.values()]))
                                for domain in DOMAINS}, trajectories=rows)


def save(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def export(path, model, stats, epoch, metrics, legacy_sha):
    save(path, dict(artifact_format=FORMAT, layout=LAYOUT, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    stats=stats, epoch=epoch, validation=metrics, legacy_sha256=legacy_sha))


class SharedLatentEncoder:
    """Streaming inference: push(force_xyz, timestamp) -> z[32]; no motion input."""
    def __init__(self, checkpoint, domain="sim", device="cpu", threads=1):
        if domain not in DOMAINS:
            raise ValueError("domain must be real or sim")
        if str(device).startswith("cpu"):
            torch.set_num_threads(threads)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("artifact_format") != FORMAT or payload.get("layout") != LAYOUT:
            raise ValueError("Expected a shared-latent checkpoint")
        self.stats, self.domain, self.device = payload["stats"], domain, device
        self.model = SharedForceModel(self.stats).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval().requires_grad_(False)
        self.reset()

    def reset(self):
        self.force, self.timestamps, self.latents = deque(maxlen=HISTORY), deque(maxlen=HISTORY), deque(maxlen=10)
        self.future = None

    @torch.inference_mode()
    def push(self, force_xyz, timestamp):
        force, timestamp = np.asarray(force_xyz, np.float32), float(timestamp)
        if (force.shape != (3,) or not np.isfinite(force).all() or not math.isfinite(timestamp) or
                (self.timestamps and timestamp <= self.timestamps[-1])):
            raise ValueError("Finite XYZ force and increasing timestamps required; reset between episodes")
        if self.timestamps and timestamp - self.timestamps[-1] > GAP:
            self.reset()
        self.force.append(force.copy()); self.timestamps.append(timestamp)
        x = window_features(np.asarray(self.force), np.asarray(self.timestamps), self.stats, self.domain)
        z, future = self.model.encode(torch.from_numpy(x[None]).to(self.device), DOMAINS.index(self.domain))
        if not torch.isfinite(z).all() or not torch.isfinite(future).all():
            raise FloatingPointError("Non-finite encoder output")
        self.future = future[0].cpu().numpy().copy()
        z = z[0].cpu().numpy().copy()
        self.latents.append(z)
        return z.copy()

    def predict_future(self):
        if self.future is None:
            raise RuntimeError("Push a force observation first")
        return self.future.copy()

    def effort_history(self):
        if not self.latents:
            raise RuntimeError("Push a force observation first")
        return np.pad(np.stack(self.latents), ((10 - len(self.latents), 0), (0, 0)), mode="edge")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--real-root", type=Path, default=Path("data"))
    p.add_argument("--sim-root", type=Path, default=Path("sim-data-aligned"))
    p.add_argument("--legacy", type=Path, default=Path("exports/force_align_latent_sim_bundle_20261004/assets/encoder.pt"))
    p.add_argument("--validation-ids", type=int, nargs="+", default=sorted(FOLDS[0]))
    p.add_argument("--diagnostic-ids", type=int, nargs="*", default=HOLDOUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if (min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or not 0 <= args.seed < 2**32 or
            not math.isfinite(args.lr) or args.lr <= 0 or
            len(set(args.validation_ids)) != len(args.validation_ids) or
            len(set(args.diagnostic_ids)) != len(args.diagnostic_ids) or
            set(args.validation_ids) & set(args.diagnostic_ids)):
        p.error("Invalid positive training parameters, seed, or disjoint trajectory IDs")
    for field in ("output_dir", "real_root", "sim_root", "legacy"):
        setattr(args, field, getattr(args, field).resolve())
    return args


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run on the GPU host, or use --device cpu for a smoke check")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = args.output_dir
    if not args.resume and root.exists() and any(p.name != "train.log" for p in root.iterdir()):
        raise FileExistsError("Output directory is not empty; use a new RUN_DIR or --resume")
    root.mkdir(parents=True, exist_ok=True)
    episodes, ids, data_sha = read_pairs(args.real_root, args.sim_root)
    val_ids, diagnostic_ids = sorted(args.validation_ids), sorted(args.diagnostic_ids)
    if not set(val_ids + diagnostic_ids) <= set(ids):
        raise ValueError("Requested validation/diagnostic trajectory IDs are missing")
    train_ids = sorted(set(ids) - set(val_ids + diagnostic_ids))
    if len(train_ids) < 2:
        raise ValueError("At least two training trajectories required")
    stats = fit_stats(episodes, train_ids)
    old_payload = torch.load(args.legacy, map_location="cpu", weights_only=True)
    if old_payload.get("artifact_format") != "tavla_force_alignment_encoder_terminal_reference_v1":
        raise ValueError("Expected the 8001 force-alignment teacher artifact")
    old_sha = sha256(args.legacy)
    identity = dict(options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                            if k not in ("output_dir", "resume", "device", "threads")},
                    train_ids=train_ids, validation_ids=val_ids, diagnostic_ids=diagnostic_ids,
                    force_data_sha256=data_sha, legacy_sha256=old_sha,
                    source_sha256={p.name: sha256(p) for p in (Path(__file__),
                        Path(__file__).parent / "insertion_alignment_encoder/model.py",
                        Path(__file__).parent / "insertion_alignment_encoder/preprocess.py")})
    if args.resume:
        import json
        if json.loads((root / "recipe.json").read_text()) != identity:
            raise ValueError("Resume requires identical data, teacher, code, split and hyperparameters")
    else:
        write_json(root / "recipe.json", identity)
    teacher = ForceAlignmentEncoder().to(args.device)
    teacher.load_state_dict(old_payload["model_state_dict"])
    print(f"Output: {root}\nOne seed {args.seed}; {len(train_ids)} train / {len(val_ids)} validation / "
          f"{len(diagnostic_ids)} diagnostic; force-only, full32 alignment", flush=True)
    print("Caching force windows and frozen 8001 teacher targets once...", flush=True)
    arrays = prepare_arrays(episodes, train_ids + val_ids + diagnostic_ids, stats, teacher, old_payload, args.device)
    del teacher
    model = SharedForceModel(stats, old_payload["model_state_dict"]).to(args.device)
    old_train = torch.cat([arrays[(i, 0)]["old"][arrays[(i, 0)]["old_valid"]] for i in train_ids])
    if not len(old_train):
        raise ValueError("Training trajectories are too short for the old teacher's baseline warmup")
    model.old_mean.copy_(old_train.mean(0).to(args.device))
    model.old_scale.copy_(old_train.std(0, unbiased=False).clamp_min(.1).to(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dataset = PairedBlocks(episodes, train_ids, arrays)
    steps = max(1, math.ceil(len(dataset) / (args.batch_size * (BLOCK + 1))))
    start_epoch, stale, history, best = 1, 0, [], (2, math.inf, math.inf)
    if args.resume:
        state = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch, stale, history, best = state["epoch"] + 1, state["stale"], state["history"], tuple(state["best"])
        for state_values in optimizer.state.values():
            for k, v in state_values.items():
                if isinstance(v, torch.Tensor) and k != "step":
                    state_values[k] = v.to(args.device)
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(args.seed + epoch * 7919)
        sampler = WeightedRandomSampler(dataset.weights, steps * args.batch_size, replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        losses = []
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, detail = objective(model, model(batch["x"]), batch, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            losses.append(detail)
        validation = evaluate(model, episodes, val_ids, arrays, args.device, epoch)
        candidate = (0 if validation["noncollapsed"] else 1, validation["alignment_score"],
                     sum(validation["future_score_n"].values()) / 2)
        if candidate < best:
            best, stale = candidate, 0
            export(root / "encoder.pt", model, stats, epoch, validation, old_sha)
        else:
            stale += 1
        record = dict(epoch=epoch, train={k: float(np.mean([r[k] for r in losses])) for k in losses[0]},
                      validation=validation, elapsed_s=time.monotonic() - started)
        history.append(record)
        write_json(root / "history.json", history)
        save(root / "last.pt", dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                   optimizer=optimizer.state_dict(), epoch=epoch, history=history, best=list(best), stale=stale))
        print(f"epoch={epoch}/{args.epochs} loss={record['train']['total']:.4f} "
              f"pair={validation['pair_mse']:.4f} align={validation['alignment_score']:.4f} "
              f"std={validation['latent_median_std']} rank={validation['latent_effective_rank']} "
              f"future_N={validation['future_score_n']} elapsed={record['elapsed_s']:.0f}s", flush=True)
        if epoch >= min(20, args.epochs) and stale >= args.patience:
            break
    payload = torch.load(root / "encoder.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    epoch = payload["epoch"]
    validation = evaluate(model, episodes, val_ids, arrays, args.device, epoch, root / "predictions/validation")
    diagnostic = evaluate(model, episodes, diagnostic_ids, arrays, args.device, epoch, root / "predictions/diagnostic") if diagnostic_ids else None
    write_json(root / "result.json", dict(completed=True, selected_epoch=epoch, seed=args.seed,
               validation=validation, diagnostic=diagnostic,
               status="single_split_diagnostic" if validation["noncollapsed"] else "collapsed_diagnostic",
               selection="full32 paired/segment alignment after noncollapse gate; future prediction breaks ties",
               diagnostic_is_blind_test=False))
    write_json(root / "manifest.json", dict(artifact_format=FORMAT, checkpoint="encoder.pt", encoder_sha256=sha256(root / "encoder.pt"),
               layout=LAYOUT, history=HISTORY, future=FUTURE, policy_history=10, input="force_xyz_N_and_timestamps",
               legacy_sha256=old_sha, requires_policy_retraining=True, status="single_split_diagnostic"))
    print(f"Complete: {root / 'encoder.pt'}; selected epoch={epoch}; noncollapsed={validation['noncollapsed']}", flush=True)


if __name__ == "__main__":
    train(parse_args())
