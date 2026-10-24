"""Inspect saved force forecasts without inference, retraining, or GPU access.

Uses target timestamps and the training evaluator's segment masks. Metrics are
averaged equally over trajectories and seeds, rather than pooling all frames.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"full": "#e87924", "motion_only": "#8c60bf", "motion_control": "#3a79b7", "hold": "#209e91",
          "arx": "#697582", "target": "#202832", "current": "#3a79b7"}
NAMES = {"full": "Full encoder", "motion_only": "Motion only", "motion_control": "Equal-capacity motion control",
         "hold": "Recovered-force hold", "arx": "ARX"}


def read_json(path):
    return json.loads(path.read_text())


def ends(time):
    boundaries = np.r_[0, np.flatnonzero(np.diff(time) > .5) + 1, len(time)]
    result = np.zeros(len(time), dtype=np.int64)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        result[lo:hi] = hi
    return result


def valid_indices(data, step):
    indices = np.arange(len(data["real_force"]))
    limits = np.minimum(ends(data["real_time"]), ends(data["time"]))
    return indices[indices + step < limits]


def rmse(error):
    return float(np.sqrt(np.mean(error ** 2))) if len(error) else None


def aligned(data, values, step):
    result = np.full_like(data["real_force"], np.nan)
    indices = valid_indices(data, step)
    result[indices + step] = values[indices]
    return result


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def save_figure(fig, destination):
    fig.savefig(destination.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def forecast_anchor(data):
    """An origin three frames before a large target change; visualization only."""
    indices = valid_indices(data, 5)
    if not len(indices):
        return 0
    changes = np.linalg.norm(data["real_force"][indices + 3] -
                             data["real_force"][indices + 2], axis=1)
    return int(indices[np.argmax(changes)])


def load_records(root, outer, folder="predictions"):
    records = {}
    for path in sorted(root.glob(f"{folder}/seed*_fold*/traj_*.npz")):
        match = re.fullmatch(r"seed(\d+)_fold(\d+)_(.+)", path.parent.name)
        if match is None:
            continue
        seed, fold, variant = int(match[1]), int(match[2]), match[3]
        tid = int(path.stem.removeprefix("traj_"))
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        # Confirm that independent recomputation reproduces the official metric.
        for step in (1, 3, 5):
            indices = valid_indices(data, step)
            actual = rmse(data["future"][indices, step - 1] - data["real_force"][indices + step])
            expected = outer["models"][variant][str(seed)][str(tid)]["all"][f"future_{step}_rmse_n"]
            if not np.isclose(actual, expected, rtol=1e-6, atol=1e-7):
                raise ValueError(f"Metric mismatch: {path}, horizon={step}")
        records[(variant, seed, tid)] = data
    return records


def make_summary(root, output, outer, records, recipe):
    seeds = recipe["seeds"]
    tids = recipe["train_ids"]
    rows = []
    for (variant, seed, tid), data in records.items():
        for step in range(1, 6):
            indices = valid_indices(data, step)
            target = data["real_force"][indices + step]
            error = data["future"][indices, step - 1] - target
            hold_error = data["current"][indices] - target
            rows.append({"variant": variant, "seed": seed, "trajectory": tid,
                         "horizon_frames": step, "valid_origins": len(indices),
                         "elapsed_mean_s": mean((data["real_time"][indices + step] -
                                                  data["real_time"][indices]).tolist()),
                         "forecast_rmse_n": rmse(error), "hold_rmse_n": rmse(hold_error),
                         **{f"forecast_{axis}_rmse_n": rmse(error[:, i])
                            for i, axis in enumerate("xyz")}})
    with (output / "forecast_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    horizons = {}
    for variant in recipe["variants"]:
        horizons[variant] = [mean([r["forecast_rmse_n"] for r in rows
                                  if r["variant"] == variant and r["horizon_frames"] == step])
                             for step in range(1, 6)]
    hold = [mean([r["hold_rmse_n"] for r in rows if r["variant"] == "full" and
                  r["horizon_frames"] == step]) for step in range(1, 6)]
    bootstrap = {}
    rng = np.random.default_rng(20261012)
    resamples = rng.integers(0, len(tids), (2000, len(tids)))
    for step in (1, 3, 5):
        paired = [mean([r["hold_rmse_n"] - r["forecast_rmse_n"] for r in rows
                        if r["variant"] == "full" and r["trajectory"] == tid and
                        r["horizon_frames"] == step]) for tid in tids]
        bootstrap[str(step)] = np.quantile(np.asarray(paired)[resamples].mean(axis=1), [.025, .975]).tolist()
    summary = {"aggregation": "mean of per-trajectory RMSE, then equal seed mean",
               "oof_trajectories": len(tids), "seeds": seeds,
               "forecast_rmse_n": horizons, "full_recovered_hold_rmse_n": hold,
               "relative_improvement_vs_hold": (1 - np.asarray(horizons["full"]) / hold).tolist(),
               "paired_trajectory_bootstrap_95_rmse_improvement_n": bootstrap,
               "validity": "Origin and target must be in the same segment in both domains."}
    (output / "forecast_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    comparison_variants = [v for v in ("full", "motion_control", "motion_only") if v in recipe["variants"]]
    for variant in comparison_variants:
        per_seed = np.array([[mean([r["forecast_rmse_n"] for r in rows if
                                    r["variant"] == variant and r["seed"] == seed and
                                    r["horizon_frames"] == step]) for step in range(1, 6)] for seed in seeds])
        ax.plot(np.arange(1, 6) * .1, horizons[variant], "o-", color=COLORS[variant], label=NAMES[variant])
        ax.fill_between(np.arange(1, 6) * .1, per_seed.min(axis=0), per_seed.max(axis=0),
                        color=COLORS[variant], alpha=.13)
    ax.plot(np.arange(1, 6) * .1, hold, "o--", color=COLORS["hold"], label=NAMES["hold"])
    arx = [mean([m["all"][f"future_{step}_rmse_n"] for m in outer["baselines"]["arx"].values()])
           for step in (1, 3, 5)]
    ax.plot([.1, .3, .5], arx, "s--", color=COLORS["arx"], label="ARX (saved horizons)")
    ax.set(title="A. Future-force error (band = range across 3 seeds)",
           xlabel="Forecast horizon (s, approximately)", ylabel="Trajectory-mean RMSE (N)")
    ax.legend(fontsize=9); ax.grid(alpha=.18)

    ax = axes[0, 1]
    phase_names = ["free", "misaligned_contact", "aligned_progress", "uncertain", "terminal", "last_8_frames"]
    x = np.arange(len(phase_names))
    offsets = np.linspace(-.25, .25, len(comparison_variants))
    width = .7 / len(comparison_variants)
    for offset, variant in zip(offsets, comparison_variants):
        values = [mean([m[phase].get("future_5_rmse_n") for s in outer["models"][variant].values()
                        for m in s.values() if phase in m]) for phase in phase_names]
        ax.bar(x + offset, values, width=width, color=COLORS[variant], label=NAMES[variant])
    ax.set_xticks(x, ["Free", "Misaligned contact", "Progress", "Uncertain", "Terminal", "Last 8 origins"])
    ax.set(title="B. +5-frame error by phase at forecast origin", ylabel="Trajectory-mean RMSE (N)")
    ax.tick_params(axis="x", rotation=15); ax.legend(fontsize=9); ax.grid(axis="y", alpha=.18)

    ax = axes[1, 0]
    for offset, variant in zip(offsets, comparison_variants):
        values = [mean([m["all"][f"current_{axis}_rmse_n"] for s in outer["models"][variant].values()
                        for m in s.values()]) for axis in "xyz"]
        ax.bar(np.arange(3) + offset, values, width=width, color=COLORS[variant], label=NAMES[variant])
    ax.set_xticks(np.arange(3), ["Fx", "Fy", "Fz"])
    ax.set(title="C. Current-force recovery: error by axis", ylabel="Trajectory-mean RMSE (N)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.18)

    ax = axes[1, 1]
    ordered = sorted(tids, key=lambda tid: mean([r["forecast_rmse_n"] for r in rows
                     if r["variant"] == "full" and r["trajectory"] == tid and r["horizon_frames"] == 5]))
    for variant in comparison_variants:
        values = [mean([r["forecast_rmse_n"] for r in rows if r["variant"] == variant and
                        r["trajectory"] == tid and r["horizon_frames"] == 5]) for tid in ordered]
        ax.plot(range(len(ordered)), values, "o-", markersize=3.5, color=COLORS[variant], label=NAMES[variant])
    ax.set_xticks(range(len(ordered)), ordered, rotation=90, fontsize=8)
    ax.set(title="D. Per-trajectory +5-frame RMSE (mean of 3 seeds)", xlabel="Trajectory ID, sorted by Full error",
           ylabel="RMSE (N)")
    ax.legend(fontsize=9); ax.grid(alpha=.18)
    fig.suptitle("Force forecast: out-of-fold evaluation on 32 paired trajectories", fontsize=16)
    save_figure(fig, output / "01_metrics_overview")
    return summary, ordered


def trajectory_plot(records, seed, tid, label, output):
    data, motion = records[("full", seed, tid)], records[("motion_only", seed, tid)]
    time = data["real_time"] - data["real_time"][0]
    future = aligned(data, data["future"][:, 4], 5)
    motion_future = aligned(motion, motion["future"][:, 4], 5)
    hold = aligned(data, data["current"], 5)
    anchor = forecast_anchor(data)
    event_time = time[anchor + 3]
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), constrained_layout=True, sharex="col")
    for i, axis in enumerate("xyz"):
        for j in range(2):
            ax = axes[i, j]
            ax.plot(time, data["real_force"][:, i], color=COLORS["target"], lw=1.7, label="Real target")
            ax.plot(time, future[:, i], color=COLORS["full"], lw=1.3, label="Full: +5 forecast")
            ax.plot(time, motion_future[:, i], color=COLORS["motion_only"], lw=1.2, label="Motion: +5 forecast")
            ax.plot(time, hold[:, i], color=COLORS["hold"], lw=1, ls="--", alpha=.85, label="Recovered hold: +5")
            if j == 1:
                ax.set_xlim(max(0, event_time - 2), min(time[-1], event_time + 2))
                visible = (time >= max(0, event_time - 2)) & (time <= event_time + 2)
                y = np.concatenate([d[visible, i] for d in [data["real_force"], future, motion_future, hold]])
                y = y[np.isfinite(y)]
                margin = max(.25, (y.max() - y.min()) * .1)
                ax.set_ylim(y.min() - margin, y.max() + margin)
                ax.axvline(event_time, color="#808080", lw=.8, ls=":")
            ax.set_ylabel(f"F{axis} (N)"); ax.grid(alpha=.2)
        axes[i, 0].set_xlim(0, time[-1])
    axes[0, 0].set_title("Full trajectory"); axes[0, 1].set_title("Zoom around a large target-force change")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("Target timestamp relative to start (s)")
    axes[-1, 1].set_xlabel("Target timestamp relative to start (s)")
    score = rmse(data["future"][valid_indices(data, 5), 4] - data["real_force"][valid_indices(data, 5) + 5])
    fig.suptitle(f"{label}: traj_{tid}, seed {seed}, out-of-fold | +5 RMSE {score:.3f} N", fontsize=15)
    save_figure(fig, output / f"02_{label.lower()}_traj_{tid}")


def fan_plot(records, seed, representatives, output):
    fig, axes = plt.subplots(len(representatives), 3, figsize=(15, 3.2 * len(representatives)), constrained_layout=True)
    for row, (label, tid) in enumerate(representatives):
        data, motion = records[("full", seed, tid)], records[("motion_only", seed, tid)]
        anchor = forecast_anchor(data)
        time = data["real_time"] - data["real_time"][anchor]
        lo = max(0, anchor - 8)
        ids = np.arange(anchor, anchor + 6)
        for axis in range(3):
            ax = axes[row, axis]
            ax.axvspan(0, time[anchor + 5], color="#eaf0f5")
            ax.plot(time[lo:anchor + 6], data["real_force"][lo:anchor + 6, axis], "o-", ms=3,
                    color=COLORS["target"], label="Real target")
            ax.plot(time[ids], np.r_[data["current"][anchor, axis], data["future"][anchor, :, axis]],
                    "o-", ms=4, color=COLORS["full"], label="Full: one origin, 5 horizons")
            ax.plot(time[ids], np.r_[motion["current"][anchor, axis], motion["future"][anchor, :, axis]],
                    "s-", ms=3, color=COLORS["motion_only"], label="Motion: same origin")
            ax.plot(time[ids], np.full(6, data["current"][anchor, axis]), "--", color=COLORS["hold"], label="Recovered hold")
            ax.axvline(0, color="#808080", ls=":", lw=1)
            ax.set(title=f"{label}: traj_{tid}, F{'xyz'[axis]}", xlabel="Time relative to forecast origin (s)", ylabel="Force (N)")
            ax.grid(alpha=.2)
            if row == 0 and axis == 0: ax.legend(fontsize=8)
    fig.suptitle(f"Forecasts issued at one fixed origin; seed {seed}; OOF trajectories", fontsize=15)
    save_figure(fig, output / "03_fixed_origin_forecasts")


def write_report(root, output, summary, representatives, outer, manifest):
    gate = read_json(root / "acceptance.json")
    metrics = ["score", "current_rmse_n", "future_1_rmse_n", "future_3_rmse_n", "future_5_rmse_n"]
    lines = ["# 力预测结果与可视化", "", "统计口径：32 条轨迹的折外评估，每条轨迹等权、3 个随机种子等权。RMSE 为三轴分量 RMSE，不是三轴向量范数。", "",
             "| 模型 | 综合误差 score ↓ | 当前力 RMSE (N) | +1 帧 RMSE (N) | +3 帧 RMSE (N) | +5 帧 RMSE (N) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in [v for v in ["full", "motion_control", "no_alignment", "no_frequency", "motion_only"] if v in outer["summaries"]]:
        values = [mean([m[k] for m in outer["summaries"][name].values()]) for k in metrics]
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    values = [gate["baseline_metrics"][k] for k in metrics]
    lines.append("| " + gate["best_baseline"] + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "1/3/5 帧的实际平均预测时间约为 0.10/0.30/0.50 秒；画图使用保存的真实时间戳，不对预测做拟合平移。",
              "", f"完整模型相对 {gate['best_baseline']} 的综合误差降低 {gate['relative_improvement']:.2%}。验收通过：{gate['passed']}；原因：{'；'.join(gate['reasons']) or '全部门槛通过'}。",
              "", "## 单独检验未来预测的价值", "", "保持当前恢复力不变：对每个预测起点 t，用 current[t] 作为各未来时刻的预测。它与完整预测头使用相同 encoder，不依赖未来输入。", "",
              "| 预测步长 | 完整预测头 RMSE (N) | 保持当前恢复力 RMSE (N) | 相对降低 | 配对轨迹 bootstrap 95% 改善区间 (N) |",
              "|---|---:|---:|---:|---:|"]
    for step in (1, 3, 5):
        interval = summary["paired_trajectory_bootstrap_95_rmse_improvement_n"][str(step)]
        lines.append(f"| +{step} 帧 | {summary['forecast_rmse_n']['full'][step-1]:.4f} | "
                     f"{summary['full_recovered_hold_rmse_n'][step-1]:.4f} | "
                     f"{summary['relative_improvement_vs_hold'][step-1]:.2%} | [{interval[0]:.4f}, {interval[1]:.4f}] |")
    lines += ["", "预测头是否优于保持当前估计，要看各步长的误差与改善区间；这个对照不能独立证明力输入优于运动输入。bootstrap 使用 32 条轨迹，先平均种子再对轨迹配对重采样 2000 次。",
              "", "## 最终模型的诊断轨迹", "", f"以下为最终选中的 seed {manifest['selected_seed']} 在 8 条已复用诊断轨迹上的结果，不能与 32 条折外轨迹直接比较。",
              "", "| 输入分支 | 当前力 RMSE (N) | +1 帧 RMSE (N) | +3 帧 RMSE (N) | +5 帧 RMSE (N) |",
              "|---|---:|---:|---:|---:|"]
    diagnostic = read_json(root / f"diagnostic_seed{manifest['selected_seed']}.json")
    for domain in ("sim", "real"):
        values = [mean([m["all"][key] for m in diagnostic[domain].values()]) for key in metrics[1:]]
        lines.append("| " + domain + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "Real 当前力直接取实测输入，所以 RMSE=0 不代表恢复质量。它的未来误差可以用于查看真实传感器输入下的预测能力。",
              "", "## 看图的方法", "", "- 打开 `force_prediction_explorer.html`：浏览器直接打开，不需要网络或服务。可选择数据集、模型、种子、轨迹、预测步长；点击曲线或拖动起点，查看单次预测的 5 个未来点。",
              "- 看 `01_metrics_overview.png`：整体误差随预测步长、阶段、力轴和轨迹变化。阴影是 3 个种子的范围，不是置信区间。",
              "- 看 `02_*`：真实力、完整预测、motion_only 和保持当前恢复力，均按未来目标时刻对齐；右侧放大目标力变化较大的窗口。",
              "- 看 `03_fixed_origin_forecasts.png`：每条折线的 5 个未来点均来自同一个预测起点，观察接触峰值是否提前预测到，峰值幅度是否被压低。",
              "", "示例轨迹按三个种子的 +5 帧 RMSE 排序选取，避免只挑最好样本：" + "，".join(f"{label}=traj_{tid}" for label, tid in representatives) + f"。图中的具体曲线使用最终选中的 seed {manifest['selected_seed']} 的折外模型。",
              "", "## 解释边界", "", "- 每个折外轨迹的曲线来自未用该轨迹训练的模型；`diagnostic_predictions` 则来自最终模型，8 条诊断轨迹此前已被复用，不是新的盲测集。",
              "- Real 输入分支的 `current` 直接使用传感器真实力，因此该分支当前力 RMSE=0 是实现设计，不能当作 encoder 恢复成功。真正的预测效果需要看 `future`。",
              "- `actual_sim_state_verified=false`：当前 sim 数据的 q/dq 来源复制或未验证。结果只支持当前配对回放数据，尚不能证明实际仿真状态或闭环控制上的泛化。",
              "- 阶段指标按预测起点的标签分组；+5 帧的目标可能进入另一个阶段。末尾没有足够未来样本时排除，并在任何一侧时间间隔超过 0.5 秒时禁止跨段。",
              "", f"最终选中权重：`{manifest['checkpoint']}`；状态：`{manifest['status']}`。"]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def write_explorer(root, output, records, recipe, manifest, ordered, real_records=None):
    reference, models = {}, {}
    annotations = {}
    labels = Path(recipe["labels"])
    if labels.is_file() and hashlib.sha256(labels.read_bytes()).hexdigest() == recipe["label_sha256"]:
        with labels.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                tid = int(row["trajectory"].removeprefix("traj_"))
                annotations.setdefault(tid, []).append([int(row["start_frame"]), int(row["end_frame"]), row["label"]])

    def rounded(array):
        return np.round(array.astype(np.float64), 4).tolist()

    def add(dataset, variant, seed, tid, data):
        ref_key = f"{dataset}|{tid}"
        if ref_key not in reference:
            reference[ref_key] = {
                "t": rounded(data["real_time"] - data["real_time"][0]),
                "y": rounded(data["real_force"]), "input": rounded(data["input_force"]),
                "end": np.minimum(ends(data["real_time"]), ends(data["time"])).tolist(),
                "phases": annotations.get(tid, []),
            }
        values = [rmse(data["current"] - data["real_force"])]
        for step in range(1, 6):
            indices = valid_indices(data, step)
            values.append(rmse(data["future"][indices, step-1] - data["real_force"][indices+step]))
        models[f"{dataset}|{variant}|{seed}|{tid}"] = {
            "c": rounded(data["current"]), "f": rounded(data["future"]), "rmse": values,
            "anchor": forecast_anchor(data),
        }

    for (variant, seed, tid), data in records.items():
        add("oof", variant, seed, tid, data)
    for (variant, seed, tid), data in (real_records or {}).items():
        add("real_oof", variant, seed, tid, data)
    for path in sorted(root.glob("diagnostic_predictions/seed*_*/*.npz")):
        match = re.fullmatch(r"seed(\d+)_(real|sim)", path.parent.name)
        if match is None: continue
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        add(match[2], "full", int(match[1]), int(path.stem.removeprefix("traj_")), data)
    payload = {"reference": reference, "models": models, "ordered": ordered,
               "seeds": recipe["seeds"], "selected": manifest["selected_seed"], "variants": recipe["variants"]}
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>力预测交互图</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#202d3c;font-family:system-ui,-apple-system,sans-serif}
main{max-width:1280px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 8px}p{line-height:1.7;margin:8px 0}
.card{background:white;border:1px solid #dce4eb;border-radius:12px;padding:18px;margin:16px 0}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}label{display:flex;flex-direction:column;gap:6px;font-size:13px}
select,button{padding:8px 10px;border-radius:6px;border:1px solid #bdcad7;background:white;color:#263547;font-size:14px}
button{cursor:pointer}.check{flex-direction:row;align-items:center;margin:10px 14px 0 0;display:inline-flex}
input[type=range]{width:min(700px,100%)}canvas{display:block;width:100%;background:white}
.muted{color:#5b6b7c;font-size:13px}.notice{background:#fff6e5;border-left:4px solid #cc9342;padding:10px 14px;font-size:14px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;margin:12px 0}.swatch{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:5px}
#stats{font-weight:600;font-variant-numeric:tabular-nums;margin:12px 0}a{color:#2f6aab}.help{font-size:14px}h2{font-size:19px;margin:4px 0 8px}
</style></head><body><main>
<h1>看清力预测的效果</h1>
<p>同时比较真实力、未来预测、保持当前恢复力和仅运动输入的模型。主图把预测放在它预测的<strong>目标时刻</strong>；下图显示<strong>同一起点发出的全部未来点</strong>。</p>
<div class="notice">__NOTICE__</div>
<div class="card"><div class="controls">
<label>数据集<select id="dataset"><option value="oof">折外评估 · 32 条轨迹</option><option value="sim">最终模型诊断 · Sim 输入 · 8 条</option><option value="real">最终模型诊断 · Real 输入 · 8 条</option></select></label>
<label>模型<select id="variant"><option value="full">完整模型 full</option><option value="motion_only">仅运动 motion_only</option><option value="no_alignment">去掉对齐 no_alignment</option><option value="no_frequency">去掉频率约束 no_frequency</option></select></label>
<label>随机种子<select id="seed"></select></label>
<label>轨迹<select id="trajectory"></select></label>
<label>预测步长<select id="horizon"><option value="1">+1 帧 · 约 0.1 秒</option><option value="2">+2 帧 · 约 0.2 秒</option><option value="3">+3 帧 · 约 0.3 秒</option><option value="4">+4 帧 · 约 0.4 秒</option><option value="5" selected>+5 帧 · 约 0.5 秒</option></select></label>
<button id="export">导出主图 PNG</button></div>
<label class="check"><input id="motion" type="checkbox" checked>比较 motion_only</label>
<label class="check" id="control-label"><input id="control" type="checkbox" checked>比较同容量运动对照</label>
<label class="check"><input id="hold" type="checkbox" checked>比较保持当前恢复力</label>
<label class="check"><input id="current" type="checkbox">显示当前恢复力</label>
<label class="check"><input id="raw" type="checkbox">显示原始输入力</label>
<label class="check"><input id="zoom" type="checkbox">局部窗口：起点前后 3 秒</label>
<div id="stats"></div><p id="dataset-note" class="muted"></p></div>
<div class="card"><h2>连续轨迹 · 预测按目标时间对齐</h2>
<div class="legend"><span><i class="swatch" style="background:#202832"></i>真实目标力</span><span><i class="swatch" style="background:#e87924"></i>所选模型：未来预测</span><span><i class="swatch" style="background:#8c60bf"></i>motion_only：相同步长</span><span id="control-legend"><i class="swatch" style="background:#3a79b7"></i>同容量对照（蓝色长虚线）</span><span><i class="swatch" style="background:#209e91"></i>保持所选模型当前恢复力</span><span><i class="swatch" style="background:#3a79b7"></i>当前恢复力（可选）</span></div>
<canvas id="mainplot" aria-label="三轴真实力和预测力曲线"></canvas>
<p class="help">点击曲线选择预测起点；启用局部窗口可查看峰值。预测曲线上的每个点由更早的起点产生，不使用目标时刻的未来输入。</p>
<label for="anchor">单次预测起点 <span id="anchor-label"></span></label><input type="range" id="anchor" min="0" step="1"><button id="event">跳到明显力变化前的起点</button></div>
<div class="card"><h2>固定一个起点 · 未来 5 帧的预测</h2><p class="muted">阴影内是真正的未来。这里的 5 个预测值都来自同一个起点，不是在每个未来时刻重新预测。横轴使用真实时间戳。</p><canvas id="fanplot" aria-label="固定预测起点的五步力预测"></canvas></div>
<p class="muted">各条轨迹的 RMSE 按有效样本计算，禁止跨越任一输入或目标数据的时间断段。页面内数据保留 4 位小数，顶部指标使用未舍入的保存数组。可离线打开，无网络依赖。</p>
</main><script id="payload" type="application/json">__PAYLOAD__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('payload').textContent);
const $=id=>document.getElementById(id), C={truth:'#202832',forecast:'#e87924',motion:'#8c60bf',control:'#3a79b7',hold:'#209e91',current:'#3a79b7',raw:'#a7a9ac'};
const phaseC={free:'#e7f3ed',misaligned_contact:'#ffe5c9',aligned_progress:'#fff0d9',uncertain:'#edf0f5',terminal:'#f8e4e7'};
let ref,model,motion,control,anchor=0,mainGeometry;
function setOptions(element,items,selected){element.innerHTML='';for(const pair of items){const o=document.createElement('option');o.value=String(pair[0]);o.textContent=pair[1];element.appendChild(o)}if(items.some(x=>String(x[0])===String(selected)))element.value=String(selected)}
setOptions($('seed'),D.seeds.map(s=>[s,String(s)+(s===D.selected?'（最终选中）':'')]),D.selected);
const variantNames={full:'完整模型 full',motion_control:'同容量运动对照 motion_control',motion_only:'仅运动 motion_only',no_alignment:'去掉对齐 no_alignment',no_frequency:'去掉频率约束 no_frequency'};
setOptions($('variant'),D.variants.map(v=>[v,variantNames[v]||v]),'full');
if(Object.keys(D.reference).some(k=>k.startsWith('real_oof|'))){const o=document.createElement('option');o.value='real_oof';o.textContent='折外评估 · Real 传感器输入 · 32 条轨迹';$('dataset').insertBefore(o,$('dataset').children[1])}
$('control-label').style.display=D.variants.includes('motion_control')?'inline-flex':'none';
$('control-legend').style.display=D.variants.includes('motion_control')?'inline':'none';
function selectData(reset){const dataset=$('dataset').value,previous=$('trajectory').value;
  const oof=dataset==='oof'||dataset==='real_oof';
  if(!oof){$('variant').value='full';$('variant').disabled=true;$('motion').disabled=true;$('control').disabled=true}else{$('variant').disabled=false;$('motion').disabled=false;$('control').disabled=false}
  const tids=Object.keys(D.reference).filter(k=>k.startsWith(dataset+'|')).map(k=>Number(k.split('|')[1])).sort((a,b)=>a-b);
  const typical=D.ordered[Math.floor((D.ordered.length-1)/2)];
  setOptions($('trajectory'),tids.map(t=>[t,'traj_'+t+(t===typical&&oof?'（按 Sim 误差选出的典型）':'')]),reset?previous:(oof?typical:tids[0]));
  updateModel(true);
}
function updateModel(reset){const dataset=$('dataset').value,variant=$('variant').value,seed=$('seed').value,tid=$('trajectory').value;
  ref=D.reference[dataset+'|'+tid];model=D.models[[dataset,variant,seed,tid].join('|')];motion=D.models[[dataset,'motion_only',seed,tid].join('|')]||null;control=D.models[[dataset,'motion_control',seed,tid].join('|')]||null;
  if(!model)throw new Error('Missing saved prediction');if(reset)anchor=model.anchor;anchor=Math.max(0,Math.min(ref.t.length-1,anchor));$('anchor').max=ref.t.length-1;$('anchor').value=anchor;
  $('dataset-note').textContent=dataset==='oof'?'Sim 折外模型：该轨迹没有用于训练。浅色背景是预测起点的阶段标签。':dataset==='real_oof'?'Real 折外模型：该轨迹没有用于训练。当前力直接取传感器，请比较未来预测与保持实测力。':dataset==='real'?'Real 输入分支：当前力直接取实测值，所以当前力 RMSE=0；请看未来预测与保持实测力的比较。诊断集不是新盲测。':'Sim 输入分支：用最终 encoder 恢复和预测真实等效力。诊断集不是新盲测。';
  drawAll();
}
function prepare(id,height){const canvas=$(id),width=canvas.clientWidth||1100,ratio=window.devicePixelRatio||1;canvas.style.height=height+'px';canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);ctx.font='12px system-ui';return{ctx,width,height}}
function format(v){return Number(v).toFixed(3)}
function mapped(step,get){const y=Array.from({length:ref.t.length},()=>null);for(let i=0;i<ref.t.length;i++)if(i+step<ref.end[i])y[i+step]=get(i);return y}
function line(ctx,t,y,axis,xmap,ymap,color,dash=[],width=1.4){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();let open=false,previous=-1;for(let i=0;i<t.length;i++){const v=y[i]&&y[i][axis];if(v===null||v===undefined||!Number.isFinite(v)){open=false;continue}const x=xmap(t[i]),z=ymap(v);if(!open||(previous>=0&&t[i]-t[previous]>.5))ctx.moveTo(x,z);else ctx.lineTo(x,z);open=true;previous=i}ctx.stroke();ctx.setLineDash([])}
function panel(ctx,box,tmin,tmax,values,axis,label){let lo=Infinity,hi=-Infinity;for(const v of values)if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v)}if(!Number.isFinite(lo)){lo=-1;hi=1}const margin=Math.max(.25,(hi-lo)*.1);lo-=margin;hi+=margin;
 const xm=t=>box.x+(t-tmin)/(tmax-tmin||1)*box.w,ym=v=>box.y+box.h-(v-lo)/(hi-lo)*box.h;ctx.strokeStyle='#d6dfe7';ctx.lineWidth=.6;ctx.fillStyle='#596a7b';ctx.font='11px system-ui';
 for(let j=0;j<=4;j++){const v=lo+(hi-lo)*j/4,z=ym(v);ctx.beginPath();ctx.moveTo(box.x,z);ctx.lineTo(box.x+box.w,z);ctx.stroke();ctx.textAlign='right';ctx.fillText(v.toFixed(1),box.x-9,z+4)}
 for(let j=0;j<=5;j++){const t=tmin+(tmax-tmin)*j/5,x=xm(t);ctx.beginPath();ctx.moveTo(x,box.y);ctx.lineTo(x,box.y+box.h);ctx.stroke();ctx.textAlign='center';ctx.fillText(t.toFixed(2),x,box.y+box.h+17)}
 ctx.fillStyle='#202d3c';ctx.font='bold 13px system-ui';ctx.textAlign='left';ctx.fillText(label,box.x,box.y-9);return{xm,ym};
}
function drawMain(){const step=Number($('horizon').value),{ctx,width}=prepare('mainplot',660);const tmin=$('zoom').checked?Math.max(0,ref.t[anchor]-3):0,tmax=$('zoom').checked?Math.min(ref.t.at(-1),ref.t[anchor]+3):ref.t.at(-1);
 const prediction=mapped(step,i=>model.f[i][step-1]),hold=mapped(step,i=>model.c[i]),m=motion?mapped(step,i=>motion.f[i][step-1]):null;
 const curves=[[ref.y,C.truth,[],1.9],[prediction,C.forecast,[],1.5]];
 if($('hold').checked)curves.push([hold,C.hold,[5,3],1.2]);if($('motion').checked&&m)curves.push([m,C.motion,[],1.3]);if($('current').checked)curves.push([model.c,C.current,[2,2],1.1]);if($('raw').checked)curves.push([ref.input,C.raw,[3,4],1]);
 if($('control').checked&&control)curves.push([mapped(step,i=>control.f[i][step-1]),C.control,[7,2],1.3]);
 for(let axis=0;axis<3;axis++){const box={x:62,y:28+axis*210,w:width-86,h:155};const values=[];for(let i=0;i<ref.t.length;i++)if(ref.t[i]>=tmin&&ref.t[i]<=tmax)for(const curve of curves)if(curve[0][i])values.push(curve[0][i][axis]);
   const scales=panel(ctx,box,tmin,tmax,values,axis,'F'+'xyz'[axis]+' (N)');ctx.save();ctx.beginPath();ctx.rect(box.x,box.y,box.w,box.h);ctx.clip();
   for(const p of ref.phases){ctx.fillStyle=phaseC[p[2]]||'#f0f3f7';const a=ref.t[p[0]],b=ref.t[Math.min(p[1]+1,ref.t.length-1)];ctx.globalAlpha=.55;ctx.fillRect(scales.xm(a),box.y,scales.xm(b)-scales.xm(a),box.h)}ctx.globalAlpha=1;
   for(const curve of curves)line(ctx,ref.t,curve[0],axis,scales.xm,scales.ym,curve[1],curve[2],curve[3]);ctx.strokeStyle='#687789';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(scales.xm(ref.t[anchor]),box.y);ctx.lineTo(scales.xm(ref.t[anchor]),box.y+box.h);ctx.stroke();ctx.setLineDash([]);ctx.restore();
   if(axis===0)mainGeometry={box,tmin,tmax};ctx.fillStyle='#596a7b';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText('目标时间 / s',box.x+box.w/2,box.y+box.h+37);
 }
 let valid=0,total=0;for(let i=0;i<ref.t.length;i++)if(i+step<ref.end[i]){valid++;total+=ref.t[i+step]-ref.t[i]}
 $('stats').textContent='当前力 RMSE '+format(model.rmse[0])+' N  ·  +'+step+' 帧 RMSE '+format(model.rmse[step])+' N  ·  实际平均预测时间 '+(total/valid).toFixed(3)+' s  ·  有效起点 '+valid;
}
function drawFan(){const {ctx,width}=prepare('fanplot',300),end=ref.end[anchor],stop=Math.min(anchor+6,end),lo=Math.max(0,anchor-6);let start=lo;while(start<anchor&&ref.end[start]!==end)start++;
 const ts=ref.t.slice(start,stop).map(t=>t-ref.t[anchor]),futureT=ref.t.slice(anchor,stop).map(t=>t-ref.t[anchor]);const truth=ref.y.slice(start,stop),pred=[model.c[anchor],...model.f[anchor]].slice(0,stop-anchor),mh=motion?[motion.c[anchor],...motion.f[anchor]].slice(0,stop-anchor):null,h=Array.from({length:stop-anchor},()=>model.c[anchor]);
 const ch=control?[control.c[anchor],...control.f[anchor]].slice(0,stop-anchor):null;
 for(let axis=0;axis<3;axis++){const box={x:55+axis*width/3,y:32,w:width/3-78,h:205},values=[...truth,...pred,...h,...(mh&&$('motion').checked?mh:[]),...(ch&&$('control').checked?ch:[])].map(v=>v[axis]);const tmin=ts[0]||0,tmax=Math.max(.1,ts.at(-1));const scales=panel(ctx,box,tmin,tmax,values,axis,'F'+'xyz'[axis]+' (N)');
  ctx.save();ctx.beginPath();ctx.rect(box.x,box.y,box.w,box.h);ctx.clip();ctx.fillStyle='#edf2f7';ctx.fillRect(scales.xm(0),box.y,scales.xm(tmax)-scales.xm(0),box.h);line(ctx,ts,truth,axis,scales.xm,scales.ym,C.truth,[],1.8);line(ctx,futureT,pred,axis,scales.xm,scales.ym,C.forecast,[],1.8);if($('hold').checked)line(ctx,futureT,h,axis,scales.xm,scales.ym,C.hold,[4,3],1.3);if(mh&&$('motion').checked)line(ctx,futureT,mh,axis,scales.xm,scales.ym,C.motion,[],1.4);
  if(ch&&$('control').checked)line(ctx,futureT,ch,axis,scales.xm,scales.ym,C.control,[7,2],1.4);
  for(let i=0;i<pred.length;i++){ctx.fillStyle=C.forecast;ctx.beginPath();ctx.arc(scales.xm(futureT[i]),scales.ym(pred[i][axis]),3.5,0,Math.PI*2);ctx.fill()}ctx.strokeStyle='#687789';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(scales.xm(0),box.y);ctx.lineTo(scales.xm(0),box.y+box.h);ctx.stroke();ctx.restore();ctx.setLineDash([]);ctx.fillStyle='#596a7b';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText('相对预测起点 / s',box.x+box.w/2,box.y+box.h+37);
 }
 $('anchor-label').textContent='第 '+anchor+' 帧 · '+ref.t[anchor].toFixed(3)+' s'+(stop-anchor<6?' · 剩余有效未来帧不足 5':'' );
}
function drawAll(){drawMain();drawFan()}
$('dataset').addEventListener('change',()=>selectData(true));for(const id of ['variant','seed','trajectory'])$(id).addEventListener('change',()=>updateModel(true));for(const id of ['horizon','motion','control','hold','current','raw','zoom'])$(id).addEventListener('change',drawAll);
$('anchor').addEventListener('input',()=>{anchor=Number($('anchor').value);drawAll()});$('event').addEventListener('click',()=>{anchor=model.anchor;$('anchor').value=anchor;drawAll()});
$('mainplot').addEventListener('click',event=>{const b=$('mainplot').getBoundingClientRect(),x=event.clientX-b.left,g=mainGeometry;if(x<g.box.x||x>g.box.x+g.box.w)return;const t=g.tmin+(x-g.box.x)/g.box.w*(g.tmax-g.tmin);let best=0;for(let i=1;i<ref.t.length;i++)if(Math.abs(ref.t[i]-t)<Math.abs(ref.t[best]-t))best=i;anchor=best;$('anchor').value=anchor;drawAll()});
$('export').addEventListener('click',()=>{const a=document.createElement('a');a.download='force_'+$('dataset').value+'_traj'+$('trajectory').value+'_seed'+$('seed').value+'_h'+$('horizon').value+'.png';a.href=$('mainplot').toDataURL('image/png');a.click()});
window.addEventListener('resize',drawAll);selectData(false);
</script></body></html>'''
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    gate = read_json(root / "acceptance.json")
    notice = f"完整模型比 {gate['best_baseline']} 综合误差低 {gate['relative_improvement']:.1%}；验收状态为 {manifest['status']}。"
    if "relative_future_improvement" in gate:
        notice = (f"完整模型未来误差相对 motion_only 改善 {gate['relative_future_improvement']['motion_only']:.1%}，"
                  f"相对同容量对照改善 {gate['relative_future_improvement']['motion_control']:.1%}；验收状态为 {manifest['status']}。")
    if gate["reasons"]:
        notice += " 验收原因：" + "；".join(gate["reasons"]) + "。"
    if not manifest.get("actual_sim_state_verified"):
        notice += " Sim q/dq 的真实仿真来源尚未验证。"
    notice += " 诊断 8 条轨迹此前已复用，不能作为新盲测。"
    import html
    (output / "force_prediction_explorer.html").write_text(template.replace("__PAYLOAD__", encoded).replace("__NOTICE__", html.escape(notice)))


def generate_visualizations(run_dir, output_dir=None):
    root = Path(run_dir).resolve()
    output = Path(output_dir) if output_dir else root / "analysis_visualizations"
    output.mkdir(parents=True, exist_ok=True)
    recipe, outer, manifest = [read_json(root / name) for name in ["recipe.json", "outer_metrics.json", "manifest.json"]]
    records = load_records(root, outer)
    print(f"Verified saved horizon metrics for {len(records)} out-of-fold records.", flush=True)
    summary, ordered = make_summary(root, output, outer, records, recipe)
    representatives = [("Best", ordered[0]), ("Typical", ordered[(len(ordered) - 1) // 2]), ("Worst", ordered[-1])]
    seed = manifest["selected_seed"]
    for label, tid in representatives:
        trajectory_plot(records, seed, tid, label, output)
    fan_plot(records, seed, representatives, output)
    write_report(root, output, summary, representatives, outer, manifest)
    # The offline explorer is generated from saved predictions as a standalone scientific figure.
    real_records = None
    if (root / "real_outer_metrics.json").is_file():
        real_records = load_records(root, read_json(root / "real_outer_metrics.json"), "real_predictions")
    write_explorer(root, output, records, recipe, manifest, ordered, real_records)
    print(f"Generated plots, PDF exports, CSV metrics and offline explorer: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    generate_visualizations(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
