"""CPU-only diagnostics for a completed single run; never updates model weights."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from force_gap_forecast_v1.common import sha256
from . import baselines
from .data import HOLDOUT, load_pairs
from .metrics import evaluate, flatten_evaluation, predict_episode
from .train import load_model, read_json


DOMAINS = ("real", "sim")
SPLITS = ("validation", "holdout")
SPLIT_NAMES = {"validation": "早停验证集", "holdout": "独立留出集"}
DOMAIN_NAMES = {"real": "真机力输入", "sim": "仿真力输入"}
MODEL_NAMES = {"full_force": "仅力编码", "full_joint": "力编码＋运动", "motion_control": "仅运动对照",
               "hold": "保持当前力 / 仿真线性校准", "ar": "线性力历史 AR"}
COLORS = {"full_force": "#2474b5", "full_joint": "#e89131", "motion_control": "#6a9b73",
          "hold": "#9298a2", "ar": "#9867a7", "real": "#2474b5", "sim": "#e89131"}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {k: saved[k] for k in saved.files}
    for key, value in result.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Non-finite saved array: {path}: {key}")
    return result


def average(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def summary_rows(metrics, split, model, domain):
    return [row["all"] for row in metrics[split]["models"][model][domain].values()]


def score(metrics, split, model, domain, key="forecast_score_n"):
    return average(summary_rows(metrics, split, model, domain), key)


def save_figure(fig, output, name, title, note=""):
    fig.suptitle(title, fontsize=16, fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, .045, 1, .965), h_pad=2, w_pad=2)
    if note:
        fig.text(.025, .012, note, fontsize=9, color="#555555")
    fig.savefig(output / name, dpi=145, facecolor="white")
    plt.close(fig)


def style_axes(axes):
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)


def export_overview_pdf(output):
    from PIL import Image
    pages = []
    try:
        for name in ("01_forecast.png", "02_frequency.png", "03_alignment.png", "04_training.png", "05_latent.png", "06_event_zoom.png"):
            with Image.open(output / name) as page:
                pages.append(page.convert("RGB"))
        temporary = output / "encoder_overview.pdf.tmp"
        pages[0].save(temporary, format="PDF", save_all=True, append_images=pages[1:], resolution=145., quality=95)
        temporary.replace(output / "encoder_overview.pdf")
    finally:
        for page in pages:
            page.close()


def pool_pair(real, sim, size):
    ii = np.array([t for t in range(size - 1, len(real["z"]), size)
                   if t - size + 1 >= max(real["starts"][t], sim["starts"][t])], dtype=int)
    if not len(ii):
        return ii, np.empty((0, 16)), np.empty((0, 16))
    r, s = [np.stack([a["z"][t - size + 1:t + 1, :16].mean(0) for t in ii]) for a in (real, sim)]
    return ii, r, s


def alignment_metrics(pairs, mean, scale):
    rng = np.random.default_rng(20261014)
    by_size, pooled = {}, {}
    for size in (4, 8, 16):
        distances, wrong, changes, all_r, all_s = [], [], [], [], []
        for tid, real in pairs["real"].items():
            sim = pairs["sim"][tid]
            ii, r, s = pool_pair(real, sim, size)
            if not len(ii):
                continue
            r, s = (r - mean[:16]) / scale[:16], (s - mean[:16]) / scale[:16]
            distances.append(float(np.mean((r - s) ** 2)))
            if len(r) > 1:
                # A nonzero circular shift is an explicitly mismatched reference, not an ablation.
                shift = int(rng.integers(1, len(r)))
                wrong.append(float(np.mean((np.roll(r, shift, axis=0) - s) ** 2)))
                valid = ((np.diff(ii) == size) &
                         (real["starts"][ii[1:]] == real["starts"][ii[:-1]]) &
                         (sim["starts"][ii[1:]] == sim["starts"][ii[:-1]]))
                if valid.any():
                    changes.append(float(np.mean((np.diff(r, axis=0)[valid] - np.diff(s, axis=0)[valid]) ** 2)))
            all_r.extend(r)
            all_s.extend(s)
        by_size[str(size)] = dict(paired_mse=float(np.mean(distances)),
                                 mismatched_reference_mse=float(np.mean(wrong)),
                                 change_mse=float(np.mean(changes)))
        pooled[size] = (np.array(all_r), np.array(all_s))
    r, s = pooled[16]
    matrix = ((s[:, None] - r[None]) ** 2).mean(-1)
    order = matrix.argsort(1)
    return dict(by_size=by_size, retrieval_top1=float(np.mean(order[:, 0] == np.arange(len(s)))),
                retrieval_top5=float(np.mean((order[:, :5] == np.arange(len(s))[:, None]).any(1))),
                retrieval_candidates=len(s)), pooled


def spectrum(a, scale):
    ii = np.flatnonzero(np.arange(len(a["z"])) - 15 >= a["starts"])
    index = ii[:, None] + np.arange(-15, 1)
    taper = np.hanning(16)[None, :, None]
    truth = a["input_force"][index]
    decoded = a["history"][ii]
    stream = a["history"][:, -1][index]
    powers = [np.mean(np.abs(np.fft.rfft(x / scale * taper, axis=1, norm="ortho")) ** 2, axis=(0, 2))
              for x in (truth, decoded, stream)]
    keep = np.fft.rfftfreq(16) >= .25
    ratios = [float(p[keep].mean() / max(powers[0][keep].mean(), 1e-12)) for p in powers[1:]]
    return np.array(powers), ratios


def forecast_figure(metrics, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    models = list(MODEL_NAMES)
    for col, domain in enumerate(DOMAINS):
        ax = axes[0, col]
        for j, split in enumerate(SPLITS):
            y = [score(metrics, split, model, domain) for model in models]
            bars = ax.bar(np.arange(len(models)) + (j - .5) * .35, y, width=.35,
                          color="#2474b5" if j == 0 else "#e89131", label=SPLIT_NAMES[split])
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_xticks(np.arange(len(models)), ["仅力编码", "编码＋运动", "仅运动", "当前力保持", "线性 AR"])
        ax.set_title(DOMAIN_NAMES[domain] + "：未来预测综合误差")
        ax.set_ylabel("轨迹等权误差（N），越低越好")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
        ax.legend(fontsize=8)
        ax = axes[1, col]
        for model in models:
            y = [score(metrics, "holdout", model, domain, f"future_{h}_rmse_n") for h in range(1, 6)]
            ax.plot(range(1, 6), y, marker="o", color=COLORS[model], label=MODEL_NAMES[model], lw=1.8)
        ax.set_title(DOMAIN_NAMES[domain] + "：独立留出集不同预测步长")
        ax.set_xlabel("预测未来帧数（每帧约 0.1 s）")
        ax.set_xticks(range(1, 6))
        ax.set_ylabel("轨迹等权 RMSE（N）")
        ax.legend(fontsize=8)
    style_axes(axes)
    save_figure(fig, output, "01_forecast.png", "未来真实力预测：编码是否有用？",
                "综合误差 = 0.25×RMSE(+1) + 0.35×RMSE(+3) + 0.40×RMSE(+5)。AR 的 α 固定为 1；仿真保持基线仅用训练集拟合线性校准。")


def frequency_figure(metrics, data, stats, output):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    spectra = {}
    for row, domain in enumerate(DOMAINS):
        scale = np.asarray(stats[domain]["scale"][:3])
        spectra[domain] = {}
        for split in SPLITS:
            values = [spectrum(a, scale) for a in data[split][domain].values()]
            spectra[domain][split] = dict(power=np.mean([v[0] for v in values], axis=0).tolist(),
                                        energy_ratios=np.mean([v[1] for v in values], axis=0).tolist())
        ax = axes[row, 0]
        p = np.asarray(spectra[domain]["holdout"]["power"])
        for power, label, color in zip(p, ("输入力", "16帧历史重建", "流式当前重建"), ("#222222", "#2474b5", "#e89131")):
            ax.semilogy(np.fft.rfftfreq(16)[1:], np.maximum(power[1:], 1e-12), "o-", label=label, color=color, ms=3)
        ax.axvspan(.25, .5, color="#e89131", alpha=.09)
        ax.set_title(DOMAIN_NAMES[domain] + "：留出集力频谱")
        ax.set_xlabel("cycles/frame（高频带为 0.25–0.5）")
        ax.set_ylabel("归一化力的窗内功率")
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        for j, (field, label) in enumerate((("reconstruction_rmse_n", "16帧历史重建"),
                                           ("current_reconstruction_rmse_n", "流式当前重建"),
                                           ("reconstruction_delta_rmse_n", "流式力差分"))):
            bars = ax.bar(np.arange(2) + (j - 1) * .24,
                          [score(metrics, s, "full_force", domain, field) for s in SPLITS], width=.24,
                          color=("#2474b5", "#e89131", "#6a9b73")[j], label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(range(2), [SPLIT_NAMES[s] for s in SPLITS])
        ax.set_ylabel("RMSE（N）")
        ax.set_title("重建波形及变化量误差")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        ax.legend(fontsize=8)
        ax = axes[row, 2]
        for j, label in enumerate(("16帧历史重建", "流式当前重建")):
            bars = ax.bar(np.arange(2) + (j - .5) * .32,
                          [spectra[domain][s]["energy_ratios"][j] for s in SPLITS], width=.32,
                          color=("#2474b5", "#e89131")[j], label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.axhline(1, color="#555555", ls="--", label="理想能量比 = 1")
        ax.set_xticks(range(2), [SPLIT_NAMES[s] for s in SPLITS])
        ax.set_ylabel("重建 / 输入高频能量（轨迹等权）")
        ax.set_title("高频被削弱还是被放大？")
        ax.legend(fontsize=8)
    style_axes(axes)
    save_figure(fig, output, "02_frequency.png", "高频与重建：能量接近不代表波形正确",
                "16帧 Hann 窗、训练集尺度归一化；频率使用 cycles/frame。输入高频能量很小时能量比不稳定，须结合波形和绝对误差判断。")
    return spectra


def alignment_figure(alignment, pools, basis, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    r, s = pools["holdout"][16]
    rp, sp = r @ basis, s @ basis
    ax = axes[0, 0]
    for a, b in zip(rp, sp):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#a5acb5", lw=.6, alpha=.4)
    ax.scatter(rp[:, 0], rp[:, 1], s=18, label="真机片段", color=COLORS["real"], alpha=.75)
    ax.scatter(sp[:, 0], sp[:, 1], s=21, marker="^", label="对应仿真片段", color=COLORS["sim"], alpha=.75)
    ax.set_title("留出集 16帧趋势片段：共享 PCA 投影")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=9)
    ax = axes[0, 1]
    for split, color in zip(SPLITS, ("#2474b5", "#e89131")):
        ax.plot((4, 8, 16), [alignment[split]["by_size"][str(n)]["paired_mse"] for n in (4, 8, 16)],
                "o-", color=color, label=SPLIT_NAMES[split] + "：真实配对")
        ax.plot((4, 8, 16), [alignment[split]["by_size"][str(n)]["mismatched_reference_mse"] for n in (4, 8, 16)],
                "x--", color=color, alpha=.7, label=SPLIT_NAMES[split] + "：同轨迹错配参考")
    ax.set_title("完整 16维趋势距离（越低越好）")
    ax.set_xlabel("片段帧数"); ax.set_ylabel("训练尺度标准化后的 MSE"); ax.set_xticks((4, 8, 16)); ax.legend(fontsize=8)
    ax = axes[1, 0]
    for split, color in zip(SPLITS, ("#2474b5", "#e89131")):
        ax.plot((4, 8, 16), [alignment[split]["by_size"][str(n)]["change_mse"] for n in (4, 8, 16)],
                "o-", color=color, label=SPLIT_NAMES[split])
    ax.set_title("相邻片段变化的跨域误差")
    ax.set_xlabel("片段帧数"); ax.set_ylabel("标准化趋势增量 MSE"); ax.set_xticks((4, 8, 16)); ax.legend()
    ax = axes[1, 1]
    for j, (field, label) in enumerate((("retrieval_top1", "精确配对 Top-1"), ("retrieval_top5", "精确配对 Top-5"))):
        bars = ax.bar(np.arange(2) + (j - .5) * .32, [100 * alignment[s][field] for s in SPLITS],
                      width=.32, color=("#2474b5", "#e89131")[j], label=label)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=3)
    for j, split in enumerate(SPLITS):
        chance = 100 / alignment[split]["retrieval_candidates"]
        ax.plot([j - .32, j + .32], [chance, chance], "--", color="#555555")
    ax.set_xticks(range(2), [f"{SPLIT_NAMES[s]}\n{alignment[s]['retrieval_candidates']} 个候选片段" for s in SPLITS])
    ax.set_title("仿真片段检索真机对应片段")
    ax.set_ylabel("命中率（%）；灰线为随机 Top-1"); ax.legend(fontsize=8)
    style_axes(axes)
    save_figure(fig, output, "03_alignment.png", "真机与仿真特征：配对趋势是否真的接近？",
                "均值、尺度和共享 PCA 均只用训练集真机特征拟合。距离与检索使用完整16维；错配参考不是无对齐消融，不能据此宣称对齐损失带来改善。")


def training_figure(root, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, stage, name in zip(axes.flat, ("teacher", "student", "motion", "control"),
                               ("真机 Encoder", "仿真 Encoder", "运动辅助预测头（Encoder 冻结）", "仅运动对照（Encoder 冻结）")):
        h = read_json(root / "single" / stage / "history.json")
        e = [r["epoch"] for r in h["records"]]
        ax.plot(e, [r["train"]["total"] for r in h["records"]], color="#2474b5", label="训练总损失")
        ax.set_ylabel("训练总损失", color="#2474b5")
        second = ax.twinx()
        second.plot(e, [r["valid"]["forecast_score_n"] for r in h["records"]], color="#e89131", label="验证预测误差")
        second.set_ylabel("验证预测误差（N）", color="#bd7226")
        ax.axvline(h["selected_epoch"], ls="--", color="#6a9b73", label=f"保留 epoch {h['selected_epoch']}")
        ax.set_title(name); ax.set_xlabel("Epoch")
        lines = ax.get_lines() + second.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    style_axes(axes)
    save_figure(fig, output, "04_training.png", "训练过程：训练损失与验证误差是否分离？",
                "四阶段训练损失定义不同，不跨阶段比较；验证集参与早停，独立留出集未参与模型选择。")


def gap_line(ax, t, y, starts, **kwargs):
    y = np.asarray(y, dtype=float).copy()
    resets = np.flatnonzero(np.diff(starts) != 0) + 1
    y[resets] = np.nan
    return ax.plot(t, y, **kwargs)


def detail_figure(split, tid, pair, output):
    fig, axes = plt.subplots(3, 4, figsize=(19, 9))
    for col, domain in enumerate(DOMAINS):
        a = pair[domain][tid]
        own_t, target_t = a["time"] - a["time"][0], a["real_time"] - a["real_time"][0]
        for axis, label in enumerate(("Fx", "Fy", "Fz")):
            ax = axes[axis, col]
            gap_line(ax, own_t, a["input_force"][:, axis], a["starts"], color="#222222", lw=1.1, label="本域输入力")
            gap_line(ax, own_t, a["history"][:, -1, axis], a["starts"], color=COLORS[domain], lw=1, label="编码后当前力重建")
            ax.set_title(f"{DOMAIN_NAMES[domain]}：{label} 重建"); ax.set_ylabel("N")
            ax = axes[axis, col + 2]
            gap_line(ax, target_t, a["real_force"][:, axis], a["real_starts"], color="#222222", lw=1.1, label="真实目标力")
            for field, model in (("future", "full_force"), ("joint", "full_joint"), ("control", "motion_control")):
                curve = np.full(len(own_t), np.nan)
                ii = np.arange(len(own_t))
                ii = ii[(ii + 5 < a["ends"]) & (ii + 5 < a["real_ends"])]
                curve[ii + 5] = a[field][ii, 4, axis]
                gap_line(ax, target_t, curve, a["real_starts"], color=COLORS[model], lw=1, label=MODEL_NAMES[model])
            ax.set_title(f"{DOMAIN_NAMES[domain]}：{label} 未来 +5帧"); ax.set_ylabel("N")
    style_axes(axes)
    for ax in axes.flat:
        ax.set_xlabel("记录时间（s）"); ax.legend(fontsize=7)
    filename = f"{split}_traj_{tid}.png"
    save_figure(fig, output, filename, f"{SPLIT_NAMES[split]} · 轨迹 {tid} · 三轴力重建与预测",
                "左两列重建各自输入域的力；右两列均预测未来真机力。预测曲线已移到目标时刻，断档不连接。")
    return filename


def latent_figure(data, selected, mean, scale, output):
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), gridspec_kw={"height_ratios": [1, 1.3, 1.3]})
    for col, (tid, desc) in enumerate(selected):
        for domain in DOMAINS:
            a = data["holdout"][domain][tid]
            t = a["time"] - a["time"][0]
            gap_line(axes[0, col], t, a["input_force"][:, 2], a["starts"], color=COLORS[domain], label=DOMAIN_NAMES[domain])
        axes[0, col].set_title(f"{desc}：轨迹 {tid} 的 Fz"); axes[0, col].set_ylabel("N"); axes[0, col].legend(fontsize=8)
        for row, domain in enumerate(DOMAINS, 1):
            a = data["holdout"][domain][tid]
            # pcolormesh preserves the actual nonuniform timestamps.
            t = a["time"] - a["time"][0]
            z = ((a["z"] - mean) / scale).T
            image = axes[row, col].pcolormesh(t, np.arange(32), z, shading="nearest", cmap="RdBu_r", vmin=-3, vmax=3)
            axes[row, col].axhline(15.5, color="#333333", ls="--", lw=.8)
            axes[row, col].set_ylabel(f"{domain} 编码维度"); axes[row, col].set_xlabel("记录时间（s）")
            fig.colorbar(image, ax=axes[row, col], pad=.015, label="训练集真机标准差单位")
    save_figure(fig, output, "05_latent.png", "32维力编码随时间的变化（上下两域共用尺度）",
                "0–15维为趋势，16–31维为瞬态；瞬态未施加跨域逐点相等约束。按仿真纯力预测误差选择中位和最差轨迹，避免只展示好例子。")


def event_figure(data, selected, output):
    fig, axes = plt.subplots(4, 2, figsize=(15, 10))
    for col, (tid, desc) in enumerate(selected):
        real = data["holdout"]["real"][tid]
        delta = np.linalg.norm(np.diff(real["input_force"], axis=0, prepend=real["input_force"][:1]), axis=1)
        valid = np.arange(len(delta)) > real["starts"]
        delta[~valid] = -1
        center = int(delta.argmax())
        lo, hi = max(real["starts"][center], center - 12), min(real["ends"][center], center + 13)
        for d, domain in enumerate(DOMAINS):
            a = data["holdout"][domain][tid]
            t = a["time"] - a["time"][0]
            for field, color, name in (("input_force", "#222222", "输入力"), ("history", COLORS[domain], "流式重建")):
                y = a[field][:, 2] if field == "input_force" else a[field][:, -1, 2]
                axes[d * 2, col].plot(t[lo:hi], y[lo:hi], "o-", color=color, ms=3, lw=1, label=name)
                dy = np.diff(y, prepend=y[:1])
                axes[d * 2 + 1, col].plot(t[lo:hi], dy[lo:hi], "o-", color=color, ms=3, lw=1, label=name)
            axes[d * 2, col].set_ylabel(f"{domain} Fz（N）")
            axes[d * 2 + 1, col].set_ylabel(f"{domain} ΔFz（N/帧）")
            for row in (d * 2, d * 2 + 1):
                axes[row, col].legend(fontsize=8); axes[row, col].set_xlabel("记录时间（s）")
        axes[0, col].set_title(f"{desc}：轨迹 {tid} 最大真实力突变附近")
    style_axes(axes)
    save_figure(fig, output, "06_event_zoom.png", "接触突变放大：高频变化有没有被抹掉？",
                "中心由真机三轴力的最大相邻帧变化确定，展示前后最多12帧；ΔF 为逐帧差值，未按时间求导。")


def write_report(root, output, metrics, alignment, spectra, details):
    rows = []
    single = read_json(root / "single/result.json")
    intro = (f"{len(single['train_ids'])}条训练、{len(metrics['validation']['ids'])}条早停验证、"
             f"{len(metrics['holdout']['ids'])}条独立留出。固定 seed {single['seed']}，无额外训练。")
    real_force = score(metrics, "holdout", "full_force", "real")
    sim_joint = score(metrics, "holdout", "full_joint", "sim")
    motion = score(metrics, "holdout", "motion_control", "sim")
    held_align = alignment["holdout"]
    findings = [f"独立留出集：真机纯力预测综合误差 {real_force:.3f} N；保持当前力为 {score(metrics, 'holdout', 'hold', 'real'):.3f} N，线性 AR 为 {score(metrics, 'holdout', 'ar', 'real'):.3f} N。",
                f"仿真编码＋运动为 {sim_joint:.3f} N，仅运动为 {motion:.3f} N，当前仿真编码没有带来额外预测收益。" if sim_joint >= motion else
                f"仿真编码＋运动为 {sim_joint:.3f} N，仅运动为 {motion:.3f} N；改善幅度需结合逐轨迹结果判断。",
                f"留出集片段精确检索 Top-1={held_align['retrieval_top1']:.1%}，随机约 {1 / held_align['retrieval_candidates']:.1%}；仍不能据此认为跨域对齐充分。",
                f"留出集真机历史重建高频能量比 {spectra['real']['holdout']['energy_ratios'][0]:.2f}，仿真为 {spectra['sim']['holdout']['energy_ratios'][0]:.2f}；需关注真机高频损失和仿真额外波动，并结合较小分母及绝对误差判断。"]
    md = ["# Force encoder 可视化诊断", "", intro, "",
          "验证集用于早停；独立留出集未参与本次训练、基线拟合或超参数选择。这里的独立性仅针对当前运行。", "",
          *["- " + finding for finding in findings], "",
          "| 集合 | 输入 | 仅力编码 | 编码＋运动 | 仅运动 | 当前力保持/校准 | 线性 AR |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for domain in DOMAINS:
            values = [score(metrics, split, model, domain) for model in MODEL_NAMES]
            md.append(f"| {SPLIT_NAMES[split]} | {DOMAIN_NAMES[domain]} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
            rows.append("<tr><td>" + "</td><td>".join([SPLIT_NAMES[split], DOMAIN_NAMES[domain], *[f"{v:.3f}" for v in values]]) + "</td></tr>")
    md += ["", "数值为轨迹等权的未来真实力综合误差，单位 N，越低越好。", "",
           "- 只评估已保存权重；AR 正则 α=1，仿真线性校准和 AR 均只在24条训练轨迹上拟合。",
           "- 未来曲线按目标时间绘制；重建目标为各自输入域的力。",
           "- 对齐距离和检索使用完整16维趋势特征；PCA只作共享视角展示，投影只在训练真机上拟合。",
           "- 错配参考不是无对齐消融；本次没有重训消融，无法归因于对齐损失。",
           "- 高频能量比接近1仍可能有相位错误；输入高频能量很小时比例会放大。",
           "- 当前 Sim q/dq 为复制或来源未独立验证，运动头比较仅代表离线配对回放。", "",
           "## 对齐检索与高频能量比", ""]
    for split in SPLITS:
        a = alignment[split]
        md.append(f"- {SPLIT_NAMES[split]}：趋势片段检索 Top-1={a['retrieval_top1']:.1%}，Top-5={a['retrieval_top5']:.1%}，候选数={a['retrieval_candidates']}。")
        for domain in DOMAINS:
            ratios = spectra[domain][split]["energy_ratios"]
            md.append(f"  - {domain} 高频能量比：历史重建 {ratios[0]:.3f}，流式重建 {ratios[1]:.3f}。")
    md += ["", "## 图表", "", "[六页总览 PDF](encoder_overview.pdf)", ""]
    figures = [("01_forecast.png", "未来真实力预测", "先看仅力编码是否优于保持当前力和线性 AR，再看运动辅助是否优于仅运动。"),
               ("02_frequency.png", "高频与重建", "同时检查频谱、波形/差分误差和高频能量比。"),
               ("03_alignment.png", "跨域特征对齐", "共享 PCA 仅供观察；完整维度距离、变化误差和精确配对检索提供定量依据。"),
               ("04_training.png", "训练过程", "四阶段曲线和最终保留的 epoch；motion/control 阶段不改变 Encoder。"),
               ("05_latent.png", "32维编码热图", "中位与最差仿真预测轨迹，两域使用训练集真机的同一尺度。"),
               ("06_event_zoom.png", "力突变放大", "查看流式重建在最大力突变附近的波形与逐帧差分。")]
    sections = []
    for file, title, note in figures:
        md.append(f"- [{title}]({file})：{note}")
        sections.append(f'<section id="{file[:2]}"><h2>{title}</h2><p>{note}</p><a href="{file}"><img src="{file}" loading="lazy" alt="{title}"></a></section>')
    md += ["", "## 重新生成", "", "在 TAVLA 根目录运行（CPU 推理，不重训）：", "", "```bash",
           "PYTHONPATH=scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \\",
           f"  .venv/bin/python -m force_predictive_encoder_v3.visualize --run-dir {root}", "```"]
    (output / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    detail_sections = []
    for split in SPLITS:
        options = "".join(f'<option value="{html.escape(file)}">轨迹 {tid}</option>' for s, tid, file in details if s == split)
        first = next(file for s, tid, file in details if s == split)
        detail_sections.append(f'<section><h2>{SPLIT_NAMES[split]}逐轨迹波形</h2><p>选择轨迹；点击图片可打开原图。左两列重建输入域力，右两列预测未来真机力。</p>'
                               f'<select aria-label="{SPLIT_NAMES[split]}轨迹" onchange="this.nextElementSibling.href=this.value;this.nextElementSibling.firstElementChild.src=this.value">{options}</select>'
                               f'<a href="{first}"><img src="{first}" loading="lazy" alt="逐轨迹三轴力波形"></a></section>')
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    body += '<title>Force Encoder 诊断</title><style>body{font:15px system-ui,sans-serif;background:#f3f5f8;color:#243142;margin:0}main{max-width:1440px;margin:32px auto;padding:0 24px}section,header{background:white;padding:24px;margin:20px 0;border-radius:14px}h1{font-size:30px;margin:0 0 12px}h2{font-size:23px}p{line-height:1.8}img{width:100%;height:auto}a{color:#2474b5}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px 10px;border-bottom:1px solid #dde3ea;text-align:right}td:first-child,td:nth-child(2){text-align:left}select{padding:10px 22px;font-size:16px;margin:10px 0}.note{padding:12px 18px;border-left:4px solid #e89131;background:#fff7eb}.scroll{overflow:auto}nav a{margin-right:18px}</style><main>'
    body += '<header><h1>Force Encoder：预测、对齐与高频诊断</h1><p>' + intro + '</p>'
    body += '<p class="note">所有未来预测都以真机力为目标。验证集参与早停；当前 Sim 运动状态来源未独立验证，运动头结果只代表离线回放。高频和跨域对齐需结合多张图判断。</p>'
    body += '<nav>' + ''.join(f'<a href="#{f[:2]}">{t}</a>' for f, t, n in figures) + '<a href="encoder_overview.pdf">总览 PDF</a><a href="analysis.md">数值报告</a><a href="diagnostics.json">完整指标</a></nav></header>'
    body += '<section><h2>未来力综合误差（N，越低越好）</h2><div class="scroll"><table><thead><tr>'
    body += ''.join(f'<th>{s}</th>' for s in ['集合','输入','仅力编码','编码＋运动','仅运动','保持/校准','线性 AR']) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    body += '<p>各轨迹等权。综合误差为 +1、+3、+5帧 RMSE 的加权和；AR 正则固定 α=1，基线只用训练集拟合。</p></section>'
    body += '<section><h2>当前观察</h2><ul>' + ''.join('<li><p>' + html.escape(f) + '</p></li>' for f in findings) + '</ul></section>'
    body += ''.join(sections + detail_sections) + '</main></html>'
    (output / "index.html").write_text(body, encoding="utf-8")


def visualize_single(root, limit=None):
    root = Path(root).resolve()
    output = root / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.titlepad": 10})
    torch.set_num_threads(2)
    recipe, single, manifest = [read_json(root / f) for f in ("recipe.json", "single/result.json", "manifest.json")]
    if not single["completed"]:
        raise ValueError("Training has not completed")
    checkpoint = root / manifest["checkpoint"]
    if sha256(checkpoint) != manifest["encoder_sha256"]:
        raise ValueError("Checkpoint does not match manifest")
    initial_hash = sha256(checkpoint)
    model, payload = load_model(checkpoint, "cpu")
    model.eval().requires_grad_(False)
    stats = payload["stats"]
    options = recipe["options"]
    episodes, audit = load_pairs(options["real_root"], options["sim_root"], options["labels"])
    for tid, row in recipe["data"]["trajectories"].items():
        if [s["sha256"] for s in row["sources"]] != [s["sha256"] for s in audit["trajectories"][tid]["sources"]]:
            raise ValueError(f"Data changed since training: trajectory {tid}")
    if sha256(options["labels"]) != recipe["label_sha256"]:
        raise ValueError("Labels changed since training")
    ids = {"validation": single["valid_ids"], "holdout": sorted(HOLDOUT)}
    train_ids = single["train_ids"]
    if set(train_ids) & set(ids["holdout"]) or set(ids["validation"]) & set(ids["holdout"]):
        raise ValueError("Holdout overlaps training or validation")
    if sorted(stats["fit_trajectory_ids"]) != sorted(train_ids):
        raise ValueError("Unexpected fitted normalizer")
    print("Data and checkpoint verified. Inferring the 8 unused paired trajectories on CPU.", flush=True)
    held = evaluate(model, episodes, ids["holdout"], stats, with_motion=True, save_dir=output / "predictions" / "holdout")
    metrics = {"validation": {"ids": ids["validation"], "models": single["models"]},
               "holdout": {"ids": ids["holdout"], "models": {name: flatten_evaluation(held, kind) for name, kind in
                            (("full_force", "force"), ("full_joint", "joint"), ("motion_control", "control"))}}}
    data = {}
    for split in SPLITS:
        parent = root / "single/predictions/validation" if split == "validation" else output / "predictions/holdout"
        data[split] = {d: {tid: load_arrays(parent / d / f"traj_{tid}.npz") for tid in ids[split]} for d in DOMAINS}
    print("Fitting two small linear references on the 24 training trajectories only.", flush=True)
    for d, domain in enumerate(DOMAINS):
        for kind in ("hold", "ar"):
            baseline = baselines.fit(episodes, train_ids, stats, d, kind, 1.)
            for split in SPLITS:
                metrics[split]["models"].setdefault(kind, {})[domain] = baselines.evaluate(baseline, episodes, ids[split], stats)
    print("Computing shared training-only projection and cross-domain statistics.", flush=True)
    train_z = np.concatenate([predict_episode(model, episodes[(tid, 0)], stats, 0)["z"] for tid in train_ids])
    mean, scale = train_z.mean(0), np.maximum(train_z.std(0), .1)
    _, _, vt = np.linalg.svd((train_z[:, :16] - mean[:16]) / scale[:16], full_matrices=False)
    basis = vt[:2].T
    np.savez_compressed(output / "training_projection.npz", mean=mean, scale=scale, trend_pca_basis=basis, train_ids=train_ids)
    alignment, pools = {}, {}
    for split in SPLITS:
        alignment[split], pools[split] = alignment_metrics(data[split], mean, scale)
    forecast_figure(metrics, output)
    spectra = frequency_figure(metrics, data, stats, output)
    alignment_figure(alignment, pools, basis, output)
    training_figure(root, output)
    ordered = sorted(ids["holdout"], key=lambda tid: metrics["holdout"]["models"]["full_force"]["sim"][str(tid)]["all"]["forecast_score_n"])
    selected = [(ordered[len(ordered) // 2], "中位预测误差"), (ordered[-1], "最差预测误差")]
    latent_figure(data, selected, mean, scale, output)
    event_figure(data, selected, output)
    details = []
    for split in SPLITS:
        for tid in ids[split][:limit]:
            file = detail_figure(split, tid, data[split], output)
            details.append((split, tid, file))
    save_json(output / "diagnostics.json", dict(checkpoint_sha256=initial_hash, train_ids=train_ids, datasets=metrics,
                                                alignment=alignment, spectra=spectra, selected_examples=selected,
                                                baseline_ar_alpha=1., projection_fit="real_training_only",
                                                actual_sim_state_verified=audit["actual_sim_state_verified"]))
    write_report(root, output, metrics, alignment, spectra, details)
    export_overview_pdf(output)
    shutil.copy2(Path(__file__), output / "visualize_single_source.py")
    if sha256(checkpoint) != initial_hash:
        raise RuntimeError("Checkpoint changed during visualization")
    with zipfile.ZipFile(root / "encoder_visualizations.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(output.iterdir()):
            if file.is_file():
                bundle.write(file, arcname="encoder_visualizations/" + file.name)
    print(f"Complete: {output / 'index.html'}", flush=True)
    for split in SPLITS:
        for domain in DOMAINS:
            print(split, domain, {m: round(score(metrics, split, m, domain), 4) for m in MODEL_NAMES}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    visualize_single(parser.parse_args().run_dir)
