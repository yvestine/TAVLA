#!/usr/bin/env python3
"""Render the domain-invariant force+pose encoder architecture as a PNG.

Read-only figure generator: writes a single PNG under eval_outputs/. Does not
touch any checkpoint, dataset, or training artifact. The layout follows the
standard DANN figure (feature extractor + task head + GRL/domain head), extended
on the left with this project's paired Real/Sim force+pose inputs and in the
middle with the paired InfoNCE alignment link.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

# Palette (colour-blind friendly, muted so text stays readable).
C_DATA = "#E8EEF7"
C_FEAT = "#D6E8D5"
C_ENC = "#BFD9C0"
C_LATENT = "#F6D7A8"
C_QUALITY = "#CFE0F3"
C_DOMAIN = "#F3CFCF"
C_INFONCE = "#E3D4EF"
C_OUT = "#FBE9C7"
EDGE = "#3A3A3A"


def box(ax, x, y, w, h, text, face, *, fontsize=11, bold=False, edge=EDGE, lw=1.4):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal", zorder=3, color="#1A1A1A",
    )
    return (x, y, w, h)


def arrow(ax, start, end, *, color=EDGE, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, linestyle=ls,
        shrinkA=2, shrinkB=2, zorder=1,
    ))


def render(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.6), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    # ---- Column 1: paired Real / Sim raw inputs ----------------------------
    box(ax, 1.5, 40, 15, 8,
        "Real 轨迹\nee_force_base (3维力)\n+ ee_pose", C_DATA, fontsize=9.5)
    box(ax, 1.5, 9, 15, 8,
        "Sim 轨迹\nee_wrench_base[:,:3]\n+ ee_pose", C_DATA, fontsize=9.5)
    ax.text(9, 50.5, "① 逐帧配对输入", ha="center", fontsize=11, fontweight="bold", color="#26456e")

    # ---- Column 2: 17-D causal features ------------------------------------
    box(ax, 21, 39.5, 17, 9,
        "因果特征 (每帧17维)\n力: 方向3 · 侧向/轴向占比2\n· 趋势方向3 · 大小2\n位姿: 速度3·位移3·深度1",
        C_FEAT, fontsize=8.6)
    box(ax, 21, 8.5, 17, 9,
        "因果特征 (每帧17维)\n力: 方向3 · 侧向/轴向占比2\n· 趋势方向3 · 大小2\n位姿: 速度3·位移3·深度1",
        C_FEAT, fontsize=8.6)
    ax.text(29.5, 50.5, "② 10维力+7维位姿", ha="center", fontsize=11, fontweight="bold", color="#26456e")
    ax.text(29.5, 29, "前20帧自由空间\n标定，去零漂", ha="center", va="center",
            fontsize=8.2, style="italic", color="#555")

    # ---- Column 3: shared encoder (weights shared across domains) ----------
    box(ax, 43, 39, 15, 10,
        "10帧历史窗\n(10 × 17)", C_DATA, fontsize=9.5)
    box(ax, 43, 8, 15, 10,
        "10帧历史窗\n(10 × 17)", C_DATA, fontsize=9.5)
    enc = box(ax, 61, 24, 15, 12,
              "共享 Transformer\nEncoder\n(2层, d=32)\n→ 32维 latent", C_ENC,
              fontsize=9.6, bold=True, lw=2.0)
    ax.text(68.5, 50.5, "③ 权重共享编码器", ha="center", fontsize=11, fontweight="bold", color="#26456e")
    ax.text(68.5, 37.4, "Real / Sim 共用同一套权重", ha="center", fontsize=8.4,
            style="italic", color="#555")

    # ---- Column 4: three training heads ------------------------------------
    box(ax, 82, 42, 16, 7,
        "质量分类头\n插歪 / 插正", C_QUALITY, fontsize=9.6, bold=True)
    box(ax, 82, 26.5, 16, 7,
        "域判别头 (经 GRL\n梯度反转)\nReal / Sim", C_DOMAIN, fontsize=9.0, bold=True)
    box(ax, 82, 11, 16, 7,
        "配对 InfoNCE\n同一(轨迹,帧)的\nReal↔Sim 拉近", C_INFONCE, fontsize=8.8, bold=True)
    ax.text(90, 50.5, "④ 三个训练损失", ha="center", fontsize=11, fontweight="bold", color="#26456e")

    # ---- flows: inputs -> features -> windows -> encoder -------------------
    arrow(ax, (16.5, 44), (21, 44))
    arrow(ax, (16.5, 13), (21, 13))
    arrow(ax, (38, 44), (43, 44))
    arrow(ax, (38, 13), (43, 13))
    arrow(ax, (58, 44), (61, 34.5))
    arrow(ax, (58, 13), (61, 25.5))

    # ---- encoder -> heads --------------------------------------------------
    arrow(ax, (76, 33), (82, 45.5), color="#2f6b8f", lw=1.8)          # quality
    arrow(ax, (76, 30), (82, 30), color="#a23b3b", lw=1.8, ls=(0, (5, 2)))  # domain (GRL)
    arrow(ax, (76, 27), (82, 14.5), color="#7a4fa0", lw=1.8, ls=(0, (2, 2)))  # infonce

    # ---- GRL reversal annotation ------------------------------------------
    ax.text(79.5, 23.2, "反向梯度\n逼 encoder\n分不出域", ha="center", va="center",
            fontsize=7.6, color="#a23b3b", style="italic")

    # ---- InfoNCE pairing bracket between the two windows -------------------
    arrow(ax, (50.5, 39), (50.5, 18), color="#7a4fa0", lw=1.4, style="<|-|>", ls=(0, (2, 2)))
    ax.text(50.5, 21.2, "配对", ha="center", va="center", fontsize=7.8,
            color="#7a4fa0", rotation=90)

    # ---- Bottom: frozen latent -> downstream products ----------------------
    box(ax, 61, 1.5, 15, 5.2,
        "冻结 32维 latent\n(域不变表示)", C_LATENT, fontsize=9.0, bold=True)
    arrow(ax, (68.5, 24), (68.5, 6.7), color="#c58a2e", lw=2.0)
    box(ax, 30, 1.2, 12.5, 5.6, "→ TAVLA\n辅助输入", C_OUT, fontsize=8.6)
    box(ax, 44, 1.2, 12.5, 5.6, "→ 仿真 residual\nRL 观测", C_OUT, fontsize=8.6)
    arrow(ax, (61, 4.1), (56.5, 4.1), color="#c58a2e", lw=1.5)
    arrow(ax, (56.5, 3.4), (42.5, 3.4), color="#c58a2e", lw=1.3, ls=(0, (3, 2)))

    # ---- title & footnote --------------------------------------------------
    ax.text(50, 56.4, "域不变 力+位姿 插入质量 Encoder — 训练流程",
            ha="center", fontsize=15, fontweight="bold", color="#1a1a1a")
    ax.text(50, 53.8,
            "推理时只保留「共享 Encoder → 32维 latent」，三个头仅训练用；结果: latent 域可分性 0.209→0.071，Real Macro-F1 0.750→0.761",
            ha="center", fontsize=8.8, color="#444")
    ax.text(99, 0.2,
            "方法: DANN (GRL, 1505.07818) + 配对对齐 (TactSpace, 2606.18959) + 力方向可迁移 (2602.14174)",
            ha="right", fontsize=7.4, color="#777", style="italic")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    render(repo / "eval_outputs" / "encoder_diagram" / "domain_invariant_encoder.png")


if __name__ == "__main__":
    main()
