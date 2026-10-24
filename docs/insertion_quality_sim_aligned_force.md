# 新版仿真力 Encoder

## 输入和验证

- 原始力来自 `sim-data-aligned/traj_*/data.h5::obs/state/ee_wrench_base[:,:3]`。力矩三列为零，不进入模型。
- 力趋势用 [force_features.py](../src/openpi/force_features.py) 生成，复用 Route A 的 10 维因果特征函数。自由空间结束帧、平滑和趋势参数直接读取 [tavla_sim_force_trend_affine.json](../configs/tavla_sim_force_trend_affine.json)，没有第二份参数表。此交叉验证对原始 Sim 力逐折拟合特征尺度；Route A 的全局 Sim→Real 仿射适配器不参与这项验证。
- 每个样本使用连续 10 帧力特征。阶段标签来自 [insertion_quality_labels_v2.csv](../annotations/insertion_quality_labels_v2.csv)，按配对轨迹和帧转移到新 Sim。只有完整落在同一阶段的 `misaligned_contact`、`aligned_progress` 窗口参与二分类；`free`、`uncertain`、`terminal` 不作为二分类标签。
- 23 条同时有两类可用窗口的轨迹逐条留出。每折的特征中心和尺度仅用其余训练轨迹估计。模型仅训练质量分类头；不会输出未经训练的终点概率。
- 这是转移标签上的离线分类验证，尚未把 encoder 接入 TAVLA 策略，也不代表新版 Sim 接触状态已独立人工复核。

## 运行

```bash
cd /data/workspace/gujiawei/TAVLA
GPU_ID=7 bash scripts/run_insertion_quality_sim_aligned_force.sh
```

`GPU_ID` 可换成空闲显卡。仅检查一折时，可使用独立输出目录：

```bash
OUTPUT_DIR=eval_outputs/insertion_quality_sim_aligned_force_check GPU_ID=7 \
  bash scripts/run_insertion_quality_sim_aligned_force.sh --max-folds 1 --epochs 1
```

完整运行写入 `eval_outputs/insertion_quality_sim_aligned_force/`：包含
`run_config.json`、每折 `model.pt` 和预处理参数、预测 CSV、汇总指标与报告。
完整运行的报告另写入项目根目录的
`insertion_quality_encoder_sim_aligned_force_report.md`。
此命令不写 `checkpoints/`、pi0 基础权重或已有 TAVLA 策略权重。

## 历史交叉验证

| 实验 | 验证方式 | pooled Macro-F1 |
|---|---|---:|
| V1 力 Encoder | 10 条轨迹逐条留出；7 折有质量标签 | 0.551 |
| V2 力 Encoder | 23 条轨迹逐条留出 | 0.690 |
| V2 手工特征 Logistic | 同上 | 0.736 |
| V3 仅力 Transformer A | 同上 | 0.643 |
| V3 力与位姿 B / 预训练与 GRL C / 仅位姿 D | 同上 | 0.741 / 0.742 / 0.757 |

这些是旧 `sim_force` 接触力代理与 Real 数据的历史结果，不是
`sim-data-aligned` 的结果。历史数据、报告和评估产物继续保留。

## 力 gap 量化（只读诊断）

先回答“这个 encoder 到底减了多少 real-sim 力 gap”。用
[measure_force_domain_gap.py](../scripts/measure_force_domain_gap.py) 在配对的
`sim-data-aligned` 上测 Real 与 Sim 的可分性（proxy-A-distance，`[0,2]`，0=不可分/无
gap，2=平凡可分），去掉前 20 帧标定窗，只看接触段。

```bash
cd /data/workspace/gujiawei/TAVLA
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/measure_force_domain_gap.py \
  --config configs/insertion_quality_sim_aligned_force_pose.json \
  --output-dir eval_outputs/force_domain_gap
```

结果（写入 `eval_outputs/force_domain_gap/`）：

| 度量对象 | proxy-A-dist | 说明 |
|---|---:|---|
| 原始力幅值 | Real 29.8 N vs Sim 5.8 N | 原始差约 5× |
| 10 维特征全部 | 1.05 | 中等 gap |
| 方向类通道（8 维） | 0.54 | **小 gap，可迁移** |
| 幅值类通道（force_norm/trend_norm） | 1.71 | **大 gap，不可迁移** |

结论与文献一致：**力的方向跨 real-sim 可迁移，力的幅值不可迁移**（见
[Direction Matters, arXiv:2602.14174](https://arxiv.org/html/2602.14174v1)）。现有 10
维特征以方向/比例为主体，已经把大部分幅值 gap 归一化掉，这解释了为什么纯 Sim
encoder 也能在 Real 上有效。

## 域不变 Encoder（核心产物）

只在 Sim 上训练/评估无法证明“减小了 gap”。因此新增一版**跨域域不变** encoder：在
Real+Sim 上联合训练，评估口径改为 held-out **Real**（sim→real 泛化），显式压低
latent 层的 real-sim gap。三种机制（都来自调研）：

- 梯度反转域对抗（DANN 式，`domain_lambda`）：把 Real/Sim latent 拉近。
- 配对 InfoNCE（`infonce_weight`）：同一 `(轨迹, 帧)` 的 Real 和 Sim 窗口拉到同一
  latent——因为两域逐帧配对，这是比单纯分布对齐更强的监督信号（见
  [TactSpace, arXiv:2606.18959](https://arxiv.org/html/2606.18959v1)）。
- 双域质量监督：转移标签同时监督 Real 和 Sim，保住 misaligned/aligned 结构，避免
  latent 坍缩（“close but not collapsed”，见
  [Sim-and-Real Co-Training 机理分析, arXiv:2604.13645](https://arxiv.org/html/2604.13645)）。

```bash
cd /data/workspace/gujiawei/TAVLA
CUDA_VISIBLE_DEVICES="4" .venv/bin/python scripts/train_insertion_quality_domain_invariant.py \
  --config configs/insertion_quality_sim_real_domain_invariant.json \
  --output-dir eval_outputs/insertion_quality_domain_invariant \
  --device cuda:0
```

23 折逐条留出，~21 分钟。结果（held-out Real）：

| 变体 | Real Macro-F1 | Real aligned F1 | Real AUROC | latent proxy-A-dist |
|---|---:|---:|---:|---:|
| force+pose 基线（无对齐） | 0.750 | 0.628 | 0.801 | 0.209 |
| **force+pose 域不变** | **0.761** | **0.649** | **0.861** | **0.071** |

域不变版把 latent gap 从 0.209 降到 **0.071**（下降约 66%），同时 Real Macro-F1 不降反
升（0.750→0.761），AUROC 从 0.801 升到 0.861。gap 显著缩小、质量不掉，说明是“close but
not collapsed”而非过度对齐坍缩。这版**冻结后的 latent** 就是要给 TAVLA 当辅助输入、给
仿真里的 residual RL 当观测的产物。完整报告写入根目录
`insertion_quality_encoder_domain_invariant_report.md`；不写 `checkpoints/` 或任何已训练权重。

冒烟检查（不覆盖正式结果）：

```bash
CUDA_VISIBLE_DEVICES="4" .venv/bin/python scripts/train_insertion_quality_domain_invariant.py \
  --config configs/insertion_quality_sim_real_domain_invariant.json \
  --output-dir eval_outputs/_smoke_domain_invariant \
  --device cuda:0 --max-folds 2 --epochs 3
```
