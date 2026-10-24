# 力觉插入对齐 Encoder（ForceAlignmentEncoder）

> 本文档描述**当前保存的** `eval_outputs/insertion_alignment_encoder/encoder.pt`。
> 它由 [`train_terminal_geometry.py`](../scripts/insertion_alignment_encoder/train_terminal_geometry.py)
> 训练，artifact 格式 `tavla_force_alignment_encoder_terminal_reference_v1`，
> `geometry_supervision = true`。
>
> 它**不是** TAVLA 的 action 模型，也**不是**后来那版 10 维手工力趋势特征
> （那套在 [`insertion_quality_sim_aligned_force.md`](insertion_quality_sim_aligned_force.md)）。

> **2026-10-10 更新**：第 1–11 节描述的是已部署的旧权重。
> 第 12 节记录包含末段插入监督的 `full_insertion_v4`。
> **后续重训使用第 13 节的 v2 A/B 入口**，分别保存旧配方复现与补充末段监督的候选。

## 1. 这个 encoder 要做什么

把**同一条轨迹、同一帧**的真机力历史和仿真力历史，分别送进**同一套**
encoder，让它们得到方向相近的 32 维表示；同时让这个表示能判断当前处于
"接触歪斜"（`misaligned_contact`）还是"对正后继续插入"（`aligned_progress`）。

```
真机 XYZ 力历史 ┐
               ├─► 基线扣除 / 变化率 / 归一化 ─► 32帧×6通道 ─► 共享 Encoder ─┬─► 32维 latent ─► 配对余弦相似损失
仿真 XYZ 力历史 ┘                                                          ├─► 歪斜 / 对正 二分类
                                                                          └─► 相对终点的位置估计（辅助）
```

目标产物是那个 **32 维 latent**，设计用途是接到 TAVLA policy 的
`observation.effort`，或给仿真里的 residual RL 当观测。

## 2. 数据与配对

| 项 | 来源 |
|---|---|
| 真机力 | `data/traj_i/data.h5` → `obs/state/ee_force_base`，只取 XYZ 三轴 |
| 仿真力 | `sim-data-aligned/traj_i/data.h5` → `obs/state/ee_force_base`，只取 XYZ 三轴 |
| 阶段标签 | `annotations/insertion_quality_labels_v2.csv` |

- 真机与仿真按**同一轨迹编号、同一帧**配对。训练代码强制两边帧数相同。
- 代码还逐帧校验两边的 `obs/state/ee_pose` 位置差；当前数据
  `max_real_sim_pose_disagreement_mm = 0.0`，即两域共用同一条末端 pose 序列。
  这点很关键，直接影响下面第 7 节的位置目标解读。
- 不用力矩（force 的后三维），只用 XYZ。

## 3. 六通道趋势输入

预处理见 [`preprocess.py`](../scripts/insertion_alignment_encoder/preprocess.py)：

1. **基线扣除**：取每条轨迹前 20 帧各轴力的中位数作为空载基线并扣除。
2. **变化率**：对基线扣除后的力做一阶差分，除以 `dt`（`dt` clip 到 `[0.02, 0.5]` s）。
3. **归一化**：用训练轨迹统计的共同尺度（各通道 `|值|` 的 95 分位，带下限
   `[0.5,0.5,0.5,1,1,1]`）相除，再 clip 到 `±10`。

每帧 6 个通道：

```
Fx, Fy, Fz,  dFx/dt, dFy/dt, dFz/dt   （基线扣除后）
```

即 encoder 看到的是"力有多大、力正在怎么变"，而不是原始传感器读数。
力的方向跨 real-sim 可迁移、幅值不可迁移，这套归一化把大部分幅值 gap 抹掉。

## 4. 窗口与截断

- **窗口**：每个样本是当前帧及之前 31 帧，共 **32 帧**；采样 10 Hz，约覆盖 3.2 s。
  右边缘是当前帧（因果，不看未来）。
- **配对**：每个窗口对应同一轨迹、同一帧的真机窗口 + 仿真窗口。
- **边界保护**：人工标注的阶段边界附近各去掉 5 帧（`boundary_guard`）。
- **截断**：仿真几何首次满足 `XY < 3 mm 且 |Z| < 3 mm`（`_first_hit`）时截断，
  后续帧不参与训练。每条轨迹的 cutoff 存在 artifact 的
  `geometry.first_hit_cutoff_frames` 里。
- **参与分类的类别**：只有人工复核的 `misaligned_contact` 和 `aligned_progress`
  两类进分类；`free`、`uncertain`、`terminal` 不参与。

## 5. 共享 Encoder 结构

实现见 [`model.py`](../scripts/insertion_alignment_encoder/model.py)。两域共用同一个模型：

```
输入 [batch, 32帧, 6通道]
  └─ Conv1d(6→48, k=1)                      stem
  └─ 3× CausalBlock(48, dilation 1/2/4)     因果残差卷积（GroupNorm+GELU+Dropout0.1）
  └─ GRU(48 → 64)                           汇总时间变化
  └─ 取最后一帧 seq[:, -1]
  └─ Linear(64→64) + GELU + Linear(64→32) + Tanh   ⇒ 32维 latent
       ├─ quality_head:  Linear(32→2)   二分类
       └─ geometry_head: Linear(32→3)   相对位置
```

> 注意 latent 过了 **Tanh**，每维被压到 `[-1,1]`。这是余弦相似度天然偏高的
> 原因之一，解读第 6 节的相似度数字时要记得这一点。

## 6. 三种训练信号

训练损失（[`train_terminal_geometry.py`](../scripts/insertion_alignment_encoder/train_terminal_geometry.py) 约 L214）：

```
loss = class_loss
     + 0.15 * consistency      # 配对 latent 余弦相似
     + 0.12 * geom_loss        # 位置 smooth-L1
     + 0.05 * direction_loss   # 位置方向余弦（仅半径≥3mm的帧）
```

- **状态分类**（主信号）：人工标为"歪斜接触/对正进展"，真机和对应仿真帧**共用**
  同一份标签。空闲、不确定、终止阶段不参与。
- **配对相似性**：对同一轨迹、同一帧的 Real/Sim latent，鼓励余弦相似度趋近 1。
  权重只有 0.15。
- **位置辅助目标**：预测当前 peg-tip 相对**终点**的位置。权重 0.12 + 0.05。

> 终点参考是从**仿真末端 EE pose** 推出的终点 tip（见第 7 节），**不是**独立测量的
> 孔中心。

训练协议：40 条轨迹里锁定 8 条做留出测试集（永不参与选 epoch / 定尺度 / 调参）。
训练集内部再做 4 折交叉验证选最佳 epoch 数，然后用全部训练集重训固定 epoch。
当前 artifact 的最终训练 epoch 数 = 9。

## 7. 到底什么变得相似了（诚实结论）

相似的是：**同一轨迹、同一时刻**的真机与仿真输入经过 encoder 后得到的 32 维
latent 的**方向**（余弦相似度）。它**不**表示两条原始力曲线相同，也**不**要求两个
latent 数值逐维相等。

留出测试轨迹上重算的配对相似度：

| 比较对象 | latent 余弦相似度中位数 |
|---|---:|
| 同轨迹、同帧的 Real/Sim 配对窗口 | **0.940** |
| 同轨迹、同阶段，错开 ≥5 帧 | 0.912 |
| 不同轨迹、同阶段 | 0.914 |

配对窗口确实更相似，但比"同阶段的非配对窗口"只高约 0.02–0.03。
**换句话说：latent 学到的主要是共同的"插入阶段"信息；现有数字还不足以证明它
精确对齐了每段独有的力变化细节。**

## 8. 实际结果（留出 8 条轨迹、370 对窗口）

留出轨迹：`[0, 2, 21, 22, 23, 33, 34, 39]`；训练窗口 1692，测试窗口 370。

| 指标 | Real | Sim | 线性力基线 Real/Sim |
|---|---:|---:|---:|
| 分类 Accuracy | 0.895 | 0.849 | 0.778 / 0.603 |
| 分类 Macro-F1 | **0.865** | **0.814** | 0.767 / 0.432 |
| `misaligned_contact` F1 | 0.928 | 0.894 | 0.819 / 0.743 |
| `aligned_progress` F1 | 0.802 | 0.733 | 0.715 / 0.120 |
| `aligned_progress` recall | 0.693 | 0.675 | — |
| 位置方向余弦 | 0.134 | 0.064 | 0.248 / 0.075 |
| 位置方向 within-90° | 0.538 | 0.555 | 0.689 / 0.555 |
| 位置 XY MAE (mm) | 1.53 | 1.96 | 1.29 / 1.35 |

两域分类的留出一致率约 **83.5%**。

三点结论：

1. **分类能力是真实的，且明显优于线性基线**，尤其 Sim 侧（0.814 vs 0.432）。
   两域主要错误模式一致：倾向把 aligned 帧漏判成 misaligned（recall 仅 0.69/0.68），
   即"保守地喊歪斜"。
2. **配对相似性主要体现共同阶段，而非精确对齐**（见第 7 节）。
3. **位置头基本不 work**：方向余弦 0.134/0.064，within-90° 近似抛硬币；
   在 Real 方向上线性基线（0.248）反而更好。这强烈提示：**从三轴力里读出插入
   偏差方向这件事，信号本身可能就不存在**，不是网络容量问题。加上终点参考来自
   共享的仿真 pose、不是独立 metrology，这一项的结论要特别保守。

## 9. 可视化

两个脚本都写同一个可替换路径
`eval_outputs/insertion_alignment_encoder/paired_latent_inspection.pdf`
（后跑的覆盖先跑的），各 2 页。

**[`visualize_decisions.py`](../scripts/insertion_alignment_encoder/visualize_decisions.py)**（决策视角）：
- 第 1 页（单条留出轨迹）：横向力 trend（实线 Real / 虚线 Sim）、`P(aligned)` 概率
  曲线 + 0.5 阈值、底部三行色带 `Label / Real预测 / Sim预测`，错判处打黑叉。
- 第 2 页：Real/Sim 混淆矩阵 + `Real P(aligned)` vs `Sim P(aligned)` 散点图，
  标注决策一致率 83.5%。左下（都判 misaligned）聚得好；有一批点落在"真机已判
  aligned、仿真还没"的下三角，直观体现仿真对正识别偏弱。

```bash
cd /data/workspace/gujiawei/TAVLA
CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python \
  -m insertion_alignment_encoder.visualize_decisions --trajectory 0
```

**[`visualize_pairs.py`](../scripts/insertion_alignment_encoder/visualize_pairs.py)**（表示层视角）：
- 第 1 页：多了 **32 通道 latent 热力图**（Real vs Sim 并排）和逐帧 cosine 曲线。
  肉眼可见两域通道模式整体相似但不逐维相等。
- 第 2 页：三组相似度箱线图（配对 / 同轨错帧 / 跨轨同阶段）+ 每条轨迹 median[IQR]。
  三组几乎重叠；图标题即"cosine alone does not show useful control information"
  ——这张图是在诚实展示配对相似性的局限。

```bash
cd /data/workspace/gujiawei/TAVLA
CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python \
  -m insertion_alignment_encoder.visualize_pairs --trajectory 0
```

## 10. 训练与推理

**旧权重当时使用的训练方式**（physical GPU 0；当前代码配方已更新）：

```bash
cd /data/workspace/gujiawei/TAVLA
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts .venv/bin/python \
  -m insertion_alignment_encoder.train_terminal_geometry
```

旧代码每次运行覆盖 `eval_outputs/insertion_alignment_encoder/encoder.pt`。
当前代码默认写入 `eval_outputs/insertion_alignment_encoder_full_insertion/encoder.pt`，
若输出已经存在则报错；建议使用第 12 节的启动脚本生成独立实验目录。

**推理适配器** [`runtime.py`](../scripts/insertion_alignment_encoder/runtime.py)：
`ForceAlignmentRuntime` 每个 10 Hz 采样调一次 `update(force_xyz)`，内部先攒满 20 帧
基线和 32 帧历史，之后每帧返回 `latent`、两个类别概率、以及位置估计
（`geometry_supervision=true` 时才非空）。不改任何现有 TAVLA policy/checkpoint 代码。

## 11. 文件清单

| 文件 | 作用 |
|---|---|
| [`model.py`](../scripts/insertion_alignment_encoder/model.py) | `ForceAlignmentEncoder` 结构 |
| [`preprocess.py`](../scripts/insertion_alignment_encoder/preprocess.py) | 基线扣除 / 变化率 / 归一化 |
| [`train.py`](../scripts/insertion_alignment_encoder/train.py) | 共享数据/窗口/指标工具 + 旧版（geometry-disabled）训练入口 |
| [`train_terminal_geometry.py`](../scripts/insertion_alignment_encoder/train_terminal_geometry.py) | **当前 artifact 的训练入口**（终点参考 + geometry 启用） |
| [`runtime.py`](../scripts/insertion_alignment_encoder/runtime.py) | TAVLA 侧逐帧推理适配器 |
| [`visualize_decisions.py`](../scripts/insertion_alignment_encoder/visualize_decisions.py) | 决策/混淆矩阵可视化 |
| [`visualize_pairs.py`](../scripts/insertion_alignment_encoder/visualize_pairs.py) | latent 热力图/相似度箱线图可视化 |

> `train.py` 仍被保留：新 `train_terminal_geometry.py` 从中复用 `Episode`、
> `PairWindows`、`LABEL_IDS`、`RECONSTRUCTED_CUTOFFS`、`classification_metrics`、
> `grouped_folds`、`read_labels`；末段窗口和尺度拟合在新入口中实现。它自带的
> `main()` 是旧版（`tavla_force_alignment_encoder_v1`，geometry 关闭）训练入口，
> 当前 artifact 不再用它，但模块本身是必须的，不能删。

## 12. 包含最后插入阶段的重训（2026-10-10）

### 12.1 新配方修复什么

入口仍为 `scripts/insertion_alignment_encoder/train_terminal_geometry.py`，
新配方在 artifact 的 `config.training_recipe` 中记为 `full_insertion_v4`。
此前 `full_insertion_v2` 已补齐末段覆盖，但一次运行最终只训练 1 epoch，
仿真分类准确率 38.1%。它的混合采样和选轮数逻辑已被下述方式替换。

| 项目 | 旧配方 | 新配方 |
| --- | --- | --- |
| 首次进入 XY<3 mm、\|Z\|<3 mm | 截断后续训练帧 | 仅记录诊断标记，继续训练到轨迹末尾 |
| 接触/插入的阶段边界 | 同时排除分类和位置监督 | 仅屏蔽分类标签，仍参与位置监督 |
| `terminal` 阶段 | 排除 | 参与位置监督与配对对齐，屏蔽二分类标签 |
| `uncertain` 阶段 | 排除 | 参与由 pose 生成的位置监督与配对对齐，屏蔽二分类标签 |
| `free` 阶段 | 不参与窗口训练 | 不参与窗口训练，可出现在后续因果窗口的历史中 |
| 归一化尺度 | 训练轨迹的截断前力特征 | 训练轨迹的全部力特征，包括末段；两域共用尺度 |
| 采样 | 按歪斜/对正类别均衡 | 分类与几何分别采样；分类还减弱长轨迹占比，置信度仅用于分类损失 |
| 内部交叉验证选 epoch | 分类 F1 与 XY 方向 | 先满足两域的分类下限，再结合末段 Z 误差选共同 epoch |
| 训练过程 | 验证早停，只保存选中结果 | 每折跑完整预算，逐轮保存损失、验证指标、实际更新数和 CV 权重 |
| 输出 | 覆盖旧 encoder | 新目录保存，已有输出拒绝覆盖 |

位置监督覆盖所有非 `free` 帧；人工阶段标签的可靠性只决定能否参与二分类。
`terminal` 不会被改标成 `aligned_progress`，避免把“已经完成并保持”教成“继续插入”。
阶段边界附近的帧虽然不参加分类，仍参加几何和配对损失。

几何采样和报告分组固定采用 `XY<3 mm` 下的 `0–3 / 3–5 / 5–8 / 8–10 mm`，
以及其余帧。每个非空深度组内再均衡阶段与轨迹，避免长时间终点保持挤占
仍在插入的样本。分类批只从经过边界保护的两个类别中采样，各类别概率为 50%。
**几何批中的样本不计算分类损失**，避免末段过采样改变分类的类别比例。
两批均参与 Real/Sim 配对损失，只有几何批计算位置/方向损失。
这里的距离参考仍然是**示教终点 tip**，不是独立测得的孔底。
如果标签把处于最后 3 mm 的帧标成 `free`，入口会报错，要求先审查标签。

损失权重继续使用 `class=1.0 / consistency=0.15 / geometry=0.12 / direction=0.05`。
分类仅对未屏蔽的 `misaligned_contact`、`aligned_progress` 帧计算。
几何损失对全部入选帧计算，避免末段因分类标签被屏蔽而失去监督。

每折完整训练 100 epoch，不提前退出。训练轨迹内部的四折验证在**同一个 epoch**
汇总：分类用累计混淆矩阵计算，几何误差按组内帧数汇总。选择条件为：

- 默认从第 1 epoch 开始比较（`MIN_EPOCHS=1`）；预热期间也可参与选择。
- Real、Sim 各自的分类 Macro-F1 ≥0.70。
- Real、Sim 的两个类别各自 recall ≥0.60。
- 有末段深度评估样本。

这些下限用于拒绝明显的分类退化，是可配置的初始标准，不是已验证的最优参数。
若所有 epoch 都不满足，保存状态为 `rejected_cv` 的 `encoder.report.json`，
打印最佳诊断轮次和各项未达标指标，以退出码 2 结束，**不生成最终 encoder 权重**。
CV 的每轮权重仍保留，它们使用独立的 artifact 格式，不能直接作为部署 encoder。
对满足条件的 epoch，以如下分数选择共同训练轮数：

```text
0.60 × min(Real_Macro_F1, Sim_Macro_F1)
+ 0.40 × mean(exp(-Real_depth_MAE_mm / 3), exp(-Sim_depth_MAE_mm / 3))
```

选轮数使用的 `selection_depth_mae_mm` 对 `0–3 mm` 与 `3–10 mm` 两部分等权，
每部分内部对非空“阶段×深度组”等权，防止较宽松深度组掩盖最后 3 mm 的问题。
同时保留旧的 `near_terminal_z_mae_mm` 统计作诊断。
几何报告新增实际 Z 均值、预测 Z 均值和预测标准差，便于检查是否只学到了常数预测。
原先锁定的 8 条留出轨迹仍为
`[0, 2, 21, 22, 23, 33, 34, 39]`，不参与模型选轮数或尺度拟合。
32 条训练轨迹内部进行 4 折验证，每折的尺度只用该折训练轨迹计算。
最终在全部 32 条训练轨迹上从头训练到选出的共同 epoch。
学习率前 5 epoch 从低值预热到 0.0003，再按 100 epoch 的预算余弦下降到 0.00003。
最终训练复用四折学习率序列的相同前缀。

`v4` 使用以下方式减少在相邻力窗口上的过拟合：

- 每个 epoch 的更新数按**有分类标签的窗口数**计算；扩大末段几何监督集不会
  再增加同一批分类窗口的重复抽样次数。每次更新仍包含独立的分类、几何两批数据。
- 分类的两类各占 50% 采样概率。每类中，轨迹概率与其窗口数的平方根成正比，
  减少长轨迹占比，同时避免把只有 1 帧的轨迹放大到与长轨迹相同的权重。
- 置信度仅乘在分类损失上，避免同时用于抽样与损失。
- 默认 AdamW `weight_decay=0.01`、时序层 `dropout=0.20`、分类 `label_smoothing=0.05`。
  两域各自的力幅值增益增强为 0.85–1.15。
- Dropout 的参数化保持已有权重的参数名称和形状，推理 `eval()` 时不生效。

这些参数是根据上一轮过拟合曲线调整的起点，尚未通过新一轮 GPU 训练确认效果。
32 条轨迹并不会因为生成数千个重叠窗口就变成数千个独立样本；训练时长也不等于质量。

### 12.2 一键训练 encoder

在 GPU 主机执行，`GPU_ID` 改成当时空闲的物理卡编号：

```bash
cd /data/workspace/gujiawei/TAVLA
GPU_ID=0 bash scripts/run_insertion_alignment_encoder_full_insertion.sh
```

只训练 encoder。默认保留 **32 帧、XYZ 力＋变化率 6 通道、32 维 latent**。
这里 `BATCH_SIZE=128` 表示**每项任务每次更新采样 128 对窗口**：
128 对分类＋128 对几何，共 256 对、512 个 Real/Sim 窗口。
与 TAVLA 的 `batch_size=8` 是不同的训练参数。
默认四折各训练完整 100 epoch，之后再进行最终训练。可通过环境变量调整：

```bash
GPU_ID=0 EPOCHS=100 MIN_EPOCHS=1 BATCH_SIZE=128 \
  bash scripts/run_insertion_alignment_encoder_full_insertion.sh
```

脚本每次创建新目录：

```text
eval_outputs/insertion_alignment_encoder_full_insertion/full_insertion_v4_<时间>_<PID>/
├── encoder.pt               # CV 达标并完成最终训练时生成
├── encoder.report.json      # CV 未达标时也保存，status=rejected_cv
├── train.log
└── encoder_training_history/
    ├── cv_fold_1.jsonl       # 每个 epoch 的损失、学习率、更新数、验证指标
    ├── cv_fold_2.jsonl
    ├── cv_fold_3.jsonl
    ├── cv_fold_4.jsonl
    ├── cv_epoch_summary.json # 四折同 epoch 汇总与资格判断
    ├── cv_fold_1_checkpoints/ # 每轮未部署的 CV 权重，另外三折同样保存
    │   └── epoch_0001.pt
    ├── run_manifest.json     # 本次参数、数据签名、划分和训练归一化尺度
    ├── final_fit.jsonl       # CV 达标后才运行
    ├── train_terminal_geometry.py
    ├── model.py
    ├── preprocess.py
    └── train.py              # 本次启动时使用的源码快照
```

可设置 `RUN_NAME` 或 `OUTPUT_DIR`，已存在的目录会被拒绝使用。
也可设置 `REAL_DIR`、`SIM_DIR`、`LABELS`、`PYTHON_BIN`。
质量下限可设置 `MIN_MACRO_F1`、`MIN_CLASS_RECALL`；学习率可设置
`LEARNING_RATE`、`MIN_LEARNING_RATE`、`WARMUP_EPOCHS`；辅助损失权重可设置
`GEOMETRY_LOSS_WEIGHT`、`CONSISTENCY_LOSS_WEIGHT`。不建议为生成权重而直接降低质量下限。
正则化可设置 `WEIGHT_DECAY`、`DROPOUT`、`LABEL_SMOOTHING`。
调整 `EPOCHS` 时，`WARMUP_EPOCHS` 必须小于它；`MIN_EPOCHS` 与预热轮数独立。

当前网络约 5 万参数、训练集 3260 对几何窗口，实际耗时与之前其他 encoder 配方或
TAVLA 训练不一定相同；请看日志中的实际 optimizer steps、elapsed_s 和学习曲线。

仅查看数据覆盖、无需 GPU 或训练：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=scripts .venv/bin/python \
  -m insertion_alignment_encoder.train_terminal_geometry --prepare-only
```

该模式打印 train/held-out 的窗口数、阶段计数和各深度组计数，不写权重。

### 12.3 训练结束后查看什么

`encoder.report.json` 保存：

- `metrics.window_coverage`：train/held-out 全部位置监督窗口数、分类窗口数、
  每阶段计数、每深度组计数，以及每条轨迹最后 3 mm 的入选帧数。
- `metrics.held_out_test.<real|sim>.geometry_by_depth`：各深度组的 Z/XY MAE。
- `metrics.held_out_test.<real|sim>.geometry_by_phase_and_depth`：区分
  `aligned_progress` 与 `terminal` 等阶段，避免完成后的保持帧掩盖插入中的误差。
- `metrics.linear_force_baseline`：在相同窗口上训练的线性力基线。
- `metrics.inner_folds`、`cv_selected_epoch_metrics` 和 `final_training_epochs`：共同选轮数结果。
- `metrics.final_optimizer_steps`、`total_cv_optimizer_steps`：实际更新次数。
- `metrics.held_out_quality_requirements`：最终留出指标是否达到上述下限，仅作报告，
  不用留出集重新选择 epoch；不满足时日志会明确提示。
- `encoder_training_history/`：完整学习曲线，即使没有合格 epoch 也保留。
- `metrics.best_diagnostic_epoch_*`：CV 未达标时，记录最佳诊断轮次的两域指标、
  失败项和四个折对应的权重路径。诊断权重没有在全部 32 条轨迹上完成最终训练。
- `by_trajectory`：各验证轨迹的分类混淆矩阵和几何误差，帮助定位困难轨迹。
- artifact config：配方、学习率、损失权重、质量下限、模型参数量、代码和标签 SHA256。

新代码修复了训练覆盖缺口；最终成功率仍需重新训练 TAVLA 后在仿真闭环比较。

### 12.4 接回 TAVLA 的顺序

新 artifact 继续使用 `tavla_force_alignment_encoder_terminal_reference_v1` 的文件结构，
另记录 `training_recipe=full_insertion_v4`；现有 `ForceAlignmentLatentEncoder` 能加载。
文件结构兼容并不表示 latent 与旧权重相同。

1. 选定新的 `encoder.pt`，保留旧 encoder 与旧策略作基线。
2. 用该新权重重新转换真机和仿真 latent 数据集，使用**新的 repo-id**。
   现有转换器入口是 `insertion_alignment_encoder.convert_force_alignment_latent`。
3. 为新数据配置独立的 TAVLA 配置、归一化统计目录和实验名，再训练 Cotraining。
4. 仿真侧使用与该新策略训练时**同一份** encoder、归一化尺度和在线预处理。

**不能只替换 8001 仿真客户端的 encoder，继续使用当前 TAVLA checkpoint。**
当前 Cotraining 启动脚本会跳过已存在的 32 维数据集；仅改 `ENCODER_PT` 不会
自动重建旧数据。新 repo-id 也必须同步进入新 TAVLA 配置，避免训练仍读取旧 latent。

### 12.5 2026-10-10 的 v3 质量拒绝记录

运行 `full_insertion_v3_20261010_205539_3975288` 的四折均完成 100 epoch，
共进行了 7800 次更新。`select_common_epoch` 在检查质量时拒绝生成最终模型，
当时的实现用 `RuntimeError` 报告拒绝。

| 四折共同轮次 | Real Macro-F1 | Sim Macro-F1 | Real 插入召回率 | Sim 插入召回率 |
| --- | --- | --- | --- | --- |
| epoch 5，全曲线最高综合分 | 0.717 | 0.661 | 0.637 | 0.604 |
| epoch 31，原 ≥20 限制下最高综合分 | 0.646 | 0.641 | 0.494 | 0.580 |

100 轮中 Sim 汇总 Macro-F1 的最高值为 0.661；所以仅取消 ≥20 的限制也无法
达到 0.70 的分类门槛。后期分类训练准确率约 100%，验证表现没有相应提升，
说明需要修正过拟合，增加轮数无法自动解决。

该轮日志与曲线保留在原目录，修改前的代码、文档、日志另备份于
`backups/encoder_cv_failure_fix_20261010_210812_676f82/`，包含 SHA256 清单。
v3 没有保存各轮权重，因此不能从这次运行直接恢复 epoch 5 的模型。

## 13. 旧配方与末段补充监督 A/B 入口（2026-10-12 更新）

实施与输出说明见 [`encoder_retraining_plan_20261010.md`](encoder_retraining_plan_20261010.md)。

```bash
cd /data/workspace/gujiawei/TAVLA && GPU_ID=0 SEEDS="42 43 44" bash scripts/run_insertion_alignment_encoder_ab.sh
```

新入口 [`train_ab.py`](../scripts/insertion_alignment_encoder/train_ab.py) 使用旧权重记录的
32 条训练轨迹、8 条留出轨迹和四个内部验证折，A/B 共享原训练尺度及基础随机数流。

- A 保留旧优化、采样、四种损失和首次命中前的训练窗口。
- B 额外抽样终点资格合格的高 / 中置信度 `aligned_progress` / `terminal` 近终点帧，单独计算几何损失。
  新增帧不参与二分类和余弦对齐损失，已完成保持帧不会被改标为继续插入。
- 每折记录完整 60 轮；模拟旧 patience=10，冻结其最佳 epoch。
  A 最终拟合轮数取四折最佳轮数的中位数。B 根据自身曲线、A 校准的分类约束及
  活动插入深度误差选共同轮次；同轮次的 A/B 对照结果单独保存。
- 终点参考失败的轨迹在 B 中屏蔽几何 / 方向损失，原分类与配对任务保留。
  默认三个训练种子，报告均值、标准差及各 seed 的分类约束状态。
- 两组候选、曲线和对照报告均写入新的 `insertion_alignment_encoder_ab` 目录。
  保存候选不代表达到部署要求，需看分类与活动插入分段深度误差，再配套重训策略。

第一轮 v1 已完成；本轮 v2 仅完成代码编辑、数据覆盖核对和静态语法检查，尚未运行训练。
