# 通用仿真/真机六维力对齐模块

本文档只针对 `TA-VLA` 仓库。模块不修改、不启动仿真仓库。

## 结论先说

当前 50 条仿真轨迹和 40 条真机轨迹可以证明存在明显的域差异，但不能组成
PolyFit 所需的监督配对集：它们不是同一时刻、同一接触姿态下的仿真/真机观测。
因此不能把两套独立轨迹逐行拼在一起训练 MLP，否则会把任务阶段差异误学成力映射，
也无法保证换任务后仍然有效。

仓库现在提供三种明确区分的方法：

1. `fit_unpaired_wrench_affine.py`：只用未配对数据估计每个通道的稳健位置/尺度变换，
   作为基线和诊断工具，不称为 PolyFit DLA。
2. `train_wrench_adapter.py`：真正的 PolyFit 风格 DLA。它要求每一行的仿真和真机
   wrench 来自同一接触状态，并按 episode 划分 train/validation/test，带有有界残差、
   dropout、输入扰动、weight decay、梯度裁剪和早停。
3. `train_unpaired_wrench_adapter.py`：当前场景使用的 PolyFit-inspired 无配对版本。
   它保留 PolyFit 的力/力矩分支 MLP，但训练目标改为无配对的分布匹配、协方差匹配、
   MMD 和时序差分统计匹配；不把不同轨迹的帧伪装成监督标签。

默认推荐方向是 `sim_to_real`：把仿真 wrench 变换到现有真机 TAVLA checkpoint 的
输入域，真机数据本身不变。若要复现论文部署方式，可使用 `real_to_sim`。

## 当前数据调研结论

当前字段已经统一为：

```text
[Fx, Fy, Fz, Tx, Ty, Tz]
力单位 N，力矩单位 N*m
真机：obs/state/ee_wrench_base
仿真：wrench_final = -wrench_base
```

已检查的结果保存在：

```text
eval_outputs/wrench_alignment/wrench_alignment.png
eval_outputs/wrench_alignment/summary.json
eval_outputs/wrench_alignment/contact_event_analysis.png
eval_outputs/wrench_alignment/contact_event_summary.json
eval_outputs/wrench_alignment/episode_event_statistics.csv
```

当前统计显示仿真接触力明显偏小且噪声形态不同：仿真力范数中位数约 `0.27 N`、
P95 约 `1.01 N`，真机约为 `3.53 N`、`6.89 N`；仿真和真机的峰值也不在同一时间
阶段。视频抽查进一步显示两边轨迹时长和接触阶段并不一致。因此仅做整体取反已经
解决符号链路，但没有解决 sim-real 的幅值、偏置、噪声和接触状态差异。

在当前数据上运行无配对 MLP 后，独立 episode 的分布指标没有稳定超过稳健仿射基线，
而且出现了力范数过冲、力矩范数偏低。因此当前默认推荐先使用稳健仿射结果；无配对
MLP 作为可复用候选和后续数据增加后的实验入口保留，不把它自动作为最终训练输入。

## PolyFit 方法在这里如何落地

论文的 DLA 使用真实 F/T 作为输入、仿真 F/T 作为监督目标；真实和仿真样本必须是
配对的。当前 TAVLA 采用 `sim_to_real` 时，将方向反过来：仿真 F/T 作为输入，真机
F/T 作为目标。模型结构仍保留 PolyFit 的核心思想：力和力矩分别编码，经过融合后
分别预测三个力和三个力矩。

为降低过拟合风险，代码增加了以下约束：

- 只使用六维 wrench，不使用 `peg-in-hole`、任务名或手工接触阶段作为输入；
- 统计量只从训练 episode 计算，验证和测试按完整 episode 隔离；
- 输出是稳健标准化空间中的有界残差，初始状态等价于稳健位置/尺度基线；
- 使用 dropout、weight decay、少量输入噪声、梯度裁剪和 validation early stopping；
- 保存 best 和 last，并报告独立 held-out test 的每通道 MAE、P95 和最大误差。

这使同一个模块可以用于其他任务；换任务时只需提供新的、真实配对的 sim-real wrench，
不需要改网络结构，也不把任务名写进映射模型。

## 配对 HDF5 格式

需要一个 HDF5 文件，最小字段如下：

```text
real_wrench    float32 [N, 6]
sim_wrench     float32 [N, 6]
episode_index  int64   [N]
```

第 `i` 行的两个 wrench 必须对应同一个接触状态；`episode_index` 只用于防止同一条
轨迹的相邻帧泄漏到验证集。可以额外保存 `timestamp`、位姿、task 和配对置信度，但
当前适配器不会把这些任务特定字段作为输入。

## 未配对基线命令

这一步不需要 GPU，使用当前已有数据只能得到一个“分布校准基线”：

```bash
cd TA-VLA
.venv/bin/python scripts/fit_unpaired_wrench_affine.py \
  --source-dir data-sim-wrench-final-50 \
  --source-pattern 'episode_*/wrench_final.csv' \
  --target-dir data \
  --target-pattern 'traj_*/data.h5' \
  --target-h5-key obs/state/ee_wrench_base \
  --output checkpoints/wrench_adapters/unpaired_sim_to_real_affine.pt
```

生成的 `unpaired_affine_report.json` 只能说明边缘分布的中位数/尺度是否靠近，不能
说明逐帧力是否正确。没有经过闭环和配对验证前，不建议用该基线替换训练数据。

## PolyFit DLA 训练命令

有真实配对 HDF5 后，在 CPU 上即可训练这个小模块；不需要占用 TAVLA 的 GPU：

```bash
cd TA-VLA
.venv/bin/python scripts/train_wrench_adapter.py \
  --paired-hdf5 data/wrench_paired.h5 \
  --output-dir checkpoints/wrench_adapters/sim_to_real_polyfit \
  --direction sim_to_real \
  --steps 5000 \
  --device cpu
```

脚本默认从 `sim_wrench` 映射到 `real_wrench`。如果要使用论文的原始方向：

```bash
.venv/bin/python scripts/train_wrench_adapter.py \
  --paired-hdf5 data/wrench_paired.h5 \
  --output-dir checkpoints/wrench_adapters/real_to_sim_polyfit \
  --direction real_to_sim \
  --steps 5000 \
  --device cpu
```

只有当 `metrics.json` 中 held-out test 的误差稳定、且不会在不同 episode 上明显恶化
时，才应把 `best.pt` 用于离线数据转换。适配器的输入必须与其 `direction` 一致，
不能在仿真侧转换一次后又在 TAVLA Server 中转换第二次。

## 当前没有严格配对时的 PolyFit-inspired 命令

这条命令可以直接使用现有 50 条仿真和 40 条真机数据。它不需要 GPU；如果 CPU 较忙，
可以降低 `--steps`。它按 episode 留出独立验证和测试轨迹：

```bash
.venv/bin/python scripts/train_unpaired_wrench_adapter.py \
  --source-dir data-sim-wrench-final-50 \
  --source-pattern 'episode_*/wrench_final.csv' \
  --target-dir data \
  --target-pattern 'traj_*/data.h5' \
  --target-h5-key obs/state/ee_wrench_base \
  --output-dir checkpoints/wrench_adapters/unpaired_polyfit_dla \
  --direction sim_to_real \
  --steps 2000 \
  --eval-every 100 \
  --patience 8 \
  --device cpu
```

输出的 `metrics.json` 中要重点比较：

```text
held_out_source_before
held_out_source_after
```

只有 `alignment_score` 在多个随机种子和 held-out episode 上都改善，并且力/力矩范数
没有过冲，才考虑使用 `best.pt`。否则使用：

```text
checkpoints/wrench_adapters/unpaired_sim_to_real_affine.pt
```

这种选择是“先用简单模型作为保守基线，复杂模型必须通过独立数据证明有效”，不是把
MLP 训练 loss 下降误认为物理对齐成功。

## 离线接入现有仿真 HDF5

例如，将 `wrench_final` 经过 `sim_to_real` 适配后导出新的 LeRobot 数据集：

```bash
.venv/bin/python scripts/convert_sim_hdf5_to_lerobot.py \
  --source-dir data-sim-wrench-final-hdf5 \
  --repo-id local/tavla_single_arm_ee_wrench_sim_polyfit \
  --effort-key wrench_final \
  --wrench-adapter checkpoints/wrench_adapters/sim_to_real_polyfit/best.pt \
  --overwrite
```

实时仿真客户端也应在构造 WebSocket payload 之前执行同一变换，然后发送：

```python
effort = adapter.transform_numpy(wrench_final[None, :])[0].astype(np.float32)
client.infer({..., "effort": effort[None, :], "prompt": task_name})
```

TAVLA Server 不应再次旋转、取反、归一化或应用适配器；它只使用 checkpoint 自带的
`norm_stats`。真机侧若采用 `sim_to_real`，继续发送原始真机
`obs/state/ee_wrench_base`，不要再经过这个仿真到真实域的适配器。

## 何时可以认为方法有效

最低验收顺序是：

1. 配对 HDF5 的字段、单位、坐标系和时间戳审计通过；
2. 按 episode 隔离的 validation/test MAE 均低于未适配基线；
3. 在未参与训练的任务或形状上，分布统计和时序峰值没有明显恶化；
4. 离线 TAVLA 动作误差没有因为适配器增加；
5. 再进行少量仿真闭环，最后才考虑联合微调。

当前数据可以立即完成第 1、2 类分布诊断，但缺少第 3 类“同一接触状态”的监督配对，
所以不能声称已经完成 PolyFit DLA 训练。
