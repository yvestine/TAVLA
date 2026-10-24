# Motion-only 主分支 + 仿真力修正

冻结现有 motion_only 的两个运动 encoder 和历史/未来 decoder。新增 GRU 只读取过去和当前
32 帧的归一化 Fxyz/ΔFxyz，加上 Real/Sim 分支标记；结合冻结的26维运动特征，分别输出最近8帧
和未来5帧的三轴力修正，以及每个时间点的门控权重。修正最后一层初始化为零。

因此 Sim 输入的初始当前力、历史力、未来力和32维特征均等价于原 motion_only；把
`residual_gain` 设为0也能恢复它。Real 输入的32维特征前6维继续使用实测 F/ΔF；
Real 未来预测的基础部分仍来自 motion_only，新增分支也有 Real 力监督，默认损失权重0.25。
训练只更新新增 GRU、修正头和门控，冻结部分训练前后校验 SHA。

## 运行

在项目根目录、外部 GPU 环境中：

```bash
cd /data/workspace/gujiawei/TAVLA
GPU_ID=0 bash scripts/run_force_gap_motion_force_v1.sh
```

建议在已有 force tmux 里执行，避免终端断开：

```bash
tmux attach -t force
cd /data/workspace/gujiawei/TAVLA
GPU_ID=0 bash scripts/run_force_gap_motion_force_v1.sh
```

默认基础目录是已完成的 `eval_outputs/force_gap_forecast_v1/20261012_155026_551478`。
原 encoder 结果不会被覆盖。新输出目录是 `eval_outputs/force_gap_motion_force_v1/时间戳_PID/`。

默认三种子42/43/44、四折、修正最多100 epoch、patience20、batch32、lr=1e-4。
内层复用18条训练轨迹的 motion_only checkpoint，用6条验证轨迹选择 epoch 和全局修正强度
0/0.25/0.5/1；外层复用24条轨迹的 motion_only checkpoint，重新训练修正并评估8条轨迹。
每个 epoch 使用与旧版本相同的轨迹等权事件采样与实际 optimizer 步数定义。
force gain 在验证阶段只调整预测组合，不改变梯度训练；修正训练本身始终 gain=1。

先做一折、一个种子的 GPU 流程检查：

```bash
GPU_ID=0 bash scripts/run_force_gap_motion_force_v1.sh --seeds 43 --folds 0 --max-epochs 3 --patience 2
```

这个缩减命令只生成该折的模型、预测与比较报告，不生成全部32条轨迹的最终 encoder。
单种子完整四折可用 `--seeds 43`；它会生成最终 encoder，但验收标为诊断，完整验收需要三个种子。

恢复时保留第一次日志打印的完整 RUN_DIR，所有训练参数也必须相同：

```bash
GPU_ID=0 RUN_DIR=/absolute/path/to/original_run RESUME=1 \
  bash scripts/run_force_gap_motion_force_v1.sh
```

已完成阶段复用；中断在半个阶段时该阶段重跑。代码、旧 checkpoint、旧评估/选择结果、数据、标签
和训练参数的 SHA/recipe 必须一致。`BASE_RUN=/absolute/path/to/completed_v1_run` 可以指定另一份
原实验；该实验必须有对应种子、四折 motion_only checkpoint、相同数据和 v1 实现。

## 最终模型与评估

四折完成后，为每个种子按旧 motion_only 内层选择轮数的中位数，在全部32条训练轨迹上训练
完整 motion teacher/student（使用原实验的 lr/batch 配方）。冻结这个基础，再按新内层选择
训练最终修正。最终强度取四折内层强度中位数的最近网格值，平局选较小值；最终轮数取使用修正
的内层轮数中位数。没有稳定选中修正时保留 gain=0，不通过额外收益验收。

输出包含 `encoder_seed*.pt`、`manifest.json`、`acceptance.json`、`report.md`、
`outer_metrics.json`、`selections.json`、原始预测 npz、诊断指标、各阶段 `history.json/model.pt/last.pt`。
旧完整模型的汇总保存在 `outer_metrics.json` 的 `previous_full_summaries`，便于比较。
默认自动生成 `analysis_visualizations/`：交互 HTML、整体指标、典型/最好/最差轨迹和单次预测图。
可以用 `--no-plots` 暂时跳过，并在训练后运行：

```bash
.venv/bin/python scripts/visualize_force_gap_forecast_v1.py --run-dir /absolute/path/to/new_run
```

主指标仍是当前与未来归一化误差的均值，轨迹和种子等权。验收要求四折三种子完整、比相同
motion_only 更好、配对轨迹 bootstrap 95% 改善下界>0、相对最佳 raw/affine/ARX 改善≥5%，
当前/未来/接触事件/峰值/变化量/高频误差不恶化>2%，每种子主指标不恶化>2%。
诊断8条轨迹不参与 epoch、gain 和种子选择，而且已被历史实验复用，不是新盲测。
融合模型与 motion_only 在同一设备上重新评估；与旧 GPU 保存指标的复用校验允许0.2%相对
加1e-4绝对数值容差，并在 `selections.json` 记录差异。这不是验收放宽，所有新验收都使用
同设备的运动基准。

当前数据的 Sim q/dq 复制自真机，结果只验证已有配对回放。波形通过且实际状态未验证时为
`replay_validated`；未通过为 `diagnostic`。本版本的模型格式和在线推理器独立于旧 v1，
旧 v1 的策略训练/服务器不能直接加载新权重。

## 推理与检查

输出继续是 Fxyz(3)+ΔFxyz(3)+运动时序特征(26)，32维；未来力是额外 `[5,3]` 输出。
未来预测包含独立的修正头，不能再通过旧 `model.future(z)` 重建，使用新推理器：

```python
from force_gap_motion_force_v1.runtime import MotionForceEncoder

encoder = MotionForceEncoder("/path/to/new_run/manifest.json", domain="sim")
z = encoder.push(force_xyz_n, q_rad, dq_rad_s, timestamp_s)
future_force = encoder.predict_future()  # float32[5,3], 单位N
effort = encoder.effort_history()        # float32[10,32]
```

32帧窗口与前序差分均为因果计算，断档>0.5秒重置，未来监督禁止跨任一侧数据断段。
无启动扣基线、无力通道剪裁。完整 checkpoint 内包含所有冻结 motion 权重，推理不依赖旧实验目录。

CPU 检查：

```bash
PYTHONPATH=scripts .venv/bin/python -m force_gap_motion_force_v1.tests
```
