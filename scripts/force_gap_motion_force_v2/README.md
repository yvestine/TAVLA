# Motion + force joint encoder v2

本版针对 v1 的三个问题：力只修补输出而没有改变隐状态，修正后的当前力没有进入未来预测头，以及 Real/Sim 共用一个增益导致 Sim 关闭时 Real 传感器也失去作用。

已新增选定 motion_only 权重的 TAVLA 训练入口（四卡、global batch8、Real/Sim50:50）；完整说明和恢复命令见 `docs/tavla_motion_only_v2_training.md`：

```bash
ENCODER_KIND=motion_only_v2 GPU_IDS=0,1,2,3 BATCH_SIZE=8 \
  bash scripts/run_tavla_force_align_latent_cotrain.sh
```

代码已经实现；本轮按用户要求没有执行 Python、测试或训练。以下命令由用户在 GPU 环境中执行，结果是否改善需要看新实验。

## 结构

输入仍为截至 t 的 32 帧 `[F, ΔF, q, dq, dt]`，输出未来 1–5 帧的真实等效力。所有历史窗口只读取 t 及之前的信息。未来观测到的 q/dq/action、未来真值和阶段标签都不是模型输入。

1. 不变的 motion_only 编码器产生 `h_motion(26)` 和原来的力历史/未来预测。旧权重、归一化、训练轨迹必须与当前数据匹配。
2. 两套独立的 Real/Sim GRU 读取力和运动历史。一个独立的运动先验头估计该域当前观测力，形成 `innovation = observed_domain_force - expected_domain_force`。这里的仿真先验和仿真力都使用 Sim 的归一化，避免直接拿仿真力减去真实等效力。
3. 适配器给出隐状态增量和力历史增量：`h_joint = h_motion + gain_domain * gated_delta_h`。Sim 当前力取修正后历史的末项；Real 当前力始终取真实传感器。
4. 未来解码器直接读取 `[corrected_current, corrected_change, h_joint]`。Real/Sim 分别复制旧解码器。先训练适配器 5 个 epoch，再以较小学习率解冻解码器副本，原 motion_only 编码器/解码器始终冻结。
5. 最终未来预测在原预测与联合预测间按本域增益混合。Sim gain=0 严格回到 motion_only；Real gain=0 仍把真实实测当前力及变化传给原未来解码器。

保留 32 维接口 `[current(3), change(3), h_joint(26)]`。新权重格式为 `tavla_motion_force_joint_v2`，流式入口为 `force_gap_motion_force_v2.runtime.JointMotionForceEncoder`。由于输出混合需要运动先验，未来预测使用该 runtime 的 `predict_future()`；不能用旧 v1 decoder 解码此 latent。这里完成的是 encoder 的训练、评估及流式推理，不包含策略服务器的自动替换。

## 对照与选择

- `motion_only`：完全相同的冻结基础权重。Sim 为原 motion_only，Real 为原 motion decoder 加真实当前力/变化。
- `motion_control`：与 full 相同参数量、初始化、优化器、训练数据和力先验辅助监督；屏蔽力历史与 innovation。运动先验头单独连接冻结的 motion latent，其监督不会进入此对照的预测路径。
- `full`：包含实际力历史与 innovation 的联合模型。
- Real 额外提供 `sensor_hold`：以 t 时刻实测力作为五步预测。

沿用 4 折 × 3 种子。外层 24 条训练/8 条评估，内层 18 条训练/6 条选择。每域独立选择最佳权重、epoch 和 gain，外层按内层结果重训。归一化、期望力先验、阈值都使用训练折。8 条已复用诊断轨迹不参与选择。

内层主要指标为 `forecast_score_n = .25*RMSE(+1) + .35*RMSE(+3) + .40*RMSE(+5)`，按真实物理 N 计算，不含当前力误差。默认至少改善 1%，并限制当前/未来/Fz/峰值/事件/峰值时间指标回退超过 2%。零增益保留为候选，若没有证据支持改善就返回基础模型。

训练主要优化未来误差，增加 Fz 权重、五步未来 Fz 峰值幅度、软峰值位置、未来变化损失；当前/历史重建为较小的辅助项。训练的峰值位置用步数，评估峰值时间用实际时间戳。短窗口事件指标噪声较大，因此同时报告原始 RMSE。

外层通过需要：相对 motion_only **及** motion_control 的未来指标都至少改善 2%，配对轨迹 bootstrap 95% 改善区间下界 >0，并满足未来、Fz、接触阶段等回退约束。旧 `score` 只为兼容以前的表和图保存，不再决定 v2 是否通过。通过也只支持对应数据来源；未验证真实 Sim 状态时为 replay_validated，失败为 diagnostic。

## 执行命令

进入工程；如需保留任务在 force 会话，先执行 `tmux new-session -A -s force`，再在该会话执行下面命令。

```bash
cd /data/workspace/gujiawei/TAVLA
PYTHONPATH=scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m force_gap_motion_force_v2.tests
```

推荐先做单折单种子短流程，检查数据、内层选择、外层重训及指标文件：

```bash
GPU_ID=0 bash scripts/run_force_gap_motion_force_v2.sh \
  --seeds 42 --folds 0 --max-epochs 6 --warmup-epochs 2 --no-plots
```

短流程只验证通路，不生成最终 encoder，也不作为效果结论。正式实验用新目录：

```bash
GPU_ID=0 bash scripts/run_force_gap_motion_force_v2.sh
```

默认读取原 motion_only 实验 `eval_outputs/force_gap_forecast_v1/20261012_155026_551478`。当数据/标签完全相同，复用其逐折 motion checkpoint；full 和同容量对照均重新训练。不会复用上一版增益为零的 residual。输出自动建在 `eval_outputs/force_gap_motion_force_v2/<时间_进程号>/`。

恢复中断需要使用原目录和完全相同的参数/代码/数据：

```bash
GPU_ID=0 RESUME=1 RUN_DIR=/绝对路径/原v2实验目录 \
  bash scripts/run_force_gap_motion_force_v2.sh
```

已经完成的 stage 会复用；正在执行而未保存完成的 stage 会从头重训，不声称按 optimizer step 断点续训。

若需要显式固定目录便于观察：

```bash
GPU_ID=0 RUN_DIR="$PWD/eval_outputs/force_gap_motion_force_v2/joint_v2_run1" \
  bash scripts/run_force_gap_motion_force_v2.sh
```

不要把短流程目录作为正式实验的 resume 目录；协议参数不同会拒绝恢复。

## 使用真实仿真状态数据

现有 `sim-data-aligned` 的 q/dq 是 Real 的复制，架构变化不会修复这点，也不会自动纠正力幅值/方向/物理定义。当前数据可用于结构实验。

新的外部仿真回放应调用已有 `force_gap_forecast_v1.capture.PairedReplayRecorder`，每个参考帧同步记录真实 **Sim** q/dq、Sim 力、Sim 时间和 `real_frame_id`，声明 force contract，引用匹配的 Real HDF5 SHA。不能把 Real q/dq 重命名为 Sim 观测；也不能离线拟合测试轨迹的时间偏移。该项目里没有外部 Isaac replay 驱动，此版不声称已经采集新数据。

40 条新配对轨迹完成后，指定新路径；归一化和全部 motion 基础权重都会重新训练，不混用旧数据的冻结 checkpoint：

```bash
SIM_DIR=/绝对路径/实际仿真回放目录 GPU_ID=0 \
  bash scripts/run_force_gap_motion_force_v2.sh \
  --base-mode fresh --require-actual-sim-state
```

目录仍需为 `traj_0/data.h5` 至 `traj_39/data.h5`，同一参考动作回放每帧一条同步观测。`--base-mode auto` 在数据变化时也会选择 fresh；显式 fresh 更便于审计。`--require-actual-sim-state` 拒绝复制/未验证的仿真状态。此模式的基础模型训练时间会显著多于复用模式。

## 看结果

正式完整实验结束后会自动生成：

- `report.md`：Sim/Real 的未来指标、每折两域 epoch/gain、验收原因。
- `outer_metrics.json` / `real_outer_metrics.json`：三种模型折外指标，以及 Real 实测力保持对照。
- `force_input_audit.json`：将 Sim 力替换为训练折平均力后的隐状态、当前力、未来力响应量。这是敏感性诊断，常值输入可能偏离训练分布，不能独立证明仿真力有用/无用。
- `analysis_visualizations/force_prediction_explorer.html`：离线交互图，可选择 Sim 折外、Real 折外或最终模型诊断；同时看 motion_only 和同容量对照。
- `analysis_visualizations/01_metrics_overview.png`：步长、轴、轨迹和阶段图，包含之前漏掉的 `misaligned_contact`。
- `manifest.json` / `encoder_seed*.pt`：最终选择及可移植权重，即使未通过也保留 diagnostic 权重和指标。

图未生成或使用 `--no-plots` 后可以自行执行：

```bash
PYTHONPATH=scripts .venv/bin/python scripts/visualize_force_gap_forecast_v1.py \
  --run-dir /绝对路径/完整v2实验目录
```

判断力路径是否有价值，先比较 full 与 motion_control 的折外未来误差，再看 force_input_audit 的未来/latent 响应，最后看接触阶段和单次五步预测。不要用 Real 当前力 RMSE=0 或诊断轨迹来证明预测改善。
