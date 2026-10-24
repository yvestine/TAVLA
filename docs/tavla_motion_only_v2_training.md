# 冻结 motion_only encoder 的 TAVLA 训练

## 2026-10-12 修正后的推荐入口

先前任务在转换Sim时抛出 `Sim action labels differ from paired Real reference: traj_0`。这是新准备流程额外要求两域动作逐值相同导致的，策略训练尚未启动。已有正常流程分别读取各域 `action/actual`，已恢复该行为。

现在沿用已有 `run_tavla_force_align_latent_cotrain.sh` 和 `convert_force_alignment_latent.py`，只增加 `ENCODER_KIND=motion_only_v2` 的输入分支；归一化仍使用原 `compute_norm_stats.py`，训练仍直接调用原 `scripts/train.py`。旧新建启动文件已改为这一入口的兼容转发。

本机已经转换完成的Real数据可直接复用，执行：

```bash
cd /data/workspace/gujiawei/TAVLA
ENCODER_KIND=motion_only_v2 \
REAL_REPO=local/tavla_motion_only_v2_54612ba4b757_a82c50c9fb76_real \
GPU_IDS=0,1,2,3 BATCH_SIZE=8 \
  bash scripts/run_tavla_force_align_latent_cotrain.sh
```

默认seed44、Sim新缓存 `local/tavla_motion_only_v2_seed44_sim`、Real/Sim50:50、四卡FSDP、每卡batch2、30000步。仅输入encoder分支变化，保留图片、关节状态和每域原有动作标签。转换时仍会打印 `Converting episodes / Compute mean / Compute std`；策略训练开始后才打印训练步数和loss。

标准统计写到 `assets/pi0_lora_user_single_arm_motion_only_v2_cotrain_50_50/<repo>/norm_stats.json`，配置按原来已运行流程合并两域统计。新模式不使用下方旧准备流程的 `POLICY_MANIFEST` 或 `CACHE_TAG`；恢复直接在上面的命令加 `RESUME=1 EXP_NAME=原实验名`，并保持相同repo。

本轮修正没有执行转换、测试或训练。下方是最初接入实现的记录；运行以本节命令为准。

本轮用户要求将最新选中的 motion_only encoder 接入 TAVLA，并给出四张 GPU、global batch=8 的命令。代码已接入；按用户要求，本轮没有执行测试、数据转换、统计计算或训练。

## 训练内容

- 默认 encoder：`eval_outputs/force_gap_motion_force_v2/20261012_215515_838437/encoder_seed44.pt`，对应 `manifest.json`。Real/Sim 增益均为零，使用 motion_only 基础模型。
- encoder 读取截至当前时刻的 32 帧力、q、dq、dt；每帧预计算 32 维 `[current(3), change(3), h(26)]`，保存到新 LeRobot 数据集的 `observation.effort`。
- Real 当前力用实测传感器，Sim 当前力用 motion_only 恢复的真实等效力。未来真实力和未来运动都不是训练输入。
- TAVLA 读取最近 10 帧，effort 为 `[batch,10,32]`。图片、8 维关节/夹爪状态仍使用现有 TAVLA 入口；原始绝对动作在 data transform 中转换为前七关节 delta，夹爪保持绝对值，策略输出再恢复为绝对动作。
- encoder 只在 CPU 转换阶段推理，参数冻结并固化成数据。JAX 策略训练不会更新 encoder。默认 Real/Sim 50:50 混合、Pi0 LoRA、FSDP 四卡、global batch8（每卡2）、30000步。
- 两域共享按50:50精确合并的 mean/std，统计在 action delta 与32维 padding 后计算。没有混用旧 encoder 的数据集或 normalization。
- 时间间隔超过0.5秒时，encoder 重置；转换也切成新的 LeRobot episode，因此 effort 查询和 action chunk 不会跨断段。

新配置：`pi0_lora_user_single_arm_motion_only_v2_cotrain_50_50`。

## 一条命令执行

在 `force` tmux 或你自己的训练会话中执行：

```bash
cd /data/workspace/gujiawei/TAVLA
GPU_IDS=0,1,2,3 BATCH_SIZE=8 \
  bash scripts/run_tavla_motion_only_v2_cotrain.sh
```

四张卡可以改成任意四个不同的可用物理索引。脚本会先在CPU上转换两域数据、生成共享归一化，再启动GPU策略训练。默认不重新训练 encoder。每10000步保存一次，最终保存29999步。

如需固定名称和准备目录，方便查看与恢复：

```bash
GPU_IDS=0,1,2,3 BATCH_SIZE=8 \
EXP_NAME=motion_only_v2_cotrain_run1 \
POLICY_DIR="$PWD/eval_outputs/tavla_motion_only_v2_policy/run1" \
  bash scripts/run_tavla_motion_only_v2_cotrain.sh
```

默认创建新的实验名，不覆盖已有策略。首次数据转换需时间和新的缓存空间；输出日志会显示准备目录、完整缓存位置及 checkpoint 目录。

## 用户可自行执行的检查

CPU接口检查使用合成数据，不运行策略训练：

```bash
PYTHONPATH=scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m force_gap_motion_force_v2.policy_tests
```

检查覆盖离线/流式effort一致性、10帧历史、断段重置、Sim零增益不依赖仿真力、错误权重/缓存/归一化拒绝，以及四卡batch8命令参数。上一轮encoder测试仍可单独运行：

```bash
PYTHONPATH=scripts .venv/bin/python -m force_gap_motion_force_v2.tests
```

只做转换和共享统计，不启动策略训练：

```bash
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD/packages/openpi-client/src"
export LEROBOT_HOME="/data/workspace/gujiawei/.cache/huggingface/lerobot"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu .venv/bin/python \
  -m force_gap_motion_force_v2.policy_prepare \
  --output-dir "$PWD/eval_outputs/tavla_motion_only_v2_policy/prepared_run1"
```

然后用已有准备结果训练：

```bash
GPU_IDS=0,1,2,3 BATCH_SIZE=8 \
POLICY_MANIFEST="$PWD/eval_outputs/tavla_motion_only_v2_policy/prepared_run1/policy_manifest.json" \
EXP_NAME=motion_only_v2_cotrain_run1 \
  bash scripts/run_tavla_motion_only_v2_cotrain.sh
```

## 恢复训练

使用原来准备目录的 policy_manifest 和原 EXP_NAME，其他训练参数保持相同：

```bash
GPU_IDS=0,1,2,3 BATCH_SIZE=8 RESUME=1 \
POLICY_MANIFEST=/绝对路径/原准备目录/policy_manifest.json \
EXP_NAME=原实验名 \
  bash scripts/run_tavla_motion_only_v2_cotrain.sh
```

可选环境变量：`TRAIN_STEPS`、`SAVE_INTERVAL`、`LOG_INTERVAL`、`NUM_WORKERS`、`MANIFEST`、`POLICY_DIR`、`CACHE_TAG`。恢复时训练参数与源码必须和原 recipe 一致；可以换四张可用卡。

特征缓存名同时包含encoder SHA和数据/转换代码身份，相同完整导出可复用。中断留下的不完整缓存拒绝复用，不自动删除。需要重新转换时，指定新的 `CACHE_TAG=retry1` 和新的 `POLICY_DIR`。

## 输出与在线使用

- 日志：`logs/<EXP_NAME>.log`，含准备与训练阶段。
- 准备及共享统计：`eval_outputs/tavla_motion_only_v2_policy/<实验名>/`，包括 `policy_manifest.json`、`assets/shared/norm_stats.json`、训练命令及 recipe。
- checkpoint：`checkpoints/pi0_lora_user_single_arm_motion_only_v2_cotrain_50_50/<EXP_NAME>/29999/`。
- 完整训练结束后，保存的各 checkpoint 的 `assets/` 附带 `motion_only_v2_policy_manifest.json`、冻结的 `motion_only_v2_encoder.pt`、可在本地加载的 `motion_only_v2_encoder_manifest.json` 和原评估证据。

在线需要用 `force_gap_motion_force_v2.runtime.JointMotionForceEncoder`，每个新观测调用 `push(F,q,dq,time)`，再将 `effort_history()` 输出的 float32 `[10,32]` 提供给策略。每次新轨迹调用 `reset()`；时间断段自动重置。旧 force-alignment latent 的服务/客户端不能当作这个 encoder 使用。本文新增的是训练入口，没有自动启动或替换旧服务器。

本次按用户明确要求进行已有回放数据的策略实验。原 encoder 的 diagnostic 状态、未通过融合验收和 `actual_sim_state_verified=false` 都保留；新的策略合同标记为 `offline_replay_experiment`。这不会把复制的 Sim q/dq 变成实测仿真状态，最终控制效果仍需新的策略评估。
