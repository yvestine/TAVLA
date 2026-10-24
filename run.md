# TA-VLA 运行记录与入口

> 第 1–7 节保留历史 LoRA 命令，其中部分命令带 `--overwrite`，且使用已存在的实验名。
> 查看历史结果可以直接阅读；要复跑策略训练，请先指定新的实验名，避免覆盖
> `checkpoints/` 中已训练的 TAVLA/pi0 权重。新版仿真力 Encoder 命令见文末。


## 已添加内容

- `scripts/audit_user_hdf5.py`：检查原始数据。
- `scripts/convert_user_hdf5_to_lerobot.py`：转成 LeRobot 数据集。
- `pi0_lora_user_single_arm_effort`：单臂力感知 LoRA 配置。
- `pi0_lora_user_single_arm_ee_wrench`：单臂末端六维力 LoRA 配置。
- `docs/franka_ee_wrench_deployment.md`：Franka 末端六维力部署说明。
- `docs/tavla_ee_wrench_finetuning_report.md`：末端六维力 LoRA 微调与数据说明。

配置要点：

- 原始 action/state 是 8 维：7 关节 + 1 夹爪
- 模型内部 `action_dim=32`：保持和 `pi0_base` checkpoint 一致，输入会自动 pad 到 32
- 策略输出 `action_output_dim=8`：部署输出时只取前 8 维
- `action_horizon=50`：训练时自动取 50 步 action chunk
- `effort_dim=7`：关节力矩
- `effort_type=EXPERT`
- 默认使用 `obs/state/joint_torque_external` 作为 `observation.effort`
- 默认 `--action-mode actual`：保留原始 `action/actual`
- 只有一个腕部相机，所以转换时会把 `wrist_camera.mp4` 同时写入左右 wrist image key

说明：LeRobot 文件里每帧只存一个 `action`。训练时 `src/openpi/training/data_loader.py` 会根据 `action_horizon=50`
自动读取连续 50 帧，组成 `actions` chunk。也就是样本 `t` 的 label 约等于：

```text
actions[t] = [action[t], action[t+1], ..., action[t+49]]
```

在这批数据里 `action[t] == state[t]`，所以经过 delta-action 转换后大致是：

```text
delta_actions[t] = [0, state[t+1]-state[t], ..., state[t+49]-state[t]]
```

## 1. 进入环境

```bash
cd TA-VLA
uv sync --python 3.11
source .venv/bin/activate
python scripts/download_base_checkpoint.py
```

新服务器的完整安装和迁移说明见 [`docs/migration.md`](docs/migration.md)。不要把 `/workspace/gujiawei` 等本机绝对路径复制到新服务器。

## 2. 检查原始数据

```bash
python scripts/audit_user_hdf5.py --raw-dir data
```

确认没有 NaN/Inf，视频帧数和 HDF5 帧数匹配，力/力矩不是全零。

## 3. 转成 LeRobot

推荐先跑这个：使用 `obs/state/joint_torque_external` 作为力输入，更偏向接触/外力信息。

```bash
python scripts/convert_user_hdf5_to_lerobot.py \
  --raw-dir data \
  --repo-id local/tavla_single_arm \
  --task "peg-in-hole" \
  --action-mode actual \
  --no-videos \
  --image-writer-processes 0 \
  --image-writer-threads 0 \
  --overwrite
```

只有在你明确想用机器人原始关节力矩 `joint_torque` 时，才跑下面这个：

```bash
python scripts/convert_user_hdf5_to_lerobot.py \
  --raw-dir data \
  --repo-id local/tavla_single_arm \
  --task "peg-in-hole" \
  --effort-key obs/state/joint_torque \
  --action-mode actual \
  --no-videos \
  --image-writer-processes 0 \
  --image-writer-threads 0 \
  --overwrite
```

如果报 `Unknown encoder 'libsvtav1'`，说明本机 ffmpeg 不支持 LeRobot 默认视频编码；保留 `--no-videos` 即可。

### 末端六维力版本

如果想用末端六维力/力矩 `[Fx,Fy,Fz,Tx,Ty,Tz]`，重新转一个独立数据集：

```bash
python scripts/convert_user_hdf5_to_lerobot.py \
  --raw-dir data \
  --repo-id local/tavla_single_arm_ee_wrench \
  --task "peg-in-hole" \
  --effort-key obs/state/ee_wrench_base \
  --action-mode actual \
  --no-videos \
  --image-writer-processes 0 \
  --image-writer-threads 0 \
  --overwrite
```

也可以把 `obs/state/ee_wrench_base` 换成 `obs/state/ee_wrench_stiffness`。一般先用 `ee_wrench_base`。

## 4. 计算归一化统计

```bash
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name pi0_lora_user_single_arm_effort
```

输出目录：

```text
assets/pi0_lora_user_single_arm_effort/local/tavla_single_arm
```

末端六维力版本：

```bash
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name pi0_lora_user_single_arm_ee_wrench
```

## 5. LoRA 微调

需要在 `nvidia-smi` 正常的 GPU 机器上跑。

初始 checkpoint 已就绪，OpenPI 会自动从本地缓存读取：

```text
$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params
```

先跑一个短 smoke test：

```bash
CUDA_VISIBLE_DEVICES=5 \
python scripts/train.py pi0_lora_user_single_arm_effort \
  --exp-name smoke_1k \
  --num-train-steps 1000 \
  --save-interval 500 \
  --batch-size 8 \
  --overwrite
```

正式训练可以增加步数，例如 30000：

```bash
CUDA_VISIBLE_DEVICES=3 \
python scripts/train.py pi0_lora_user_single_arm_effort \
  --exp-name first_lora \
  --num-train-steps 30000 \
  --save-interval 30000 \
  --log-interval 50 \
  --batch-size 8 \
  --overwrite 2>&1 | tee logs/first_lora.log
```

不用 wandb 时，训练后画本地曲线：

```bash
python scripts/plot_training_log.py logs/first_lora.log --out logs/first_lora_curves.png
```

末端六维力版本训练：

```bash
CUDA_VISIBLE_DEVICES=5 \
python scripts/train.py pi0_lora_user_single_arm_ee_wrench \
  --exp-name ee_wrench_ckpt10k \
  --num-train-steps 30000 \
  --save-interval 10000 \
  --log-interval 50 \
  --batch-size 8 \
  --overwrite 2>&1 | tee logs/ee_wrench_ckpt10k.log
```

会保存约这些 checkpoint：

```text
checkpoints/pi0_lora_user_single_arm_ee_wrench/ee_wrench_ckpt10k/10000
checkpoints/pi0_lora_user_single_arm_ee_wrench/ee_wrench_ckpt10k/20000
checkpoints/pi0_lora_user_single_arm_ee_wrench/ee_wrench_ckpt10k/29999
```

续训：

```bash
python scripts/train.py pi0_lora_user_single_arm_effort \
  --exp-name first_lora \
  --num-train-steps 30000 \
  --save-interval 30000 \
  --resume
```

## 6. 离线看训练效果

用一条采集数据做 replay，对比模型预测动作和真实动作：

```bash
CUDA_VISIBLE_DEVICES=3 \
python scripts/eval_policy_on_user_episode.py \
  --episode-dir data/traj_0 \
  --checkpoint-dir checkpoints/pi0_lora_user_single_arm_effort/first_lora/29999 \
  --out-dir eval_outputs/traj_0_first_lora_29999 \
  --stride 5 \
  --max-frames 40 \
  --chunk-index 1
```

输出：

```text
eval_outputs/traj_0_first_lora_29999/actions_pred_vs_true.png
eval_outputs/traj_0_first_lora_29999/mae_per_frame.png
eval_outputs/traj_0_first_lora_29999/pred_vs_true.csv
eval_outputs/traj_0_first_lora_29999/summary.txt
```

说明：`--chunk-index 1` 比较下一帧动作；不要默认看第 0 帧，因为你的数据里 `action[t]` 接近 `state[t]`。

末端六维力版本评估：

```bash
CUDA_VISIBLE_DEVICES=5 \
python scripts/eval_policy_on_user_episode.py \
  --config-name pi0_lora_user_single_arm_ee_wrench \
  --checkpoint-dir checkpoints/pi0_lora_user_single_arm_ee_wrench/first_lora_ee_wrench/29999 \
  --effort-key obs/state/ee_wrench_base \
  --out-dir eval_outputs/traj_0_first_lora_ee_wrench_29999 \
  --stride 5 \
  --max-frames 40 \
  --chunk-index 1
```

## 7. 启动训练后的策略服务

`--policy.dir` 要指向具体 checkpoint step 目录。比如 `num-train-steps=30000` 的最终 checkpoint 通常是 `29999`。

```bash
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi0_lora_user_single_arm_effort \
  --policy.dir checkpoints/pi0_lora_user_single_arm_effort/first_lora/29999
```

如果不确定有哪些 step：

```bash
ls checkpoints/pi0_lora_user_single_arm_effort/first_lora
```

服务会监听：

```text
0.0.0.0:8000
```

末端六维力版本服务：

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi0_lora_user_single_arm_ee_wrench \
  --policy.dir checkpoints/pi0_lora_user_single_arm_ee_wrench/first_lora_ee_wrench/29999
```

## 建议

- 先检查当前数据集的轨迹数，再跑 smoke test。
- 保持默认 `--action-mode actual`，让模型和 data loader 自己处理 action chunk。
- 优先用 `joint_torque_external` 做力输入。
- 如果用末端六维力，优先试 `ee_wrench_base`。
- 后续如果只有一个腕部相机，最好再做更干净的单 wrist 配置，而不是长期复制成左右 wrist。

## 可复用训练脚本

仓库内的长时间训练命令已整理成可配置脚本，均从仓库根目录执行：

```bash
# 单臂、无 effort 的 30k LoRA 微调
bash scripts/run_pi0_no_effort_30k.sh 0 my_exp

# 真机/仿真 50:50 联合训练；GPU 参数使用逗号分隔
bash scripts/run_pi0_sim_real_cotrain_30k.sh 0,1,2,3 my_cotrain 8

# 真机末端六维力微调；通过环境变量覆盖默认参数
GPU_ID=5 EXP_NAME=real_ee_wrench bash scripts/run_tavla_real_ee_wrench_gpu457.sh
```

训练数据、norm stats、checkpoint、日志和本地缓存均不提交 Git。仿真 CSV/MP4/HDF5
放在本地 `sim-data/`，完整仿真数据转换流程见
[`docs/tavla_sim_finetune_runbook.md`](docs/tavla_sim_finetune_runbook.md)。


## 新版仿真力 Encoder

独立的 Sim 力质量分类交叉验证使用 `sim-data-aligned/`，与已训练的 TAVLA/pi0
策略权重分开。配置、数据来源和历史 V1–V3 结果见
[仿真力 Encoder 说明](docs/insertion_quality_sim_aligned_force.md)。

```bash
cd /data/workspace/gujiawei/TAVLA
GPU_ID=7 bash scripts/run_insertion_quality_sim_aligned_force.sh
```

## 真机/新版仿真力的插入段对比（只读）

只比较 `misaligned_contact` 与 `aligned_progress`，排除自由和接近阶段；
同时保留原三维力模长与沿插入轴的有符号力，避免把换指标误认为完整力已对齐。

```bash
cd /data/workspace/gujiawei/TAVLA
PYTHONPATH=scripts:src .venv/bin/python scripts/analyze_insertion_force_alignment.py
```

结果写入 `eval_outputs/real_sim_force_comparison_20260929/insertion_alignment/`。
34 条至少有 20 帧活动插入数据的轨迹中，原三维力模长同帧相关系数中位数为
0.325；沿插入轴的有符号力、7 帧因果平滑后为 0.476。
该命令只生成对比图与 CSV，不训练 encoder，也不改 TAVLA/pi0 checkpoint。

## Force-alignment latent 纯真机策略的仿真部署（2026-10-04）

已训练实验 `force_align_latent_gpu0123_fsdp/29999` 使用纯真机数据，输入为
最新 terminal-reference `ForceAlignmentEncoder` 的 32 维 latent。模型配置名
仍为 `pi0_lora_user_single_arm_control_latent`，实际数据集为
`local/tavla_single_arm_force_align_latent`。

仿真侧传输包：
[`force_align_latent_sim_bundle_20261004.zip`](exports/force_align_latent_sim_bundle_20261004.zip)。
详细输入协议、力来源/符号、10 Hz 采样、encoder 32 帧历史、策略 10 帧 latent 历史、
夹爪单位及调用示例见包内
[`README`](exports/force_align_latent_sim_bundle_20261004/README.md)。
包中包含 encoder checkpoint 及轻量推理/客户端代码，TAVLA 权重留在训练机。

训练机专用 server 入口（使用 checkpoint 自带的归一化统计）：

```bash
cd /data/workspace/gujiawei/TAVLA
CUDA_VISIBLE_DEVICES=5 \
OPENPI_DATA_HOME=/data/workspace/gujiawei/TAVLA/.cache/openpi \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/serve_force_align_latent_real.py --port 8000
```

本次部署运行在已有的 `force:0` tmux 窗口，日志为
`logs/force_align_latent_real_server.log`。重启前确认端口与 GPU 可用；部署记录和验证
结果在包内 `deployment_status.json` / `verification.json`。

## Force-alignment latent 50:50 co-training 策略部署（2026-10-06）

Co-training 实验 `force_align_latent_cotrain_50_50_v1/29999` 已完成。策略使用
真机和 `sim-data-aligned/` 两个数据集，采样权重各为 0.5；沿用相同的
terminal-reference `ForceAlignmentEncoder`。最终 checkpoint：

```text
checkpoints/pi0_lora_user_single_arm_force_align_latent_cotrain_50_50/force_align_latent_cotrain_50_50_v1/29999
```

专用 server 正运行于已有的 `tavla:0` tmux 窗口，使用 GPU 7，监听 8001；同机 8000
仍提供纯真机策略以供对照。启动命令（在已有 `tavla` 会话的 TAVLA shell 中执行）：

```bash
cd /data/workspace/gujiawei/TAVLA
CUDA_VISIBLE_DEVICES=7 \
OPENPI_DATA_HOME=/data/workspace/gujiawei/TAVLA/.cache/openpi \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONUNBUFFERED=1 \
.venv/bin/python scripts/serve_force_align_latent_cotrain.py --port 8001
```

测试端连接 `ws://114.214.164.36:8001`。已传的仿真接入包中 encoder 与预处理可复用，
但旧示例会拒绝 co-training 的 server variant；先在解压目录应用
[`force_align_latent_cotrain_client_compat.patch`](exports/force_align_latent_cotrain_client_compat.patch)，
或重新传当前更新的接入包。注意检查 websocket metadata 中的策略名、两个 dataset、
50:50 权重和 encoder SHA256；策略输入仍为 `(10,32)` latent history，输出为 50×8
绝对关节位置目标。当前完成了参考观测推理验证，还需测试端连接后做仿真闭环评测。

## 运动背景补偿 Encoder：motion_v1

在外部 GPU 主机已有的 `train` tmux shell 中运行完整 encoder 实验：

```bash
cd /data/workspace/gujiawei/TAVLA && GPU_ID=7 bash scripts/run_force_gap_motion_v1.sh
```

训练使用物理 GPU 7；固定4折、初始化 seeds42/43/44、对齐权重0/0.01/0.1，
包含重训旧 encoder、双向未来3/5帧运动预测、背景/对齐消融及随后 CPU 软件检查。
结果复用 `eval_outputs/force_gap_encoder_v2/`，覆盖 `motion_encoder.pt`、
`motion_encoder.report.json`、`motion_encoder.log`，只保存一个新候选 checkpoint。
首20帧允许运动，但应为未接触的自由空间；最终以 report 的 `qualified` 判断是否达标。
输入、预处理、划分和报告字段见
[motion_v1 说明](scripts/force_gap_encoder_v2/MOTION_V1.md)。代码由用户运行。
