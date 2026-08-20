# TA-VLA LoRA 微调步骤

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

- 先用当前 51 条数据跑 smoke test。
- 保持默认 `--action-mode actual`，让模型和 data loader 自己处理 action chunk。
- 优先用 `joint_torque_external` 做力输入。
- 如果用末端六维力，优先试 `ee_wrench_base`。
- 后续如果只有一个腕部相机，最好再做更干净的单 wrist 配置，而不是长期复制成左右 wrist。
