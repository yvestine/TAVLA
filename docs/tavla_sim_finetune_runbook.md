# TAVLA 仿真数据接入与联合微调运行说明

本文档只针对 `TA-VLA` 仓库。不会修改或启动仿真仓库，也不包含仿真闭环测试。

本轮已完成数据转换、字段校验、LeRobot 统计信息生成和 90/10 数据加载自检；按当前
要求没有在此环境执行长时间训练，因此下面的 checkpoint 路径是执行命令后应生成的
新路径，不代表已覆盖或替换原始 checkpoint。

## 已确定的数据语义

- `ppo_joint_targets.csv` 是 TAVLA 的 8 维动作标签：
  `[joint_0, ..., joint_6, gripper]`。
- 前 7 维是绝对关节位置目标，单位 rad，关节顺序与真机 `joint_0` 到
  `joint_6` 一致。
- 第 8 维是归一化夹爪目标，范围 `[0, 1]`，`0` 闭合、`1` 张开；当前 PPO
  数据通常为 `0`。
- `actions.csv` 是 PPO 原始 6 维笛卡尔增量动作，不是 TAVLA 的 8 维监督标签。
  已知 `episode_0/actions.csv` 混入过 6 维和 8 维行，因此转换脚本保留它但完全
  不把它写成训练标签。
- `wrench_final.csv` 是当前六维力输入，顺序 `[Fx, Fy, Fz, Tx, Ty, Tz]`，单位
  `[N, N, N, N·m, N·m, N·m]`，位于 robot base 坐标系，力矩参考点为
  robot base 原点。它等于已完成符号校准的 `-wrench_base`；旧的
  `wrench_model.csv` 不再作为当前 TAVLA 的默认训练字段。
- 源数据按 30 Hz 记录，PPO 每 3 帧更新一次；转换为 10 Hz，只保留
  `0, 3, 6, ...` 帧，避免重复动作标签。

## 1. CSV/MP4 转规范 HDF5

在仓库根目录执行：

```bash
cd TA-VLA
.venv/bin/python scripts/convert_sim_csv_to_hdf5.py \
  --source-dir data-sim \
  --output-dir data-sim-hdf5 \
  --task "peg-in-hole" \
  --decision-stride 3 \
  --decision-offset 0 \
  --overwrite
```

脚本会检查所有数值 CSV 的列数、长度和 NaN/Inf，检查时间戳递增，并检查两路视频
是否能读取。当前导出中有 49 条轨迹的视频比 CSV 少 1 帧；默认按最短流截断尾帧并
记录在 HDF5 属性和 `data-sim-hdf5/conversion_summary.json` 中。若希望遇到长度差异
直接失败，增加 `--strict-lengths`。

输出的关键字段为：

```text
obs/state/joint_pos             (N, 7)
obs/state/gripper_pos           (N, 1)
obs/state/wrench_model           (N, 6)
action/ppo_joint_targets         (N, 8)
decision/obs/state/...           10 Hz 版本
decision/action/ppo_joint_targets 10 Hz 版本
decision/images/...              10 Hz 图像
attrs["task"]                   episode 级指令
```

当前 CSV 导出没有 `data.attrs["task"]`，所以转换命令显式使用仓库默认指令
`peg-in-hole`，并在 `task_source` 中标记了这一事实。若之后拿到每条轨迹
的真实语言指令，应在转换时逐 episode 写入 `attrs["task"]`，不要继续使用这个默认值。

真机原始 HDF5 同样没有语言属性；如果需要重建真机 LeRobot 数据并统一 task，执行：

```bash
.venv/bin/python scripts/convert_user_hdf5_to_lerobot.py \
  --raw-dir data \
  --repo-id local/tavla_single_arm_ee_wrench \
  --task "peg-in-hole" \
  --effort-key obs/state/ee_wrench_base \
  --action-mode actual \
  --no-videos \
  --image-writer-processes 0 \
  --image-writer-threads 8 \
  --overwrite
```

该命令会重建 40 条真机轨迹；不要在重建未完成时启动联合训练。

## 2. 转为 TAVLA 使用的 LeRobot 数据

完整 50 条轨迹：

```bash
.venv/bin/python scripts/convert_sim_hdf5_to_lerobot.py \
  --source-dir data-sim-hdf5 \
  --repo-id local/tavla_single_arm_ee_wrench_sim \
  --effort-key wrench_final \
  --overwrite
```

LeRobot 数据位于：

```text
${HF_HOME:-$HOME/.cache/huggingface}/lerobot/local/tavla_single_arm_ee_wrench_sim
```

## 3. 先做 20 条轨迹过拟合测试

生成前 20 条轨迹的独立数据集：

```bash
.venv/bin/python scripts/convert_sim_hdf5_to_lerobot.py \
  --source-dir data-sim-hdf5 \
  --repo-id local/tavla_single_arm_ee_wrench_sim_overfit \
  --effort-key wrench_final \
  --max-episodes 20 \
  --overwrite
```

计算该子集自己的 norm stats：

```bash
JAX_PLATFORMS=cpu \
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_lora_user_single_arm_ee_wrench_sim_overfit
```

确认加载后的张量：

```bash
JAX_PLATFORMS=cpu .venv/bin/python - <<'PY'
import dataclasses
import numpy as np
from openpi.training import config, data_loader

cfg = dataclasses.replace(
    config.get_config("pi0_lora_user_single_arm_ee_wrench_sim_overfit"),
    batch_size=1,
    num_workers=0,
)
loader = data_loader.create_data_loader(cfg, num_batches=1, shuffle=False, num_workers=0)
obs, actions = next(iter(loader))
print(obs.state.shape, obs.effort.shape, actions.shape)
print({k: v.shape for k, v in obs.images.items()})
assert obs.state.shape == (1, 32)
assert obs.effort.shape == (1, 1, 6)
assert actions.shape == (1, 50, 32)
assert np.isfinite(np.asarray(obs.state)).all()
assert np.isfinite(np.asarray(obs.effort)).all()
assert np.isfinite(np.asarray(actions)).all()
PY
```

从原真机六维力 checkpoint 初始化，训练 1000 steps。该配置不会从头训练，且输出
目录是新的：

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/train.py \
  pi0_lora_user_single_arm_ee_wrench_sim_overfit \
  --exp-name sim_overfit20 \
  --num-train-steps 1000 \
  --save-interval 250 \
  --log-interval 25 \
  --batch-size 8 \
  --overwrite \
  2>&1 | tee logs/sim_overfit20.log
```

预期 checkpoint 目录：

```text
checkpoints/pi0_lora_user_single_arm_ee_wrench_sim_overfit/sim_overfit20/999
```

过拟合通过标准：在 20 条轨迹上，离线评估的 8 维预测应明显贴近
`decision/action/ppo_joint_targets`，并且夹爪维保持在合理的 `[0, 1]` 范围。训练 loss
下降本身不是充分标准；若不贴近标签，不要继续联合微调。

使用过拟合 checkpoint 做离线动作误差检查。该步骤需要 GPU，建议放入 tmux：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/eval_policy_on_sim_hdf5.py \
  --source-dir data-sim-hdf5 \
  --checkpoint-dir checkpoints/pi0_lora_user_single_arm_ee_wrench_sim_overfit/sim_overfit20/999 \
  --config-name pi0_lora_user_single_arm_ee_wrench_sim_overfit \
  --prompt "peg-in-hole" \
  --max-episodes 20 \
  --out eval_outputs/sim_overfit20_action_error.json
```

## 4. 90% 真机 + 10% 仿真联合微调

联合配置使用：

```text
real: local/tavla_single_arm_ee_wrench       90%
sim:  local/tavla_single_arm_ee_wrench_sim   10%
```

数据加载器不是按两个数据集原始帧数拼接，而是按显式权重抽样，仿真数据允许重复采样，
因此不会因真机帧数更多而稀释到小于 10%。执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/train.py \
  pi0_lora_user_single_arm_ee_wrench_joint_finetune \
  --exp-name sim10pct_ft \
  --num-train-steps 1000 \
  --save-interval 250 \
  --log-interval 25 \
  --batch-size 8 \
  --overwrite \
  2>&1 | tee logs/sim10pct_ft.log
```

该配置从以下已有真机 checkpoint 继续：

```text
checkpoints/pi0_lora_user_single_arm_ee_wrench/first_lora_ee_wrench/29999/params
```

新的 checkpoint 预计在：

```text
checkpoints/pi0_lora_user_single_arm_ee_wrench_joint_finetune/sim10pct_ft/999
```

不会覆盖原始 checkpoint。

## 5. 启动现有 TAVLA Server

完成联合训练并确认实际保存目录后：

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config pi0_lora_user_single_arm_ee_wrench_joint_finetune \
  --policy.dir checkpoints/pi0_lora_user_single_arm_ee_wrench_joint_finetune/sim10pct_ft/999
```

Server 使用仓库已有 WebSocket 协议。episode 开始时，client 调用：

```python
client.reset()
```

Server 会重置 policy RNG/记录状态并返回 `{"reset": True}`。

## 6. Server 输入输出

输入为：

```python
{
    "images": {
        "cam_high": high_rgb,          # HxWx3 或 3xHxW, uint8, RGB
        "cam_left_wrist": wrist_rgb,   # HxWx3 或 3xHxW, uint8, RGB
    },
    "state": np.asarray([q0, q1, q2, q3, q4, q5, q6, gripper], np.float32),
    "effort": np.asarray([[Fx, Fy, Fz, Tx, Ty, Tz]], np.float32),
    "prompt": "peg-in-hole",
}
```

`state` 的 7 个关节为 rad，夹爪为归一化值。`effort` 必须使用训练时同一坐标系和
单位；当前仿真和配置的输入都是 tool/fingertip frame、fingertip origin、N/N·m。
Server 不要求客户端手动做 norm stats 归一化。

输出为：

```python
{
    "actions": np.ndarray,  # shape (50, 8)
}
```

每行是 `[joint_0_target, ..., joint_6_target, gripper_target]`，前 7 维是绝对关节
目标，最后一维为 `[0, 1]` 夹爪目标。
