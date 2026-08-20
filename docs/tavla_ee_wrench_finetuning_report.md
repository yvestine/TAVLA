# TAVLA 末端六维力 LoRA 微调说明

本文档用于让其他项目理解当前 TAVLA 末端六维力版本是如何采集数据、转换数据、配置模型、进行 LoRA 微调以及部署推理的。

## 1. 当前版本概览

当前使用的是单臂 Franka 数据，包含视觉、机器人状态、动作标签和末端六维力/力矩。

核心配置：

```text
训练配置名: pi0_lora_user_single_arm_ee_wrench
LeRobot repo_id: local/tavla_single_arm_ee_wrench
基础 checkpoint: pi0_base
LoRA 模型: gemma_2b_lora + gemma_300m_lora
力输入类型: EffortType.EXPERT
力输入维度: 6
动作输出维度: 8
动作 chunk 长度: 50
控制模式: absolute_joint
频率: 10 Hz
```

当前原始数据：

```text
episode 数量: 40
总帧数: 7188
成功 episode: 40
单臂: right / single arm
相机: front_camera + wrist_camera
图像尺寸: [3, 480, 640]
夹爪语义: normalized_0_closed_1_open
```

当前末端六维力 checkpoint：

```text
checkpoints/pi0_lora_user_single_arm_ee_wrench/ee_wrench_ckpt10k/10000
checkpoints/pi0_lora_user_single_arm_ee_wrench/ee_wrench_ckpt10k/20000
checkpoints/pi0_lora_user_single_arm_ee_wrench/ee_wrench_ckpt10k/29999
```

另有早期完整训练版本：

```text
checkpoints/pi0_lora_user_single_arm_ee_wrench/first_lora_ee_wrench/29999
```

## 2. 原始数据格式

每条轨迹目录类似：

```text
data/traj_0/data.h5
data/traj_0/front_camera.mp4
data/traj_0/wrist_camera.mp4
```

HDF5 主要字段：

```text
obs/state/joint_pos                  (T, 7)  关节角，rad
obs/state/joint_vel                  (T, 7)  关节速度
obs/state/ee_pose                    (T, 7)  [x,y,z,qx,qy,qz,qw]
obs/state/gripper_pos                (T, 1)  夹爪归一化，0=闭合, 1=张开
obs/state/gripper_width_m            (T, 1)  夹爪宽度，m

obs/state/joint_torque               (T, 7)  原始关节力矩
obs/state/joint_torque_external      (T, 7)  外部估计关节力矩

obs/state/ee_wrench_base             (T, 6)  base 坐标系末端六维力/力矩
obs/state/ee_wrench_stiffness        (T, 6)  stiffness/K 坐标系末端六维力/力矩
obs/state/ee_force_base              (T, 3)  ee_wrench_base 前 3 维
obs/state/ee_torque_base             (T, 3)  ee_wrench_base 后 3 维

action/actual/arm                    (T, 7)  录制时 arm action
action/actual/gripper                (T, 1)  录制时 gripper action
action/policy/arm                    (T, 7)
action/policy/gripper                (T, 1)

timestamps                           (T,)    Unix timestamp
info                                 JSON
meta/env_meta                        JSON
meta/env_kwargs                      JSON
attrs["success"]                     episode 是否成功
```

元数据要点：

```text
hz = 10
control_mode = absolute_joint
controller_backend = absolute_joint
rgb cameras = front_camera, wrist_camera
gripper_semantics = normalized_0_closed_1_open
is_dual_arm = false
```

## 3. 末端六维力数据

本版本训练使用：

```text
obs/state/ee_wrench_base
```

顺序为：

```text
[Fx, Fy, Fz, Tx, Ty, Tz]
```

含义：

```text
Fx, Fy, Fz: base 坐标系下末端外力，单位按 Franka wrench 约定为 N
Tx, Ty, Tz: base 坐标系下末端外力矩，单位按 Franka wrench 约定为 Nm
```

重要说明：

```text
1. 当前转换脚本不做额外零偏、滤波、裁剪或旋转。
2. 训练时直接读取 HDF5 里的 obs/state/ee_wrench_base。
3. 部署端必须尽量复现同一个坐标系、符号、单位和计算来源。
4. 如果部署端拿到的是法兰/tool/stiffness 坐标系下的 wrench，需要先转换到与训练一致的 base 坐标系。
5. 当前训练没有使用 joint_torque_external，也没有使用 ee_wrench_stiffness。
```

当前 40 条数据中 `ee_wrench_base` 的粗略范围：

```text
Fx: min -7.6692, max  4.8195, mean -2.5152, std 1.4933
Fy: min -3.9835, max  3.7986, mean  0.4785, std 1.0489
Fz: min -29.8733, max 5.4080, mean -1.2109, std 2.6253
Tx: min -9.1062, max  3.3299, mean -1.1754, std 1.4152
Ty: min -4.1316, max 16.8903, mean -1.2037, std 1.3630
Tz: min -0.6037, max  0.9201, mean  0.2494, std 0.1673
```

归一化统计里 `effort` 的均值和标准差：

```text
mean = [-2.5423, 0.4312, -1.0318, -1.1241, -1.2457, 0.2434]
std  = [ 1.6714, 1.2777,  2.8150,  1.5958,  1.4829, 0.1926]
```

## 4. State 和 Action 语义

state 定义：

```text
state = [q0, q1, q2, q3, q4, q5, q6, gripper]
shape = (8,)
q 单位 = rad
gripper = 0 闭合, 1 张开
```

action 标签来自：

```text
action/actual/arm
action/actual/gripper
```

合并后：

```text
action = [target_q0, ..., target_q6, target_gripper]
shape = (8,)
```

当前数据中 `action/actual` 基本等于同帧 `state`。训练时不是只学单帧 identity，因为 OpenPI data loader 会根据 `action_horizon=50` 自动构造未来动作 chunk：

```text
actions[t] = [action[t], action[t+1], ..., action[t+49]]
```

训练配置中：

```text
delta_action_mask = (7, -1)
```

含义：

```text
前 7 维关节动作在训练输入中转成相对当前 state 的 delta
最后 1 维 gripper 保持绝对值
```

推理输出时再通过 `AbsoluteActions` 转回 absolute joint target。因此部署端收到的是绝对关节目标，不是关节增量。

## 5. 图像数据

原始数据有两路 RGB 视频：

```text
front_camera.mp4
wrist_camera.mp4
```

TAVLA/OpenPI 模型侧需要三个 image key：

```text
cam_high
cam_left_wrist
cam_right_wrist
```

由于当前只有一个 wrist camera，转换和部署都采用：

```text
cam_high = front_camera
cam_left_wrist = wrist_camera
cam_right_wrist = wrist_camera
```

部署输入可以直接传：

```text
shape = (480, 640, 3)
dtype = uint8
color = RGB
```

server 内部会 resize 到 224x224。若用 OpenCV 读图，必须从 BGR 转 RGB。

## 6. 数据转换到 LeRobot

转换脚本：

```text
scripts/convert_user_hdf5_to_lerobot.py
```

末端六维力版本转换命令：

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

转换后 LeRobot 数据集位置：

```text
~/.cache/huggingface/lerobot/local/tavla_single_arm_ee_wrench
```

转换后的关键字段：

```text
observation.state              (8,)
observation.effort             (6,)
action                         (8,)
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

其中：

```text
observation.effort = raw_h5["obs/state/ee_wrench_base"]
```

## 7. 归一化统计

归一化统计命令：

```bash
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name pi0_lora_user_single_arm_ee_wrench
```

训练前 assets 位置：

```text
assets/pi0_lora_user_single_arm_ee_wrench/local/tavla_single_arm_ee_wrench/norm_stats.json
```

checkpoint 内保存的位置：

```text
checkpoints/pi0_lora_user_single_arm_ee_wrench/<exp>/<step>/assets/norm_stats.json
```

当前统计维度：

```text
state:   32  # 原始 8 维 pad 到模型内部 action_dim=32
actions: 32  # 原始 8 维 pad 到模型内部 action_dim=32
effort:   6  # 末端六维力不 pad
```

部署端不需要手动归一化。policy server 会加载 checkpoint 中的 norm stats 自动处理。

## 8. TAVLA 模型配置

训练配置在：

```text
src/openpi/training/config.py
```

配置名：

```text
pi0_lora_user_single_arm_ee_wrench
```

核心参数：

```python
model = pi0.Pi0Config(
    action_dim=32,
    effort_dim=6,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    effort_type=EffortType.EXPERT,
)

data = LeRobotTavlaDataConfig(
    repo_id="local/tavla_single_arm_ee_wrench",
    effort_history=(0,),
    delta_action_mask=(7, -1),
    action_output_dim=8,
    padding_stat=True,
    default_prompt="peg-in-hole",
    base_config=DataConfig(local_files_only=True),
)
```

含义：

```text
action_dim=32:
  保持与 pi0_base checkpoint 兼容，8 维 action/state 会 pad 到 32。

effort_dim=6:
  使用末端六维力/力矩。

effort_history=(0,):
  只使用当前帧 wrench，不使用历史力序列。

effort_type=EXPERT:
  将 effort 投影成 token，加入 action expert / decoder 侧。

action_output_dim=8:
  部署时只输出前 8 维，即 7 关节 + 1 夹爪。
```

## 9. 力数据如何进入模型

数据流：

```text
HDF5 obs/state/ee_wrench_base
  -> LeRobot observation.effort
  -> data transform 顶层字段 "effort"
  -> model Observation.effort
  -> Pi0 effort token
  -> action expert / decoder suffix
```

关键代码位置：

```text
scripts/convert_user_hdf5_to_lerobot.py
  将 obs/state/ee_wrench_base 写成 observation.effort

src/openpi/training/config.py
  LeRobotTavlaDataConfig 在 effort_history 非空时 repack "effort"

src/openpi/policies/tavla_policy.py
  TavlaInputs 把 data["effort"] 传给模型输入

src/openpi/models/pi0.py
  EffortType.EXPERT 时，effort token 加到 suffix/action expert 侧
```

当前不是把 wrench 拼进 state，也不是让模型预测 future effort；它只是作为当前观测输入参与动作预测。

## 10. LoRA 微调

基础权重：

```text
s3://openpi-assets/checkpoints/pi0_base/params
```

本地缓存：

```text
$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params
```

训练命令：

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

实际保存：

```text
10000
20000
29999
```

注意：当前保存逻辑按 loop step 保存，所以不是 `9999/19999/29999`。

训练参数说明：

```text
num_train_steps=30000
save_interval=10000
batch_size=8
log_interval=50
```

可训练部分主要包括 LoRA 参数，以及新增的 effort projection 层。基础 pi0 权重从 pi0_base 加载。

## 11. 离线评估

脚本：

```text
scripts/eval_policy_on_user_episode.py
```

示例：

```bash
CUDA_VISIBLE_DEVICES=5 \
python scripts/eval_policy_on_user_episode.py \
  --config-name pi0_lora_user_single_arm_ee_wrench \
  --checkpoint-dir checkpoints/pi0_lora_user_single_arm_ee_wrench/ee_wrench_ckpt10k/10000 \
  --effort-key obs/state/ee_wrench_base \
  --out-dir eval_outputs/traj_0_ee_wrench_10000 \
  --stride 5 \
  --max-frames 40 \
  --chunk-index 1
```

输出：

```text
actions_pred_vs_true.png
mae_per_frame.png
pred_vs_true.csv
summary.txt
```

建议为了后续仿真稀疏奖励探索，分别测试：

```text
10000
20000
29999
```

目标不是选成功率最高的，而是选仿真中能偶尔成功的 checkpoint，例如成功率 20%~40%。

## 12. 部署输入输出

部署时启动 server：

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi0_lora_user_single_arm_ee_wrench \
  --policy.dir checkpoints/pi0_lora_user_single_arm_ee_wrench/ee_wrench_ckpt10k/10000
```

Franka/robot client 每次发送：

```python
obs = {
    "images": {
        "cam_high": front_rgb,         # (480,640,3) uint8 RGB
        "cam_left_wrist": wrist_rgb,   # (480,640,3) uint8 RGB
        "cam_right_wrist": wrist_rgb,  # 同一张 hand 图
    },
    "state": np.array([q0, q1, q2, q3, q4, q5, q6, gripper], dtype=np.float32),
    "effort": np.array([[Fx, Fy, Fz, Tx, Ty, Tz]], dtype=np.float32),
    "prompt": "peg-in-hole",
}
```

server 返回：

```python
result = {
    "actions": np.ndarray,  # shape (50, 8)
}
```

动作语义：

```text
actions[i] = [target_q0, ..., target_q6, target_gripper]
```

这是 absolute joint target，不是 delta。部署初期建议从 `actions[1]` 开始执行，因为 `actions[0]` 往往接近当前 state。

## 13. 需要其他项目特别注意

```text
1. 力数据必须是 base 坐标系 [Fx,Fy,Fz,Tx,Ty,Tz]。
2. 图像必须是 RGB，不是 OpenCV 默认 BGR。
3. hand 图当前复制到 cam_left_wrist 和 cam_right_wrist。
4. gripper 是归一化开合度，不是米制宽度。
5. state/action 都是 Franka 关节顺序 q0...q6。
6. 模型返回 absolute joint target，控制端必须做限幅和安全检查。
7. 部署端不要手动归一化 state/effort/action。
8. 如果仿真增强需要偶尔成功，不一定用最终 29999 checkpoint，可以从 10000 或 20000 开始选。
```

## 14. 相关文件

```text
run.md
docs/franka_ee_wrench_deployment.md
docs/tavla_ee_wrench_finetuning_report.md

scripts/convert_user_hdf5_to_lerobot.py
scripts/compute_norm_stats.py
scripts/train.py
scripts/eval_policy_on_user_episode.py
scripts/serve_policy.py

src/openpi/training/config.py
src/openpi/policies/tavla_policy.py
src/openpi/models/pi0.py
src/openpi/transforms.py
```
