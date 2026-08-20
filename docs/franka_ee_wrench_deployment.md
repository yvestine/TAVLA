# Franka 末端六维力 TAVLA 部署说明

这份文档给机械臂 Linux 主机端使用。目标是把 GPU 服务器上的 TAVLA policy server 接到 Franka 控制程序。

## 1. 总体结构

```text
Franka 主机/control loop  --websocket-->  GPU 服务器/TAVLA server
Franka 主机/control loop  <--actions-----  GPU 服务器/TAVLA server
```

注意：TAVLA server 不会主动连接 Franka，也不会主动推控制命令。Franka 主机必须作为 client，每次发送当前观测 `obs`，server 返回预测动作 `actions`，然后 Franka 主机本地代码再把动作发给 libfranka / ROS 控制器。

## 2. GPU 服务器启动 policy server

在 TA-VLA 仓库中启动末端六维力版本：

```bash
cd TA-VLA
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=4 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi0_lora_user_single_arm_ee_wrench \
  --policy.dir checkpoints/pi0_lora_user_single_arm_ee_wrench/first_lora_ee_wrench/29999
```

如果 checkpoint 名字不同，把 `--policy.dir` 改成实际目录。

服务监听：

```text
0.0.0.0:8000
```

Franka 主机需要能访问：

```text
ws://<GPU服务器IP>:8000
```

## 3. Franka 主机安装 client

只需要轻量 client，不需要在 Franka 主机上装完整训练环境。

如果已经把 TA-VLA 仓库同步到 Franka 主机：

```bash
cd /path/to/TA-VLA/packages/openpi-client
pip install -e .
```

依赖主要是：

```text
numpy
websockets
msgpack
pillow
dm-tree / tree
```

## 4. 发送给模型的输入格式

每次调用 server，需要发送一个 Python dict：

```python
obs = {
    "images": {
        "cam_high": front_rgb,
        "cam_left_wrist": wrist_rgb,
        "cam_right_wrist": wrist_rgb,
    },
    "state": state,
    "effort": effort,
    "prompt": "peg-in-hole",
}
```

字段要求：

```text
front_rgb:
  shape: H x W x 3 或 3 x H x W
  dtype: uint8
  color: RGB，不是 OpenCV 默认 BGR
  训练数据原始尺寸: 480 x 640

wrist_rgb:
  shape: H x W x 3 或 3 x H x W
  dtype: uint8
  color: RGB
  目前训练时只有一个腕部相机，所以 cam_left_wrist 和 cam_right_wrist 都传同一张 wrist 图

state:
  shape: (8,)
  dtype: float32
  内容: [joint_0, ..., joint_6, gripper]
  joint 单位: rad
  gripper: 0=闭合, 1=张开

effort:
  shape: (1, 6)
  dtype: float32
  内容: [[Fx, Fy, Fz, Tx, Ty, Tz]]
  坐标系: 必须和训练时一致，当前配置默认使用 obs/state/ee_wrench_base

prompt:
  字符串，建议固定为 "peg-in-hole"
```

不要在 Franka 端手动做归一化。server 会使用 checkpoint 里的 norm stats 自动归一化。

## 5. 最小 client 示例

```python
import cv2
import numpy as np

from openpi_client import websocket_client_policy


SERVER_IP = "填GPU服务器IP"
SERVER_PORT = 8000

client = websocket_client_policy.WebsocketClientPolicy(
    host=SERVER_IP,
    port=SERVER_PORT,
)


def bgr_to_rgb_uint8(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def build_obs(front_bgr, wrist_bgr, joint_pos, gripper_pos, ee_wrench_base):
    front_rgb = bgr_to_rgb_uint8(front_bgr)
    wrist_rgb = bgr_to_rgb_uint8(wrist_bgr)

    state = np.concatenate(
        [
            np.asarray(joint_pos, dtype=np.float32),          # (7,)
            np.asarray([gripper_pos], dtype=np.float32),      # (1,)
        ],
        axis=0,
    )

    effort = np.asarray(ee_wrench_base, dtype=np.float32)[None, :]  # (1, 6)

    return {
        "images": {
            "cam_high": front_rgb,
            "cam_left_wrist": wrist_rgb,
            "cam_right_wrist": wrist_rgb,
        },
        "state": state,
        "effort": effort,
        "prompt": "peg-in-hole",
    }


obs = build_obs(
    front_bgr=front_camera_frame,
    wrist_bgr=wrist_camera_frame,
    joint_pos=current_joint_pos,
    gripper_pos=current_gripper_pos,
    ee_wrench_base=current_ee_wrench_base,
)

result = client.infer(obs)
action_chunk = result["actions"]

print(action_chunk.shape)  # 通常是 (50, 8)
```

## 6. server 返回什么

server 返回：

```python
{
    "actions": np.ndarray,
}
```

当前配置输出：

```text
actions.shape = (50, 8)
```

每一行是一个 absolute joint command：

```text
[target_joint_0, ..., target_joint_6, target_gripper]
```

含义：

```text
target_joint_i:
  单位 rad
  目标关节位置

target_gripper:
  0=闭合
  1=张开
```

模型内部 `action_dim=32`，但部署输出已经裁成前 8 维，不需要 robot 端处理 padding。

## 7. 控制循环怎么用 action

最简单的闭环方式：

```python
result = client.infer(obs)
action_chunk = result["actions"]       # (50, 8)
cmd = action_chunk[1]                  # 推荐先用第 1 个未来动作
target_joints = cmd[:7]
target_gripper = cmd[7]
```

为什么不是默认 `action_chunk[0]`：你的训练数据里 `action[t]` 基本等于当前 `state[t]`，所以第 0 个动作往往接近当前状态。部署初期建议先用 `action_chunk[1]`，或者根据实际延迟使用 `action_chunk[2]`。

如果要降低推理频率，可以一次请求一个 chunk，然后开环执行其中多步：

```text
第 0 次请求 obs[t]，执行 actions[1], actions[2], ..., actions[k]
第 k 步后重新请求新的 obs
```

建议先从小的 `k` 开始，例如 `k=3~5`。不要一开始直接开环执行完整 50 步。

## 8. 控制安全注意事项

必须在 Franka 主机侧做安全限制，不能直接无保护地把模型输出发给机器人：

```text
1. joint target 必须 clamp 到 Franka 关节限位内。
2. 每周期 joint delta 必须限幅，例如 abs(target - current) 不超过一个小阈值。
3. 速度/加速度/jerk 由控制器限制，不能只依赖模型。
4. gripper 输出 clamp 到 [0, 1]。
5. server 超时、网络断开、返回 NaN/Inf 时，立即停止发送新目标或进入 hold position。
6. 第一轮只 dry-run：打印 action，不驱动真机。
7. 第二轮低速小范围测试，手放急停。
8. 确认相机、state、wrench 都和采集时同源、同顺序、同单位、同坐标系。
```

## 9. 末端六维力注意事项

当前训练配置使用：

```text
obs/state/ee_wrench_base
```

也就是 base 坐标系下：

```text
[Fx, Fy, Fz, Tx, Ty, Tz]
```

Franka 部署端要尽量使用和采集脚本完全一样的计算来源。如果部署端拿到的是法兰/工具坐标系力，或者符号方向不同，模型效果会明显变差。

上线前建议打印 5 秒数据，检查范围是否接近训练数据：

```text
Fx roughly: -8 ~ 5
Fy roughly: -6 ~ 5
Fz roughly: -31 ~ 4
Tx roughly: -10 ~ 4
Ty roughly: -4 ~ 19
Tz roughly: -2 ~ 2
```

这些范围只是当前采集数据的粗略范围，不是硬限位。

## 10. 联通测试

在 Franka 主机：

```bash
ping <GPU服务器IP>
```

检查端口：

```bash
nc -vz <GPU服务器IP> 8000
```

如果连不上，检查：

```text
1. GPU 服务器 server 是否启动成功。
2. 端口是否是 8000。
3. 防火墙是否放行。
4. Franka 主机和 GPU 服务器是否在同一网络。
5. client 里 host 是否填的是 GPU 服务器 IP，不是 0.0.0.0。
```

## 11. 常见错误

```text
KeyError: images / state / effort
  obs dict 字段名不对。

shape mismatch for effort
  末端六维力版本必须传 effort shape (1, 6)。

图像颜色异常
  OpenCV 读出来是 BGR，需要转 RGB。

返回 actions 但机器人不动
  server 只返回 action，不会自动发给 Franka；需要 Franka 控制程序自己执行。

动作方向明显不对
  检查 joint 顺序、gripper 语义、ee_wrench 坐标系和符号。
```
