# 8001：真机发送六维力，服务端运行冻结 encoder

2026-10-24 更新：`scripts/serve_force_align_latent_cotrain.py` 默认使用
`--input-mode raw`。server-006 上地址为 `ws://10.0.40.113:8001`，tmux 为
`tavla_8001`。策略 checkpoint 和 encoder 权重保持同一训练配套组合。

## 输入与输出

- `images.cam_high`、`images.cam_left_wrist`：RGB uint8 `(H,W,3)`。
- `state`：float32 `(8,)`，七个关节角 rad + 归一化夹爪。
- `effort`：原始 float32 `(6,)` 或 `(1,6)`，顺序为 `[Fx,Fy,Fz,Tx,Ty,Tz]`；也支持
  一批**尚未发送过的**样本 `(N,6)`。
- `prompt`：`peg-in-hole`。
- 可选 `timestamp_s`：单个样本的采集时间，单位秒；批量样本使用 `timestamps_s (N,)`。
  时间戳在一个 episode 内严格递增。使用与传感器采集对应的时间，不使用网络响应完成时间。

力必须与真机训练字段 `ee_wrench_base` 的坐标、符号和单位一致。服务器接收六维力，
但这份旧 encoder **只使用 XYZ 力**，不使用后三维力矩。8005 仍使用原始六通道策略。

服务器在 CPU 上执行：首 20 帧固定基线扣除 → 力变化率 → 训练保存的特征尺度归一化
和截断 → 左侧补齐的 32 帧窗口 → 原冻结 encoder → 最近 10 帧 latent → GPU 策略。
客户端发送物理量，不再计算 encoder，也不再做策略归一化。

## 每个 episode 的初始化

1. 建立一次 WebSocket 连接并保持连接，调用 `client.reset()`。
2. 保持机械臂、避免新的插入接触，以 10 Hz 收集并发送 20 个空载样本，约 2 秒。
   可用 `update_only=True` 仅更新力历史，不需要图片与 state。
3. 预热返回 `ready=False`、`samples_needed`，此时没有 `actions`，客户端应保持位置。
   第 20 个样本后 `ready=True`。首段 latent 会按离线训练转换的规则自动回填。
4. 之后发送新图像、state 和新的力样本进行推理；返回 `ready=True` 和
   `actions (50,8)`，语义仍为绝对关节角目标和夹爪目标。

```python
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="10.0.40.113", port=8001)
assert client.get_server_metadata()["encoder_location"] == "server_cpu"
client.reset()

# 以下读取函数由真机采集程序提供；每次读取一个新的 10 Hz 样本。
for _ in range(20):
    wrench6, sensor_time_s = read_new_wrench_sample()
    status = client.infer({
        "effort": np.asarray(wrench6, dtype=np.float32).reshape(1, 6),
        "timestamp_s": float(sensor_time_s),
        "update_only": True,
    })
assert status["ready"]

front_rgb, wrist_rgb, state8, wrench6, sensor_time_s = read_new_observation()
result = client.infer({
    "images": {"cam_high": front_rgb, "cam_left_wrist": wrist_rgb},
    "state": np.asarray(state8, dtype=np.float32),
    "effort": np.asarray(wrench6, dtype=np.float32).reshape(1, 6),
    "timestamp_s": float(sensor_time_s),
    "prompt": "peg-in-hole",
})
if result["ready"]:
    actions = result["actions"]
    # 交给已有的限位、限速和超时处理控制器，先做不驱动机器的观测回放。
```

兼容只发送六维力的旧客户端时，可以省略时间戳；服务器按相邻力样本间隔 0.1 秒处理。
这要求客户端实际按 10 Hz 连续采样。一个 episode 内不要切换有/无时间戳模式。
一个样本只发送一次；不要每次重复发送包含旧样本的整段历史。
若一次执行多个动作，期间也须持续用 `update_only=True` 提交新力样本。

每次任务重置调用 `client.reset()`，重新收集空载基线。新连接拥有独立的基线、
特征窗口、latent 历史和策略随机状态；重连不会恢复旧历史。

## 权重与旧接口

encoder SHA256 为 `db793ebd4f3158b8a8add895cf04368b2289df7fad450f1d2a39b09bb46acd80`。
原来的客户端 latent `(10,32)` 协议可在启动时显式选择 `--input-mode latent`。
默认 raw 模式会拒绝 latent 输入，以免把两种力输入混用。
