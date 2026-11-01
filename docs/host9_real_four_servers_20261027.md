# Host9 四个真机策略服务

服务运行在 server-006 的物理 GPU 0，WebSocket 地址如下：

| 端口 | 任务 | 力输入 | tmux |
| --- | --- | --- | --- |
| 8000 | 双孔插入 | 原始六维力 | `host9_server_8000` |
| 8001 | USB 插入 | 原始六维力 | `host9_server_8001` |
| 8002 | 双孔插入 | 服务端 encoder | `host9_server_8002` |
| 8003 | USB 插入 | 服务端 encoder | `host9_server_8003` |

机器 IP 为 `10.0.40.113`。这里的 8001 与之前旧的 8001 模型不同：现在是 USB 原始六维力模型。
所有策略都使用对应任务的 `20261026_v1/29999` checkpoint 和 checkpoint 内的归一化统计。

## 真机输入

四个端口使用相同的接口：

- `images.cam_high`：前视相机，RGB `uint8 (H,W,3)`。
- `images.cam_left_wrist`：腕部相机，RGB `uint8 (H,W,3)`；服务端按训练数据将其复制给右腕输入。
- `state`：`float32 (8,)`，七个关节位置（rad）和训练数据中的 `gripper_pos`，夹爪数值使用采集数据的同一格式。
- `effort`：原始六维力 `[Fx,Fy,Fz,Tx,Ty,Tz]`，`float32 (6,)`、`(1,6)` 或一批新的 `(N,6)`。
  力的坐标系、符号和单位必须与真机训练数据的 `obs/state/ee_wrench_base` 一致：前三维为机器人基座坐标系下的 N，后三维为 Nm。

8000 和 8001 使用全部六维。8002 和 8003 在服务端 CPU 运行新训练的 Real encoder，使用前三维 XYZ 力，构造 32 帧力窗口和 10 帧 latent 历史；不使用旧 encoder 的 20 帧基线扣除。训练时用的 encoder SHA256 为
`06e5f3ee3ef69b3e6c9e2ef4bc0ebea249f6d3ef7da6da18b6171acfa2a6112e`。

每个端口固定使用该任务训练数据中的提示词；客户端传入的 `prompt` 会被任务对应的训练提示词替换。客户端无须计算 latent 或做策略归一化。

## 力采样、时间与重置

每个新 WebSocket 连接都有独立历史。每次新 episode 调用 `client.reset()`，或建立新连接。
力按训练频率 **10 Hz** 连续采样；每个采样只发送一次，不能反复发送含旧样本的滑动窗口。

客户端可以省略时间戳，服务端把连续发送的每个样本按 0.1 秒间隔处理。也可以给单帧发送 `timestamp_s`，给一批发送 `timestamps_s`，单位秒，使用真实采样时间并严格递增。一个 episode 内固定使用同一种时间模式；显式时间间隔超过 0.5 秒会清空 encoder 历史。

如果一次执行多个动作，期间仍需采集全部新的力样本；下次请求将这些样本按时间顺序组成 `(N,6)` 一并发送，图片和 `state` 对应最后一个力样本。也可在不需要推理时使用 `update_only=True` 发送新增样本，服务端返回 `updated` 和 `ready`，不返回动作。

```python
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host='10.0.40.113', port=8002)  # 按上表选择任务和模型
client.reset()

# front_rgb、wrist_rgb、state8、wrench6 由真机采集程序提供。
result = client.infer({
    'images': {'cam_high': front_rgb, 'cam_left_wrist': wrist_rgb},
    'state': np.asarray(state8, dtype=np.float32),
    'effort': np.asarray(wrench6, dtype=np.float32),
})
actions = result['actions']  # (50,8)，前 7 维绝对关节目标，末维夹爪目标
```

返回的 `actions` 是绝对目标，不要再叠加当前关节角。控制端应继续执行已有的关节限位、限速和超时处理。

## 验证与操作

已用两项任务的真实录制数据回放四个 WebSocket 服务，检查六维力输入、动作形状与有限值、重置、连接隔离及单帧/分批更新。encoder 在线输入还与实际训练 parquet 的 latent 做过对照。
本机单样本预热推理约 0.12–0.13 秒，实际网络和控制延迟会增加；力应由采集端持续采样并缓存，策略推理不能保证 10 Hz。
这些检查没有驱动机械臂，也不能证明真机插入成功；首次上机先进行无动作回放，再低速测试并保持急停可用。

回放结果：`logs/host9_real_servers_smoke_20261027.json`。
各服务日志：`logs/host9_server_8000.log` 至 `logs/host9_server_8003.log`。
四个会话都不存在且端口空闲时，可重新启动：

```bash
cd /workspace/gujiawei/TA-VLA
PYTHONPATH=scripts:src:packages/openpi-client/src .venv/bin/python scripts/launch_host9_real_servers.py
```

重启后再做回放预热；首个推理包含模型编译，可能需要十几秒。
