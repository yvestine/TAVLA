# 力预测 v3 TAVLA：2 号卡与服务端 encoder

服务地址：`ws://114.214.164.36:8006`。默认使用 **sim encoder 分支**。

- 策略配置：`pi0_lora_user_single_arm_force_predictive_v3_cotrain_50_50`。
- 最终 checkpoint：`force_predictive_v3_cotrain_20261022_120747_3643501_thpfix/29999`，已完成 30000 步。
- Encoder 从该 checkpoint 的 `assets/` 加载，SHA256 为 `da9117c7091b910e70a6efa6fa09fd68dd77dfa5997353e06e8b5ea11d176edb`。
- TAVLA 在物理 GPU 2 上运行，启动脚本按 GPU UUID 绑定；冻结 encoder 在 server CPU 上运行。
- 仿真客户端只需 NumPy、现有 `openpi_client` 及其网络依赖，不需要加载 encoder 或安装 PyTorch。

## 仿真端发送什么

图片、状态和 prompt 延续已有协议，`effort` 改为原始六维 wrench，额外发送仿真时间戳：

```python
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="114.214.164.36", port=8006)
assert client.get_server_metadata()["encoder_location"] == "server_cpu"
client.reset()  # 每次 env.reset 后调用；同时清空 server encoder 的两种历史

response = client.infer({
    "images": {
        "cam_high": front_rgb,           # RGB uint8 (H,W,3)
        "cam_left_wrist": wrist_rgb,     # RGB uint8 (H,W,3)
    },
    "state": np.asarray(state8, dtype=np.float32),  # (8,)，7 个关节 rad + 归一化夹爪
    "effort": np.asarray(wrench6, dtype=np.float32).reshape(1, 6),
    "timestamp_s": float(simulation_time_s),
    "force_source": "panda_link7_wrench_calibrated",
    "prompt": "peg-in-hole",
})
actions = np.asarray(response["actions"], dtype=np.float32)  # (50,8)，绝对目标
predicted_force = response["predicted_force"]               # (5,3)，N
```

`wrench6` 的列为 `[Fx,Fy,Fz,Tx,Ty,Tz]`。前三维力须使用 robot base 坐标轴、单位 N。
`force_source` 有两个明确的选择：

| 来源 | `force_source` | server 转换 |
| --- | --- | --- |
| 已与训练定义一致的 `obs/state/ee_wrench_base` | `ee_wrench_base`（默认） | `wrench[:3]` |
| 与原采集器定义、标定、base 坐标轴相同的 `sim/force/panda_link7_wrench_calibrated` | `panda_link7_wrench_calibrated` | `-wrench[:3]`，倍率 1 |

如果传的是已经对齐的 `ee_wrench_base`，示例中应改用 `"force_source": "ee_wrench_base"`。
同一个 episode 首次选择来源后，后续请求可以省略该字段，不能在 episode 中途换来源。
其他 local/world/body 坐标的 wrench 需要先转换为上述定义；server 无法从六个数自动推断坐标系。

该 checkpoint 的 encoder **只使用前三维力，后三维力矩不参与编码**。
Server 完成源信号取负（如需）、域归一化、趋势/瞬态分解、32 帧 encoder 窗口和 10 帧 latent 历史。
这是 v3 的训练预处理，**不减首 20 帧基线，也不使用旧 encoder 的 3.4 倍映射**。首个样本即可请求动作，历史不足时按训练规则补齐。

## 时间、动作分块与重置

按训练的 10 Hz 控制采样，每 0.1 秒仿真时间采集一次力。时间戳使用仿真秒数，严格递增；不要使用网络等待的墙钟时间。
即使 policy 并非每步请求，仍需保留所有控制步的力采样，不能直接把多步执行后的最后一个样本当作完整历史。

一次执行多个动作时，可以在下次推理请求中批量上传期间的所有新增样本：

```python
response = client.infer({
    "images": latest_images,
    "state": np.asarray(latest_state8, dtype=np.float32),
    "effort": np.asarray(new_wrenches, dtype=np.float32),  # (N,6)，旧到新
    "timestamps_s": np.asarray(new_sim_times_s, dtype=np.float64),  # (N,)
    "force_source": "ee_wrench_base",
})
```

图片与 state 对应最后一个力样本的时刻。每个样本只发一次，不发送已上传的重叠滑动窗口，不发送未来观测。
也可在控制步单独发送力更新，暂不运行 TAVLA：

```python
ack = client.infer({
    "effort": np.asarray(wrench6, dtype=np.float32),
    "timestamp_s": float(simulation_time_s),
    "update_only": True,
})  # 返回 {"updated": 1, "timestamp_s": ...}
```

调用推理时需要包含**尚未发送的新样本**，不能再次发送刚刚 update_only 的同一时刻。
每次环境 reset 后调用 `client.reset()`；新 WebSocket 连接会建立独立的 encoder 历史。
相邻时间戳间隔超过 0.5 秒时，encoder 自动清空力历史和 latent 历史，与训练的数据分段一致。

## 返回值

- `actions`: `(50,8)`，前 7 维为绝对关节目标 rad，第 8 维为归一化夹爪目标。不再叠加当前关节角。
- `predicted_force`: `(5,3)`，force-only 预测头的未来 5 帧 **Real 域对齐 base XYZ 力**，单位 N；10 Hz 下名义对应未来 0.1–0.5 秒。它不是仿真原始传感器坐标下的六维 wrench。
- `timestamp_s`: 本次最后一个输入样本的时间戳。

沿用已有评测约定时可先同步闭环执行 `actions[1]`，每次执行一控制步后重规划。
首请求包含 JIT 编译，应预留等待时间；实际延迟见下述 smoke 报告，不保证墙钟 10 Hz。

## 启动、检查和验证

服务在 `tmux` 会话 `tavla-force-v3` 的 `server` 窗口中运行。手动启动命令：

```bash
cd /data/workspace/gujiawei/TAVLA
bash scripts/serve_force_predictive_v3_gpu2.sh --port 8006
```

启动脚本默认绑定物理 GPU 2。调整时可设置 `GPU_INDEX`；`--domain real` 则切换为 checkpoint 同附的 Real encoder 分支。
进程与 GPU 身份见 `logs/deployment_8006_force_predictive_v3.json`，服务日志为 `logs/server_8006_force_predictive_v3.log`。

验证入口：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=scripts .venv/bin/pytest -q scripts/test_serve_force_predictive_v3.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:packages/openpi-client/src:scripts \
  .venv/bin/python scripts/smoke_force_predictive_v3_server.py
```

CPU 检查覆盖训练离线编码与分块流式编码一致性、来源转换、力矩忽略、初始化补齐、时间戳检查、reset、断帧和连接隔离。
`logs/server_8006_encoder_training_parity.json` 对照记录仿真轨迹与实际训练 parquet 的 latent。
`logs/server_8006_force_predictive_v3_smoke.json` 记录真实 WebSocket/GPU 推理结果、延迟、reset 一致性、两种来源一致性以及批量/逐步更新一致性。
这些检查验证部署和接口；仿真闭环任务成功率需在仿真端运行评测。
