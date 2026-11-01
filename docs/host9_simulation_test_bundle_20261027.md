# Host9 仿真推理测试包

本包将四个已训练的 TAVLA policy、共享力 encoder 和启动服务所需文件放在一起。服务只做推理；仿真端继续提供与训练输入一致的 RGB 图片、8 维 state 和原始六维力。

## 包含内容

- 双孔 raw6、USB raw6、双孔 encoder、USB encoder 四个推理 checkpoint，包含模型 `params` 和 `assets/` 中的归一化统计。训练续跑用的 `train_state` 不包含在推理包内。
- `encoder_seed42.pt` 和 encoder manifest。encoder SHA256：`06e5f3ee3ef69b3e6c9e2ef4bc0ebea249f6d3ef7da6da18b6171acfa2a6112e`。
- 完整 `src/`、`packages/`、`scripts/`、`configs/` 源码快照，以及 `pyproject.toml` 和 `uv.lock`。这些包括本地尚未推送 GitHub 的 Host9 服务及配置改动。
- 八份对应 Real/Sim LeRobot `meta/` 元数据和 PaliGemma tokenizer 文件。
- 本说明和模型、端口对应关系。

仿真推理不需要原始训练视频、HDF5 数据或 Pi0 base checkpoint。启动时需要 Linux x86_64、Python 3.11、NVIDIA CUDA 12 兼容 GPU、tmux 和网络连接（用于 `uv sync` 下载 Python 依赖）。压缩包不包含 `.venv`。

## 下载和解包

Windows PowerShell 将压缩包下载到 `D:\codes` 后，可在 Linux 仿真服务器上的 TAVLA 仓库根目录解包。压缩包根目录里的 `checkpoints/` 和 `eval_outputs/` 会放到仓库对应路径；再将 `simulation_test_kit/deployment_overlay/` 合入代码根目录。

```bash
cd /path/to/TA-VLA
tar -xf /path/to/tavla_host9_sim_test_20261027.tar -C .
cp -a simulation_test_kit/deployment_overlay/. .
export LEROBOT_HOME="$PWD/simulation_test_kit/lerobot_metadata"
export OPENPI_DATA_HOME="$PWD/simulation_test_kit/openpi_cache"
uv sync --locked
```

`uv sync` 完成后，在 GPU 0 启动四个服务：

```bash
PYTHONPATH=scripts:src:packages/openpi-client/src .venv/bin/python scripts/launch_host9_real_servers.py
```

服务地址与策略如下：

| 端口 | 仿真任务 | 力表示 | 模型配置 |
|---|---|---|---|
| 8000 | 双孔插入 | 原始六维力 | `pi0_lora_host9_dual_peg_raw6_cotrain` |
| 8001 | 插 USB | 原始六维力 | `pi0_lora_host9_usb_raw6_cotrain` |
| 8002 | 双孔插入 | 服务端 encoder | `pi0_lora_host9_dual_peg_encoder_cotrain` |
| 8003 | 插 USB | 服务端 encoder | `pi0_lora_host9_usb_encoder_cotrain` |

客户端 WebSocket 地址为 `ws://<推理服务器IP>:8000` 至 `8003`。同一台机器上测试时可用 `127.0.0.1`。启动脚本使用物理 GPU 0，并要求该卡至少有约 50 GB 空闲显存。

仿真客户端每条 observation 发送：

- `images.cam_high`、`images.cam_left_wrist`：RGB `uint8 (H,W,3)`。
- `state`：`float32 (8,)`，七个关节位置（rad）和训练记录格式的夹爪位置。
- `effort`：原始 `[Fx,Fy,Fz,Tx,Ty,Tz]`，`float32 (6,)`，使用训练数据相同的 base frame、符号和单位（N、Nm）。encoder 端口在服务端将 XYZ 力转换成 latent，仿真客户端不要自行编码。

每个 episode 保持同一 WebSocket 连接并调用客户端 `reset()`。encoder 模型需要按 10 Hz 维护新力样本历史；详细协议见 `docs/host9_real_four_servers_20261027.md`。

服务由 `scripts/serve_host9_real.py` 提供；它目前的文件名保留了真机部署时的命名，输入协议也适用于仿真。服务器仅返回 50 步绝对关节目标，不会替仿真器执行动作。
