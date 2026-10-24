# 五个模型服务的仿真串行测试交接清单

核对日期：2026-10-08。以下端口已逐一读取正在运行服务的 WebSocket 握手；checkpoint 均为 `29999`。GPU 主机为 `linke5`，主网卡地址为 `114.214.164.36`。仿真侧的跨主机网络连通性需要从仿真主机确认。

## 1. 五个模型及测试顺序

请按 `8000 → 8001 → 8002 → 8003 → 8004` 串行测试：完成一个模型的全部测试 episode，再切换到下一个模型。同一时刻只有一个评测环境和一个模型客户端执行推理。

| 顺序 | 端口 / 测试 ID | 模型 | 训练数据 | 除共同输入外的力输入 | 输出 |
| --- | --- | --- | --- | --- | --- |
| 1 | `8000` / `force_align_latent_real_only` | ForceAlignmentEncoder TAVLA，纯真机 | 真机 | `effort`: float32 `(10,32)`，encoder latent 历史 | `actions`: `(50,8)`，绝对关节位置与夹爪目标 |
| 2 | `8001` / `force_align_latent_cotrain_50_50` | ForceAlignmentEncoder TAVLA，Cotraining | 真机/仿真，采样权重 50:50 | 同上；使用与 8000 完全相同的冻结 encoder | 同上 |
| 3 | `8002` / `pi0_real_no_force` | Pi0，无力，纯真机 | 真机 | 不发送 `effort` | 同上 |
| 4 | `8003` / `pi0_cotrain_no_force` | Pi0，无力，Cotraining | 真机/仿真，采样权重 50:50 | 不发送 `effort` | 同上 |
| 5 | `8004` / `tavla_real_raw_6d_wrench` | 原始六维末端力 TAVLA，纯真机 | 真机 | `effort`: float32 `(1,6)`，当前帧 `[Fx,Fy,Fz,Tx,Ty,Tz]` | 同上 |

本次测试集包含这五个模型。原始六维力 TAVLA 的 Cotraining checkpoint 当前未部署、也不在此清单中。

连接地址为 `ws://114.214.164.36:<端口>`。服务器监听的 `0.0.0.0` 是绑定地址；外部客户端请使用服务器 IP。

### 训练数据的区别

- 8000：`local/tavla_single_arm_force_align_latent`。
- 8001：上述真机数据 + `local/tavla_single_arm_force_align_latent_sim`，后者来自 `sim-data-aligned`。
- 8002：`local/tavla_single_arm`。
- 8003：上述真机数据 + `local/tavla_single_arm_ee_wrench_sim_hand_noise_50`；模型不读取该数据集的力通道。
- 8004：`local/tavla_single_arm_ee_wrench`，读取真机原始 `obs/state/ee_wrench_base`。

8001 和 8003 使用的仿真训练数据版本不同。比较两者时，结果同时受模型输入表示和训练数据版本影响。

## 2. 共同输入：所有五个模型都需要

每次请求发送一个 Python 字典，经现有 `openpi_client` 序列化。接口是 WebSocket + 支持 NumPy 的二进制 MsgPack。

```python
import numpy as np

obs = {
    "images": {
        "cam_high": front_rgb,
        "cam_left_wrist": wrist_rgb,
        "cam_right_wrist": wrist_rgb,
    },
    "state": np.asarray([q0, q1, q2, q3, q4, q5, q6, gripper_pos],
                        dtype=np.float32),
    "prompt": "peg-in-hole",
    # 根据端口，在第 3 节添加 effort，或完全省略 effort。
}
```

| 字段 | shape / dtype | 语义与单位 |
| --- | --- | --- |
| `images.cam_high` | `(H,W,3)` / uint8 | 当前前视相机 RGB 图像 |
| `images.cam_left_wrist` | `(H,W,3)` / uint8 | 当前腕部相机 RGB 图像 |
| `images.cam_right_wrist` | `(H,W,3)` / uint8 | 单臂训练使用同一张腕部图像；直接复制 `cam_left_wrist` |
| `state` | `(8,)` / float32 | `[joint_0,...,joint_6,gripper_pos]`；七个关节角单位 rad，顺序对应 Panda joint1–joint7 |
| `prompt` | 非空字符串 | 五个模型统一固定为 `peg-in-hole` |

具体要求：

1. 使用当前**测量到的**关节位置和夹爪位置构造 `state`，而不是上一帧发出的目标。
2. 夹爪位置采用 `0=闭合、1=最大张开` 的归一化总开口宽度：`gripper_pos = total_width_m / 0.08`。若仿真只给一个手指的位移，要先得到两指总宽度。真机 traj_0 中约 `0.0865` 对应总宽度约 `0.00692 m`，可作单位核对。
3. 图像必须 RGB uint8。OpenCV BGR 应先转 RGB，RGBA 应去掉 alpha；不要把 `[0,1]` 浮点图像直接发给 encoder 服务。
4. 训练原始图像为 `480×640`。服务会进行模型需要的缩放；建议仿真侧保持相同分辨率、相机方向和合理匹配的视角。
5. 两路相机、state、当前力都来自同一仿真时刻。发送单个观测，无 batch 维度。
6. 服务已支持省略 `cam_right_wrist` 时镜像左腕图像。为了统一五个模型的请求，建议明确发送同一张腕部图像到两个键。
7. 所有数值应有限。8000/8001 对 state 和 effort **严格要求 float32 及指定 shape**，Python list 或 float64 会被拒绝。
8. 客户端发送物理量或 encoder 原始 latent。VLA 输入归一化由服务使用 checkpoint 自带 norm stats 完成；客户端不要再次做 z-score、quantile norm 或补到 32 维。

## 3. 每个模型的力输入和注意事项

### 3.1 8000：纯真机 ForceAlignmentEncoder TAVLA

完整输入 = 第 2 节共同输入 + 下列字段：

```python
obs["effort"] = latent_history.astype(np.float32)  # (10,32)
```

- 仿真侧先采集**已对齐定义的 base 坐标系三维力** `[Fx,Fy,Fz]`，单位 N。
- 在仿真侧用第 4 节的冻结 encoder 生成 32 维 latent，再形成 10 帧 latent 历史发送给服务器。
- 服务器接收 latent，不会把原始三维/六维力自动变成 encoder latent。
- encoder 在线输入只有力及时间戳；不要求末端 pose 输入给 encoder。
- 模型训练数据只有真机，仿真测试仍须使用相同 encoder 和力定义。
- 输出为第 5 节定义的 `(50,8)` 绝对目标序列。

### 3.2 8001：Cotraining ForceAlignmentEncoder TAVLA

输入和输出与 8000 完全一致：共同输入 + float32 `(10,32)` 的 `effort`，输出 `(50,8)` 绝对目标序列。

- 使用与 8000 **同一份 encoder 权重和同一套因果预处理**。
- 真机/仿真采样权重为 50:50；仿真训练数据的力来自 `sim-data-aligned`。
- 不能为该端口更换为另一套 encoder，也不能仅因是 Cotraining 就省略力的定义对齐或启动标定。
- 8000/8001 比较时，相机、初始状态、force 来源、encoder、控制周期和动作行索引保持一致。

### 3.3 8002：纯真机 Pi0，无力输入

完整输入就是第 2 节字典，**省略 `effort` 键**。

- 需要相机、七关节位置、归一化夹爪位置和 prompt。
- 不需要 encoder、力历史、三维力、六维力或关节力矩。
- 不要为了适配统一接口添加零 `(10,32)` 或零 `(1,6)` 的 `effort`。
- 仿真仍可计算和记录接触力，作为评测指标；这些力不作为该模型的输入。
- 输出仍是 `(50,8)` 绝对目标序列。

### 3.4 8003：Cotraining Pi0，无力输入

输入和输出与 8002 完全一致：共同输入、不发送 `effort`，输出 `(50,8)` 绝对目标序列。

- 真机/仿真采样权重为 50:50。
- 仿真训练数据目录名虽然含 `ee_wrench`，当前 Pi0 的配置是 `EffortType.NO`，不读取力。
- 不要调用 force encoder，也不要额外发送力字段。

### 3.5 8004：纯真机原始六维力 TAVLA

完整输入 = 第 2 节共同输入 + 当前帧六维 wrench：

```python
wrench_base = np.asarray([Fx, Fy, Fz, Tx, Ty, Tz], dtype=np.float32)
obs["effort"] = wrench_base[None, :]  # (1,6)，当前帧，无时间历史
```

| 通道 | 含义 | 单位 |
| --- | --- | --- |
| 0–2 | 末端受力，robot base XYZ 坐标轴 | N |
| 3–5 | 末端力矩，robot base XYZ 坐标轴 | N·m |

注意事项：

1. 对应真机训练字段 `obs/state/ee_wrench_base`。力和力矩的坐标轴、符号、传感器定义、力矩参考点都必须与该字段匹配。仅把 local 力旋转到 base 轴，不能保证参考点不同的力矩也已匹配。
2. 使用完整六维 wrench。七维关节力矩、三维力、encoder latent、10 帧历史均不属于这个模型的输入。
3. 该模型没有采用 8000/8001 的 encoder 预处理。不要自动套用 encoder 的首 20 帧减基线，也不要套用旧的幅值仿射映射或手动标准化。保留传感器本身必要的标定，使其定义符合训练字段。
4. `sim-data-aligned/build_aligned_sim.py` 会**将后三维力矩置零**，用于当前 force-only encoder 数据。若把该版本的 `ee_wrench_base` 直接拿来测 8004，实际上是“缺失力矩”的输入；须标注这个限制，不能当作完整六维传感器测试。
5. 第 4 节 `-calibrated_link7[:3]` 的公式只核实了 encoder 的三维力合同，不能据此推定原始六维 wrench 的六个通道都应取负。8004 的完整 wrench 转换须由仿真传感器定义和真机字段共同确认。
6. 输出为 `(50,8)` 绝对目标序列。该模型只有真机训练，仿真力幅值或力矩定义差异会影响结果，应在评测记录中注明实际采用的 wrench 来源和转换。

## 4. 8000/8001 的 encoder 合同

### 4.1 交接文件和权重身份

已核对当前 zip 的 adapter、client、示例和 encoder 权重与目录中相应文件一致。仿真侧可直接复制：

```text
/data/workspace/gujiawei/TAVLA/exports/force_align_latent_sim_bundle_20261004.zip
```

解压目录含 `sim_adapter.py`、`assets/encoder.pt`、`openpi_client/`、encoder 推理实现、依赖文件、force 合同和示例。

权重格式为 `tavla_force_alignment_encoder_terminal_reference_v1`，SHA256 必须为：

```text
db793ebd4f3158b8a8add895cf04368b2289df7fad450f1d2a39b09bb46acd80
```

不要用 `force_gap_encoder_v2/encoder.pt`、`motion_encoder.pt` 或其他新权重替换：这两个 policy 训练时使用的是上面的冻结 encoder。encoder 可在仿真侧 CPU 运行，建议 `torch.set_num_threads(1)`。

### 4.2 仿真力来源、坐标和符号

已经核实的 `sim-data-aligned` 合同是：

```text
obs/state/ee_force_base
    = -1.0 × sim/force/panda_link7_wrench_calibrated[..., :3]

scale = 1.0
单位 = N
坐标轴 = robot base XYZ
```

- 如果仿真输出已经是同定义的 `obs/state/ee_force_base`，直接送给 adapter，不再次取负。
- 如果来源是**与原采集器同定义、同标定且已处于 base 轴下**的 calibrated link7 wrench，使用 `aligned_force_from_calibrated_link7(wrench)`；该函数截取前三维并取负。
- 原始 body wrench、local/world contact force 或不同传感器的 wrench，须先核对坐标与受力方向，不能仅凭名称套用负号。
- 使用 `scale=1.0`。源文件开头旧说明中的 `3.4` 与当前可执行代码及合同不一致，不使用该旧倍率。
- encoder 只使用三维力；传感器后三维力矩不进入此 encoder。

### 4.3 每个 episode 的初始化和连续更新

1. `env.reset(seed=...)` 后，调用 `adapter.reset()` 和 `client.reset()`。
2. 保持姿态，避免新的插入接触，以 **10 Hz 仿真时间采样**收集前 20 个三维力样本，约 2 秒。不要用正在接触/插入的样本估计自由空间基线。
3. adapter 对前 20 个样本逐轴取 median 作为本 episode 的固定基线；第 1–19 次更新返回 `None`，第 20 次返回可用的 `(10,32)` effort 历史。
4. 首次准备好时会回填启动阶段 latent；32 帧 encoder 窗口不足时按训练规则左侧补齐，不需要另等 32+10 个样本。
5. 每推进 0.1 秒仿真时间，提供一次新的力样本和严格递增的**仿真时间戳（秒）**。网络等待的墙钟时间不进入导数。
6. adapter 内部完成：基线残差 → 按秒求导 → 六个 feature 的保存尺度归一化与截断 → 32 帧窗口 → 32 维 latent → 最近 10 帧 latent 历史。
7. effort 行顺序为**旧到新**，最后一行为当前时刻：`[z(t-9),...,z(t)]`。不要转置成 `(32,10)`，也不要加 batch 维度。
8. 若一次执行多个动作，期间仍须每 0.1 秒更新 force encoder；不能只在请求模型时更新。
9. 一个力样本只 update 一次。不要在 `build_observation` 前重复喂当前样本，也不要提前读取未来力。
10. 切换 episode、重置环境或时间戳回跳时，清空 encoder 的基线和两个历史队列。`client.reset()` 只 reset 服务端 policy 的随机状态，不会重置仿真侧 encoder。

最小使用方式：

```python
import torch
from sim_adapter import TavlaSimulationAdapter

torch.set_num_threads(1)
adapter = TavlaSimulationAdapter(device="cpu")
adapter.reset()

# 每个新仿真 tick 调用一次；force_xyz_n 已满足上述 force 合同。
history = adapter.update_force(force_xyz_n, simulation_timestamp_s)
if history is not None:
    obs = adapter.build_observation(
        front_rgb=front_rgb,
        wrist_rgb=wrist_rgb,
        joint_pos_rad=joint_pos_rad,  # (7,)
        gripper_pos=gripper_pos,
        prompt="peg-in-hole",
    )
    # obs["effort"] 已经是 float32 (10,32)。
```

## 5. 共同输出与动作执行

五个服务都返回：

```python
{
    "actions": np.ndarray,  # shape (50,8)
}
```

每行：

```text
[q0_target, q1_target, q2_target, q3_target,
 q4_target, q5_target, q6_target, gripper_target]
```

- 前七列是**绝对关节位置目标，单位 rad**。服务已完成动作反归一化和 delta→absolute 转换；仿真侧不要再加当前关节角。
- 第八列是归一化夹爪目标，`0=闭合，1=最大张开`。若控制器需要总宽度米，转换为 `clip(gripper_target,0,1) * 0.08`；若需要两指各自位置，按控制器定义分配。
- 这是关节位置序列；不能直接当成末端位姿、速度、关节力矩或无单位的增量。
- 输出已裁为八维，不需要处理内部的 32 维 padding。
- 实际新服务 smoke test 返回 float64；建议统一 `np.asarray(..., dtype=np.float32)`，并检查 `(50,8)` 与全部 finite。
- 50 行对应 10 Hz 的约 5 秒名义动作窗口。首轮评测建议同步闭环、每次只执行一个未来动作后重规划。

### 5.1 首轮统一使用 `actions[1]`

转换器保存同一帧的 `action/actual`。本次抽查五个训练数据来源的 episode_000000，`action[t]` 均与 `state[t]` 相同；现有部署示例也使用 `actions[1]`。因此第 0 行主要对应当前时刻，第 1 行约对应 `t+0.1 s`。

```python
result = client.infer(obs)
actions = np.asarray(result["actions"], dtype=np.float32)
if actions.shape != (50, 8) or not np.isfinite(actions).all():
    raise ValueError("Invalid model action chunk")

target = actions[1].copy()  # 索引 1，即第二行
target[7] = np.clip(target[7], 0.0, 1.0)
# 调用仿真已有的关节位置目标控制器，推进 0.1 秒仿真时间。
# 所有模型使用相同的关节限位、速度限制和夹爪控制规则。
```

如果后续改成每次执行 `k` 步，统一使用 `actions[1:k+1]`，每步 0.1 秒，encoder 每步更新。五个模型使用相同的 `k`，并将策略写入结果。首次测试不要直接开环执行全部 50 行，也不要把多个模型的动作队列混在一起。

### 5.2 仿真时间与实际耗时

建议同步执行：取观测 → 请求并等待模型 → 应用目标 → 推进 0.1 秒仿真 → 取新观测。等待网络期间暂停仿真时间。若物理引擎 timestep 为 0.002 秒，每个模型 tick 对应 50 个物理子步。

10 Hz 描述训练/控制时间间隔，不代表这些服务可保证墙钟实时 10 Hz。新部署的 8002/8003/8004 已记录热请求约 `0.190/0.204/0.205 s`；首个请求因 JIT 约 `29.9/35.1/30.3 s`。仿真侧为首请求留足等待时间，例如 120 秒；将编译热身耗时与任务指标分开记录。

## 6. 客户端协议、身份核对与 reset

解压上述 bundle 后，可直接使用其中的 `openpi_client`；同一个客户端协议适用于全部五个端口。

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="114.214.164.36", port=8002)
metadata = client.get_server_metadata()
client.reset()  # 每个 episode 开始前调用
result = client.infer(obs)
# 全部 episode 完成/切换端口时关闭连接；当前 client 没有独立 public close 方法。
client._ws.close()
```

- 客户端连接后首先接收一次 metadata，客户端实现已处理这一帧。自行实现协议时，不能把 metadata 当动作返回值。
- 每个 infer 请求等到完整响应再发送下一次。不要使用 HTTP POST、JSON ndarray list 或并发请求替代现有协议。
- reset 的 wire 请求是 `{"reset": True}`，响应是 `{"reset": True}`；使用当前带 `.reset()` 的客户端版本。
- 8000 的 `policy_variant` 应为 `force_align_latent_real_only`；8001 应为 `force_align_latent_cotrain_50_50`。二者都应有第 4 节 hash，以及 `effort_shape=[10,32]`、`action_shape=[50,8]`。
- 8002/8003/8004 当前使用通用 server，握手 metadata 为 `{}`，这是当前实现的正常表现。按端口与部署清单核对模型，不对这三个端口套用 encoder metadata 检查。
- policy 随机状态属于该服务的 policy 实例。评测期间每个端口保持一个评测客户端，避免其他客户端插入请求/reset，影响随机序列和时延。
- 出错时服务可能返回文本 traceback 并关闭当前连接；当前 client 会抛出异常。将该 episode 标记为接口/推理失败，不继续执行上一份动作。

## 7. 串行测试步骤与公平比较

为全部模型固定一份测试清单，包括 seed、机器人初始状态、夹持状态、物体/孔初始位姿、相机参数、物理参数、控制器限制、成功判据和任务时长。

对 `8000 → 8001 → 8002 → 8003 → 8004` 依次执行：

1. 只连接当前端口，核对身份；如需 JIT 热身，使用一份符合该模型输入的观测，热身后重新 reset policy 和环境再开始计分。
2. 对每一个固定 seed，恢复相同环境初始状态，清空上一 episode 的动作队列、观测缓存和统计，调用 `client.reset()`；encoder 模型还须 `adapter.reset()`。
3. 所有模型先执行相同的约 2 秒姿态保持阶段。8000/8001 用此阶段收集 20 个基线样本；另外三个模型保持相同姿态和时长，使开始插入时的环境状态可比。保持末端与目标接触区分离，允许正常夹持物体。
4. 20 个采样点可为 `t=0.0,0.1,...,1.9`。随后保持推进到 `t=2.0`，从新的观测进入正式闭环；encoder 在新的 tick 正常再更新一次。
5. 从正式闭环开始计任务时长。使用同一策略：10 Hz 仿真时间、`actions[1]`、每次执行一行、同步重规划。
6. encoder 模型持续更新力与 latent；Pi0 不向模型传力；raw6D 模型每次只传当前完整六维 wrench。
7. 达到相同成功判据、超时、控制异常或接口失败时结束 episode，保存记录，再 reset 到下一个 seed。
8. 当前模型完成全部 seed 后关闭客户端，清空缓存，再切换下一个端口。五个服务器保持运行，无需在测试过程中反复启动。

任务时长、episode 数和成功判据由仿真任务定义统一给定；例如使用固定 300 个正式控制 tick，则是 30 秒仿真任务窗口，初始约 2 秒保持阶段另记。

至少记录：

| 类别 | 字段 |
| --- | --- |
| 模型身份 | 测试 ID、端口、checkpoint、encoder hash（适用时） |
| 复现信息 | seed、初始关节/夹爪、物体与孔位姿、相机与物理配置 |
| 接口信息 | state/effort shape、force/wrench 来源、坐标变换与符号、是否完整包含力矩 |
| 控制方式 | tick=0.1 s、action index=1、每次执行步数、夹爪换算和目标限幅策略 |
| 任务结果 | 成功/超时/接口失败/控制失败、结束原因、正式控制 tick 数与仿真时长 |
| 性能和轨迹 | 首次/热推理耗时、关节和目标轨迹、插入指标、接触力峰值；按需保存 RGB 与 latent |

## 8. 精确 checkpoint 与现有服务位置

以下相对路径均以 `/data/workspace/gujiawei/TAVLA/` 为根。

### 8000

```text
checkpoints/pi0_lora_user_single_arm_control_latent/force_align_latent_gpu0123_fsdp/29999
```

config 名称虽含 `control_latent`，正在运行的服务已经通过 metadata 核实为当前 force-only ForceAlignmentEncoder 版本。

### 8001

```text
checkpoints/pi0_lora_user_single_arm_force_align_latent_cotrain_50_50/force_align_latent_cotrain_50_50_v1/29999
```

### 8002

```text
checkpoints/pi0_lora_user_single_arm/pi0_no_effort_30k/29999
```

### 8003

```text
checkpoints/pi0_lora_user_single_arm_cotrain_no_effort_50_50/no_effort_cotrain_50_50_bs8_30k/29999
```

### 8004

```text
checkpoints/pi0_lora_user_single_arm_ee_wrench/tavla_real_gpu45_bs4_30k_20260913/29999
```

| 端口 | 物理 GPU 索引 | tmux pane |
| --- | --- | --- |
| 8000 | 5 | `force:0.0` |
| 8001 | 7 | `tavla:0.0` |
| 8002 | 4 | `pi:0.0` |
| 8003 | 5 | `train:0.0` |
| 8004 | 7 | `test:0.0` |

8000/8001 为原有 encoder 服务；本次读取握手未执行推理或 reset。

## 9. 已有检查的范围

- 本次逐一连接五个端口，已获取握手 metadata。
- 新部署 8002/8003/8004 的记录数据离线 WebSocket 推理、reset、输出 shape 与 finite 检查已通过，见 `logs/offline_three_servers_20261008_210843.json`。
- 8000/8001 的已有 smoke 报告分别为 `logs/force_align_latent_real_server_smoke.json`、`logs/force_align_latent_cotrain_server_smoke.json`。
- 服务部署记录为 `logs/deployment_three_servers_20261008_210843.json`。
- 这些检查证明协议和推理可运行；仿真闭环成功率、插入效果及原始六维仿真传感器的完整定义匹配仍由本次串行评测验证。

## 10. 核对依据

输入/输出依据当前 `scripts/serve_force_align_latent_real.py`、`scripts/serve_force_align_latent_cotrain.py`、`scripts/serve_policy.py`、`src/openpi/policies/tavla_policy.py`、`src/openpi/policies/policy_config.py`、`src/openpi/transforms.py` 与训练配置。

encoder 合同依据 bundle 中的 `sim_adapter.py`、`reference/sim_force_source_contract.json`、encoder 推理/预处理代码及 `sim-data-aligned/build_aligned_sim.py` 的当前常量。动作时间对应关系同时核对了五个训练数据来源的 episode_000000 parquet、真机 traj_0 HDF5、转换器和现有部署示例。
