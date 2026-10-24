# Force gap forecast v1

目标：把仿真力编码成真机等效力特征，恢复近期真机力、预测未来真机力，最后接入 TAVLA。
这是一条新流程，旧 encoder、旧数据集、8001/8005 server 和已有 checkpoint 保留。

## 1. 一条命令训练 encoder 原型

在 TAVLA 根目录运行，GPU_ID 指向外部 GPU 环境中的一张可用卡：

```bash
GPU_ID=0 bash scripts/run_force_gap_forecast_v1.sh
```

默认完整协议：种子 42/43/44、四折外层验证、18/6 内层选 epoch、四个模型变体、最多 200 epoch、patience 20、batch32（每个样本有8个相邻时刻）。
`--max-epochs` 等参数可以追加到命令末尾。减少种子、缺少消融或 motion-only 对照的运行只能保存诊断产物。
一次 epoch 的步数为 `ceil(训练帧数 / (32*8))`，每条轨迹有相同采样总质量；不靠重复无变化数据制造训练时长。
完整协议包括 teacher/student 选 epoch、外层重训和最终重训，比旧单次训练多；耗时以日志中的 optimizer steps 和 elapsed 为准。

输出是独立时间戳目录，包含：

- `manifest.json`：选中模型、SHA、特征布局、数据根路径、可用状态。
- `encoder_seed42/43/44.pt`：三个完整 teacher/student 和 decoder 权重。
- `data_audit.json`：40 对轨迹的 HDF SHA、复制状态判定和实际仿真状态来源。
- `outer_metrics.json`：逐种子、逐轨迹、逐轴、逐阶段指标，最后8帧单独报告。
- `acceptance.json`、`report.md`：预先固定门槛和未通过原因。
- `stages/*/history.json`、`curves.png`、`last.pt`：学习曲线、实际步数、所选及最后诊断权重。
- `predictions/`、`diagnostic_predictions/`：波形数据，真机8条诊断集不参与选模型。

中断后从已完成阶段恢复，未完成阶段重新训练：

```bash
GPU_ID=0 RUN_DIR=/absolute/path/to/run RESUME=1 bash scripts/run_force_gap_forecast_v1.sh
```

恢复时数据、代码、标签、参数必须与原 run 一致。修改后使用新目录。

### 数据限制与验收

当前 `sim-data-aligned` 的 q/dq 是复制真机状态，7188 帧全部保留用于原型。即使波形指标通过，状态最多为 `replay_validated`，不能据此启动新的 TAVLA 训练。
波形未通过状态为 `diagnostic`，权重和报告仍然保存，流程正常完成。

输入：32帧，每帧 `[Fxyz, ΔFxyz, q7, dq7, dt]`，21维；ΔF 是逐帧 N 差值，不是除以 dt 的导数。
真机目标为原始 `obs/state/ee_force_base`，单位 N，没有启动基线扣除、平滑、窗口幅值归一化或力通道截断。
历史 decoder 恢复最近8帧真机力；未来 decoder 预测未来5帧真机力。仿真力从不作为 decoder 监督目标。
高频表示当前约10Hz帧序列中的尖峰/快速变化；频带按 cycles/frame 报告，不假定时间戳完全均匀。

验收使用外层轨迹指标：相对 raw/affine/ARX 中最佳基线，当前+未来误差改善≥5%、2000次轨迹 bootstrap 下界>0；变化量、尖峰和高频误差不恶化>2%；优于同容量 motion-only 预测器；三个种子公开且无种子主误差恶化>2%。
配对特征只对齐26维学习通道；另外6维由物理力监督。学生训练时 teacher 与两个 decoder 的参数哈希必须保持不变。

## 2. 仿真主机补采实际状态

仿真入口在另一台主机，使用 `PairedReplayRecorder` 接入你现有的 replay 循环。不要把新 q/dq 拼到旧力文件。

```python
from force_gap_forecast_v1.capture import PairedReplayRecorder

recorder = PairedReplayRecorder(
    "sim-data-observed-v1/traj_0", "data/traj_0/data.h5",
    force_source="already_aligned",  # 或同标定 base 轴的 calibrated_link7
)
for real_frame_id in range(number_of_reference_frames):
    # 从同一次 replay、同一观测时刻读取；q/dq 必须是实际仿真状态。
    recorder.append(real_frame_id, force_xyz, observed_q, observed_dq,
                    sim_timestamp, normalized_gripper_pos, front_rgb, wrist_rgb)
recorder.close()
```

两路 RGB 图像须为 uint8 (480,640,3)。每条轨迹逐个参考 frame ID 采样；动作目标由 recorder 从配对真机参考轨迹读取，观测状态使用仿真实测值。
重复目录拒绝覆盖。缺失帧只保留诊断文件，不生成完整 `data.h5`。完成40条后复制整个新目录到训练主机：

```bash
GPU_ID=0 SIM_DIR=/absolute/path/to/sim-data-observed-v1 bash scripts/run_force_gap_forecast_v1.sh
```

此版本为可复现性完整重跑协议，不跨不同 run 自动复用 teacher；相同真机输入和种子会重新训练同一真机配方。
通过波形验收和实际仿真状态来源检查后，状态为 `policy_train_ready`。

## 3. 一条命令转换、统计并训练 TAVLA

```bash
MANIFEST=/absolute/path/to/accepted/run/manifest.json \
GPU_IDS=0,1,2,3 bash scripts/run_tavla_force_gap_forecast_v1.sh
```

脚本在任何训练前检查验收证据和 SHA。新 LeRobot repo ID 带 encoder SHA，拒绝覆盖旧缓存；已有相同完整合同可以复用。
真机和仿真共享一份 50:50 mean/std 策略归一化。新配置为
`pi0_lora_user_single_arm_force_gap_forecast_cotrain_50_50`。
从 Pi0 基础参数训练 LoRA，global batch8、4卡、30000步；本流程只保存新策略最终29999步，不删除任何旧权重。

两个入口分别控制 encoder 和策略训练；没有自动选择/占用 GPU，也没有启动旧 server。
策略中断恢复需指定已有 policy manifest 和相同 EXP_NAME：

```bash
POLICY_MANIFEST=/absolute/path/to/policy_manifest.json EXP_NAME=original_name \
GPU_IDS=0,1,2,3 RESUME=1 bash scripts/run_tavla_force_gap_forecast_v1.sh
```

## 4. Server 与在线合同

```bash
export PYTHONPATH=scripts:src:packages/openpi-client/src
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m force_gap_forecast_v1.serve \
  --policy-manifest /absolute/path/to/policy_manifest.json \
  --checkpoint-dir /absolute/path/to/checkpoints/29999 --port 8006
```

启动检查端口、策略合同和归一化；不会停止8001/8005。客户端必须发送 encoder SHA。

```python
from force_gap_forecast_v1.sim_adapter import TavlaAdapter, EpisodeTrace
from openpi_client.websocket_client_policy import WebsocketClientPolicy

adapter = TavlaAdapter("manifest.json", domain="sim")  # 真机使用 real
client = WebsocketClientPolicy(host="SERVER_IP", port=8006)
adapter.check_server(client.get_server_metadata())
adapter.reset()
client.reset()
trace = EpisodeTrace("episode0.npz")
# 每个新的10Hz观测都更新，即使该时刻没有请求策略。
adapter.update(aligned_force_xyz_n, actual_q_rad, actual_dq_rad_s, sim_timestamp_s)
obs = adapter.observation(front_rgb, wrist_rgb, normalized_gripper_pos)
actions = client.infer(obs)["actions"]
execute_absolute_joint_target(actions[1])
trace.append(adapter, actions[1])
trace.close()
```

`state=float32[8]`：7个关节rad +归一化夹爪。
`effort=float32[10,32]`：最旧在前，当前在最后；不足时复制首行。
32维布局为 **Fxyz(3)+ΔFxyz(3)+时序特征(26)**，前6维不是六维 wrench。
真机 F/ΔF 直接取测量值；仿真取 decoder 恢复的真机等效值。
`actions[50,8]` 为绝对关节位置+夹爪，沿用10Hz执行 `actions[1]` 的对比协议。
首帧立即可用且ΔF=0；不等待20帧扣基线。新 episode 调用 reset；时间戳重复/倒退报错，断档>0.5秒自动清空历史。
未来真机力输出 `[5,3]`，单位N，仅用于诊断。

仿真力合同：已经对齐的 `obs/state/ee_force_base` 不再取负。同定义、同标定且处于机器人 base 轴的 calibrated link7 wrench 使用 `aligned_force_from_calibrated_link7`，取前三维乘-1；不能直接用于另一种原始传感器。

## 5. 导出与验收

```bash
PYTHONPATH=scripts:src .venv/bin/python -m force_gap_forecast_v1.export \
  --manifest /absolute/path/to/run/manifest.json --output-dir exports/force_gap_forecast_new
```

把整个导出目录复制到仿真主机，安装 requirements.txt，在目录内调用 adapter/recorder。
目标主机运行：

```bash
python -m force_gap_forecast_v1.validate --manifest manifest.json --output runtime_checks.json
```

检查真实波形之外的接口性质，包括在线/离线≤1e-5、训练预处理一致、因果性、尖峰保留、断档，以及单线程CPU p95≤20ms。
本机静态/合成单元验收：`PYTHONPATH=scripts:src .venv/bin/python -m unittest force_gap_forecast_v1.tests -v`。

闭环入口是你仿真环境的 callback `runner(variant,host,port,seed,trace_path,encoder_manifest)`。
它重置环境、使用对应版本的 adapter、执行绝对动作、记录trace，并返回 `depth_error_m, remaining_depth_m, xy_error_m, failure_reason`。
`remaining_depth_m` 为有符号的剩余深度，正值表示未插够；`depth_error_m=abs(remaining_depth_m)`。这样深度不足与插过深不会混在一起。
旧两版保留原来的输入合同，新版使用上述 adapter。

```bash
python -m force_gap_forecast_v1.serial_eval --runner your_sim_runner:run_episode \
  --manifest manifest.json --host SERVER_IP --output-dir evaluation_new --fresh-seeds 100
```

顺序为8005→8001→8006，先相同40 seeds，再100个新 seeds。严格成功条件为深度误差≤3mm且XY<3mm。
输出逐episode结果和分组成功率，强制新版trace保存力、恢复力、未来力、q/dq、时间戳、实际执行动作和SHA。
另一台主机的仿真环境入口不在本仓库，callback需要接入当地环境；本机不宣称已完成闭环或真机成功率验收。
