# 多任务力 encoder：A100 上先训练 Real，再训练 Sim

本入口沿用 predictive v3 的双分支 TCN、32 维表示和原有力预测／重建损失。
只训练一次固定划分、一个种子；不训练 motion/control，不做多折、参数搜索或最终重训。
默认 alignment=0，Real 先训练，Sim 从所选 Real 权重初始化。

## 1. 获取代码和真机数据

在 A100 主机的仓库目录执行 `git pull --ff-only`。代码不包含原始数据和权重。
当前源主机已准备好以下数据包，包含旧插孔和三个新任务，保留原始 HDF5 和视频：

```bash
scp gujiawei@114.214.164.36:/data/workspace/gujiawei/TAVLA/exports/force_predictive_v3_host9_real_20261026.tar.gz /tmp/
mkdir -p ../tavla-host9-real-20261026
tar -xzf /tmp/force_predictive_v3_host9_real_20261026.tar.gz -C ../tavla-host9-real-20261026
```

数据包约 230 MiB，SHA256 为
`51ba3b17a62b4dd86d4ce67fa2ed38b414644cf80f2ee18cbd4951edf05ec087`。
可用 `sha256sum /tmp/force_predictive_v3_host9_real_20261026.tar.gz` 核对。

数据包解压目录应为：

```text
tavla-host9-real-20261026/
  data/traj_*/data.h5                         # 原有 peg-in-hole，40 条
  data-host9-three-20261025/
    dual_peg_insert_right/traj_*/data.h5       # 40 条
    plate_sliding_right/traj_*/data.h5         # 30 条
    usb-insertio_right/traj_*/data.h5          # 40 条，保留原目录名拼写
```

也可自行迁移 `TAVLA/data/` 和 `TAVLA/data-host9-three-20261025/`，保持目录结构。
配置为 `configs/force_predictive_v3_host9_multitask.json`；相对路径以 `DATA_ROOT` 为根，
绝对路径也可使用。缺少任务会报错，不会悄悄跳过。

复用 A100 上已有的 CUDA PyTorch 环境，需安装 `numpy`、`h5py`、`scipy`；不需要为
encoder 单独安装 JAX、LeRobot 或视频依赖。设置 `PYTHON_BIN=/path/to/python` 可选择环境；
默认使用仓库的 `.venv/bin/python`。先确认该环境的 `torch.cuda.is_available()` 为 True。
在 `tmux` 中运行以下训练，或自行用 `nohup` 保持进程。

## 2. 现在就可以训练 Real teacher

```bash
DATA_ROOT="$PWD/../tavla-host9-real-20261026" \
GPU_ID=0 STAGE=teacher \
RUN_DIR="$PWD/eval_outputs/force_predictive_v3_multitask/host9_teacher" \
  bash scripts/run_force_predictive_encoder_v3_multitask.sh
```

这一阶段完全不需要 Sim 文件，只读取每条轨迹的：

- `obs/state/ee_force_base`：`[N,3]`，robot base 坐标轴、N；
- `timestamps`：`[N]`，秒，严格递增。

`q/dq`、动作、图像和阶段标签均不参与该入口。encoder 历史为 32 帧，重建最近 16 帧，
共享预测头预测未来 5 个采样点的真机力。它们都在 Real 阶段训练。
不做启动前 20 帧扣基线，也不乘 3.4。

默认 seed42，按任务分别以整条轨迹进行 80/20 划分：150 条中 120 条训练、30 条验证。
采样先保证任务等权、任务内轨迹等权，再使用一半均匀／一半事件采样；验证也先按轨迹、
再按任务等权。两域不拆开划分，同一真机轨迹与回放始终属于同一组。
归一化只由训练轨迹拟合。断档超过 0.5 秒时切分片段，历史与未来标签不跨断档。

默认 AdamW lr=3e-4、weight decay=1e-4、batch32、每阶段最多200轮、patience20。
按 `task_equal_future_score_N / mean(real_train_std) + 0.25*task_equal_reconstruction_nrmse`
选最佳权重；保留 v3 原有差分、频谱、峰值等损失。

运行参数可以加在脚本后面，例如 `--max-epochs 200 --patience 20 --batch-size 32`。
单次使用一张可见 GPU；`GPU_ID` 或预先设置的 `CUDA_VISIBLE_DEVICES` 控制卡号。
尚无 A100 耗时保证，日志每10轮记录一次实测耗时。

只检查真实数据、划分和统计而不开始训练：

```bash
DATA_ROOT="$PWD/../tavla-host9-real-20261026" \
RUN_DIR="$PWD/eval_outputs/force_predictive_v3_multitask/host9_audit" \
  bash scripts/run_force_predictive_encoder_v3_multitask.sh --audit-only
```

检查和正式训练使用不同输出目录；若要在检查目录继续，添加 `RESUME=1`。

## 3. rollout 完成后训练 Sim student

配置中的 `sim_root` 也以同一个 `DATA_ROOT` 为根。将旧插孔回放放到 `sim-data-aligned/`，
三个新任务的回放放到 `sim-data-host9-three-20261025/<task>/traj_*/data.h5`，
或者修改配置中的 Sim 路径。Real 路径可以随机器迁移，但 HDF5 内容必须与 teacher 阶段一致。

每个任务要求 Real/Sim 轨迹 ID、帧数对应。新任务默认要求 `pairing/real_frame_id` 为
`0..N-1`；若存在 `pairing/real_timestamp_s`、`paired_real_sha256`，也会与真机文件核对。
若已有采集器确实保证每个参考帧恰好对应一条回放观测，可在该任务配置中明确添加
`"pairing": "frame_index"`。这只是采集合同声明，不代表程序从波形推断或验证了物理接触一致。
旧插孔数据已按这种声明配置。程序不做 DTW 或事后曲线配准。

默认 Sim 输入同样为 `obs/state/ee_force_base`。其他键可用 `sim_force_key` 指定；
只有同定义、同标定且已处于 base 轴下的 calibrated link7 力，才设置
`"sim_force_source": "calibrated_link7"`，程序对前三维取负、倍率1。
默认 `aligned_base_xyz_n_v1` 来源不会再次取负。

```bash
DATA_ROOT="$PWD/../tavla-host9-real-20261026" \
GPU_ID=0 STAGE=student \
TEACHER_RUN="$PWD/eval_outputs/force_predictive_v3_multitask/host9_teacher" \
RUN_DIR="$PWD/eval_outputs/force_predictive_v3_multitask/host9_student" \
  bash scripts/run_force_predictive_encoder_v3_multitask.sh
```

Student 直接继承 teacher 的种子、轨迹划分、Real 输入统计和真机目标统计；只新增
Sim 训练集输入统计。冻结 Real encoder、Real 重建头和共享未来力预测头，训练 Sim
encoder 与本域重建头；冻结预测头仍允许梯度传回 Sim encoder。

两阶段使用同一 Git revision。变更 Real 数据、输入力字段或训练源码会拒绝复用 teacher。
此纯力入口不认证 Sim 状态、动作和视频的来源，`actual_sim_state_verified` 保持 False。

## 4. 输出与恢复

每个阶段的 `RUN_DIR` 包含：

- `encoder_seed42.pt`、`manifest.json`：所选模型和可移植接口；不包含运动辅助头训练成果。
- `history.json`、`train.log`：逐轮 loss、按任务验证误差、所选 epoch 和实测时间。
- `metrics.json`、`report.md`：逐任务／轨迹预测、重建、频谱及 latent 诊断。
- `split.json`、`normalization.json`、`recipe.json`、`data_audit.json`：划分、训练集尺度、源码／数据哈希。
- `predictions/validation/<task>/traj_*.npz`：所选模型的验证预测、latent 与时间戳。
- `best.pt`：当前最佳模型；`last.pt`：最后完整 epoch 的模型、优化器、最佳权重和 RNG 状态。

中断后使用相同数据、超参数和 `RUN_DIR`，添加 `RESUME=1`。从最后完整 epoch 恢复，
未完成的 epoch 重跑。恢复支持更换物理 GPU；精确的浮点重现仍取决于设备和环境。
已完成阶段会校验并复用，不重复训练。不要把 teacher 的输出目录当作 student 的输出目录。

Teacher 导出只允许 `domain="real"`；请求 Sim 分支会报错。
Student 导出允许两域，通过现有 `ForcePredictiveEncoder` 加载，latent32、effort历史 `[10,32]`：

```python
from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
encoder = ForcePredictiveEncoder("/path/to/host9_student/manifest.json", domain="sim")
z = encoder.push(force_xyz_n, timestamp_s)
future_real = encoder.predict_future()
```

此处完成的是 encoder 训练。多任务 TAVLA 仍需用该固定 encoder 生成各任务数据和统计，
保留各任务自己的 prompt；原有固定40条、固定插孔 prompt 的策略脚本不能原样用于这批数据。

CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest force_predictive_encoder_v3.test_multitask force_predictive_encoder_v3.tests -v
```

2026-10-26 已通过4项多任务集成测试和15项原有 v3 测试，并用本批150条真实轨迹在 CPU
完成1个 epoch 的流程检查（120训练／30验证，含导出及逐任务报告）。该短跑只验证代码与
数据衔接，不用于判断收敛效果；尚未在 A100 上测量吞吐。
