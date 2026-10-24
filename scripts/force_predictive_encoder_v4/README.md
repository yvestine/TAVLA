# Force predictive encoder v4

32 维纯力表示，共享预测头输出当前及未来 5 步的**真机三轴力**。复用 v3 的因果输入与双分支 TCN encoder；不读取 q/dq/动作，不做 Sim 自身重建、latent 对齐或旧模型蒸馏。

```text
loss = current_huber + future_huber + 0.5 * increment_huber
```

current/future 分别按有效元素取均值，使用真机训练集力标准差；increment 使用连续帧真机力增量标准差，覆盖当前到 +1 以及 +1 到 +5。不对增量除以不均匀 dt。全部尺度只用训练轨迹拟合并保存；缺失、补齐和跨断档目标不参与 loss。

先训练 Real encoder 和共享六步预测头，再固定两者、训练 Sim encoder。固定 head 的参数仍允许梯度传回 Sim encoder。两阶段都只按验证集未来真实力 `.25*RMSE(+1)+.35*RMSE(+3)+.40*RMSE(+5)` 选 epoch，记录当前力和变化误差。

在外部 GPU 主机的 TAVLA 根目录运行：

```bash
GPU_IDS=6 bash scripts/run_force_predictive_encoder_v4.sh
```

默认 seed 42、fold 0：24 条训练、8 条早停验证、8 条另外诊断；200 epoch/阶段、patience 20、batch 32、AdamW lr 3e-4。轨迹等权，一半均匀、一半偏向变化事件的块采样。每个块 16 个端点，每个端点读取最多 32 帧历史；首帧历史左补齐，但补齐端点不作为监督。不是多折交叉验证，诊断轨迹过去已被查看，不称为盲测。

指定输出位置和训练参数：

```bash
GPU_IDS=6 RUN_DIR="$PWD/eval_outputs/force_predictive_encoder_v4/run1" \
  bash scripts/run_force_predictive_encoder_v4.sh \
  --seed 42 --validation-fold 0 --max-epochs 200 --patience 20 --batch-size 32
```

该入口只使用一张明确指定的物理 GPU，不启动策略训练。`REAL_DIR`、`SIM_DIR`、`LABELS`、`PYTHON_BIN` 可覆盖；默认数据是 `data/` 和 `sim-data-aligned/`。阶段标签仅用于分组指标，不参与主 loss，注释文件缺省时仍能运行。

恢复：

```bash
GPU_IDS=6 RESUME=1 RUN_DIR="$PWD/eval_outputs/force_predictive_encoder_v4/run1" \
  bash scripts/run_force_predictive_encoder_v4.sh
```

恢复时需重复原来传入的超参数，代码、数据、标签和超参数必须一致。已完成阶段校验并复用，中断阶段从固定父模型重新训练，不恢复中断 epoch 的 optimizer。新实验使用新目录，不覆盖旧模型。`--delta-weight 0` 可单独做无变化项对照。

产物：

- `report.md` / `result.json`：当前力、未来 +1/+3/+5、覆盖第一步的增量、逐轴、事件、阶段和流式变化误差；latent 有效秩、第一主成分占比与通道标准差。
- `teacher/history.json` / `student/history.json`：每轮分项损失、梯度范数、验证指标；`model.pt` 是选择权重，`last.pt` 是最后状态。
- `encoder_seed42.pt` / `manifest.json`：带新格式的权重，记录目标域为 Real；仅标为单次划分诊断。
- `predictions/{validation,diagnostic}/{real,sim}/traj_*.npz`：当前/未来预测、latent、真值、有效掩码和两域时间戳。
- `normalization.json` / `recipe.json` / `data_audit.json` / `source_snapshot/`：训练尺度、损失/选模配置、来源和源码快照。
- `baseline_coefficients.npz`：训练集拟合的仿真当前力仿射校准，以及两域 force-only AR，固定 ridge alpha=1。Real 保持基线使用真实当前力，Sim 基线只使用 Sim 当前/历史力。

流式推理：

```python
from force_predictive_encoder_v4.runtime import ForcePredictiveEncoder

encoder = ForcePredictiveEncoder("path/to/run/manifest.json", domain="sim")
z = encoder.push(force_xyz_n, timestamp_s)  # [32]
current_real = encoder.predict_current()   # [3], N
future_real = encoder.predict_future()     # [5,3], N
effort = encoder.effort_history()           # [10,32]
encoder.reset()
```

时间戳递增；断档超过 0.5 秒自动重置。+5 指五个采样步，不保证精确 0.5 秒。接口不接收实测真机当前力的旁路。新 latent 的语义不同，后续策略需要重新生成特征、统计并训练。

CPU 合成检查：

```bash
PYTHONPATH=scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest force_predictive_encoder_v4.tests -v
```

实际数据端到端快速检查可在新目录运行 `--device cpu --max-epochs 1`；它只验证流程，不用于判断训练效果。
