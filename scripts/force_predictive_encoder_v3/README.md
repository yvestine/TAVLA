# Force predictive encoder v3

独立的新实验；不修改旧权重、策略数据集或服务器。Encoder 只读取历史三轴力和 dt；q/dq 只进入最后单独训练的运动辅助预测头。

多任务、只训练 Real teacher、之后再接 Sim student 的入口见
[A100 多任务训练说明](../../docs/force_predictive_v3_multitask_a100.md)。新入口不要求固定40条，
不读取 q/dq 或阶段标签，也不训练 motion/control；本页原有运行入口保持原协议。

## 运行

在外部 GPU 主机的 TAVLA 根目录运行。默认只跑 seed 42 和一个固定留出组；GPU_IDS 是明确指定的物理卡号，每卡最多一个 worker：

```bash
GPU_IDS=6 bash scripts/run_force_predictive_encoder_v3.sh
```

默认使用 fold 0 的8条轨迹作固定验证，其他24条训练；seed=42、趋势对齐权重=0.1。按依赖顺序训练 teacher、student、motion、control 四个阶段，验证集用于早停。不会做多折/多 seed、权重搜索、无瞬态消融或最终重训；验证指标是单次留出诊断，不是交叉验证结论。

如需重跑完整对照协议，再显式加 `--full-cv`：

```bash
GPU_IDS=6 bash scripts/run_force_predictive_encoder_v3.sh --full-cv
```

完整协议为4折 × 3种子及最终重训，耗时远高于默认单次训练。`--seed`、`--validation-fold`、`--single-align-weight` 可调整默认单次训练；`--max-epochs`、`--patience`、`--batch-size` 同时适用于两种模式。

固定目录与恢复：

```bash
GPU_IDS=6 RUN_DIR="$PWD/eval_outputs/force_predictive_encoder_v3/run1" \
  bash scripts/run_force_predictive_encoder_v3.sh --single-run

GPU_IDS=6 RESUME=1 RUN_DIR="$PWD/eval_outputs/force_predictive_encoder_v3/run1" \
  bash scripts/run_force_predictive_encoder_v3.sh --single-run
```

恢复要求代码、数据、标签、超参数一致。已完成的任务/阶段复用；中断阶段从其固定父权重重新训练，不声称逐 optimizer step 恢复。可改变卡号和是否画图。
`REAL_DIR`、`SIM_DIR`、`LABELS`、`PYTHON_BIN` 可覆盖输入。默认旧版对照来源为 `eval_outputs/force_gap_forecast_v1/20261012_155026_551478`；`BASE_RUN` 可修改。
新数据与旧版对照不匹配时显式传 `--no-old-baseline`，仍然训练同容量运动对照和力自回归基线；报告不会声称已与旧版比较。
`--require-actual-sim-state` 可强制要求独立实测的 Sim 状态。现有复制状态保留 `actual_sim_state_verified=false`。

## 表示与训练

- 输入：截至 t 的32帧 `(Fx,Fy,Fz)`，base坐标系、N；当前已对齐的 `obs/state/ee_force_base` 不再取负，不做启动扣基线或裁剪。
- 训练折统计提供两域各自固定的线性力归一化。EMA在每个32帧窗口内首帧初始化，使用实际dt、时间常数0.3秒；不足32帧先左侧补首帧。`slow + fast = normalized raw force`。
- 趋势输入为 `[slow, Δslow, dt/0.1]`，瞬态输入为 `[fast, Δraw, dt/0.1]`。两个宽度64、kernel3、dilation1/2/4/8/16的因果TCN各输出16维，无时间降采样。输出的全部32维是学习特征，不能按旧版 `[F3,ΔF3,h26]` 解读。
- 本域两个重建头分别从趋势/瞬态latent恢复最近16帧分量，求和恢复本域原始力。Sim重建目标是Sim输入力，不是“恢复当前真机力”。共同预测头从32维特征预测未来5帧真机力。
- 真机预训练后固定真机Encoder和未来预测头，仿真Encoder由真机初始化。仿真同时学习本域重建、未来真机力预测和4/8/16帧趋势片段对齐。瞬态不施加跨域逐帧相等约束。
- 全程无原始力直达decoder的旁路。运动辅助头/屏蔽latent的同容量对照在Encoder冻结后分别训练，使用相同初始化、采样和优化配置；不得把未来q/dq或动作作为输入。
- 无瞬态分支的对照独立重训并执行相同对齐权重搜索，不是仅在推理时将latent置零。

损失：未来力MSE + 0.25未来差分 + 0.1未来峰值，再加0.5本域重建、0.25历史差分、0.1频谱、0.1历史峰值、0.01方差/协方差正则和趋势对齐。
重建同时监督总波形和两个分量；频谱采用8/16帧Hann窗的复数误差，0.25–0.5 cycles/frame权重加倍；不只比较频谱幅值。
峰值同时约束幅值和软峰值位置。所有目标按训练折尺度归一化，评估返回N和实际秒；不将不均匀采样的频带声称为精确Hz。

AdamW lr=3e-4、weight_decay=1e-4、梯度裁剪3，最多200epoch、patience20、batch32个16端点块。轨迹等权，块采样一半均匀、一半偏向快速变化。
teacher/student选epoch用 `future_score_N / mean(real_train_std) + 0.25*reconstruction_nrmse`；运动头只按未来score选择。默认固定对齐权重0.1，不搜索候选；完整协议才搜索0/0.03/0.1/0.3并训练消融。

## 指标与产物

每条轨迹/每个种子等权。未来score为 `.25 RMSE(+1) + .35 RMSE(+3) + .40 RMSE(+5)`。
纯力预测分别比较保持当前力与force-only线性AR；Sim保持基线先用训练数据拟合当前力的仿射校准。AR正则只在内层选择。
运动辅助预测比较同容量运动对照及可用的旧motion-only。改善目标5%，2000次轨迹bootstrap改善下界>0；轴、事件、峰值和接触阶段限制回退2%。
旧版Real对照的decoder还读取实测当前力及变化，Sim对照则只使用运动；报告保留这一区别。接触阶段验收按实际标签执行，包含 `misaligned_contact`、`aligned_progress`、`terminal` 和末8帧。
趋势对齐要求距离较无对齐改善5%，另报告片段检索；重建检查高频复数误差、流式当前重建频谱、差分与峰值，禁止仅以latent cosine为验收标准。
瞬态分支须相对重训的无瞬态对照改善高频重建，bootstrap下界>0。teacher维度标准差低于0.1不通过验收。

- 默认单次训练输出 `single/single_report.md`、`single/result.json`、`single/predictions/validation/`，以及根目录的 `encoder_seed42.pt` 和 `manifest.json`。
- 完整协议输出 `report.md` / `acceptance.json`：总结果和所有未通过原因。
- `outer_metrics.json`：按域、模型、种子、轨迹、阶段、步长、轴保存的指标。
- `selections.json`：各折epoch、对齐权重及候选结果。
- `force_input_audit.json`：分段打乱力、保持运动不变时的latent和预测响应；仅作为敏感性诊断。
- `jobs/seed*_fold*/predictions/{full,no_alignment,no_fast}/{real,sim}/traj_*.npz`：完整折外波形、历史重建、latent与未来预测。full还包含运动辅助/运动对照。
- `jobs/*/**/history.json`：逐epoch训练损失、验证指标、步数、时间；`last.pt`保留最后状态，`model.pt`保留选择状态。
- `logs/seed*_fold*.log`：每个GPU任务的日志；主日志为 `train.log`。
- `encoder_seed42/43/44.pt` / `manifest_seed*.json` / `manifest.json`：最终权重，失败也按diagnostic保留。
- `final/seed*/diagnostic_predictions`：不参与选择的8条旧诊断轨迹。
- `visualizations/index.html`：固定seed42的全部折外波形图片索引，可本地打开；未来曲线按目标时间绘制。

完整结果通过也只说明离线波形证据；复制Sim运动数据通过时标为replay_validated，独立Sim状态通过时标为waveform_validated。本版不启动策略训练或替换任何服务器。

完成单次训练后，可生成可视化报告（CPU 推理，不重训）：

```bash
PYTHONPATH=scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m force_predictive_encoder_v3.visualize --run-dir /path/to/run
```

输出 `visualizations/index.html`、6张总览图及六页 `encoder_overview.pdf`、验证集和另外8条未参与本次训练/早停的轨迹波形、`analysis.md`、`diagnostics.json`，以及根目录的 `encoder_visualizations.zip`。均值、尺度和共享PCA只用训练集真机编码拟合；仿真当前力校准与固定α=1的线性AR也只拟合训练集。错配片段参考不等同于无对齐消融。单次训练的独立留出仅针对当前运行，不代表这些历史轨迹从未被其他实验查看。

## 流式接口

```python
from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder

encoder = ForcePredictiveEncoder("/path/to/run/manifest.json", domain="sim")
z = encoder.push(force_xyz_n, timestamp_s)   # float32[32]，首帧即可用
effort = encoder.effort_history()            # float32[10,32]，不足时左补首项
future = encoder.predict_future()           # float32[5,3]，仅靠力特征，单位N
joint_future = encoder.predict_future(q_history, dq_history)
history = encoder.reconstruct_history()     # float32[16,3]，本域力重建
encoder.reset()
```

q/dq历史须与内部力窗口一一对应：不足32帧时为当前episode已收集的帧数，否则为最近32帧；单位rad和rad/s。
重复/倒退/非有限时间戳报错，断档>0.5s自动重置；训练目标不会跨断档。首帧的补齐重建点仅作接口占位，不作为有效历史评价。
后续接TAVLA须使用新特征缓存、重新计算统计并训练策略，即使effort形状仍为[10,32]也不能直接替换旧模型输入。

CPU合成检查（不读取真实训练数据，不启动正式训练）：

```bash
PYTHONPATH=scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest force_predictive_encoder_v3.tests -v
```
