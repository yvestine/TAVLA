# motion_v1：运动背景补偿 Force Encoder

本轮实现和验证 encoder。使用已有的 40 对真机/接触分离仿真轨迹，不要求绝对静止。
代码已准备好，由用户在外部 GPU 主机运行。模型是否有效以运行生成的报告为准。

## 1. 一条命令运行完整实验

在已有 `train` tmux 的 shell 中执行：

```bash
cd /data/workspace/gujiawei/TAVLA && GPU_ID=7 bash scripts/run_force_gap_motion_v1.sh
```

脚本在当前 shell 前台运行，物理 GPU 7 映射为程序内 `cuda:0`。
训练、重训基线、未来运动预测和消融使用这一张卡；之后软件检查使用 CPU。
默认 Python 为仓库 `.venv/bin/python`，可通过 `PYTHON_BIN` 指定已有环境的解释器。
脚本启用 `pipefail`；训练或软件检查发生错误时会停止，检查输出追加到同一份日志。

只写入现有目录 `eval_outputs/force_gap_encoder_v2/`：

| 文件 | 内容 |
| --- | --- |
| `motion_encoder.pt` | 一个最终候选：网络、背景模型、特征定义、归一化、输入约定和数据划分 |
| `motion_encoder.report.json` | 数据指纹、各折结果、消融、统计验收和软件检查 |
| `motion_encoder.log` | 本轮训练及软件检查日志；重新运行覆盖 |

各折模型、预测头和旧 encoder 的重训模型只保留在内存。最终候选按统计规则选取，
即使没有通过验收也会保存供诊断；此时 `qualified` 为 `false`。
写 checkpoint/report 时临时文件会原子替换为上述文件。

## 2. 数据输入与范围

默认输入：

- 真机：`data/traj_*/data.h5`。
- 仿真：`sim-data-aligned/traj_*/data.h5`。
- 力：`obs/state/ee_force_base`，前三轴，牛顿，机器人基座坐标系。
- 状态：`obs/state/joint_pos` 七轴弧度、`obs/state/joint_vel` 七轴弧度/秒。
- 时间：`timestamps`，秒，严格递增，名义 10 Hz；报告实际时间跨度及采样抖动。
- 标注：`annotations/insertion_quality_labels_v2.csv`。
- 原 first-hit cutoff 和旧模型配方：`eval_outputs/insertion_alignment_encoder/encoder.pt`。

读入时检查形状、有限值、首 20 帧 `high/free` 标注、配对长度与时间。
仿真声明的力字段必须带有 `counterfactual_no_contact_subtracted=True` 和
`frame=robot_base_O`，所选三轴必须等于该字段的前三轴。
本入口使用这套接触分离来源检查，不依赖旧 `sim_force_saved` 属性。
这些检查验证文件约定的一致性，物理坐标和传感器方向仍需按采集定义保持一致。

重建覆盖第 20 个样本开始、原 first-hit cutoff 之前的全部窗口。
阶段监督保留 `misaligned_contact` / `aligned_progress`，跳过 `low` 和段边界各 5 帧。
同帧 action 字段只做诊断；预测目标使用 `q[t+3]-q[t]` 和 `q[t+5]-q[t]`。
两种未来目标均在 cutoff 前的样本才参与预测评估。

## 3. 运动背景补偿

关节状态只用于计算背景，神经 encoder 输入为补偿力的 17 维特征。

```text
φ_t = [sin(q_t), cos(q_t), joint_vel_t, abs(joint_vel_t)]  # 28 维
b0  = median(force[:20], axis=0)
φ0  = median(φ[:20], axis=0)
background_t = b0 + clip(((φ_t-φ0)/feature_scale) @ coefficient, -P99, P99)
residual_t = force_t - background_t
```

`coefficient` 是对缩放后的 28 维状态求得的三轴系数，等价于原始基底上的矩阵 A。
Huber IRLS Ridge 只拟合训练轨迹的高置信度真机自由空间帧，每条轨迹权重相同。
正则从 `0.001/0.01/0.1/1/10` 内部按轨迹分组选择。
自由空间 OOF MSE 相对固定 `b0` 的改善不足 20% 时禁用 A；报告保留各正则的 OOF
结果和禁用原因。禁用只表示本次数据未支持学习补偿，实验继续评估其余表示。

每一外折和用于 epoch 选择的内部划分分别拟合自己的背景与归一化；最终候选只拟合
32 条开发轨迹。背景变化限幅使用对应训练自由空间每轴绝对变化的 P99。
首 20 帧参考可以包含运动，必须在接触之前；参考和参数在本 episode 内固定，
持续受力不会被在线基线更新吸收。

仿真来源已扣除无接触反事实背景，仅减去本 episode 的 `b0`。
它的关节状态用于统一输入校验，不进入其背景变化预测。

## 4. 特征、网络与训练

残差因果 3 帧均值，计算 1 帧和 5 帧差分，保留原始顺序。
`s_force/s_delta1/s_delta5` 是首 20 帧对应向量模长的中位数，各自下限 `0.01 N`；
含运动的这些量表示背景变化尺度，不解释为纯传感器噪声。

| 索引 | 特征 |
| --- | --- |
| 0–2 | 平滑残差的软方向 `v/(norm(v)+s_force)` |
| 3–5 | 1 帧差分的软方向 |
| 6–8 | 5 帧差分的软方向 |
| 9 | 沿基座 `(0,0,-1)` 的有符号分量 / `(force_norm+s_force)` |
| 10 | 横向模长 / `(force_norm+s_force)` |
| 11–13 | 三种模长相对各自背景尺度的 `log1p` |
| 14 | `log1p(force_norm_in_newtons)` |
| 15–16 | 实际 1 帧秒数 / 0.1、5 帧秒数 / 0.5 |

差分不除以很短的采样间隔；时间跨度独立作为特征。
归一化使用训练帧 median 和 `max(IQR, 0.1*(P95-P05), 0.001)`，截断到 `[-10,10]`。
不足 32 帧的历史在左侧重复该 episode 的第一帧特征。

结构支路输入 `0..10,15,16`，Conv48 膨胀率 `1/2/4`、逐帧 LayerNorm、GRU64，输出24。
强度支路输入 `11..14`，Conv16、GRU16，输出8。拼接输出 float32 的32维 latent。
阶段头使用完整32维；重建头恢复整个32帧窗口的前15个特征。

```text
L = phase + 0.1*reconstruction + 0.05*gain_consistency
          + 0.01*variance + lambda*conditional_alignment
```

仿真阶段权重再乘 `force_norm²/(force_norm²+s_force²)`。
条件对齐仅约束结构24维，使用高置信度且连续信号可信度至少0.5的同阶段样本。
每域组内至少16个不同帧才计算加权均值距离和0.25倍协方差距离。
比较 `lambda=0/0.01/0.1`，报告每 epoch 的非零对齐 batch 数量。
增益视图以 log-uniform 抽取 `0.5–2`，缩放残差及差分后、归一化前重算特征，
保留原起始背景尺度；结构支路做一致性约束，强度支路保留增益响应。
方差项使用结构通道标准差下限0.1的 hinge。

## 5. 冻结表示验证与验收

固定32条开发轨迹的4折，分折 seed42；初始化 seeds42/43/44，所有比较复用划分。
epoch 只由训练折内部轨迹选择，之后在整训练折重训。
原先反复检查过的 `0/2/21/22/23/33/34/39` 作为8条开发诊断轨迹披露，
不参与候选选择，不称作未使用过的最终测试集。

冻结 encoder 后使用相同 `64→32→14` MLP 预测器，比较：

- `state_only`：32帧七轴位置和速度。
- `state_background`：状态 + 32帧背景预测。
- `state_raw_force`：状态 + 原始三轴力历史。
- `state_manual`：状态 + 17维手工特征历史。
- `state_legacy`：状态 + 同折同 seed 从头重训的 terminal-reference 旧 encoder。
- `state_new`：状态 + 新 encoder；另做移除背景、移除对齐消融。

旧 encoder 重训保留原阶段、配对余弦、terminal geometry 和 XY 方向损失；
旧 checkpoint 的网络权重不直接作为该交叉验证比较结果。

预测器仅用源域训练，分别测试 Real→Sim / Sim→Real 的未来3、5帧关节变化。
同时报告弧度 RMSE、由源域训练目标标准差归一化的 RMSE 和实际秒数。
验收误差是按轨迹等权、seed求均值、双向和双跨度平均的归一化 RMSE。
仿真目标代表配对示范的未来变化，不是独立仿真动力学真值。

验收条件：

1. 相对五个基线中平均误差最小者，新表示平均改善至少5%。
2. 任一迁移方向平均误差恶化不超过2%。
3. 按整条轨迹分组、配对 bootstrap 2000次，平均误差改善95%区间下界 >0。
4. 两域阶段 macro-F1 相对重训旧 encoder 各下降不超过0.02。
5. 同阶段打乱 force latent 后误差增加，配对轨迹 bootstrap 区间下界 >0。
6. 软件检查通过，并完整完成3个初始化 seed。

候选先保留满足阶段保护者，再按预测误差选择；没有满足者仍选一个供诊断，
验收不会因此放宽。打乱使用同标注阶段内的随机排列，保留该阶段 latent 的经验分布；
优先保留其他轨迹或同轨迹至少间隔32帧的匹配，只在这些合格帧上配对比较。
域线性分类 AUC、条件均值/协方差距离、同帧余弦作为辅助诊断。
结论限定为现有数据上的表示和预测收益，闭环策略收益需要后续另行验证。

## 6. 在线输入接口

```python
from force_gap_encoder_v2.motion_infer import MotionForceLatentEncoder

encoder = MotionForceLatentEncoder(
    "eval_outputs/force_gap_encoder_v2/motion_encoder.pt",
    source="real",  # 当前接触分离仿真用 "sim_contact_isolated"
    device="cpu",
)
encoder.reset()
latent = encoder.push(force_xyz_n, joint_pos_rad, joint_vel_rad_s, timestamp_s)
if latent is not None:
    assert latent.dtype.name == "float32" and latent.shape == (32,)
    history = encoder.effort_history()  # float32 [10,32]，从旧到新
```

每100ms发送一次同步观测，F为基座坐标的3轴牛顿，q/dq为7轴弧度和弧度/秒，
时间为秒且严格递增。来源在构造时指定，不是网络输入的域标签。
传感器坐标系不同需先按物理旋转转换到基座系，保持训练时的力符号和接触分离定义。
无需调用方另外减基线、平滑、归一化或手调 Real/Sim 乘数。

首19次调用返回 `None`，第20次起输出32维。`effort_history()` 只保存已经生成的
latent，未满10个时在左侧重复第一个可用输出；尚未完成校准时调用会报错。
每个 episode 都应 `reset()`，它清除校准、所有历史和时间状态。
启动后只缓存必要的3/5/32帧输入历史，固定背景参考和尺度。

批量接口 `encoder.batch(force[T,3],q[T,7],dq[T,7],time[T])` 返回 `[T-19,32]`，
第0行对应原始第19帧。批量调用不改变流式历史。
新 checkpoint 使用 `tavla_force_gap_encoder_motion_v1` 格式及上述状态输入约定。

## 7. 软件检查与报告阅读

总命令会在训练后运行下列检查，验证完整8条诊断轨迹的两域：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=scripts:src .venv/bin/python \
  -m force_gap_encoder_v2.validate_motion --device cpu
```

检查 checkpoint SHA256 与报告匹配、数据数组指纹匹配、校准输出、
批量/流式差异≤`1e-5`、未来观测扰动不改变之前的输出、NumPy/Torch 增益特征一致、
仿真状态不进入力特征、持续5N注入不会改变背景预测且会产生 latent 响应、
reset重放一致、非法力和倒退时间被拒绝。检查结果写回同一份 report。

主要读取：`acceptance.checks`、`acceptance.mean_relative_improvement`、
`acceptance.direction_relative_improvements`、`acceptance.paired_bootstrap`、
`acceptance.same_phase_shuffle_bootstrap`、`final_training.background.selection`、
`ablation_no_background`、`ablation_no_alignment` 和 `software_checks`。
只有训练完整、统计条件全部满足并通过软件检查，报告的 `qualified` 才为 `true`。
`status=failed` 表示代码/数据运行错误；`status=complete, qualified=false` 表示完成实验
但当前候选尚未满足有效性门槛。
