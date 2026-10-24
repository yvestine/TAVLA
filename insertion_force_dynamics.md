# Paired insertion force dynamics extraction

## 目标

从同一 rollout 的 Real/Sim 配对轨迹中提取‘插歪候选模式’和‘插对候选模式’，不训练网络，也不把启发式结果当作 correct/wrong ground truth。Sim torque 没有参与；只使用三维 translational force。

## 特征定义

- `F_lateral = sqrt(Fx^2 + Fy^2)`：把 xy 方向作为横向/偏心受力代理。
- `F_axial = abs(Fz)`：把 z 方向作为插入轴向受力代理；这是基于当前 base-frame 和视频运动方向的工程假设。
- `lateral_ratio = F_lateral / (||F|| + eps)`：区分横向边缘接触和轴向插入。
- `delta_norm`：三帧平滑后的力模长变化。
- `direction_delta`：相邻力方向的夹角变化。
- `Fx/Fy_reversals`：横向力分量符号反转次数，作为纠偏/振荡候选信号。
- Real 使用各自 free-space 段的向量 median 做 baseline subtraction；Sim 保持 `contact_total_base` 原值，因为其 free-space 基本为零。

## 如何解释‘插歪’和‘插对’

### 插歪/边缘接触的候选力学模式

通常应同时出现：横向力 `F_lateral` 上升、`lateral_ratio` 较高、力方向快速旋转或 Fx/Fy 反复换向、EE xy 出现纠偏路径；若随后重新对准，接触阶段会持续一段时间而不是单个 impulse。单个大力峰不够作为 wrong-insertion 证据。

### 插对/稳定插入的候选力学模式

通常更接近：z 方向持续推进、轴向力 `F_axial` 占主导、横向比例下降或保持较低、力方向变化减小，最后出现轴向 terminal event 并保持。末端大轴向峰表示插入/到底事件，不单独等价于成功。

## 当前 10 条结果

|轨迹|Real 接触后横向/轴向|Real 末端|Sim 接触后横向/轴向|当前判断|
|---|---|---|---|---|
|traj_0|ratio p95=0.754, Δ||F|| p95=2.310 N|axial p95=15.196 N, z进度=-0.0001 m|ratio p95=0.972, ||F|| p95=23.59 N|可见接触/插入事件，但无足够 wrong 证据|
|traj_3|ratio p95=0.408, Δ||F|| p95=6.198 N|axial p95=22.367 N, z进度=0.0001 m|ratio p95=1.000, ||F|| p95=73.66 N|可见接触/插入事件，但无足够 wrong 证据|
|traj_8|ratio p95=0.272, Δ||F|| p95=3.342 N|axial p95=11.819 N, z进度=-0.0002 m|ratio p95=1.000, ||F|| p95=67.86 N|可见接触/插入事件，但无足够 wrong 证据|
|traj_14|ratio p95=0.977, Δ||F|| p95=0.870 N|axial p95=2.478 N, z进度=0.0001 m|ratio p95=0.993, ||F|| p95=221.63 N|存在横向纠偏/边缘接触候选|
|traj_18|ratio p95=0.810, Δ||F|| p95=1.998 N|axial p95=9.458 N, z进度=0.0011 m|ratio p95=0.990, ||F|| p95=511.89 N|存在横向纠偏/边缘接触候选|
|traj_19|ratio p95=0.955, Δ||F|| p95=0.967 N|axial p95=18.693 N, z进度=0.0002 m|ratio p95=0.000, ||F|| p95=0.00 N|存在横向纠偏/边缘接触候选|
|traj_25|ratio p95=0.728, Δ||F|| p95=4.542 N|axial p95=28.160 N, z进度=-0.0004 m|ratio p95=0.999, ||F|| p95=144.50 N|可见接触/插入事件，但无足够 wrong 证据|
|traj_30|ratio p95=0.473, Δ||F|| p95=8.918 N|axial p95=28.285 N, z进度=-0.0006 m|ratio p95=1.000, ||F|| p95=43.45 N|可见接触/插入事件，但无足够 wrong 证据|
|traj_32|ratio p95=1.000, Δ||F|| p95=1.399 N|axial p95=27.085 N, z进度=0.0004 m|ratio p95=1.000, ||F|| p95=52.16 N|可见接触/插入事件，但无足够 wrong 证据|
|traj_39|ratio p95=0.712, Δ||F|| p95=9.531 N|axial p95=28.710 N, z进度=-0.0002 m|ratio p95=0.986, ||F|| p95=424.12 N|存在横向纠偏/边缘接触候选|

## 重要限制

1. 这 10 条轨迹目前没有人工 success/failure 或 wrong-insertion 标签，所以本报告只能提取阶段和候选信号，不能训练‘插歪/插对’分类器。
2. 这些轨迹的 Sim 是对应 Real rollout 的视觉 replay，配对主要用于比较事件顺序和趋势；不能要求 Real 与 Sim 的 Newton 数值相等。
3. 当前最有价值的跨域公共信息不是绝对力幅值，而是‘横向/轴向比例、方向变化、持续时间、纠偏路径、末端轴向事件’。
4. 真正要学习 correct/wrong，下一步应在这些事件窗口上补 outcome 标签，或者利用明确的纠偏动作作为 weak label；不能仅靠 free/contact onset 代替。

## 输出

- `insertion_force_dynamics.csv`：每条轨迹、每个域、每个阶段的统计。
- `eval_outputs/insertion_force_dynamics/paired_insertion_force_dynamics.png`：Real/Sim 横向力、轴向力和总力的对齐图。
- `eval_outputs/insertion_force_dynamics/traj_*_{real,sim}_dynamics.npz`：逐帧提取结果。
