# 10 条代表轨迹人工辅助审阅

本审计把 Real front/wrist 视频、Sim front/wrist replay、Real EE force、Sim `contact_total_base` 和 EE pose 放在同一帧轴上查看。已有 free-space cutoff 只表示高置信度 free-space；cutoff 之后先保留为 transition/uncertain，不能直接当作 contact，更不能据此生成 wrong-insertion ground truth。

## 总结结论

1. 10 条轨迹都表现出相同的宏观流程：free-space approach → 目标孔附近过渡 → 接触/对准后的继续下压或保持。视频中的 Sim 是对应 Real rollout 的视觉 replay，画面运动大体一致；这说明不能用视频画面本身解释 Sim force 的大幅差异。
2. Real 的接触信息主要表现为时间连续的过程：接触 onset 时力分量重新分配，部分轨迹随后有轻微横向调整，最终插入/末端保持时出现明显轴向峰值。它不是一个单帧 contact bit。
3. Sim 的 `contact_total_base` 在 free-space 基本严格为 0，接触后出现稀疏且明显放大的 impulse；很多轨迹的 Sim 第一非零帧早于人工确认的 Real stable-contact onset，但通常落在 Real 的 transition/uncertain 区间或之前，不能把它直接解释成错误插入。
4. 这 10 条视频没有提供足够可靠的‘插对/插歪’结果标签。能较有把握标的是阶段边界和 terminal force event；除 traj_0 之外，没有一条能仅凭当前视频+力数据安全地标为 wrong insertion。
5. 因此当前数据更适合先构造‘接触后力变化/插入阶段’的时序特征，再寻找或补充真正的 outcome 标签；不建议把这 10 条的 contact onset 直接当作 correct/wrong supervision。

## 逐条审阅

|轨迹|free end|人工审阅 onset|Real terminal peak|Sim 第一非零|Sim 峰值|置信度|判定|
|---|---:|---:|---:|---:|---:|---|---|
|traj_0|95|151|257 (15.7391 N)|116|220 (56.2506 N)|high|有明确 alignment/correction search；仍不是二分类 wrong 标签|
|traj_3|95|156|158 (23.758 N)|103|149 (113.4425 N)|medium|阶段边界可用；wrong 不可判|
|traj_8|88|110|177 (12.3835 N)|87|149 (268.6719 N)|medium|阶段边界可用；wrong 不可判|
|traj_14|68|94|147 (4.134 N)|76|101 (256.4002 N)|medium|阶段边界可用；wrong 不可判|
|traj_18|66|130|130 (9.3739 N)|65|129 (704.6669 N)|medium|阶段边界可用；wrong 不可判|
|traj_19|50|58|141 (19.2121 N)|76|133 (77.723 N)|medium|阶段边界可用；wrong 不可判|
|traj_25|77|114|191 (30.1023 N)|94|108 (332.4656 N)|medium|阶段边界可用；wrong 不可判|
|traj_30|53|98|101 (29.8743 N)|60|76 (99.7085 N)|medium|阶段边界可用；wrong 不可判|
|traj_32|81|99|151 (29.8796 N)|95|128 (146.8832 N)|medium|阶段边界可用；wrong 不可判|
|traj_39|43|70|72 (30.0152 N)|46|68 (814.5809 N)|medium|阶段边界可用；wrong 不可判|

## 解释口径

- `free_end`: 用户确认的 high-confidence free-space 末帧，包含该帧。
- `contact_onset`: 视频、EE z/xy 运动变化和 Real wrench 时间响应共同支持的首次明显接触/插入转折；它不是自动阈值结果。
- `transition`: free_end 与 contact_onset 之间的帧，默认 near-contact/uncertain。
- `terminal peak`: 后续最大的 Real axial/force event，用于描述插入末端变化，不等同于成功标签。
- `Sim 第一非零`: 仅表示 `contact_total_base` 第一次离开严格零；它不是 Real stable-contact onset，也不是 wrong-insertion onset。

## 可复查文件

- 逐轨迹数值表：`representative_trajectory_audit.csv`
- 生成脚本：`scripts/audit_representative_trajectories.py`
- 关键帧和力曲线：`eval_outputs/representative_trajectory_audit/`（本地生成目录，已被 gitignore 忽略）

## 最终判断

这 10 条可以支持‘不同阶段的力变化模式’审计，但不能单独支持‘正确/错误插入’监督。当前最可靠的共同信号是：接触 onset 后的力方向/分量重排、持续时间、以及末端轴向峰值；Sim 的幅值和脉冲形态不能直接与 Real 一一对应。
