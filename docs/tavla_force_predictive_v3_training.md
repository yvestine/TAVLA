# 力预测 encoder 的 TAVLA 联合训练

默认冻结 `force_predictive_encoder_v3/20261014_160410_2018868/encoder_seed42.pt`，
seed42、alignment=0。同一组早停验证轨迹的 Sim 纯力 future score 为 1.83676 N，
alignment=0.1 为 1.83941 N，v4 为 1.8952 N。这里按现有验证误差选权重，
不代表独立盲测或下游策略成功率比较。

在 TAVLA 根目录运行：

```bash
cd /data/workspace/gujiawei/TAVLA
ENCODER_KIND=force_predictive_v3 \
MANIFEST="$PWD/eval_outputs/force_predictive_encoder_v3/20261014_160410_2018868/manifest.json" \
GPU_IDS=2,3 BATCH_SIZE=8 TRAIN_STEPS=30000 \
  bash scripts/run_tavla_force_align_latent_cotrain.sh
```

流程复用现有转换、统计和训练脚本。先在 CPU 上将两域各自的历史三轴力和时间戳
编码为 32 维 latent，使用各域自己的 encoder 分支；不向 encoder 提供运动或未来量。
保留各域原有图片、状态和 `action/actual`。时间间隔超过 0.5 秒时同时重置 encoder
并切分 LeRobot episode。新缓存以 encoder SHA 前缀区分，完成标记保存来源 HDF5 哈希。

策略使用 Pi0 LoRA、Real/Sim 50:50、10 帧 effort 历史、双卡 FSDP、全局 batch8
（每卡4）、30000步，每10000步保存一次。两域重新计算统计，再按50:50合并。
配置名为 `pi0_lora_user_single_arm_force_predictive_v3_cotrain_50_50`。

默认自动生成新的实验名。日志在 `logs/<实验名>.log`，checkpoint 在
`checkpoints/pi0_lora_user_single_arm_force_predictive_v3_cotrain_50_50/<实验名>/`。
训练完成后会把冻结 encoder、可移植 manifest 和输入说明附在 checkpoint 的 `assets/`。

恢复时沿用原来的 repo、manifest 和参数，增加 `RESUME=1 EXP_NAME=原实验名`。
`REAL_REPO`、`SIM_REPO`、`ASSETS_DIR` 可指定新的独立缓存和统计目录。

训练入口默认设置 `NUMPY_MADVISE_HUGEPAGE=0`。2026-10-22 此主机发生透明大页
同步内存整理，导致预训练权重的 NumPy 类型转换长期占用 CPU，未到达首个训练步。
该设置只影响当前进程的大数组分配，不修改系统设置；已准备的数据缓存和统计可复用。

修复后运行 `force_predictive_v3_cotrain_20261022_120747_3643501_thpfix` 复用了两域缓存和统计，
权重加载/转换/校验耗时23.4秒，双卡初始化耗时17.6秒，首个训练步 loss=0.0805。
该运行已于 2026-10-23 完成 30000 步，最终 checkpoint 为 `29999`，日志以该实验名命名。

2026-10-24 已将最终 checkpoint 部署到物理 GPU 2、WebSocket 端口 8006，
encoder 在 server 处理，仿真端可直接发送六维力与仿真时间戳。
协议、启动方式和验证记录见 [服务端 encoder 部署说明](tavla_force_predictive_v3_server.md)。
