# TA-VLA 迁移与复现指南

本仓库包含在 OpenPI/TA-VLA 基础上的本地修改，重点是单臂数据、关节力矩/末端六维力输入、LoRA 微调、离线评估和策略服务。模型权重、数据集、缓存和实验输出不放入 Git；迁移到新服务器时重新下载基础权重，并按下面流程生成本地数据资产。

## 1. 新服务器要求

- Ubuntu 22.04/24.04，建议 NVIDIA GPU、驱动支持 CUDA 12。
- Python 3.11。
- Git、`curl`、`ffmpeg`，训练还需要能正常执行 `nvidia-smi`。
- 至少约 30 GB 可用空间用于基础模型缓存；训练和 checkpoint 需要更多空间。

先确认：

```bash
nvidia-smi
python3 --version
git --version
ffmpeg -version
```

## 2. 安装环境

```bash
git clone <你的 GitHub 仓库地址>.git TA-VLA
cd TA-VLA

# 安装 uv（如果服务器已有 uv，可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# 项目固定 Python 3.11；uv.lock 用于复现依赖版本
uv python install 3.11
uv sync --python 3.11
source .venv/bin/activate
```

如果服务器不能访问默认 PyPI，需要提前配置内部镜像；不要手工混用系统 pip 和项目环境。CUDA 版 JAX 依赖由 `pyproject.toml`/`uv.lock` 管理。

## 3. 下载基础权重

基础权重不在 GitHub 中。显式预下载：

```bash
source .venv/bin/activate
python scripts/download_base_checkpoint.py
```

默认缓存目录是 `~/.cache/openpi`；空间不足时可指定：

```bash
export OPENPI_DATA_HOME=/data/$USER/openpi-cache
python scripts/download_base_checkpoint.py
```

训练配置使用公开的 `s3://openpi-assets/checkpoints/pi0_base/params`。因此即使不预下载，第一次训练/推理也会自动下载；建议先显式下载以便提前发现网络或权限问题。

## 4. 准备数据和 norm stats

原始真机数据不随仓库迁移。默认格式为：

```text
data/
└── traj_0/
    ├── data.h5
    ├── front_camera.mp4
    └── wrist_camera.mp4
```

将数据复制到新服务器后执行：

```bash
python scripts/audit_user_hdf5.py --raw-dir data
python scripts/convert_user_hdf5_to_lerobot.py \
  --raw-dir data \
  --repo-id local/tavla_single_arm_ee_wrench \
  --task "peg-in-hole" \
  --effort-key obs/state/ee_wrench_base \
  --action-mode actual \
  --no-videos \
  --image-writer-processes 0 \
  --image-writer-threads 0 \
  --overwrite

JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py \
  --config-name pi0_lora_user_single_arm_ee_wrench
```

若使用七维关节外力矩，把配置名改为 `pi0_lora_user_single_arm_effort`，并使用 `--effort-key obs/state/joint_torque_external`。转换脚本会将单个腕部视频复制到模型需要的两个 wrist image key。

## 5. 训练和推理

先运行短 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  pi0_lora_user_single_arm_ee_wrench \
  --exp-name smoke_1k \
  --num-train-steps 1000 \
  --save-interval 500 \
  --batch-size 8 \
  --overwrite
```

正式训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  pi0_lora_user_single_arm_ee_wrench \
  --exp-name first_lora_ee_wrench \
  --num-train-steps 30000 \
  --save-interval 10000 \
  --log-interval 50 \
  --batch-size 8 \
  --overwrite 2>&1 | tee logs/first_lora_ee_wrench.log
```

训练结果位于 `checkpoints/<config>/<experiment>/<step>/`，其中 `assets/` 包含训练数据的 norm stats。该目录不提交 Git；迁移已训练模型时需要单独复制或上传 checkpoint。

启动策略服务：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi0_lora_user_single_arm_ee_wrench \
  --policy.dir checkpoints/pi0_lora_user_single_arm_ee_wrench/first_lora_ee_wrench/29999
```

服务端会加载 checkpoint 内的 norm stats，不需要在机器人端再次归一化。Franka/末端六维力输入协议见 [`franka_ee_wrench_deployment.md`](franka_ee_wrench_deployment.md)。

## 6. 迁移检查清单

```bash
python -m pytest src/openpi/shared/wrench_adapter_test.py src/openpi/training/data_loader_test.py -q
python -m compileall -q src packages scripts
git status --short --ignored | head -100
```

提交前应确认：

- `git status` 中没有 `checkpoints/`、`assets/`、`data*/`、`eval_outputs/`、`.venv/` 或模型文件。
- 新服务器执行 `uv sync` 后能导入 `openpi`。
- `download_base_checkpoint.py` 成功打印 `params` 和 `assets` 路径。
- 至少完成一次 norm stats 计算和 1k steps smoke test。

完整的实验记录和专项说明见：

- [`run.md`](../run.md)
- [`tavla_ee_wrench_finetuning_report.md`](tavla_ee_wrench_finetuning_report.md)
- [`tavla_sim_finetune_runbook.md`](tavla_sim_finetune_runbook.md)
- [`wrench_polyfit_alignment.md`](wrench_polyfit_alignment.md)
