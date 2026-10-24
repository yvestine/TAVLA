# 在 A100 服务器部署 8001 / 8005 并继续训练

代码从 `https://github.com/yvestine/TAVLA` 获取，模型与训练数据通过 Windows / 移动硬盘中转。以下命令在目标 Linux GPU 服务器、你的账号下执行。GPU 编号和路径按实际机器修改。

## 1. 获取代码和安装环境

```bash
mkdir -p "$HOME/codes"
cd "$HOME/codes"
git clone https://github.com/yvestine/TAVLA.git
cd TAVLA

# 已安装 uv 时可跳过下面两行。
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --python 3.11

nvidia-smi
JAX_PLATFORMS=cuda .venv/bin/python -c 'import jax; print(jax.devices())'
```

使用 `uv.lock` 固定的环境，包括 JAX 0.5.0、CUDA 12 依赖、Flax 0.10.2 和 Orbax 0.11.1。先确认 JAX 能列出目标 GPU，再启动服务或训练。驱动与 CUDA 环境故障的排查参考 [JAX 官方安装说明](https://docs.jax.dev/en/latest/installation.html)。

## 2. 导入已训练模型

原服务器生成的 `tavla_8001_8005_transfer_20261024` 迁移目录已准备为 4 个分片，约 11.49 GiB。在 Windows 下载后，将完整目录复制到目标服务器，例如 `~/tavla_8001_8005_transfer_20261024/`。

```bash
cd "$HOME/tavla_8001_8005_transfer_20261024"
sha256sum -c SHA256SUMS
mkdir -p "$HOME/tavla-transfer"
cat tavla-8001-8005.tar.part-* | tar -xf - -C "$HOME/tavla-transfer"

cd "$HOME/codes/TAVLA"
TRANSFER_ROOT="$HOME/tavla-transfer/TAVLA"
mkdir -p checkpoints eval_outputs/insertion_alignment_encoder .cache/openpi/big_vision
cp -a "$TRANSFER_ROOT/checkpoints/." checkpoints/
cp "$TRANSFER_ROOT/eval_outputs/insertion_alignment_encoder/encoder.pt" \
  eval_outputs/insertion_alignment_encoder/encoder.pt
cp "$TRANSFER_ROOT/.cache/openpi/big_vision/paligemma_tokenizer.model" \
  .cache/openpi/big_vision/paligemma_tokenizer.model
```

只从包内导入模型及配套文件，继续使用 GitHub 拉取的代码。包中保存的是创建迁移包时的代码快照。两个 checkpoint 对应：

| 端口 | 配置名 | 实验 / step |
| --- | --- | --- |
| 8001 | `pi0_lora_user_single_arm_force_align_latent_cotrain_50_50` | `force_align_latent_cotrain_50_50_v1/29999` |
| 8005 | `pi0_lora_user_single_arm_ee_wrench_cotrain_aligned_50_50` | `raw6_aligned_cotrain_50_50_gpu1245_30k_20261009/29999` |

8001 的 encoder SHA256 应为 `db793ebd4f3158b8a8add895cf04368b2289df7fad450f1d2a39b09bb46acd80`；客户端沿用此 encoder 和原输入预处理。checkpoint 中的完整 `params/`、`assets/norm_stats.json` 及 `assets/datasets.txt` 均需保留。

## 3. 启动两个服务

以下示例分别使用目标服务器上的 GPU 0 和 GPU 1。只需要各选一张 GPU 来运行服务，训练时使用另外选定的空闲卡。

```bash
cd "$HOME/codes/TAVLA"
mkdir -p logs
nohup bash scripts/serve_transferred_policy.sh 8001 0 \
  > logs/server_8001.log 2>&1 < /dev/null &
nohup bash scripts/serve_transferred_policy.sh 8005 1 \
  > logs/server_8005.log 2>&1 < /dev/null &

tail -n 40 logs/server_8001.log logs/server_8005.log
ss -ltn '( sport = :8001 or sport = :8005 )'
```

脚本根据代码所在目录设置路径，并禁用 JAX 显存预分配。端口监听完成后再从现有客户端测试实际推理。首次推理可能需要编译。服务使用 checkpoint 自带的 norm stats；启动时先出现一次训练资产路径的 `Norm stats not found ... skipping` 后仍可继续，应结合后续日志判断。

客户端需能访问新服务器的 `8001` / `8005` 端口。两个内网完全不通时，原内网的客户端无法直接连接新服务器，需要在目标服务器可达的网络运行客户端，或另外建立网络连接。

## 4. 训练还需要迁移哪些文件

当前 4 分片包只包含推理所需文件。根据后续训练方式补充以下文件：

| 用途 | 额外文件 |
| --- | --- |
| 从现有模型参数开始一个新的微调实验 | 相应 LeRobot 数据集；训练用 norm stats，可复制原资产或重新计算 |
| 恢复原实验优化器和训练步数 | 上述文件，以及原实验完整 checkpoint（含 `train_state/`、`params/`、step 元数据）；使用 W&B 时还需要实验目录的 `wandb_id.txt` |
| 从基础模型开始新实验 | 数据集、norm stats，以及可下载的 `pi0_base` 初始化权重 |

现有四个预处理数据集在源服务器的 `/data/workspace/gujiawei/.cache/huggingface/lerobot/` 下：

```text
local/tavla_single_arm_force_align_latent
local/tavla_single_arm_force_align_latent_sim
local/tavla_single_arm_ee_wrench
local/tavla_single_arm_sim_aligned
```

8001 使用前两个，8005 使用后两个。复制整个数据集目录，包含 `meta/`、Parquet 数据和图片 / 视频子目录。放到目标服务器 `~/tavla-data/lerobot/local/` 下，并设置：

```bash
export LEROBOT_HOME="$HOME/tavla-data/lerobot"
```

训练用 norm stats 可以把源服务器以下两个目录复制到目标 Git checkout 的相同相对位置：

```text
assets/pi0_lora_user_single_arm_force_align_latent_cotrain_50_50/
assets/pi0_lora_user_single_arm_ee_wrench_cotrain_aligned_50_50/
```

仅迁移原始数据时，需要先按对应配置转换为 LeRobot 数据集，再计算 norm stats。8001 必须用与 checkpoint 匹配的旧 encoder 生成 32 维 latent；8005 使用六维原始 wrench。修改训练数据时，按新数据重新计算资产，并为新数据选择独立路径与实验名。

## 5. 用现有权重继续微调，建立新实验

下面以 8001 为例，使用 GPU 0、1，共两张卡。将 `GPU_IDS` 换成实际用于训练的空闲卡列表，FSDP 数自动等于选择的 GPU 数。初始全局 batch 取每卡 2 个样本；这只是起始设置，显存和吞吐需要在目标 A100 上确认。

```bash
cd "$HOME/codes/TAVLA"
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD/scripts"
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
export LEROBOT_HOME="$HOME/tavla-data/lerobot"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

CONFIG='pi0_lora_user_single_arm_force_align_latent_cotrain_50_50'
CHECKPOINT="checkpoints/$CONFIG/force_align_latent_cotrain_50_50_v1/29999"
GPU_IDS='0,1'
IFS=',' read -r -a TRAIN_GPUS <<< "$GPU_IDS"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
BATCH_SIZE=$((2 * ${#TRAIN_GPUS[@]}))
EXP_NAME="a100_finetune_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"

# 先做 20 步验证；成功后用另一个新实验名运行正式步数。
.venv/bin/python scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --weight-loader.params-path "$CHECKPOINT/params" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "${#TRAIN_GPUS[@]}" \
  --num-train-steps 20 \
  --save-interval 20 \
  --no-wandb-enabled
```

此方式沿用模型参数，重新初始化优化器、学习率调度和训练步数。`--weight-loader.params-path` 指向已迁移的参数；省略该选项会沿用配置中的基础模型初始化路径。正式训练时根据目标数据和收敛情况设置总步数、学习率及保存间隔。

训练 8005 时只替换模型配置与 checkpoint：

```bash
CONFIG='pi0_lora_user_single_arm_ee_wrench_cotrain_aligned_50_50'
CHECKPOINT="checkpoints/$CONFIG/raw6_aligned_cotrain_50_50_gpu1245_30k_20261009/29999"
```

然后使用新的 `EXP_NAME` 执行同样的 `train.py` 命令。全局 batch 必须能被可见 GPU 数整除；FSDP 设备数也要与可见 GPU 拓扑相容。

## 6. 恢复原实验状态

补齐原实验的完整 checkpoint 后，保留其配置名和实验目录名。例如恢复 8001：

```bash
.venv/bin/python scripts/train.py \
  pi0_lora_user_single_arm_force_align_latent_cotrain_50_50 \
  --exp-name force_align_latent_cotrain_50_50_v1 \
  --resume \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "${#TRAIN_GPUS[@]}" \
  --num-train-steps 60000 \
  --save-interval 10000 \
  --no-wandb-enabled
```

总步数必须高于 checkpoint 中已完成的步数。例中的 60000 表示训练到总计 60000 步，学习率调度也应根据延长的训练计划单独设置。恢复状态不使用 `--overwrite`；该选项会删除原实验目录。跨 GPU 数或 GPU 型号恢复时，仍需在目标机器验证 Orbax 恢复和第一步训练。

## 7. 本次代码同步的验证范围

同步前执行代码语法与相关 CPU 测试；原服务器的 8001 / 8005 服务已确认模型路径与握手可用。A100 上的驱动、显存用量、服务推理和多 GPU 训练尚需在目标机器验证。
