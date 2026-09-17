#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_ID="${1:-0}"
EXP_NAME="${2:-pi0_no_effort_30k}"

cd "$PROJECT_DIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

export OPENPI_DATA_HOME="$PROJECT_DIR/.cache/openpi"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "[1/2] Computing normalization statistics (without effort)..."
JAX_PLATFORMS=cpu "$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name pi0_lora_user_single_arm \
  --batch-size 8 \
  --num-workers 0

echo "[2/2] Starting pi0 LoRA fine-tuning on GPU ${GPU_ID}..."
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train.py pi0_lora_user_single_arm \
  --exp-name "$EXP_NAME" \
  --batch-size 8 \
  --num-train-steps 30000 \
  --save-interval 1000000 \
  --no-wandb-enabled \
  --overwrite

echo "Training complete. Final checkpoint:"
echo "$PROJECT_DIR/checkpoints/pi0_lora_user_single_arm/$EXP_NAME/29999"
