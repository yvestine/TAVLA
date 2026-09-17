#!/usr/bin/env bash
set -Eeuo pipefail

# TAVLA real-robot fine-tuning following run.md: one GPU, no FSDP.
# Default physical GPU: 5. Override with GPU_ID=4 or GPU_ID=7.
# The model uses 6-D end-effector wrench data:
#   obs/state/ee_wrench_base -> (Fx, Fy, Fz, Tx, Ty, Tz)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_ID="${GPU_ID:-5}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/.cache/openpi}"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

CONFIG_NAME="pi0_lora_user_single_arm_ee_wrench"
EXP_NAME="${EXP_NAME:-real_ee_wrench_runmd_gpu${GPU_ID}_30k}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

if (( BATCH_SIZE <= 0 )); then
  echo "ERROR: BATCH_SIZE must be positive; got $BATCH_SIZE" >&2
  exit 1
fi

DATASET_INFO="$LEROBOT_HOME/local/tavla_single_arm_ee_wrench/meta/info.json"
STATS_FILE="$PROJECT_DIR/assets/$CONFIG_NAME/local/tavla_single_arm_ee_wrench/norm_stats.json"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME"
LOG_FILE="$PROJECT_DIR/logs/$EXP_NAME.log"

if [[ ! -f "$DATASET_INFO" ]]; then
  echo "ERROR: real wrench dataset not found: $DATASET_INFO" >&2
  echo "Run the real-data conversion before starting this script." >&2
  exit 1
fi

"$PYTHON_BIN" - "$DATASET_INFO" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    info = json.load(handle)
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
if shape != [6]:
    raise SystemExit(f"ERROR: expected 6-D observation.effort, got {shape}")
print("Validated observation.effort shape: [6]")
PY

if [[ ! -f "$STATS_FILE" ]]; then
  echo "Normalization statistics not found; computing them on CPU..."
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/compute_norm_stats.py" \
    --config-name "$CONFIG_NAME" \
    --batch-size 8 \
    --num-workers 0
fi

if [[ ! -e "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]]; then
  echo "pi0_base checkpoint not found; downloading it..."
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/download_base_checkpoint.py"
fi

if [[ -e "$CHECKPOINT_DIR" && "$RESUME" != "1" && "$OVERWRITE" != "1" ]]; then
  echo "ERROR: checkpoint directory already exists: $CHECKPOINT_DIR" >&2
  echo "Use RESUME=1 to continue it, or OVERWRITE=1 to replace it." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
train_args=(
  "$PROJECT_DIR/scripts/train.py" "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --batch-size "$BATCH_SIZE"
  --num-train-steps "$TRAIN_STEPS"
  --save-interval "$SAVE_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --no-wandb-enabled
)
[[ "$RESUME" == "1" ]] && train_args+=(--resume)
[[ "$OVERWRITE" == "1" ]] && train_args+=(--overwrite)

echo "Starting TAVLA Pi0 LoRA fine-tuning on physical GPU $GPU_ID (run.md mode, no FSDP)"
echo "Batch size: $BATCH_SIZE"
echo "Log: $LOG_FILE"
"$PYTHON_BIN" "${train_args[@]}" 2>&1 | tee "$LOG_FILE"
