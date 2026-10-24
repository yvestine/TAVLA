#!/usr/bin/env bash
set -Eeuo pipefail

# Real-first TAVLA force-history ablation. This script does not mix Sim force.
# Uses two-GPU FSDP by default because Pi0 does not fit comfortably at batch 4
# on one 3090 with three image streams and force history.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-4,5}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
FSDP_DEVICES="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/.cache/openpi}"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

CONFIG_NAME="pi0_lora_user_single_arm_real_force_trend_v1"
REPO_ID="local/tavla_single_arm_real_force_trend_v1"
EXP_NAME="${EXP_NAME:-real_force_trend_v1_gpu45_fsdp}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RAW_DIR="${RAW_DIR:-$PROJECT_DIR/data}"
FORCE_CONFIG="$PROJECT_DIR/configs/tavla_real_force_trend_v1.json"
USE_VIDEOS="${USE_VIDEOS:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

if (( FSDP_DEVICES != 2 )); then
  echo "ERROR: this launcher expects exactly two GPUs; got GPU_IDS=$GPU_IDS" >&2
  exit 1
fi
if (( BATCH_SIZE <= 0 || BATCH_SIZE % FSDP_DEVICES != 0 )); then
  echo "ERROR: global batch size ($BATCH_SIZE) must be divisible by GPU count ($FSDP_DEVICES)" >&2
  exit 1
fi

DATASET_INFO="$LEROBOT_HOME/$REPO_ID/meta/info.json"
DATASET_READY=0
if [[ -f "$DATASET_INFO" ]]; then
  if "$PYTHON_BIN" - "$DATASET_INFO" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    info = json.load(handle)
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
episodes = info.get("total_episodes")
if shape != [10] or episodes != 40:
    raise SystemExit(1)
PY
  then
    DATASET_READY=1
  fi
fi

if [[ "$DATASET_READY" != "1" ]]; then
  echo "Converting Real HDF5 to causal force-trend dataset..."
  conversion_args=(
    "$PROJECT_DIR/scripts/convert_user_hdf5_to_lerobot.py"
    --raw-dir "$RAW_DIR"
    --repo-id "$REPO_ID"
    --effort-key "obs/state/ee_force_base"
    --effort-representation real_force_trend_v1
    --force-config "$FORCE_CONFIG"
    --overwrite
  )
  if [[ "$USE_VIDEOS" != "1" ]]; then
    echo "Using image storage because ffmpeg libsvtav1 is unavailable."
    conversion_args+=(--no-videos)
  fi
  "$PYTHON_BIN" "${conversion_args[@]}"
fi

"$PYTHON_BIN" - "$DATASET_INFO" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    info = json.load(handle)
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
if shape != [10]:
    raise SystemExit(f"ERROR: expected 10-D force trend effort, got {shape}")
print("Validated observation.effort shape: [10]")
PY

STATS_FILE="$PROJECT_DIR/assets/$CONFIG_NAME/$REPO_ID/norm_stats.json"
RECOMPUTE_STATS="${RECOMPUTE_STATS:-1}"
if [[ "$RECOMPUTE_STATS" == "1" || ! -f "$STATS_FILE" ]]; then
  echo "Computing force-trend normalization statistics on CPU..."
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/compute_norm_stats.py" \
    --config-name "$CONFIG_NAME" --batch-size "$BATCH_SIZE" --num-workers 0
fi

if [[ ! -e "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]]; then
  echo "pi0_base checkpoint not found; downloading it..."
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/download_base_checkpoint.py"
fi

LOG_FILE="$PROJECT_DIR/logs/$EXP_NAME.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Starting Real force-trend TAVLA FSDP training on physical GPUs $GPU_IDS"
echo "Config: $CONFIG_NAME"
echo "Global batch size: $BATCH_SIZE (per GPU: $((BATCH_SIZE / FSDP_DEVICES)))"
echo "History: 10 causal frames (about 1 s at 10 Hz)"
echo "Log: $LOG_FILE"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train.py" "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-train-steps "$TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --no-wandb-enabled 2>&1 | tee "$LOG_FILE"
