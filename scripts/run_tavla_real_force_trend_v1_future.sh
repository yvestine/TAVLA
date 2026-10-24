#!/usr/bin/env bash
set -Eeuo pipefail

# Real-first TAVLA with future force-trend auxiliary prediction.
# The model predicts actions and the next action-horizon force features during
# training; inference still returns actions only.
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

CONFIG_NAME="pi0_lora_user_single_arm_real_force_trend_v1_future"
REPO_ID="local/tavla_single_arm_real_force_trend_v1"
EXP_NAME="${EXP_NAME:-real_force_trend_v1_future_gpu45_fsdp}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RAW_DIR="${RAW_DIR:-$PROJECT_DIR/data}"
FORCE_CONFIG="$PROJECT_DIR/configs/tavla_real_force_trend_v1.json"
RECOMPUTE_STATS="${RECOMPUTE_STATS:-1}"

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
  echo "Force-trend dataset is missing or incomplete; rebuilding it..."
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/convert_user_hdf5_to_lerobot.py" \
    --raw-dir "$RAW_DIR" \
    --repo-id "$REPO_ID" \
    --effort-key "obs/state/ee_force_base" \
    --effort-representation real_force_trend_v1 \
    --force-config "$FORCE_CONFIG" \
    --no-videos \
    --overwrite
fi

"$PYTHON_BIN" - "$DATASET_INFO" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    info = json.load(handle)
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
episodes = info.get("total_episodes")
if shape != [10] or episodes != 40:
    raise SystemExit(f"ERROR: expected 40 episodes with 10-D effort, got episodes={episodes}, shape={shape}")
print("Validated Real force-trend dataset: 40 episodes, observation.effort=[10]")
PY

STATS_FILE="$PROJECT_DIR/assets/$CONFIG_NAME/$REPO_ID/norm_stats.json"
if [[ "$RECOMPUTE_STATS" == "1" || ! -f "$STATS_FILE" ]]; then
  echo "Computing history + future force-trend statistics on CPU..."
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/compute_norm_stats.py" \
    --config-name "$CONFIG_NAME" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0
fi

if [[ ! -e "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]]; then
  echo "pi0_base checkpoint not found; downloading it..."
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/download_base_checkpoint.py"
fi

LOG_FILE="$PROJECT_DIR/logs/$EXP_NAME.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Starting Real force-trend + future-force TAVLA training"
echo "Physical GPUs: $GPU_IDS"
echo "Global batch size: $BATCH_SIZE (per GPU: $((BATCH_SIZE / FSDP_DEVICES)))"
echo "Future force horizon: action horizon (currently 50 frames, about 5 s at 10 Hz)"
echo "Future force loss weight: 0.1"
echo "Log: $LOG_FILE"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train.py" "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-train-steps "$TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --no-wandb-enabled 2>&1 | tee "$LOG_FILE"
