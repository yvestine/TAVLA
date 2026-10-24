#!/usr/bin/env bash
set -Eeuo pipefail

# One-click TAVLA fine-tune on the frozen control-encoder latent as effort.
#
# observation.effort here is the 32-D control-supervised domain-invariant latent
# (built by export_control_encoder.py, emitted per-frame by control_encoder_infer.py
# via convert_user_hdf5_to_lerobot.py --effort-representation control_latent).
#
# This launcher does everything: freeze/export the encoder if missing, build the
# 32-D-effort LeRobot dataset, compute norm stats, download pi0_base if needed, and
# start FSDP training. It NEVER reads or writes any existing TAVLA/pi0 checkpoint
# except the read-only pi0_base weight loader; the encoder work stays under its
# own eval_outputs/ tree.
#
# Needs 2 GPUs (Pi0 + three image streams + effort history does not fit at batch 4
# on a single 3090). On 4x3090, set GPU_IDS to a pair and raise BATCH_SIZE.
#
#   GPU_IDS=4,5 BATCH_SIZE=4 bash scripts/run_tavla_control_latent.sh

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

CONFIG_NAME="pi0_lora_user_single_arm_control_latent"
REPO_ID="local/tavla_single_arm_control_latent"
EXP_NAME="${EXP_NAME:-control_latent_gpu45_fsdp}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RAW_DIR="${RAW_DIR:-$PROJECT_DIR/data}"
ENCODER_DIR="${ENCODER_DIR:-$PROJECT_DIR/eval_outputs/control_encoder_deploy}"
ENCODER_PT="$ENCODER_DIR/encoder.pt"
EFFORT_KEY="${EFFORT_KEY:-obs/state/ee_force_base}"
USE_VIDEOS="${USE_VIDEOS:-0}"
EXPECTED_EFFORT_DIM="${EXPECTED_EFFORT_DIM:-32}"

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

# Step A: freeze the deployable encoder once, if it is not already there.
if [[ ! -f "$ENCODER_PT" ]]; then
  echo "Frozen control encoder not found; training + exporting it (offline, ~1 min on 1 GPU)..."
  PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src" "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/export_control_encoder.py" \
    --output-dir "$ENCODER_DIR" \
    --device "cuda:0"
fi

# Step B: build the 32-D-effort dataset if missing or wrong-shaped.
DATASET_INFO="$LEROBOT_HOME/$REPO_ID/meta/info.json"
DATASET_READY=0
if [[ -f "$DATASET_INFO" ]]; then
  if EXPECTED_EFFORT_DIM="$EXPECTED_EFFORT_DIM" "$PYTHON_BIN" - "$DATASET_INFO" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    info = json.load(handle)
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
episodes = info.get("total_episodes")
want = int(os.environ["EXPECTED_EFFORT_DIM"])
if shape != [want] or episodes != 40:
    raise SystemExit(1)
PY
  then
    DATASET_READY=1
  fi
fi

if [[ "$DATASET_READY" != "1" ]]; then
  echo "Converting Real HDF5 to control-latent (32-D effort) dataset..."
  conversion_args=(
    "$PROJECT_DIR/scripts/convert_user_hdf5_to_lerobot.py"
    --raw-dir "$RAW_DIR"
    --repo-id "$REPO_ID"
    --effort-key "$EFFORT_KEY"
    --effort-representation control_latent
    --encoder-path "$ENCODER_PT"
    --overwrite
  )
  if [[ "$USE_VIDEOS" != "1" ]]; then
    echo "Using image storage because ffmpeg libsvtav1 is unavailable."
    conversion_args+=(--no-videos)
  fi
  PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src" "$PYTHON_BIN" "${conversion_args[@]}"
fi

EXPECTED_EFFORT_DIM="$EXPECTED_EFFORT_DIM" "$PYTHON_BIN" - "$DATASET_INFO" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    info = json.load(handle)
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
want = int(os.environ["EXPECTED_EFFORT_DIM"])
if shape != [want]:
    raise SystemExit(f"ERROR: expected {want}-D control-latent effort, got {shape}")
print(f"Validated observation.effort shape: [{want}]")
PY

STATS_FILE="$PROJECT_DIR/assets/$CONFIG_NAME/$REPO_ID/norm_stats.json"
RECOMPUTE_STATS="${RECOMPUTE_STATS:-1}"
if [[ "$RECOMPUTE_STATS" == "1" || ! -f "$STATS_FILE" ]]; then
  echo "Computing control-latent normalization statistics on CPU..."
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
echo "Starting control-latent TAVLA FSDP training on physical GPUs $GPU_IDS"
echo "Config: $CONFIG_NAME"
echo "Global batch size: $BATCH_SIZE (per GPU: $((BATCH_SIZE / FSDP_DEVICES)))"
echo "Effort: 32-D frozen control-encoder latent, 10 causal frames of history"
echo "Log: $LOG_FILE"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train.py" "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-train-steps "$TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --no-wandb-enabled 2>&1 | tee "$LOG_FILE"
