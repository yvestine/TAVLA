#!/usr/bin/env bash
set -Eeuo pipefail

# Run from inside the already existing `tavla` tmux session. This script does
# not create or attach to tmux sessions; it performs the encoder preparation,
# LeRobot conversion, norm-stat generation, then the two policy runs in order.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
ENCODER_GPU="${ENCODER_GPU:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
FSDP_DEVICES="${#GPU_ARRAY[@]}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/.cache/openpi}"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

REAL_DIR="${REAL_DIR:-$PROJECT_DIR/data}"
SIM_DIR="${SIM_DIR:-$PROJECT_DIR/sim-data-aligned}"
FORCE_KEY="${FORCE_KEY:-obs/state/ee_force_base}"
ENCODER_PT="${ENCODER_PT:-$PROJECT_DIR/eval_outputs/force_gap_encoder_v2/encoder.pt}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if (( FSDP_DEVICES < 1 || BATCH_SIZE % FSDP_DEVICES != 0 )); then
  echo "ERROR: BATCH_SIZE=$BATCH_SIZE must be divisible by GPU count ($FSDP_DEVICES)" >&2
  exit 1
fi

echo "[1/5] Train force-gap v2 encoder on GPU $ENCODER_GPU (trajectory-grouped CV)"
CUDA_VISIBLE_DEVICES="$ENCODER_GPU" "$PYTHON_BIN" -m force_gap_encoder_v2.train \
  --real-root "$REAL_DIR" \
  --sim-root "$SIM_DIR" \
  --force-key "$FORCE_KEY" \
  --device cuda:0 \
  --output "$ENCODER_PT"

build_v2_dataset() {
  local raw_dir="$1" repo_id="$2"
  echo "[2/5] Convert $raw_dir -> $repo_id"
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m force_gap_encoder_v2.convert_to_lerobot \
    --raw-dir "$raw_dir" \
    --repo-id "$repo_id" \
    --force-key "$FORCE_KEY" \
    --encoder-path "$ENCODER_PT" \
    --overwrite \
    --no-videos
}

build_v2_dataset "$REAL_DIR" "local/tavla_single_arm_force_gap_v2"
build_v2_dataset "$SIM_DIR" "local/tavla_single_arm_force_gap_v2_sim"

compute_stats() {
  local config_name="$1"
  echo "[3/5] Compute norm stats for $config_name"
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/compute_norm_stats.py" \
    --config-name "$config_name" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0
}

train_policy() {
  local config_name="$1" exp_name="$2"
  local log_file="$PROJECT_DIR/logs/$exp_name.log"
  mkdir -p "$(dirname "$log_file")"
  echo "[5/5] Train $config_name ($exp_name), GPUs=$GPU_IDS, steps=$TRAIN_STEPS"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" "$PROJECT_DIR/scripts/train.py" "$config_name" \
    --exp-name "$exp_name" \
    --batch-size "$BATCH_SIZE" \
    --fsdp-devices "$FSDP_DEVICES" \
    --num-train-steps "$TRAIN_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --overwrite \
    --no-wandb-enabled 2>&1 | tee "$log_file"
}

# A uses the already-converted current force-alignment latent datasets. It
# changes only the 50:50 sampler behavior and uses a separate stats directory.
if [[ ! -f "$LEROBOT_HOME/local/tavla_single_arm_force_align_latent/meta/info.json" || \
      ! -f "$LEROBOT_HOME/local/tavla_single_arm_force_align_latent_sim/meta/info.json" ]]; then
  echo "ERROR: current A datasets are missing under $LEROBOT_HOME/local/." >&2
  echo "Build them first with scripts/run_tavla_force_align_latent_cotrain.sh, or set LEROBOT_HOME." >&2
  exit 1
fi
compute_stats "pi0_lora_user_single_arm_force_gap_sampling_a"
compute_stats "pi0_lora_user_single_arm_force_gap_encoder_b"

echo "[4/5] Run the matched policy A/B retrains sequentially"
train_policy "pi0_lora_user_single_arm_force_gap_sampling_a" "force_gap_sampling_a"
train_policy "pi0_lora_user_single_arm_force_gap_encoder_b" "force_gap_encoder_b"

echo "Finished. A: checkpoints/pi0_lora_user_single_arm_force_gap_sampling_a/force_gap_sampling_a/"
echo "         B: checkpoints/pi0_lora_user_single_arm_force_gap_encoder_b/force_gap_encoder_b/"
