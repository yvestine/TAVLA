#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_IDS="${1:-0,1,2,3,5,6,7,8}"
EXP_NAME="${2:-sim_real_cotrain_50_50_30k}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
FSDP_DEVICES="${#GPU_ARRAY[@]}"
BATCH_SIZE="${3:-$FSDP_DEVICES}"

REAL_RAW_DIR="$PROJECT_DIR/data"
SIM_RAW_DIR="$PROJECT_DIR/sim_force"
REAL_REPO_ID="local/tavla_single_arm_ee_wrench"
SIM_REPO_ID="local/tavla_single_arm_ee_wrench_sim_hand_noise_50"
CONFIG_NAME="pi0_lora_user_single_arm_ee_wrench_cotrain_base_50_50"
TRAIN_STEPS=30000
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
STATS_DIR="$PROJECT_DIR/assets/pi0_lora_user_single_arm_ee_wrench_cotrain_50_50"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

if (( FSDP_DEVICES < 2 )); then
  echo "ERROR: Co-training needs at least two GPUs; got: $GPU_IDS" >&2
  exit 1
fi
if (( BATCH_SIZE <= 0 || BATCH_SIZE % FSDP_DEVICES != 0 )); then
  echo "ERROR: global batch size ($BATCH_SIZE) must be positive and divisible by GPU count ($FSDP_DEVICES)" >&2
  exit 1
fi

for raw_dir in "$REAL_RAW_DIR" "$SIM_RAW_DIR"; do
  episode_count="$(find "$raw_dir" -mindepth 2 -maxdepth 2 -type f -path '*/traj_*/data.h5' | wc -l)"
  if [[ "$episode_count" -eq 0 ]]; then
    echo "ERROR: no traj_*/data.h5 episodes found under $raw_dir" >&2
    exit 1
  fi
  echo "Found $episode_count episodes under $raw_dir"
done

cd "$PROJECT_DIR"

export OPENPI_DATA_HOME="$PROJECT_DIR/.cache/openpi"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[provenance] Validating Sim contact-force source..."
SIM_REPO_PATH="$LEROBOT_HOME/$SIM_REPO_ID"
SIM_VALIDATION_ARGS=(--raw-dir "$SIM_RAW_DIR")
if [[ -f "$SIM_REPO_PATH/meta/info.json" ]]; then
  SIM_VALIDATION_ARGS+=(--lerobot-repo "$SIM_REPO_PATH")
fi
"$PYTHON_BIN" scripts/validate_sim_force_source.py "${SIM_VALIDATION_ARGS[@]}"

if [[ -f "$LEROBOT_HOME/$REAL_REPO_ID/meta/info.json" ]]; then
  echo "[1/4] Reusing converted real-robot dataset."
else
  echo "[1/4] Converting real-robot data to LeRobot..."
  "$PYTHON_BIN" scripts/convert_user_hdf5_to_lerobot.py \
    --raw-dir "$REAL_RAW_DIR" \
    --repo-id "$REAL_REPO_ID" \
    --task "peg-in-hole" \
    --effort-key obs/state/ee_wrench_base \
    --action-mode actual \
    --no-videos \
    --image-writer-processes 0 \
    --image-writer-threads 0 \
    --overwrite
fi

if [[ -f "$LEROBOT_HOME/$SIM_REPO_ID/meta/info.json" ]]; then
  echo "[2/4] Reusing converted simulation dataset."
else
  echo "[2/4] Converting simulation data to LeRobot..."
  "$PYTHON_BIN" scripts/convert_user_hdf5_to_lerobot.py \
    --raw-dir "$SIM_RAW_DIR" \
    --repo-id "$SIM_REPO_ID" \
    --task "peg-in-hole" \
    --effort-key sim/force/contact_training_wrench \
    --action-mode actual \
    --no-videos \
    --image-writer-processes 0 \
    --image-writer-threads 0 \
    --overwrite
fi

if [[ -f "$STATS_DIR/$REAL_REPO_ID/norm_stats.json" && -f "$STATS_DIR/$SIM_REPO_ID/norm_stats.json" ]]; then
  echo "[3/4] Reusing normalization stats."
else
  echo "[3/4] Computing normalization stats for the 50/50 mixture..."
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" scripts/compute_norm_stats.py \
    --config-name "$CONFIG_NAME" \
    --batch-size 8 \
    --num-workers 0
fi

echo "[4/4] Starting Pi0 LoRA Co-training with FSDP on GPUs $GPU_IDS (global batch size $BATCH_SIZE)..."
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-train-steps "$TRAIN_STEPS" \
  --save-interval 1000000 \
  --no-wandb-enabled \
  --overwrite

echo "Training complete. Checkpoint after 30,000 updates:"
echo "$PROJECT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME/29999/params"
