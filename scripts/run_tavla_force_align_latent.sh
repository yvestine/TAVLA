#!/usr/bin/env bash
set -Eeuo pipefail

# One-click TAVLA fine-tune on the frozen ForceAlignmentEncoder latent as effort.
#
# observation.effort here is the 32-D FORCE-ONLY insertion-alignment latent from
# eval_outputs/insertion_alignment_encoder/encoder.pt (artifact format
# tavla_force_alignment_encoder_terminal_reference_v1). This is a DIFFERENT
# encoder from the control-latent launcher (that one is ControlEncoder, force+pose).
#
# Nothing existing is modified. We reuse the shipped config
# pi0_lora_user_single_arm_control_latent unchanged (identical 32-D-effort /
# 10-frame-history / Pi0 plumbing) and only redirect its dataset + norm-stats to
# a new repo id via tyro CLI overrides, so no new TrainConfig is needed.
#
# Default: 4 GPUs 0,1,2,3, global batch 8 (per-GPU 2), FSDP.
#
#   bash scripts/run_tavla_force_align_latent.sh
#   GPU_IDS=0,1,2,3 BATCH_SIZE=8 bash scripts/run_tavla_force_align_latent.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
FSDP_DEVICES="${#GPU_ARRAY[@]}"
BATCH_SIZE="${BATCH_SIZE:-8}"             # global; per-GPU = BATCH_SIZE / FSDP_DEVICES
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

# We borrow the control-latent config's machinery but point it at our own repo id
# and a private assets dir so neither the control-latent dataset nor its norm
# stats are ever touched.
CONFIG_NAME="pi0_lora_user_single_arm_control_latent"
REPO_ID="${REPO_ID:-local/tavla_single_arm_force_align_latent}"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_DIR/assets/pi0_lora_user_single_arm_force_align_latent}"
EXP_NAME="${EXP_NAME:-force_align_latent_gpu0123_fsdp}"

RAW_DIR="${RAW_DIR:-$PROJECT_DIR/data}"
ENCODER_PT="${ENCODER_PT:-$PROJECT_DIR/eval_outputs/insertion_alignment_encoder/encoder.pt}"
EFFORT_KEY="${EFFORT_KEY:-obs/state/ee_force_base}"
USE_VIDEOS="${USE_VIDEOS:-0}"
EXPECTED_EFFORT_DIM="${EXPECTED_EFFORT_DIM:-32}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/.cache/openpi}"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
CONVERT_PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if (( FSDP_DEVICES < 1 )); then
  echo "ERROR: no GPUs in GPU_IDS=$GPU_IDS" >&2
  exit 1
fi
if (( BATCH_SIZE % FSDP_DEVICES != 0 )); then
  echo "ERROR: global batch size ($BATCH_SIZE) must be divisible by GPU count ($FSDP_DEVICES)" >&2
  exit 1
fi
if [[ ! -f "$ENCODER_PT" ]]; then
  echo "ERROR: frozen encoder not found: $ENCODER_PT" >&2
  echo "       Train it with: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts $PYTHON_BIN -m insertion_alignment_encoder.train_terminal_geometry" >&2
  exit 1
fi

echo "=================================================================="
echo "TAVLA force-alignment-latent fine-tune"
echo "  GPUs=$GPU_IDS  FSDP=$FSDP_DEVICES  global batch=$BATCH_SIZE (per-GPU $((BATCH_SIZE / FSDP_DEVICES)))"
echo "  config=$CONFIG_NAME  repo=$REPO_ID  exp=$EXP_NAME"
echo "  effort=32-D force-only ForceAlignmentEncoder latent, 10 causal history frames"
echo "=================================================================="

# ---- step 1: base checkpoint ------------------------------------------
if [[ ! -e "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]]; then
  echo ">>> downloading pi0_base checkpoint"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/download_base_checkpoint.py"
fi

# ---- step 2: build the 32-D force-align-latent dataset ----------------
DATASET_INFO="$LEROBOT_HOME/$REPO_ID/meta/info.json"
DATASET_READY=0
if [[ -f "$DATASET_INFO" ]]; then
  if EXPECTED_EFFORT_DIM="$EXPECTED_EFFORT_DIM" "$PYTHON_BIN" - "$DATASET_INFO" <<'PY'
import json, os, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
episodes = info.get("total_episodes")
want = int(os.environ["EXPECTED_EFFORT_DIM"])
raise SystemExit(0 if (shape == [want] and episodes == 40) else 1)
PY
  then
    DATASET_READY=1
  fi
fi

if [[ "$DATASET_READY" != "1" ]]; then
  echo ">>> converting Real HDF5 -> force-align-latent (32-D effort) dataset"
  convert_args=(
    "$PROJECT_DIR/scripts/insertion_alignment_encoder/convert_force_alignment_latent.py"
    --raw-dir "$RAW_DIR"
    --repo-id "$REPO_ID"
    --effort-key "$EFFORT_KEY"
    --encoder-path "$ENCODER_PT"
    --overwrite
  )
  [[ "$USE_VIDEOS" != "1" ]] && convert_args+=(--no-videos)
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$CONVERT_PYTHONPATH" "$PYTHON_BIN" "${convert_args[@]}"
else
  echo ">>> $REPO_ID already present with 32-D effort, skipping conversion"
fi

# Validate the dataset effort dim before training.
EXPECTED_EFFORT_DIM="$EXPECTED_EFFORT_DIM" "$PYTHON_BIN" - "$DATASET_INFO" <<'PY'
import json, os, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
want = int(os.environ["EXPECTED_EFFORT_DIM"])
if shape != [want]:
    raise SystemExit(f"ERROR: expected {want}-D effort, got {shape}")
print(f"Validated observation.effort shape: [{want}]")
PY

# ---- step 3: norm stats (into our private assets dir) -----------------
# The stock compute_norm_stats.py cannot take --data.* overrides (its tyro main
# only accepts config_name/batch_size/...), so use the sibling helper that
# reuses its stats logic but accepts an explicit repo id + assets dir.
STATS_FILE="$ASSETS_DIR/$REPO_ID/norm_stats.json"
RECOMPUTE_STATS="${RECOMPUTE_STATS:-1}"
if [[ "$RECOMPUTE_STATS" == "1" || ! -f "$STATS_FILE" ]]; then
  echo ">>> computing norm stats on CPU -> $ASSETS_DIR"
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH="$CONVERT_PYTHONPATH" "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/insertion_alignment_encoder/compute_norm_stats_for_repo.py" \
    --config-name "$CONFIG_NAME" \
    --repo-id "$REPO_ID" \
    --assets-dir "$ASSETS_DIR" \
    --batch-size "$BATCH_SIZE" --num-workers 0
fi

# ---- step 4: FSDP fine-tune -------------------------------------------
LOG_FILE="$PROJECT_DIR/logs/$EXP_NAME.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo ">>> training exp=$EXP_NAME -> $LOG_FILE"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train.py" "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-train-steps "$TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --overwrite \
  --no-wandb-enabled \
  --data.repo-id "$REPO_ID" \
  --data.assets.assets-dir "$ASSETS_DIR" 2>&1 | tee "$LOG_FILE"

echo "=================================================================="
echo "Done. Checkpoints: $PROJECT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME/"
echo "=================================================================="
