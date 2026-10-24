#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot Route A launcher: prepare data + norm stats, then serially fine-tune
# the three models on GPUs 0,1,2,3 (FSDP, per-GPU batch 2, global batch 8).
#
#   Model 1  pi0 pure-sim (no effort)          local/tavla_single_arm_sim_aligned
#   Model 2  TAVLA cotrain (affine force-trend) real trend + sim affine trend, 50/50
#   Model 3  TAVLA pure-sim (affine force-trend) sim affine trend only
#
# Force representation: sim 6-D wrench -> robust affine (sim->real amplitude)
# -> 10-D causal force-trend features. Torque is never used. Gripper is a
# constant channel in both domains, so it is carried through unchanged.
#
# Only the final 30000-step checkpoint is kept (save_interval is 1_000_000 in
# every config, and the train loop always writes the last step).

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
FSDP_DEVICES="${#GPU_ARRAY[@]}"
BATCH_SIZE="${BATCH_SIZE:-8}"          # global; per-GPU = BATCH_SIZE / FSDP_DEVICES
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"   # checkpoint every 10k steps; keep_period=10k retains them
LOG_INTERVAL="${LOG_INTERVAL:-100}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/.cache/openpi}"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

RAW_SIM_DIR="${RAW_SIM_DIR:-$PROJECT_DIR/sim-data-aligned}"
ADAPTER="$PROJECT_DIR/assets/wrench_adapters/sim_aligned_to_real_affine.pt"
SIM_FORCE_CONFIG="$PROJECT_DIR/configs/tavla_sim_force_trend_affine.json"

SIM_ALIGNED_REPO="local/tavla_single_arm_sim_aligned"
SIM_TREND_REPO="local/tavla_single_arm_sim_force_trend_affine"

CFG_SIM_ONLY="pi0_lora_user_single_arm_sim_only"
CFG_COTRAIN="pi0_lora_user_single_arm_force_trend_cotrain_affine_50_50"
CFG_SIM_TREND="pi0_lora_user_single_arm_force_trend_sim_affine"

EXP_SIM_ONLY="${EXP_SIM_ONLY:-sim_only_30k}"
EXP_COTRAIN="${EXP_COTRAIN:-cotrain_affine_trend_50_50_30k}"
EXP_SIM_TREND="${EXP_SIM_TREND:-sim_affine_trend_30k}"

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

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/assets/wrench_adapters"

echo "=================================================================="
echo "Route A: three-model serial fine-tune"
echo "  GPUs=$GPU_IDS  FSDP=$FSDP_DEVICES  global batch=$BATCH_SIZE (per-GPU $((BATCH_SIZE / FSDP_DEVICES)))"
echo "  steps=$TRAIN_STEPS  save_interval=$SAVE_INTERVAL (final-step only)"
echo "=================================================================="

# ---- helpers -----------------------------------------------------------

# effort_shape REPO -> prints the observation.effort last dim, or "MISSING".
effort_shape() {
  local repo="$1"
  local info="$LEROBOT_HOME/$repo/meta/info.json"
  [[ -f "$info" ]] || { echo "MISSING"; return; }
  "$PYTHON_BIN" - "$info" <<'PY'
import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
eps = info.get("total_episodes")
print(f"{shape}:{eps}")
PY
}

compute_stats() {
  local cfg="$1"
  echo ">>> [$cfg] computing norm stats on CPU"
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/compute_norm_stats.py" \
    --config-name "$cfg" --batch-size "$BATCH_SIZE" --num-workers 0
}

train_one() {
  local cfg="$1" exp="$2"
  local log="$PROJECT_DIR/logs/$exp.log"
  echo ">>> [$cfg] training exp=$exp  ->  $log"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train.py" "$cfg" \
    --exp-name "$exp" \
    --batch-size "$BATCH_SIZE" \
    --fsdp-devices "$FSDP_DEVICES" \
    --num-train-steps "$TRAIN_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --overwrite \
    --no-wandb-enabled 2>&1 | tee "$log"
}

# ---- step 1: base checkpoint ------------------------------------------
if [[ ! -e "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]]; then
  echo ">>> downloading pi0_base checkpoint"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/download_base_checkpoint.py"
fi

# ---- step 2: sim datasets ---------------------------------------------
# Model 1 dataset: raw sim (no-effort model ignores effort, but the 6-D wrench
# is harmless to store). Reused for norm-stats only.
if [[ "$(effort_shape "$SIM_ALIGNED_REPO")" != "[6]:40" ]]; then
  echo ">>> converting $SIM_ALIGNED_REPO (raw sim, 40 eps)"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/convert_user_hdf5_to_lerobot.py" \
    --raw-dir "$RAW_SIM_DIR" --repo-id "$SIM_ALIGNED_REPO" \
    --task "peg-in-hole" --effort-key obs/state/ee_wrench_base \
    --action-mode actual --no-videos \
    --image-writer-processes 0 --image-writer-threads 0 --overwrite
else
  echo ">>> $SIM_ALIGNED_REPO already present, skipping conversion"
fi

# Models 2 & 3 dataset: affine-aligned sim -> 10-D force-trend.
if [[ "$(effort_shape "$SIM_TREND_REPO")" != "[10]:40" ]]; then
  echo ">>> converting $SIM_TREND_REPO (affine + force-trend, 40 eps)"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/convert_user_hdf5_to_lerobot.py" \
    --raw-dir "$RAW_SIM_DIR" --repo-id "$SIM_TREND_REPO" \
    --task "peg-in-hole" --effort-key obs/state/ee_wrench_base \
    --effort-representation real_force_trend_v1 \
    --force-config "$SIM_FORCE_CONFIG" \
    --wrench-adapter "$ADAPTER" \
    --action-mode actual --no-videos \
    --image-writer-processes 0 --image-writer-threads 0 --overwrite
else
  echo ">>> $SIM_TREND_REPO already present, skipping conversion"
fi

# The cotrain real half (local/tavla_single_arm_real_force_trend_v1) is assumed
# already built. Guard so we fail early with a clear message if it is missing.
if [[ "$(effort_shape "local/tavla_single_arm_real_force_trend_v1")" != "[10]:40" ]]; then
  echo "ERROR: real force-trend dataset local/tavla_single_arm_real_force_trend_v1 is missing or wrong shape." >&2
  echo "       Build it with scripts/run_tavla_real_force_trend_v1_gpu3.sh (conversion section) first." >&2
  exit 1
fi

# ---- step 3: norm stats (all three) -----------------------------------
compute_stats "$CFG_SIM_ONLY"
compute_stats "$CFG_COTRAIN"
compute_stats "$CFG_SIM_TREND"

# ---- step 4: serial fine-tune -----------------------------------------
train_one "$CFG_SIM_ONLY"  "$EXP_SIM_ONLY"
train_one "$CFG_COTRAIN"   "$EXP_COTRAIN"
train_one "$CFG_SIM_TREND" "$EXP_SIM_TREND"

echo "=================================================================="
echo "All three models finished. Kept checkpoints (steps 10000, 20000, 29999):"
echo "  $PROJECT_DIR/checkpoints/$CFG_SIM_ONLY/$EXP_SIM_ONLY/"
echo "  $PROJECT_DIR/checkpoints/$CFG_COTRAIN/$EXP_COTRAIN/"
echo "  $PROJECT_DIR/checkpoints/$CFG_SIM_TREND/$EXP_SIM_TREND/"
echo "=================================================================="
