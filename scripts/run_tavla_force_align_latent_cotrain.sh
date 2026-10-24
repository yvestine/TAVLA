#!/usr/bin/env bash
set -Eeuo pipefail

# TAVLA real+sim cotrain on the frozen ForceAlignmentEncoder latent as effort.
#
# observation.effort is the 32-D FORCE-ONLY insertion-alignment latent from
# eval_outputs/insertion_alignment_encoder/encoder.pt. The SAME frozen encoder
# encodes both domains:
#   Real half: data/traj_*          -> local/tavla_single_arm_force_align_latent
#   Sim  half: sim-data-aligned/*   -> local/tavla_single_arm_force_align_latent_sim
# 50/50 mixture, mirroring the Route A force-trend cotrain.
#
# Nothing existing is modified except the one appended TrainConfig
# pi0_lora_user_single_arm_force_align_latent_cotrain_50_50 (list repo_id +
# dataset_sampling_weights). The stock compute_norm_stats.py / train.py read the
# repo list and per-dataset assets from that config directly, so no --data.*
# CLI override is needed here.
#
# Default: 4 GPUs 0,1,2,3, global batch 8 (per-GPU 2), FSDP.
#
#   bash scripts/run_tavla_force_align_latent_cotrain.sh
#   GPU_IDS=0,1,2,3 BATCH_SIZE=8 bash scripts/run_tavla_force_align_latent_cotrain.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
FSDP_DEVICES="${#GPU_ARRAY[@]}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

ENCODER_KIND="${ENCODER_KIND:-force_alignment}"
if [[ "$ENCODER_KIND" == motion_only_v2 ]]; then
  CONFIG_NAME="pi0_lora_user_single_arm_motion_only_v2_cotrain_50_50"
  REAL_REPO="${REAL_REPO:-local/tavla_motion_only_v2_seed44_real}"
  SIM_REPO="${SIM_REPO:-local/tavla_motion_only_v2_seed44_sim}"
  ENCODER_PT="${ENCODER_PT:-${MANIFEST:-$PROJECT_DIR/eval_outputs/force_gap_motion_force_v2/20261012_215515_838437/manifest.json}}"
  EXP_NAME="${EXP_NAME:-motion_only_v2_cotrain_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
  export MOTION_ONLY_V2_REAL_REPO="$REAL_REPO" MOTION_ONLY_V2_SIM_REPO="$SIM_REPO"
  unset MOTION_ONLY_V2_POLICY_MANIFEST
elif [[ "$ENCODER_KIND" == force_predictive_v3 ]]; then
  CONFIG_NAME="pi0_lora_user_single_arm_force_predictive_v3_cotrain_50_50"
  REAL_REPO="${REAL_REPO:-local/tavla_force_predictive_v3_da9117c7091b_real}"
  SIM_REPO="${SIM_REPO:-local/tavla_force_predictive_v3_da9117c7091b_sim}"
  ENCODER_PT="${ENCODER_PT:-${MANIFEST:-$PROJECT_DIR/eval_outputs/force_predictive_encoder_v3/20261014_160410_2018868/manifest.json}}"
  EXP_NAME="${EXP_NAME:-force_predictive_v3_cotrain_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
  export FORCE_PREDICTIVE_V3_REAL_REPO="$REAL_REPO" FORCE_PREDICTIVE_V3_SIM_REPO="$SIM_REPO"
elif [[ "$ENCODER_KIND" == force_alignment ]]; then
  CONFIG_NAME="pi0_lora_user_single_arm_force_align_latent_cotrain_50_50"
  REAL_REPO="${REAL_REPO:-local/tavla_single_arm_force_align_latent}"
  SIM_REPO="${SIM_REPO:-local/tavla_single_arm_force_align_latent_sim}"
  EXP_NAME="${EXP_NAME:-force_align_latent_cotrain_50_50_gpu0123}"
else
  echo "ERROR: ENCODER_KIND must be force_alignment, motion_only_v2 or force_predictive_v3" >&2
  exit 1
fi
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_DIR/assets/$CONFIG_NAME}"
[[ "$ENCODER_KIND" == motion_only_v2 ]] && export MOTION_ONLY_V2_ASSETS_DIR="$ASSETS_DIR"
[[ "$ENCODER_KIND" == force_predictive_v3 ]] && export FORCE_PREDICTIVE_V3_ASSETS_DIR="$ASSETS_DIR"

REAL_DIR="${REAL_DIR:-$PROJECT_DIR/data}"
SIM_DIR="${SIM_DIR:-$PROJECT_DIR/sim-data-aligned}"
ENCODER_PT="${ENCODER_PT:-$PROJECT_DIR/eval_outputs/insertion_alignment_encoder/encoder.pt}"
EFFORT_KEY="${EFFORT_KEY:-obs/state/ee_force_base}"
USE_VIDEOS="${USE_VIDEOS:-0}"
EXPECTED_EFFORT_DIM="${EXPECTED_EFFORT_DIM:-32}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/.cache/openpi}"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export PYTHONUNBUFFERED=1
CONVERT_PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src"
export PYTHONPATH="$CONVERT_PYTHONPATH:$PROJECT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

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
  exit 1
fi
if [[ "$ENCODER_KIND" == force_predictive_v3 ]]; then
  # Resolve nvidia-smi physical indices; do not rely on CUDA enumeration order.
  export CUDA_VISIBLE_DEVICES="$("$PYTHON_BIN" - "$GPU_IDS" <<'PY'
import csv
import subprocess
import sys
selectors = sys.argv[1].split(',')
rows = csv.reader(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits',
], text=True).splitlines())
lookup = {}
for row in rows:
    index, uuid, used = (value.strip() for value in row)
    lookup[index] = lookup[uuid] = (index, uuid, int(used))
uuids = []
for selector in selectors:
    if selector not in lookup:
        raise SystemExit(f'GPU not found: {selector}')
    index, uuid, used = lookup[selector]
    if uuid in uuids or used > 1024:
        raise SystemExit(f'GPU {index} is duplicated or occupied ({used} MiB used)')
    print(f'Physical GPU {index}: {uuid}', file=sys.stderr)
    uuids.append(uuid)
print(','.join(uuids))
PY
)"
  [[ -n "$CUDA_VISIBLE_DEVICES" ]] || exit 1
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" - "$ENCODER_PT" <<'PY'
import sys
from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
ForcePredictiveEncoder(sys.argv[1], domain="real", device="cpu")
print('Verified frozen force-predictive-v3 encoder format and SHA-256')
PY
fi

echo "=================================================================="
echo "TAVLA $ENCODER_KIND latent cotrain (real+sim, 50/50)"
echo "  GPUs=$GPU_IDS  FSDP=$FSDP_DEVICES  global batch=$BATCH_SIZE (per-GPU $((BATCH_SIZE / FSDP_DEVICES)))"
echo "  config=$CONFIG_NAME  exp=$EXP_NAME"
echo "  real=$REAL_REPO  sim=$SIM_REPO"
echo "=================================================================="

# repo_present REPO DIR -> builds a 32-D-effort dataset from DIR unless already present.
build_dataset() {
  local repo="$1" raw="$2" domain="$3"
  local info="$LEROBOT_HOME/$repo/meta/info.json"
  local ready=0
  if [[ -f "$info" ]]; then
    if EXPECTED_EFFORT_DIM="$EXPECTED_EFFORT_DIM" "$PYTHON_BIN" - "$info" "$ENCODER_KIND" "$domain" "$ENCODER_PT" "$raw" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
info = json.loads(path.read_text())
shape = info.get("features", {}).get("observation.effort", {}).get("shape")
episodes = info.get("total_episodes")
want = int(os.environ["EXPECTED_EFFORT_DIM"])
if shape != [want] or episodes != 40:
    raise SystemExit(1)
if sys.argv[2] in ("motion_only_v2", "force_predictive_v3"):
    encoder = json.loads(Path(sys.argv[4]).read_text())
    markers = ([path.parent / "force_predictive_v3_encoder.json"] if sys.argv[2] == "force_predictive_v3"
               else [path.parent / "motion_only_encoder.json", path.parent / "motion_only_v2_contract.json"])
    marker = next((p for p in markers if p.is_file()), None)
    if marker is None:
        raise SystemExit(1)
    saved = json.loads(marker.read_text())
    if (saved.get("encoder_sha256") != encoder["encoder_sha256"] or saved.get("domain") != sys.argv[3]
            or saved.get("completed") is not True or saved.get("layout") != encoder["layout"]):
        raise SystemExit(1)
    sources = saved.get("source_h5_sha256", {})
    if len(sources) != 40:
        raise SystemExit(1)
    for tid, digest in sources.items():
        source = Path(sys.argv[5]) / f"traj_{tid}/data.h5"
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise SystemExit(1)
raise SystemExit(0)
PY
    then
      ready=1
    fi
  fi
  if [[ "$ready" == "1" ]]; then
    echo ">>> $repo already present with 32-D effort, skipping conversion"
    return
  fi
  if [[ "$ENCODER_KIND" != force_alignment && -e "$LEROBOT_HOME/$repo" ]]; then
    echo "ERROR: existing $ENCODER_KIND cache is incomplete or uses a different encoder/source; choose a new repo ID: $repo" >&2
    exit 1
  fi
  echo ">>> converting $raw -> $repo (32-D $ENCODER_KIND latent)"
  local args=(
    "$PROJECT_DIR/scripts/insertion_alignment_encoder/convert_force_alignment_latent.py"
    --raw-dir "$raw" --repo-id "$repo"
    --effort-key "$EFFORT_KEY" --encoder-path "$ENCODER_PT"
  )
  if [[ "$ENCODER_KIND" != force_alignment ]]; then
    args+=(--encoder-kind "$ENCODER_KIND" --domain "$domain")
  else
    args+=(--overwrite)
  fi
  [[ "$USE_VIDEOS" != "1" ]] && args+=(--no-videos)
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$CONVERT_PYTHONPATH" "$PYTHON_BIN" "${args[@]}"
}

# ---- step 1: base checkpoint ------------------------------------------
if [[ ! -e "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]]; then
  echo ">>> downloading pi0_base checkpoint"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/download_base_checkpoint.py"
fi

# ---- step 2: build both datasets --------------------------------------
build_dataset "$REAL_REPO" "$REAL_DIR" real
build_dataset "$SIM_REPO" "$SIM_DIR" sim

# ---- step 3: norm stats (stock script handles the list repo_id) -------
# The cotrain config names both repos and its own assets dir, so the stock
# compute_norm_stats.py iterates them and writes <assets_dir>/<repo>/norm_stats.
RECOMPUTE_STATS="${RECOMPUTE_STATS:-1}"
STATS_REAL="$ASSETS_DIR/$REAL_REPO/norm_stats.json"
STATS_SIM="$ASSETS_DIR/$SIM_REPO/norm_stats.json"
if [[ "$RECOMPUTE_STATS" == "1" || ! -f "$STATS_REAL" || ! -f "$STATS_SIM" ]]; then
  echo ">>> computing norm stats on CPU for both datasets"
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/compute_norm_stats.py" \
    --config-name "$CONFIG_NAME" --batch-size "$BATCH_SIZE" --num-workers 0
fi

# ---- step 4: FSDP cotrain ---------------------------------------------
LOG_FILE="$PROJECT_DIR/logs/$EXP_NAME.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo ">>> training exp=$EXP_NAME -> $LOG_FILE"
if [[ "$ENCODER_KIND" != force_alignment && "${RESUME:-0}" != 1 && -e "$PROJECT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME" ]]; then
  echo "ERROR: checkpoint experiment exists; choose a new EXP_NAME or set RESUME=1." >&2
  exit 1
fi
TRAIN_FLAGS=()
if [[ "${RESUME:-0}" == 1 ]]; then
  TRAIN_FLAGS+=(--resume)
elif [[ "$ENCODER_KIND" == force_alignment ]]; then
  TRAIN_FLAGS+=(--overwrite)
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train.py" "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-train-steps "$TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  "${TRAIN_FLAGS[@]}" \
  --no-wandb-enabled 2>&1 | tee "$LOG_FILE"

if [[ "$ENCODER_KIND" != force_alignment ]]; then
  "$PYTHON_BIN" - "$ENCODER_PT" "$PROJECT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME" "$REAL_REPO" "$SIM_REPO" "$ENCODER_KIND" <<'PY'
import json
from pathlib import Path
import shutil
import sys
source = Path(sys.argv[1]).resolve()
manifest = json.loads(source.read_text())
kind = sys.argv[5]
for checkpoint in Path(sys.argv[2]).iterdir():
    if not checkpoint.name.isdecimal() or not (checkpoint / "assets/norm_stats.json").is_file():
        continue
    assets = checkpoint / "assets"
    shutil.copy2(source.parent / manifest["checkpoint"], assets / f"{kind}_encoder.pt")
    portable = dict(manifest, checkpoint=f"{kind}_encoder.pt")
    (assets / f"{kind}_encoder_manifest.json").write_text(json.dumps(portable, indent=2) + "\n")
    (assets / f"{kind}_training_inputs.json").write_text(json.dumps({
        "encoder_sha256": manifest["encoder_sha256"], "encoder_frozen": True,
        "repo_ids": sys.argv[3:5], "sampling_weights": [.5,.5], "effort_shape": [10,32],
        "action_source": "domain_native_actual", "actual_sim_state_verified": manifest["actual_sim_state_verified"],
    }, indent=2) + "\n")
PY
fi

echo "=================================================================="
echo "Done. Checkpoints: $PROJECT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME/"
echo "=================================================================="
