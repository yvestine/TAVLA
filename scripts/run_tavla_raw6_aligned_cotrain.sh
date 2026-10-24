#!/usr/bin/env bash
set -Eeuo pipefail

# Four-GPU raw six-channel TAVLA cotrain using the SAME aligned simulation
# recordings as the force-alignment encoder policy. Simulation torque is zero
# in this source. No encoder, trend features or extra wrench adapter is used.
# Run manually; this launcher prepares CPU norm stats, then starts training.
# It creates a fresh run and saves only the final checkpoint at step 29999.
#
# GPU_IDS=1,2,4,5 bash scripts/run_tavla_raw6_aligned_cotrain.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-1,2,4,5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EXP_NAME="${EXP_NAME:-raw6_aligned_cotrain_50_50_30k_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
CONFIG_NAME="pi0_lora_user_single_arm_ee_wrench_cotrain_aligned_50_50"
REAL_REPO="local/tavla_single_arm_ee_wrench"
SIM_REPO="local/tavla_single_arm_sim_aligned"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/.cache/openpi}"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || (( BATCH_SIZE % 4 != 0 )); then
  echo "ERROR: BATCH_SIZE must be a positive multiple of four." >&2
  exit 1
fi
if [[ -z "$EXP_NAME" || "$EXP_NAME" == */* || "$EXP_NAME" == "." || "$EXP_NAME" == ".." ]]; then
  echo "ERROR: EXP_NAME must be a nonempty directory name without slashes." >&2
  exit 1
fi

CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME"
LOG_FILE="$PROJECT_DIR/logs/$EXP_NAME.log"
if [[ -e "$CHECKPOINT_DIR" || -e "$LOG_FILE" ]]; then
  echo "ERROR: run or log already exists; choose a new EXP_NAME: $EXP_NAME" >&2
  exit 1
fi

# Resolve physical nvidia-smi indices to UUIDs, avoiding CUDA index ambiguity.
# Refuse to start on an occupied card. This check does not reserve the GPUs.
GPU_UUIDS="$("$PYTHON_BIN" - "$GPU_IDS" <<'PY'
import csv
import subprocess
import sys

selectors = sys.argv[1].split(',')
if len(selectors) != 4 or len(set(selectors)) != 4:
    raise SystemExit('ERROR: GPU_IDS must select exactly four distinct GPUs.')
inventory = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
    '--format=csv,noheader,nounits',
], text=True, timeout=45)
lookup = {}
for row in csv.reader(inventory.splitlines()):
    row = [value.strip() for value in row]
    lookup[row[0]] = lookup[row[1]] = row
processes = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits',
], text=True, timeout=45)
active = {}
for row in csv.reader(processes.splitlines()):
    if len(row) >= 2:
        active.setdefault(row[0].strip(), []).append(row[1].strip())
uuids = []
for selector in selectors:
    if selector not in lookup:
        raise SystemExit(f'ERROR: GPU not found: {selector}')
    index, uuid, used, free, utilization = lookup[selector]
    if uuid in uuids:
        raise SystemExit(f'ERROR: duplicate physical GPU: {selector}')
    if uuid in active or int(used) > 1024 or int(free) < 20000 or int(utilization) != 0:
        raise SystemExit(f'ERROR: GPU {index} is occupied: used={used} MiB, free={free} MiB, '
                         f'utilization={utilization}%, PIDs={active.get(uuid, [])}')
    print(f'GPU {index}: {uuid}, free={free} MiB', file=sys.stderr)
    uuids.append(uuid)
print(','.join(uuids))
PY
)"
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"

# The raw simulation dataset was compared with all 40 aligned recordings.
# Validate both datasets' six-channel schema before creating stats or a new run.
"$PYTHON_BIN" - "$REAL_REPO" "$SIM_REPO" <<'PY'
import json
import os
from pathlib import Path
import sys

for repo in sys.argv[1:]:
    info_path = Path(os.environ['LEROBOT_HOME']) / repo / 'meta/info.json'
    if not info_path.is_file():
        raise SystemExit(f'ERROR: prepared dataset not found: {info_path}')
    info = json.loads(info_path.read_text())
    for key, shape in [('observation.effort', [6]), ('observation.state', [8]), ('action', [8])]:
        actual = info['features'].get(key, {}).get('shape')
        if actual != shape:
            raise SystemExit(f'ERROR: {repo} {key}: expected {shape}, got {actual}')
    if info['total_episodes'] != 40 or info['total_frames'] != 7188 or info['fps'] != 10:
        raise SystemExit(f'ERROR: {repo} differs from the matched 40-episode, 7188-frame, 10-Hz data.')
    print(f'{repo}: 40 episodes, 7188 frames, 10 Hz, raw effort [6]')
PY

if [[ ! -e "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]]; then
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$PROJECT_DIR/scripts/download_base_checkpoint.py"
fi

STATS_ROOT="$PROJECT_DIR/assets/$CONFIG_NAME"
if [[ ! -f "$STATS_ROOT/$REAL_REPO/norm_stats.json" || ! -f "$STATS_ROOT/$SIM_REPO/norm_stats.json" ]]; then
  echo "Computing six-channel normalization statistics for both datasets on CPU..."
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" \
    "$PROJECT_DIR/scripts/compute_norm_stats.py" \
    --config-name "$CONFIG_NAME" --batch-size 8 --num-workers 0
fi

mkdir -p "$PROJECT_DIR/logs"
echo "Raw6 TAVLA cotrain: real/sim 50:50, four-GPU FSDP, global batch=$BATCH_SIZE, 30000 steps"
echo "Simulation source: sim-data-aligned; torque channels are zero in that source."
echo "Final checkpoint: $CHECKPOINT_DIR/29999"
echo "Log: $LOG_FILE"
JAX_PLATFORMS=cuda "$PYTHON_BIN" "$PROJECT_DIR/scripts/train.py" "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices 4 \
  --num-train-steps 30000 \
  --save-interval 1000000 \
  --log-interval 100 \
  --no-wandb-enabled 2>&1 | tee -a "$LOG_FILE"
