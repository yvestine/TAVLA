#!/bin/bash
# Start plate_sliding raw6 (8004) and encoder (8005) servers on physical GPU 0.
set -euo pipefail
ROOT=/workspace/gujiawei/TA-VLA
cd "$ROOT"
GPU=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
for PORT in 8004 8005; do
  NAME=host9_server_$PORT
  tmux has-session -t $NAME 2>/dev/null && { echo "$NAME exists"; exit 1; }
  CMD="env CUDA_VISIBLE_DEVICES=$GPU JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENPI_DATA_HOME=/workspace/gujiawei/.cache/openpi LEROBOT_HOME=/workspace/gujiawei/.cache/huggingface/lerobot HOST9_DATASET_VERSION=20261026_v1 PYTHONPATH=$ROOT/scripts:$ROOT/src:$ROOT/packages/openpi-client/src $ROOT/.venv/bin/python scripts/serve_host9_plate.py --port $PORT 2>&1 | tee -a logs/$NAME.log"
  tmux new-session -d -s $NAME -c "$ROOT" "bash -o pipefail -c '$CMD'"
  tmux set-window-option -t $NAME:0 remain-on-exit on
  echo "STARTED $NAME on GPU0 ($GPU)"
done
