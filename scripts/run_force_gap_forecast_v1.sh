#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_ID="${GPU_ID:?Specify one physical idle GPU, e.g. GPU_ID=0}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/eval_outputs/force_gap_forecast_v1/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src:$PROJECT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p "$RUN_DIR"
ARGS=(--real-root "${REAL_DIR:-$PROJECT_DIR/data}" --sim-root "${SIM_DIR:-$PROJECT_DIR/sim-data-aligned}"
      --output-dir "$RUN_DIR" --labels "${LABELS:-$PROJECT_DIR/annotations/insertion_quality_labels_v2.csv}")
[[ "${RESUME:-0}" == 1 ]] && ARGS+=(--resume)
"$PYTHON_BIN" -m force_gap_forecast_v1.train "${ARGS[@]}" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
echo "Encoder manifest: $RUN_DIR/manifest.json"
