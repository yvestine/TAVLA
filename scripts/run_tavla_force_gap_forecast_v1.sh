#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_IDS="${GPU_IDS:?Specify four physical GPUs, e.g. GPU_IDS=0,1,2,3}"
export PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src:$PROJECT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/.cache/openpi}"
export LEROBOT_HOME="${LEROBOT_HOME:-$PROJECT_DIR/../.cache/huggingface/lerobot}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN_NAME="${EXP_NAME:-force_forecast_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
POLICY_DIR="${POLICY_DIR:-$PROJECT_DIR/eval_outputs/force_gap_forecast_policy/$RUN_NAME}"
if [[ -z "${POLICY_MANIFEST:-}" ]]; then
  MANIFEST="${MANIFEST:?Set MANIFEST to the encoder manifest.json}"
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" -m force_gap_forecast_v1.prepare_policy \
    --manifest "$MANIFEST" --output-dir "$POLICY_DIR"
  POLICY_MANIFEST="$POLICY_DIR/policy_manifest.json"
fi
# Validate readiness before downloading or starting any GPU job.
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" - "$POLICY_MANIFEST" <<'PY'
import sys
from force_gap_forecast_v1.policy_train import validate_policy_manifest
validate_policy_manifest(sys.argv[1])
PY
if [[ ! -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]]; then
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$PROJECT_DIR/scripts/download_base_checkpoint.py"
fi
ARGS=(--policy-manifest "$POLICY_MANIFEST" --exp-name "$RUN_NAME" --gpu-ids "$GPU_IDS")
[[ "${RESUME:-0}" == 1 ]] && ARGS+=(--resume)
mkdir -p "$PROJECT_DIR/logs"
"$PYTHON_BIN" -m force_gap_forecast_v1.policy_train "${ARGS[@]}" "$@" 2>&1 | tee -a "$PROJECT_DIR/logs/$RUN_NAME.log"
