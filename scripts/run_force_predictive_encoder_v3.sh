#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
if [[ "${RESUME:-0}" == 1 && -z "${RUN_DIR:-}" ]]; then
    echo "RESUME=1 requires the original RUN_DIR" >&2
    exit 1
fi
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/eval_outputs/force_predictive_encoder_v3/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
export PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
ARGS=(--output-dir "$RUN_DIR" --gpu-ids "$GPU_IDS")
[[ "${RESUME:-0}" == 1 ]] && ARGS+=(--resume)
[[ -n "${REAL_DIR:-}" ]] && ARGS+=(--real-root "$REAL_DIR")
[[ -n "${SIM_DIR:-}" ]] && ARGS+=(--sim-root "$SIM_DIR")
[[ -n "${LABELS:-}" ]] && ARGS+=(--labels "$LABELS")
[[ -n "${BASE_RUN:-}" ]] && ARGS+=(--old-base-run "$BASE_RUN")
mkdir -p "$RUN_DIR"
printf 'Force predictive encoder v3\nGPUs: %s\nOutput: %s\n' "$GPU_IDS" "$RUN_DIR"
"$PYTHON_BIN" -m force_predictive_encoder_v3.run "${ARGS[@]}" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
