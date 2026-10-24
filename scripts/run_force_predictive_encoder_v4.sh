#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
[[ "$GPU_IDS" =~ ^[0-9]+$ ]] || { echo 'This single-run launcher needs one physical GPU, e.g. GPU_IDS=6' >&2; exit 1; }
if [[ "${RESUME:-0}" == 1 && -z "${RUN_DIR:-}" ]]; then
    echo 'RESUME=1 requires the original RUN_DIR' >&2
    exit 1
fi
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/eval_outputs/force_predictive_encoder_v4/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
ARGS=(--output-dir "$RUN_DIR")
[[ "${RESUME:-0}" == 1 ]] && ARGS+=(--resume)
[[ -n "${REAL_DIR:-}" ]] && ARGS+=(--real-root "$REAL_DIR")
[[ -n "${SIM_DIR:-}" ]] && ARGS+=(--sim-root "$SIM_DIR")
[[ -n "${LABELS:-}" ]] && ARGS+=(--labels "$LABELS")
mkdir -p "$RUN_DIR"
printf 'Force predictive encoder v4; physical GPU %s\nOutput: %s\n' "$GPU_IDS" "$RUN_DIR"
"${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}" -m force_predictive_encoder_v4.train "${ARGS[@]}" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
