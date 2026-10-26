#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
STAGE="${STAGE:-teacher}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/eval_outputs/force_predictive_v3_multitask/host9_$STAGE}"
DEVICE="${DEVICE:-cuda:0}"
if [[ "$DEVICE" == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
fi
export PYTHONPATH="$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
ARGS=(--stage "$STAGE" --output-dir "$RUN_DIR" --device "$DEVICE")
[[ -n "${CONFIG:-}" ]] && ARGS+=(--config "$CONFIG")
[[ -n "${DATA_ROOT:-}" ]] && ARGS+=(--data-root "$DATA_ROOT")
[[ -n "${TEACHER_RUN:-}" ]] && ARGS+=(--teacher-run "$TEACHER_RUN")
[[ "${RESUME:-0}" == 1 ]] && ARGS+=(--resume)
exec "$PYTHON_BIN" -m force_predictive_encoder_v3.multitask "${ARGS[@]}" "$@"
