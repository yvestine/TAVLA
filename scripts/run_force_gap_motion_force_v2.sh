#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_ID="${GPU_ID:?Specify a physical GPU, e.g. GPU_ID=0}"
BASE_RUN="${BASE_RUN:-$PROJECT_DIR/eval_outputs/force_gap_forecast_v1/20261012_155026_551478}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/eval_outputs/force_gap_motion_force_v2/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_DIR/scripts:$PROJECT_DIR/src:$PROJECT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
if [[ ! -f "$BASE_RUN/recipe.json" || ! -f "$BASE_RUN/selections.json" ]]; then
    echo "Missing original motion foundation recipe/selections: $BASE_RUN" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"
ARGS=(--base-run "$BASE_RUN" --output-dir "$RUN_DIR")
[[ "${RESUME:-0}" == 1 ]] && ARGS+=(--resume)
[[ -n "${SIM_DIR:-}" ]] && ARGS+=(--sim-root "$SIM_DIR")
[[ -n "${REAL_DIR:-}" ]] && ARGS+=(--real-root "$REAL_DIR")
[[ -n "${LABELS:-}" ]] && ARGS+=(--labels "$LABELS")
printf 'Motion foundation: %s\nOutput directory: %s\n' "$BASE_RUN" "$RUN_DIR"
"$PYTHON_BIN" -m force_gap_motion_force_v2.train "${ARGS[@]}" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
