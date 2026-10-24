#!/usr/bin/env bash
# User-run encoder experiment. Uses the current shell; creates no tmux session.
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/scripts:$TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
GPU_ID="${GPU_ID:-7}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m force_gap_encoder_v2.train \
  --recipe motion_v1 \
  --real-root data --sim-root sim-data-aligned \
  --labels annotations/insertion_quality_labels_v2.csv \
  --force-key obs/state/ee_force_base \
  --cutoff-checkpoint eval_outputs/insertion_alignment_encoder/encoder.pt \
  --folds 4 --fold-seed 42 --seeds 42 43 44 \
  --alignment-weights 0 0.01 0.1 \
  --holdout-trajectories 0 2 21 22 23 33 34 39 \
  --device cuda:0 \
  --output eval_outputs/force_gap_encoder_v2/motion_encoder.pt

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m force_gap_encoder_v2.validate_motion \
  --checkpoint eval_outputs/force_gap_encoder_v2/motion_encoder.pt \
  --report eval_outputs/force_gap_encoder_v2/motion_encoder.report.json \
  --device cpu \
  2>&1 | tee -a eval_outputs/force_gap_encoder_v2/motion_encoder.log

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path("eval_outputs/force_gap_encoder_v2/motion_encoder.report.json")
report = json.loads(path.read_text(encoding="utf-8"))
acceptance = report["acceptance"]
summary = {
    "qualified": report["qualified"],
    "best_baseline": acceptance["best_baseline"],
    "selected_alignment_weight": acceptance["selected_alignment_weight"],
    "relative_error_improvement": acceptance["mean_relative_improvement"],
    "direction_improvements": acceptance["direction_relative_improvements"],
    "statistical_checks": acceptance["checks"],
    "software_checks_passed": report["software_checks"]["passed"],
    "background_enabled": report["final_training"]["background"]["enabled"],
    "report": str(path),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
