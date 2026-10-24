#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_ID="${GPU_ID:-7}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/eval_outputs/insertion_quality_sim_aligned_force}"

cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_insertion_quality_v3_transformer.py \
  --config configs/insertion_quality_sim_aligned_force.json \
  --annotations annotations/insertion_quality_labels_v2.csv \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  "$@"
