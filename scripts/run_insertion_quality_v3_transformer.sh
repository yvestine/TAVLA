#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

GPU_ID="${GPU_ID:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/insertion_quality_v3_transformer}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train_insertion_quality_v3_transformer.py \
  --config configs/insertion_quality_v3_transformer.json \
  --annotations annotations/insertion_quality_labels_v2.csv \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  "$@"
