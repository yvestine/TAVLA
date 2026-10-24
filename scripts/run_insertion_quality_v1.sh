#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_ID="${GPU_ID:-3}"

cd "$PROJECT_DIR"

echo "[1/2] Auditing human/motion weak labels"
"$PYTHON_BIN" scripts/audit_insertion_quality_labels_v1.py

echo "[2/2] Running 10-fold trajectory-level evaluation on physical GPU $GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_insertion_quality_v1.py --device cuda:0 "$@"
