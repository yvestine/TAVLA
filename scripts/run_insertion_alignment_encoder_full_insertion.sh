#!/usr/bin/env bash
set -Eeuo pipefail

# Train only the force-only encoder. Each invocation uses a new output folder.
# Usage: GPU_ID=0 bash scripts/run_insertion_alignment_encoder_full_insertion.sh
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_NAME="${RUN_NAME:-full_insertion_v4_$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/eval_outputs/insertion_alignment_encoder_full_insertion/$RUN_NAME}"
REAL_DIR="${REAL_DIR:-$PROJECT_DIR/data}"
SIM_DIR="${SIM_DIR:-$PROJECT_DIR/sim-data-aligned}"
LABELS="${LABELS:-$PROJECT_DIR/annotations/insertion_quality_labels_v2.csv}"
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: output folder already exists; choose a new RUN_NAME or OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

echo "Encoder training: full_insertion_v4, physical GPU=$GPU_ID, history=32, paired Real/Sim"
echo "Output: $OUTPUT_DIR/encoder.pt"
echo "Report: $OUTPUT_DIR/encoder.report.json"
"$PYTHON_BIN" -m insertion_alignment_encoder.train_terminal_geometry \
  --real-root "$REAL_DIR" \
  --sim-root "$SIM_DIR" \
  --labels "$LABELS" \
  --output "$OUTPUT_DIR/encoder.pt" \
  --device cuda:0 \
  --history-length 32 \
  --baseline-frames 20 \
  --batch-size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-100}" \
  --min-selection-epoch "${MIN_EPOCHS:-1}" \
  --min-macro-f1 "${MIN_MACRO_F1:-0.70}" \
  --min-class-recall "${MIN_CLASS_RECALL:-0.60}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --min-learning-rate "${MIN_LEARNING_RATE:-0.00003}" \
  --warmup-epochs "${WARMUP_EPOCHS:-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --dropout "${DROPOUT:-0.20}" \
  --label-smoothing "${LABEL_SMOOTHING:-0.05}" \
  --geometry-loss-weight "${GEOMETRY_LOSS_WEIGHT:-0.12}" \
  --consistency-loss-weight "${CONSISTENCY_LOSS_WEIGHT:-0.15}" \
  --seed "${SEED:-42}" \
  --selection-depth-scale-mm 3.0 \
  "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
