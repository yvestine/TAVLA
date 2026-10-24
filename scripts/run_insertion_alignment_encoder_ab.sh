#!/usr/bin/env bash
set -Eeuo pipefail

# One visible GPU, sequential A then B; never overwrite an existing experiment.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_NAME="${RUN_NAME:-legacy_terminal_ab_v2_$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/eval_outputs/insertion_alignment_encoder_ab/$RUN_NAME}"
SEEDS="${SEEDS:-42 43 44}"
read -r -a TRAINING_SEEDS <<< "$SEEDS"
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: choose a new RUN_NAME or OUTPUT_DIR; directory exists: $OUTPUT_DIR" >&2
  exit 1
fi
if [[ ${#TRAINING_SEEDS[@]} -eq 0 ]]; then
  echo "ERROR: SEEDS must contain at least one integer" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

echo "Encoder A/B: physical GPU=$GPU_ID, seeds=${TRAINING_SEEDS[*]}, paired batch=${BATCH_SIZE:-128}"
echo "A: legacy recipe. B: qualified high/medium terminal geometry and own constrained CV epoch. CV epochs=${EPOCHS:-60}."
echo "Candidates: $OUTPUT_DIR/seed_<seed>/{A,B}/encoder.pt"
echo "Comparison: $OUTPUT_DIR/comparison.md and comparison.json"
"$PYTHON_BIN" -m insertion_alignment_encoder.train_ab \
  --real-root "${REAL_DIR:-$PROJECT_DIR/data}" \
  --sim-root "${SIM_DIR:-$PROJECT_DIR/sim-data-aligned}" \
  --labels "${LABELS:-$PROJECT_DIR/annotations/insertion_quality_labels_v2.csv}" \
  --reference-encoder "${REFERENCE_ENCODER:-$PROJECT_DIR/eval_outputs/insertion_alignment_encoder/encoder.pt}" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --seeds "${TRAINING_SEEDS[@]}" \
  --epochs "${EPOCHS:-60}" \
  --patience "${PATIENCE:-10}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --terminal-batch-size "${TERMINAL_BATCH_SIZE:-32}" \
  --terminal-loss-weight "${TERMINAL_LOSS_WEIGHT:-0.03}" \
  --reference-stability-mm "${REFERENCE_STABILITY_MM:-3.0}" \
  --selection-class-quantile "${SELECTION_CLASS_QUANTILE:-0.10}" \
  --selection-bootstrap-samples "${SELECTION_BOOTSTRAP_SAMPLES:-1000}" \
  "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
