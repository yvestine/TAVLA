#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 2 || ( "$1" != "8001" && "$1" != "8005" ) || -z "$2" ]]; then
  echo "Usage: bash scripts/serve_transferred_policy.sh <8001|8005> <GPU index or UUID>" >&2
  exit 2
fi

PORT="$1"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment missing: run uv sync --frozen --python 3.11 in $PROJECT_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$2"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME="$PROJECT_DIR/.cache/openpi"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$PORT" == "8001" ]]; then
  exec "$PYTHON_BIN" scripts/serve_force_align_latent_cotrain.py \
    --tavla-root "$PROJECT_DIR" --port 8001
else
  exec "$PYTHON_BIN" scripts/serve_policy.py \
    --port 8005 --default-prompt peg-in-hole \
    policy:checkpoint \
    --policy.config pi0_lora_user_single_arm_ee_wrench_cotrain_aligned_50_50 \
    --policy.dir "$PROJECT_DIR/checkpoints/pi0_lora_user_single_arm_ee_wrench_cotrain_aligned_50_50/raw6_aligned_cotrain_50_50_gpu1245_30k_20261009/29999"
fi
