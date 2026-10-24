#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/packages/openpi-client/src:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export NUMPY_MADVISE_HUGEPAGE=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Select physical nvidia-smi index 2 by UUID, independent of CUDA enumeration.
GPU_INDEX="${GPU_INDEX:-2}"
GPU_UUID="$(nvidia-smi --id="$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
exec .venv/bin/python scripts/serve_force_predictive_v3.py "$@"
