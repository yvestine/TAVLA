#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Compatibility alias only. Conversion, stats and training use the existing launcher.
export ENCODER_KIND=motion_only_v2
exec bash "$PROJECT_DIR/scripts/run_tavla_force_align_latent_cotrain.sh" "$@"
