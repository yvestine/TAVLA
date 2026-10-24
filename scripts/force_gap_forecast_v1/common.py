from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import FORMAT, LAYOUT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_manifest(path, require_ready=False):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("artifact_format") != FORMAT or value.get("layout") != LAYOUT:
        raise ValueError("Not a force_gap_forecast_v1 manifest / feature layout")
    checkpoint = Path(value["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    if sha256(checkpoint) != value["encoder_sha256"]:
        raise ValueError("Encoder SHA mismatch")
    if require_ready and value.get("status") != "policy_train_ready":
        raise ValueError(f"Policy training requires actual-Sim-state acceptance; status={value.get('status')}. "
                         "See acceptance.json and data_audit.json; copied Real q/dq cannot be promoted.")
    if require_ready:
        for key in ("acceptance", "audit"):
            evidence = path.parent / value[key]
            if sha256(evidence) != value[f"{key}_sha256"]:
                raise ValueError(f"Changed {key} evidence")
        gate = json.loads((path.parent / value["acceptance"]).read_text())
        audit = json.loads((path.parent / value["audit"]).read_text())
        if not gate.get("passed") or not audit.get("actual_sim_state_verified"):
            raise ValueError("Readiness requires accepted waveform metrics and actual Sim states")
    return value, checkpoint


def finite_array(value, shape, name, dtype=np.float32):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite {shape}, got {result.shape}")
    return result


def aligned_force_from_calibrated_link7(wrench_base_n):
    value = np.asarray(wrench_base_n, np.float32)
    if value.shape not in ((3,), (6,)) or not np.isfinite(value).all():
        raise ValueError("Expected calibrated robot-base wrench (3,) or (6,)")
    return -value[:3].copy()
