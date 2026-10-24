#!/usr/bin/env python3
"""Control-relevant force targets for the sim->real transfer probe.

These targets are deterministic functions of a *single domain's* force signal,
computed with an identical functional form for Real and Sim.  They are NOT human
labels.  Each one is a quantity a contact/insertion controller actually needs:

  * ``contact_active``  -- is the tool in contact at all (force above the
    per-trajectory free-space noise gate)?
  * ``lateral_dir``     -- which way is the lateral contact force pushing, in the
    plane perpendicular to the insertion axis (the corrective direction a
    residual policy would nudge along)?
  * ``axial_resist``    -- is the axial force opposing insertion (jam/backdrive)
    rather than assisting it?

The transfer probe trains a small predictor of these targets on Sim
representations and evaluates it on held-out Real.  A representation that has
aligned the two domains in a control-relevant way lets a Sim-trained probe work
on Real; raw force (whose magnitude differs ~5x) should not transfer.  This is
the offline precondition for a residual policy learned in sim to run on the real
robot, and it does not use, touch, or modify any TAVLA/pi0 checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openpi.force_features import causal_mean
from openpi.force_features import _robust_norm_threshold


def plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane perpendicular to ``axis``.

    The basis is deterministic given the axis, so the 2-D lateral direction is
    expressed in the same frame for every trajectory and both domains.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ axis) * axis
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(axis, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return e1.astype(np.float64), e2.astype(np.float64)


TARGET_NAMES = ("contact_active", "lateral_dir", "axial_resist")


@dataclass
class ControlTargets:
    """Per-frame control-relevant targets derived from one domain's force.

    ``contact_active`` and ``axial_resist`` are binary (values in {0, 1}, or -1
    where undefined, e.g. no contact for the axial-sign target).  ``lateral_dir``
    is a unit 2-vector in the perpendicular plane, zeroed where there is no
    resolvable lateral force.  ``lateral_valid`` marks frames where the lateral
    direction is defined.
    """

    contact_active: np.ndarray
    axial_resist: np.ndarray
    lateral_dir: np.ndarray
    lateral_valid: np.ndarray


def build_control_targets(
    force_xyz: np.ndarray,
    *,
    free_end: int,
    insertion_axis: np.ndarray,
    smoothing_frames: int = 3,
    active_noise_mad_multiplier: float = 6.0,
    active_noise_floor: float = 1e-6,
    lateral_floor_ratio: float = 0.25,
) -> ControlTargets:
    """Compute control targets from a single domain's raw force.

    The construction mirrors ``build_real_force_trend_features``: subtract the
    per-trajectory free-space baseline, causal-smooth, and gate on a robust
    free-space noise threshold.  Every step is identical for Real and Sim, so the
    targets share one definition across domains -- what differs is only the force
    that goes in.  ``free_end`` estimates the free-space baseline/gate and is not
    a label.
    """
    force_xyz = np.asarray(force_xyz, dtype=np.float64)
    if force_xyz.ndim != 2 or force_xyz.shape[1] != 3:
        raise ValueError(f"Expected force_xyz [time, 3], got {force_xyz.shape}")
    if not 0 <= free_end < len(force_xyz):
        raise ValueError(f"free_end={free_end} outside trajectory length {len(force_xyz)}")
    axis = np.asarray(insertion_axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    e1, e2 = plane_basis(axis)

    baseline = np.median(force_xyz[: free_end + 1], axis=0)
    residual = force_xyz - baseline
    smoothed = causal_mean(residual, smoothing_frames).astype(np.float64)
    norm = np.linalg.norm(smoothed, axis=1)
    _, _, gate = _robust_norm_threshold(norm[: free_end + 1], active_noise_mad_multiplier, active_noise_floor)

    contact = (norm > gate).astype(np.int64)

    axial_signed = smoothed @ axis
    lateral_vec = smoothed - axial_signed[:, None] * axis[None, :]
    lateral_norm = np.linalg.norm(lateral_vec, axis=1)

    # Axial resistance: does the axial force oppose the insertion direction?
    # Insertion moves along +axis, so an opposing (resisting) contact force
    # points along -axis, i.e. axial_signed < 0.  Defined only under contact.
    axial_resist = np.full(len(smoothed), -1, dtype=np.int64)
    axial_resist[contact == 1] = (axial_signed[contact == 1] < 0.0).astype(np.int64)

    # Lateral corrective direction: unit 2-vector in the (e1, e2) plane, defined
    # only when the lateral force is a clear fraction of the total contact force.
    lateral_valid = (contact == 1) & (lateral_norm > lateral_floor_ratio * np.maximum(norm, 1e-12)) & (lateral_norm > gate)
    lateral_dir = np.zeros((len(smoothed), 2), dtype=np.float64)
    comp1 = lateral_vec @ e1
    comp2 = lateral_vec @ e2
    planar = np.stack([comp1, comp2], axis=1)
    planar_norm = np.linalg.norm(planar, axis=1)
    safe = lateral_valid & (planar_norm > 1e-12)
    lateral_dir[safe] = planar[safe] / planar_norm[safe, None]

    return ControlTargets(
        contact_active=contact,
        axial_resist=axial_resist,
        lateral_dir=lateral_dir.astype(np.float32),
        lateral_valid=lateral_valid.astype(bool),
    )
