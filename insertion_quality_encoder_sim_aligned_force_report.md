# Aligned Sim Force Encoder: Trajectory-level Cross-validation

## Setup

- History window: `10` frames.
- Force input: `policy_force_trend_v1` derived from Fx/Fy/Fz; torque excluded.
- Input domains: `['sim']`; supervised domains: `['sim']`; evaluation domain: `sim`.
- Sim source: `sim-data-aligned/obs/state/ee_wrench_base`.
- History must remain within one reviewed phase: `True`.
- Labels: reviewed `misaligned_contact` and `aligned_progress`; uncertain/free/terminal endpoints are masked for quality classification.
- No TAVLA policy code was modified.

## Results

### sim_force_transformer

- Pooled Macro-F1: `0.696`
- Misaligned F1: `0.845`
- Aligned F1: `0.547`
- Accuracy: `0.770`
- AUROC: `0.779`
- Fold Macro-F1 mean: `0.662`
- Domain-head accuracy: n/a

## Interpretation

This is held-out Sim classification using phase labels transferred by paired trajectory/frame from the reviewed Real/older-Sim set. It does not validate the new Sim contact state independently.
The encoder is standalone; policy integration and Real-domain performance require separate evaluation.
