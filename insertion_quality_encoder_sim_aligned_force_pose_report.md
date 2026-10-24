# Aligned Sim Force+Pose Ablation: Trajectory-level Cross-validation

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

### sim_force_pose_transformer

- Pooled Macro-F1: `0.767`
- Misaligned F1: `0.875`
- Aligned F1: `0.658`
- Accuracy: `0.817`
- AUROC: `0.834`
- Fold Macro-F1 mean: `0.742`
- Domain-head accuracy: n/a

### sim_pose_only_transformer

- Pooled Macro-F1: `0.720`
- Misaligned F1: `0.844`
- Aligned F1: `0.597`
- Accuracy: `0.775`
- AUROC: `0.829`
- Fold Macro-F1 mean: `0.664`
- Domain-head accuracy: n/a

## Interpretation

This is held-out Sim classification using phase labels transferred by paired trajectory/frame from the reviewed Real/older-Sim set. It does not validate the new Sim contact state independently.
The encoder is standalone; policy integration and Real-domain performance require separate evaluation.
