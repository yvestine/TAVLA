# Domain-Invariant Force+Pose Encoder: Real+Sim Cross-validation with Latent Gap

## Setup

- History window: `10` frames; force calibration: first `20` frames.
- Input domains: `['real', 'sim']`; supervised: `['real', 'sim']`; evaluation domain: `real` (sim->real generalization).
- Sim source: `sim-data-aligned/obs/state/ee_wrench_base`; Real force: `obs/state/ee_force_base`.
- Quality metrics on held-out REAL windows; latent gap on held-out Real-vs-Sim windows (out-of-sample).
- No TAVLA/pi0 checkpoint or policy code was modified.

## Results

| variant | Real Macro-F1 | Real aligned F1 | Real AUROC | latent proxy-A-dist | latent std-mean-diff |
|---|---|---|---|---|---|
| force_pose_paired_baseline | 0.750 | 0.628 | 0.801 | 0.209 | 0.045 |
| force_pose_domain_invariant | 0.761 | 0.649 | 0.861 | 0.071 | 0.025 |

## Interpretation

- Lower `latent proxy-A-distance` = Real and Sim latents are less separable = smaller sim-real force gap.
- The baseline variant (domain_lambda=0, infonce=0) is the reference; the domain-invariant variant should lower the latent gap while keeping Real Macro-F1 comparable.
- A gap that collapses to ~0 with a large Macro-F1 drop indicates over-alignment (collapse); the goal is 'close but not collapsed' (surveyed co-training analysis).
- The frozen encoder latent is the artifact to feed TAVLA as an auxiliary input and to expose as a residual-RL observation in sim.

This encoder is standalone; policy integration is a separate, additive step.
