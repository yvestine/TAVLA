# Control-Supervised Domain-Invariant Force+Pose Encoder

## Setup

- History window: `10` frames; force calibration: first `20` frames.
- Input domains: `['real', 'sim']`; evaluation domain: `real` (sim->real generalization).
- Adds contact/axial-resist/lateral-direction auxiliary heads supervised on BOTH domains with force-derived control targets.
- Quality metrics on held-out REAL windows; latent gap on held-out Real-vs-Sim windows (out-of-sample).
- No TAVLA/pi0 checkpoint or policy code was modified.

## Results

| variant | Real Macro-F1 | Real aligned F1 | Real AUROC | latent proxy-A-dist |
|---|---|---|---|---|
| force_pose_control_no_adv | 0.757 | 0.642 | 0.882 | 0.532 |
| force_pose_control_domain_invariant | 0.728 | 0.606 | 0.867 | 0.592 |

## Interpretation

- The decisive metric for this encoder is the sim->real transfer probe, not these numbers: run
  `run_sim_to_real_force_probe.py --latent-run-dir <this output dir>` and compare `latent_control` against `direction_only` / `force_feats_10d`.
- The auxiliary control heads exist only to shape the latent; at deployment the frozen `encode` output is the artifact.

This encoder is standalone; policy integration is a separate, additive step.
