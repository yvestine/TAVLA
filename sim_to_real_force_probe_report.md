# Sim->Real Force Transfer Probe (control-relevant targets)

## Setup

- Probe trained on **Sim windows only**, evaluated on **held-out Real** (leave-one-trajectory-out); no Real labels used for fitting.
- Targets are deterministic control quantities from each domain's own force, not transferred human phase labels:
  - `contact_active` (binary): tool in contact vs free, above the free-space noise gate.
  - `axial_resist` (binary, contact frames only): axial force opposes insertion (jam/backdrive).
  - `lateral_dir` (unit 2-vector): corrective lateral push direction; scored by cosine to truth.
- Representations: `raw_force` (magnitude-bearing), `direction_only` (8 scale-free channels = the Direction-Matters representation), `force_feats_10d` (direction+magnitude), and the frozen encoder latents `latent_baseline` / `latent_dinv`.
- No TAVLA/pi0 checkpoint or policy code was read or modified.

## Results (higher = better transfer)

| representation | contact F1 | axial-resist F1 | axial AUROC | lateral cos | lateral <90° |
|---|---|---|---|---|---|
| raw_force | 0.623 | 0.781 | 0.905 | 0.816 | 0.974 |
| direction_only | 0.593 | 0.931 | 0.992 | 0.947 | 0.991 |
| force_feats_10d | 0.322 | 0.899 | 0.973 | 0.952 | 0.991 |
| latent_baseline | 0.380 | 0.453 | 0.730 | 0.234 | 0.670 |
| latent_dinv | 0.518 | 0.606 | 0.845 | 0.113 | 0.571 |
| latent_control_no_adv | 0.999 | 0.977 | 0.998 | 0.979 | 0.998 |
| latent_control | 0.998 | 0.969 | 0.996 | 0.975 | 0.999 |

## How to read this

- If `raw_force` transfers poorly but a representation transfers well, that representation carries **control-relevant force information that survives the sim->real gap** -- the precondition for a residual policy learned in sim to run on the real robot.
- `direction_only` vs `force_feats_10d` / `latent_dinv` is the Direction-Matters ablation: if keeping magnitude (and aligning it) beats dropping it, the encoder does more than re-derive 'direction transfers, magnitude does not'.
- `latent_dinv` vs `latent_baseline` isolates the effect of the domain-adversary + paired-InfoNCE alignment on downstream transfer, beyond the raw latent-gap number.

This probe is offline and standalone; it does not by itself integrate the encoder into TAVLA.
