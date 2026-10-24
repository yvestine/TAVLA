# TAVLA documentation index

Use this page to find the current run instructions and the related design notes.
The repository contains historical experiment reports at its root; they remain
there so existing relative links and uncommitted work keep working.

## Start here

- [A100 deployment and further training](a100_deployment_training.md): GitHub checkout, offline model transfer, 8001/8005 serving, and multi-GPU fine-tuning or resume.
- [Run guide](../run.md): environment, datasets, conversion, and policy training.
- [Project overview](../README.md): installation and repository structure.
- [Real force trend input](tavla_real_force_trend_v1.md): current force-input path.
- [Franka wrench deployment](franka_ee_wrench_deployment.md): robot-side inference setup.

## Insertion quality and force research

- [Label review audit](../insertion_quality_label_audit_v2.md)
- [Force dynamics audit](../insertion_force_dynamics.md)
- [Sim-to-real force probe](../sim_to_real_force_probe_report.md)
- [Control-supervised encoder](../insertion_quality_encoder_control_report.md)
- [Domain-invariant encoder](../insertion_quality_encoder_domain_invariant_report.md)
- [Sim-aligned force encoder](../insertion_quality_encoder_sim_aligned_force_report.md)
- [Sim-aligned force and pose encoder](../insertion_quality_encoder_sim_aligned_force_pose_report.md)
- [Encoder reports: V1, V2, and V3](../insertion_quality_encoder_v1_report.md),
  [V2](../insertion_quality_encoder_v2_report.md), and
  [V3 transformer](../insertion_quality_encoder_v3_transformer_report.md)
- [Sim-aligned force experiment notes](insertion_quality_sim_aligned_force.md)
- [Representative trajectory audit](../representative_trajectory_audit.md)
- [Visualization index](../insertion_quality_v1_visualization_index.md)

## Existing operations and design notes

- [Sim fine-tuning runbook](tavla_sim_finetune_runbook.md)
- [EE wrench deployment](franka_ee_wrench_deployment.md)
- [Migration](migration.md)
- [Docker](docker.md)
- [Remote inference](remote_inference.md)
- [Normalization statistics](norm_stats.md)
- [Wrench alignment](wrench_polyfit_alignment.md)

## New force-trend encoder work

- [Insertion alignment encoder](insertion_alignment_encoder.md): force-only aligned/misaligned classifier and TAVLA latent interface.
- Implementation: `scripts/insertion_alignment_encoder/`.
- Single replaceable model artifact: `eval_outputs/insertion_alignment_encoder/encoder.pt`.
