# Host9 encoder and four task policies

The pipeline is `scripts/run_host9_training_pipeline.py`; its fixed plan is
`configs/host9_training_plan_20261026.json`. It finishes the Sim force encoder,
creates the policy datasets and normalization statistics, and launches four
independent tmux training sessions as each task's datasets and statistics become
ready. Each policy uses one A100, batch 8,
30,000 steps, the same Pi0 base weights and LoRA variants, and Real/Sim 50:50.

| GPU | Task | Force representation | tmux |
|---|---|---|---|
| 0 | Dual peg insertion | Original six dimensional wrench | host9_dual_raw6_gpu0 |
| 4 | USB insertion | Original six dimensional wrench | host9_usb_raw6_gpu4 |
| 5 | Dual peg insertion | Frozen encoder, history [10,32] | host9_dual_encoder_gpu5 |
| 6 | USB insertion | Frozen encoder, history [10,32] | host9_usb_encoder_gpu6 |

The previous 8001/8005 servers are stopped. Other users' GPU processes are
left running. A policy waits until its assigned GPU has at least 50,000 MiB
free. GPU UUIDs determine device assignment. The single GPU batch-8 capacity
probes use the full production architecture, all three 224x224 image inputs,
action horizon 50, and the actual pretrained weights; their results are in
`logs/tavla_batch8_{raw6,latent}_probe.json`.

## Encoder

The completed Real teacher used all 150 original Real trajectories, with a
fixed 120/30 split. Its selected epoch is 35, SHA256
`6bb21347d6585a887d510c08c60cbd2d61bbb4d7935c8dd71eac80d519fe8b01`.
The original encoder training modules are preserved; the new student entry
checks their original hashes and every Real HDF5 identity before continuing.

There are 145 available Sim trajectories: peg 40, dual peg 36, plate 30, USB 39.
Their native inputs contain 38,482 frames. The inherited split has 116 training
and 29 validation trajectories. Dual peg 31–34 and USB 30 have no eligible Sim
pair and are explicitly excluded. Existing teacher weights and Real input and
target statistics are frozen. Only the Sim encoder and Sim reconstruction heads
are optimized, with trend alignment weight 0.1, batch 32, maximum 200 epochs,
and patience 20. Input statistics use only the inherited Sim training split.

The three new tasks are retimed rollouts, not one-recorded-frame-per-Real-frame
captures. The adapter verifies each saved original force/time array against its
Real source and uses `recorded_reference/source_frame_index` for supervision.
Sim inputs remain on their native 10Hz clock. Future targets are original Real
frame offsets +1 through +5 at the recorded source position. Targets are linearly
interpolated only within a continuous original recording segment. They never
cross Real gaps greater than 0.5 seconds. Force waveforms are not used for DTW
or correspondence estimation. This is reference-progress supervision and does
not establish identical physical contact events in the two domains.

The new Sim streams are native startup-tared incoming link7 wrench, in base
axes, with the recorded sign retained. The old peg export already contains the
documented `-1` conversion, so its XYZ sign is undone for the Sim encoder to
recover its calibrated incoming-link7 source. No 3.4 amplitude multiplier is
used. Metadata for the new tasks explicitly marks known-load sign calibration
as unverified; the learned Sim branch is domain specific. Real inference uses
the unchanged Real branch. Torque does not enter the force encoder.

The output is `eval_outputs/force_predictive_v3_multitask/host9_student_rollouts_v1`.
The completed run stopped at epoch 56 and selected epoch 36: validation
selection score 1.056973, future-force error 2.160338 N and reconstruction
NRMSE 0.178475. All three quality checks passed. The exported encoder SHA256 is
`06e5f3ee3ef69b3e6c9e2ef4bc0ebea249f6d3ef7da6da18b6171acfa2a6112e`.
The pipeline requires a valid final checkpoint hash, unchanged frozen Real
weights, improved validation selection score, and improved future-force error
against the initialized Sim branch before preparing policies. These checks
establish offline training progress; robot success still requires evaluation.

## Policy data

Only eligible rollout IDs from `training_manifest.json` are used in both domains:
36 dual peg Real/Sim trajectories and 39 USB Real/Sim trajectories. The original
Real USB trajectory 30 is retained in the source bundle but omitted from these
matched policy comparisons because its Sim rollout failed.

Real/Sim native frame counts are 7,719/9,447 for dual peg, and 8,982/14,132 for
USB. Both representations use identical images, episode boundaries, state,
actions, prompts and source files. Real uses its recorded `action/actual` labels;
Sim uses the applied absolute `action` with the pre-action observation and image.
Episodes split at recording gaps >0.5 seconds. Wrench remains six dimensional
for raw policies. Encoded data is regenerated from the final shared encoder,
using the Real branch on Real and the Sim branch on native incoming Sim XYZ.
Old `observations/force_latent` is not reused.

RGB images receive the model's exact 224x224 aspect-preserving resize before
being cached; the wrist camera is mirrored into the right-wrist input. Dataset
contracts record all input HDF5/video hashes, exact encoder SHA and segment
boundaries. Interrupted conversions are preserved before rebuilding. Each
dataset has an external file lock, allowing parallel converters and the pipeline
to safely reuse the same completed output. The `host9_prepare_parallel` tmux
session prepares the remaining datasets concurrently; its logs end in
`_parallel.log`.
Normalization statistics can also run concurrently. Host9 statistics use a lock
per dataset and are reused only when dataset provenance, model settings,
transformation code, batch settings and the saved statistics hash match.
Each epoch samples Real/Sim 50:50 without replacement, and redraws the larger source
at the next epoch so a fixed subset is not permanently omitted. This mode
requires `num_workers=0`.

## Monitoring and recovery

```bash
tmux attach -t tavla_host9_pipeline
tail -f logs/host9_encoder_student.log
cat eval_outputs/host9_training_pipeline/20261026_v1/status.json
```

Encoder checkpoints include optimizer and RNG state; rerunning the pipeline
resumes the last complete epoch. A completed stage is reused only when the
data, code, teacher and hyperparameters match. Existing policy tmux sessions
must be inspected before restarting orchestration. Policy checkpoints are saved
every 5,000 steps and at final step 29999. Policy resume restores optimizer
state; it does not promise an identical data-iterator position.
