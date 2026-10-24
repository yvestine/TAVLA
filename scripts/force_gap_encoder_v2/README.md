# Force-gap encoder v2 and matched TAVLA A/B

## 当前 encoder 实验：`motion_v1`

运动背景补偿方案、离线验收和新的四参数在线接口见
[MOTION_V1.md](MOTION_V1.md)。在外部 GPU 主机、已有的 `train` tmux shell 中运行：

```bash
cd /data/workspace/gujiawei/TAVLA && GPU_ID=7 bash scripts/run_force_gap_motion_v1.sh
```

该入口仅运行 encoder 实验，复用 `eval_outputs/force_gap_encoder_v2/`，覆盖同名的
`motion_encoder.pt`、`motion_encoder.report.json`、`motion_encoder.log`。
下面的 12 维特征、单参数力接口、静止校准和策略 A/B 说明对应旧 `legacy_v2` 配方。

This directory contains a frozen, force-only encoder intended to reduce the
Real/Sim force gap while keeping insertion phase useful to the policy. It does
not change the TAVLA server or policy architecture: `observation.effort` remains
32-D and TAVLA still receives the last 10 latent frames.

## Input contract

The encoder expects one force vector per frame:

| Item | Required value |
| --- | --- |
| Array | `float32 [N,3]`, ordered `(Fx,Fy,Fz)` |
| Frame | Robot base frame, matching `obs/state/ee_force_base` in the training HDF5 |
| Unit | Newtons |
| Rate | 10 Hz (100 ms between samples) |
| Calibration | First 20 samples collected before policy actions begin |
| Timestamp | Optional for batch inference; strictly increasing for streaming inference |

For a simulator sensor that reports force in sensor/link coordinates, rotate it
into the robot base frame before calling the encoder:

```text
force_base = rotation_base_from_sensor @ force_sensor
```

Resolve the sensor sign convention before conversion: the training field must
mean the same physical direction as the simulator field. Do not feed raw link-7
force if that field has not been rotated and sign-checked. Do not apply a
hand-tuned Real/Sim force multiplier; scale robustness is learned with training
augmentation. The first 20 samples should be stationary/pre-contact samples,
because they define the episode's force baseline.

## Preprocessing and output

For each episode, the encoder:

1. Subtracts the coordinate-wise median of the first 20 force vectors.
2. Applies a causal 3-frame mean to the residual.
3. Computes a 5-frame force trend and robust activity gates from the calibration
   interval (6 MAD, with a `1e-6 N` floor).
4. Builds 12 features in the exact order below.
5. Applies the train-only median/IQR normalizer and clips each channel to
   `[-10,10]`.
6. Encodes a causal 32-frame window as a 32-D `tanh` latent.

Feature order:

```text
force_dir_x, force_dir_y, force_dir_z,
trend_dir_x, trend_dir_y, trend_dir_z,
axial_signed_fraction, lateral_ratio,
log1p_force_norm, log1p_trend_norm,
force_valid, trend_valid
```

The insertion axis defaults to base-frame `(0,0,-1)`. Verify that it matches
the task's insertion direction before training or deployment. Direction and
ratio channels preserve force direction; the two log-magnitude channels retain
force intensity without requiring a separate simulator-specific gain. Invalid
low-force directions are set to zero and identified by their validity flags.

Offline conversion produces `observation.effort [N,32]`. For online use:

```python
from force_gap_encoder_v2.deploy_infer import ForceGapLatentEncoder, StreamingForceGapEncoder

encoder = ForceGapLatentEncoder("eval_outputs/force_gap_encoder_v2/encoder.pt", device="cuda:0")
stream = StreamingForceGapEncoder(encoder, output_history=10)

latent = stream.push(force_base_newtons, timestamp_seconds)  # None for first 19 samples
if latent is not None:
    effort_history = stream.effort_history()  # chronological [10,32], ready for TAVLA
```

The first latent is returned after calibration sample 20. Hold the robot safely
at its initial pose while collecting those samples. Each later call returns the
current 32-D latent. `effort_history()` edge-pads its earliest values until 10
outputs exist. The policy's 10-frame history corresponds to approximately one
second at 10 Hz.

The same frozen checkpoint and preprocessor must be used for Real and Sim. Do
not normalize each domain independently at deployment. The artifact stores the
feature order, training normalizer, insertion axis, calibration length, gates,
and model weights.

## Encoder training and conversion

Before another encoder fit, audit the calibration assumption with the current
checkpoint's preprocessing parameters:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=scripts:src .venv/bin/python \
  -m force_gap_encoder_v2.audit_calibration \
  --output eval_outputs/force_gap_encoder_v2/calibration_audit.json
```

This CPU audit compares the first calibration interval with the earliest
low-motion candidate in the leading reviewed free-space interval. The default
1 mm and 0.005 rad limits are diagnostic heuristics. It records stage-specific
validity rates, force distributions and timestamp spans. The phase statistics
include all reviewed post-calibration frames, including frames beyond the
training first-hit cutoff. Sim labels/poses/timestamps may be transferred from
Real, so their agreement is not independent physical verification. The audit
does not modify preprocessing, annotations or checkpoints, and alternative
calibration is never applied to frames before that calibration finishes.
Repeated runs overwrite this report in the existing result directory.

The training entry point below still implements the simplified phase/cosine
experiment. Feature reconstruction, conditional mean/covariance alignment and
bidirectional action prediction with a state-only baseline must be added before
it can implement the original force-gap evaluation plan.

The training script uses the existing phase labels and terminal-reference
first-hit cutoffs from
`eval_outputs/insertion_alignment_encoder/encoder.pt`. It skips `low`
confidence intervals, uses high-confidence labels at weight 1.0 and other
reviewed labels at 0.65, ignores `free`/`uncertain` phases, and keeps trajectory groups intact. The eight
previously held-out trajectories are not used to select the alignment strength
or fit the final encoder. Each candidate is evaluated by Real and Sim grouped
out-of-fold macro-F1, raw-feature paired cosine, and paired-latent cosine.
Compare raw-feature cosine with latent cosine in the saved report to check
whether the encoder actually narrows the paired representation gap. Selection first
protects the lower of the two domain F1 scores; within 0.02 of the best score it
selects the candidate with the highest paired cosine. Domain-probe accuracy is
not used as a standalone success criterion.

From the TAVLA repository root, the full encoder fit is:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:src .venv/bin/python \
  -m force_gap_encoder_v2.train \
  --real-root data \
  --sim-root sim-data-aligned \
  --force-key obs/state/ee_force_base \
  --cutoff-checkpoint eval_outputs/insertion_alignment_encoder/encoder.pt \
  --output eval_outputs/force_gap_encoder_v2/encoder.pt \
  --device cuda:0
```

Convert the Real and aligned-Sim HDF5 files with that same artifact:

```bash
export PYTHONPATH=scripts:src
export LEROBOT_HOME="$PWD/../.cache/huggingface/lerobot"

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m force_gap_encoder_v2.convert_to_lerobot \
  --raw-dir data --repo-id local/tavla_single_arm_force_gap_v2 \
  --encoder-path eval_outputs/force_gap_encoder_v2/encoder.pt \
  --force-key obs/state/ee_force_base --overwrite --no-videos

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m force_gap_encoder_v2.convert_to_lerobot \
  --raw-dir sim-data-aligned --repo-id local/tavla_single_arm_force_gap_v2_sim \
  --encoder-path eval_outputs/force_gap_encoder_v2/encoder.pt \
  --force-key obs/state/ee_force_base --overwrite --no-videos
```

## Matched policy A/B

Two configs are registered in `src/openpi/training/config.py`:

| Run | Encoder/data | Sampler | Purpose |
| --- | --- | --- | --- |
| A `pi0_lora_user_single_arm_force_gap_sampling_a` | Existing `force_align_latent` Real and Sim datasets | 50:50, without replacement | Isolate the sampling change |
| B `pi0_lora_user_single_arm_force_gap_encoder_b` | New v2 Real and Sim datasets | 50:50, without replacement | Isolate the encoder change against A |

Both use the same Pi0 LoRA config, 32-D effort, 10-frame effort history,
30,000 steps, batch 8, and base checkpoint. Existing source datasets must be
present for A. The source sampler draws each source frame at most once per
loader pass; equal-size datasets therefore have equal representation and full
frame coverage in a pass. The training loop can make multiple passes over the
same dataset during 30,000 steps.

The prepared script computes separate norm stats, then trains A followed by B
in the already-running `tavla` tmux session. It does not create a tmux session:

```bash
cd /data/workspace/gujiawei/TAVLA
GPU_IDS=0,1,2,3 BATCH_SIZE=8 TRAIN_STEPS=30000 \
  bash scripts/run_force_gap_ab.sh
```

Set `LEROBOT_HOME`, `OPENPI_DATA_HOME`, `PYTHON_BIN`, `GPU_IDS`, or
`ENCODER_GPU` in the shell if your training host uses different paths/devices.
The script uses GPUs 0–3 for policy training and GPU 0 for the encoder stage;
change those values if any of them are occupied. It serializes A and B so they
do not compete for GPU memory. This command is provided for you to run; it has
not been started from this session.

## Comparing policy results in simulation

Run A and B with the same initial states, randomized episode order, force
protection limit, and action-speed cap. Preserve the existing cap at or below
the demonstrated safe speed; do not increase it during the encoder comparison.
Record per episode: success, peg/hole XY error per frame, maximum force, force
limit status, and the exact policy checkpoint plus encoder artifact hash. Use
the same evaluation set and controller parameters for both.

The acceptance gate is at least 36 successful rollouts out of 40, pooled XY
tracking error P95 below 3 mm, and no force-protection violations. The simulator
project is not mounted in this TAVLA repository, so the rollout controller must
write the report contract expected by `check_sim_eval.py`; this repository
cannot rotate the external simulator's sensor data or execute its rollouts.
Also check that the held-out encoder report does not show a large Real/Sim
phase-F1 drop. A higher latent cosine alone is not sufficient if either phase
classification or closed-loop performance regresses.

Report JSON format:

```json
{
  "schema": "tavla_force_gap_rollout_v1",
  "model": "A or B",
  "episodes": [
    {
      "trajectory_id": 0,
      "success": true,
      "xy_error_mm": [0.8, 1.1, 1.4],
      "force_safety_ok": true
    }
  ]
}
```

Check an externally produced report with:

```bash
PYTHONPATH=scripts:src .venv/bin/python -m force_gap_encoder_v2.check_sim_eval \
  /path/to/rollout_report.json
```
