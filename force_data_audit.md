# TAVLA 6D force data audit

- Audit date: 2026-09-17
- Scope: read-only inspection and plotting. No training logic, model, dataset, or normalization code was modified.
- Deterministic plot selection: `np.random.default_rng(20260917).choice(40, 3, replace=False)` -> `traj_25`, `traj_3`, `traj_18`.
- Important naming distinction: this audit treats `sim-data/` (visual replay H5) and `sim_force/` (contact-force H5 / current local Sim LeRobot source) as two separate Sim artifacts because their force arrays are different.

## Executive answer

1. Real force is read from `data/traj_*/data.h5 :: obs/state/ee_wrench_base`, shape `(T, 6)`, `float32`, named `[Fx,Fy,Fz,Tx,Ty,Tz]`. Repository documentation identifies the convention as robot-base-frame force/torque with units `[N,N,N,N·m,N·m,N·m]`; the HDF5 itself does not carry explicit unit attributes.
2. The raw `sim-data/traj_*/data.h5` files do **not** contain independent IsaacLab force for the selected training field. For all 40 paired trajectories, `sim-data/.../obs/state/ee_wrench_base` is byte-for-byte equal to `data/.../obs/state/ee_wrench_base`; state, action, and timestamps are also byte-for-byte equal. Its metadata explicitly says synthetic visual rollout and paired Real source.
3. A separate `sim_force/` set exists. Its `obs/state/ee_wrench_base` is byte-for-byte equal to `sim/force/contact_training_wrench`, which is `[sim/force/contact_total_base, zeros(3)]` for all 40 trajectories. Thus it has simulator contact-force values in Fx/Fy/Fz and structurally zero Tx/Ty/Tz; it is not a complete simulated 6D wrench. The H5 provenance attributes strongly identify this as a simulator contact-force proxy, but the IsaacLab generator is not present in this TAVLA repository, so the exact IsaacLab backend/version cannot be independently verified here.
4. The current local LeRobot dataset used by the hand-noise Sim/cotrain config, `local/tavla_single_arm_ee_wrench_sim_hand_noise_50`, is byte-for-byte equal to `sim_force/` for effort, state, and action on all 40 episodes. Its first camera frame also matches `sim_force` video, not Real video. Therefore the currently present local training dataset is contact-force Sim data, while the launcher’s raw fallback path points at `sim-data/`, which would import copied Real force if the local dataset were rebuilt.
5. Raw conversion performs no force filtering, baseline subtraction, clipping, rotation, or normalization. During TAVLA train/inference, `observation.effort` is normalized by stored per-dataset z-score stats before the model; with the current config it is `(B,1,6)` and `EffortType.EXPERT` projects the current six values into one effort token. So the model sees processed/normalized force, not raw force.

## 1. Read path: H5 -> dataset -> preprocessing -> model

| stage | location | function/config | force field and shape |
|---|---|---|---|
| Raw H5 | `data/traj_*/data.h5` | HDF5 dataset | `obs/state/ee_wrench_base`, `(T,6)`, `float32` |
| Raw Sim visual replay | `sim-data/traj_*/data.h5` | HDF5 dataset | same path and shape, but array copied from paired Real |
| Sim contact-force source | `sim_force/traj_*/data.h5` | HDF5 dataset | `sim/force/contact_total_base`, `(T,3)`; `sim/force/contact_training_wrench`, `(T,6)`; copied into `obs/state/ee_wrench_base` |
| H5 -> LeRobot | [`scripts/convert_user_hdf5_to_lerobot.py`](scripts/convert_user_hdf5_to_lerobot.py):142-168 | `_load_episode` | `effort = ep[effort_key][:].astype(np.float32)`; current command uses `obs/state/ee_wrench_base` |
| LeRobot feature | same file:101-129 | `_create_dataset` | `observation.effort`, `(6,)`, names `[Fx,Fy,Fz,Tx,Ty,Tz]` |
| Frame write | same file:201-214 | `convert` | writes `dataset.add_frame({"observation.effort": effort[i], ...})`; no force operation |
| Repack | [`src/openpi/training/config.py`](src/openpi/training/config.py):374-393 | `LeRobotTavlaDataConfig.__post_init__` | maps `observation.effort` -> `effort` when `effort_history` is nonempty |
| Temporal query | [`src/openpi/training/data_loader.py`](src/openpi/training/data_loader.py):251-275 | `create_dataset` | current `(0,)` history at dataset `fps=10`; returned effort shape `(history,6)` = `(1,6)` per sample |
| TAVLA input repack | [`src/openpi/policies/tavla_policy.py`](src/openpi/policies/tavla_policy.py):33-71 | `TavlaInputs.__call__` | passes `data["effort"]` through; state is padded to model `action_dim=32` |
| Normalize | [`src/openpi/training/data_loader.py`](src/openpi/training/data_loader.py):307-327 and [`src/openpi/transforms.py`](src/openpi/transforms.py):114-143 | `transform_dataset` / `Normalize` | `(x-mean)/(std+1e-6)`; current effort is normalized using stored stats |
| Inference Normalize | [`src/openpi/policies/policy_config.py`](src/openpi/policies/policy_config.py):58-78 | `create_trained_policy` | deployment path applies the same `Normalize` before model inference |
| Model observation | [`src/openpi/models/model.py`](src/openpi/models/model.py):88-95, 143-220 | `Observation` / `preprocess_observation` | non-`NO` effort passes through as `effort: (B,N,6)`; current `EffortType.EXPERT` keeps it separate from state |
| Effort token | [`src/openpi/models/pi0.py`](src/openpi/models/pi0.py):209-237 | `_process_effort_tokens` | current effort `obs.effort[:,-1]` `(B,6)` -> one projected suffix token `(B,1,width)` |
| Current config | [`src/openpi/training/config.py`](src/openpi/training/config.py):1055-1074 | `pi0_lora_user_single_arm_ee_wrench` | `effort_dim=6`, `effort_history=(0,)`, `EffortType.EXPERT` |
| Current cotrain config | [`src/openpi/training/config.py`](src/openpi/training/config.py):638-673 | `_make_tavla_hand_noise_cotrain_config` | repo order Real then `local/tavla_single_arm_ee_wrench_sim_hand_noise_50`; force dim 6 |

## 2. Field, order, length, sampling, preprocessing, quality

### Field and order

The converter defines `WRENCH_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")` and infers these names for a six-dimensional field. The current selected field is `obs/state/ee_wrench_base`; `obs/state/ee_wrench` is numerically identical to it in the checked raw H5 files. `sim_force` additionally stores the simulator contact fields described above.

All three raw directories have 40 episodes and 7,188 rows. Trajectory lengths range from 75 to 368 frames; median 170.5 and mean 179.7 frames. The stored metadata and LeRobot datasets declare 10 Hz. Raw H5 timestamp differences have median 0.100043 s (9.99571 Hz) across the Real set and are identical in both Sim sets. Timestamps are not perfectly uniform: the global minimum/maximum per-step differences are 0.008075/0.20025 s, and many episodes contain steps outside 0.08-0.12 s. The converter does not use H5 timestamps; it writes LeRobot timestamps at the declared 10 Hz.

### Raw Real statistics

|axis|n|mean|std|median|p05|p95|min|max|
|---|---|---|---|---|---|---|---|---|
|Fx|7188|-2.54811|1.65681|-2.52251|-5.52536|0.0639928|-7.66918|4.81951|
|Fy|7188|0.43718|1.2733|0.697489|-2.17731|2.11007|-3.9835|3.79861|
|Fz|7188|-1.01719|2.82249|-0.879913|-4.30358|2.39908|-29.8733|5.40804|
|Tx|7188|-1.12196|1.58937|-1.48467|-3.24933|2.18373|-9.10618|3.32992|
|Ty|7188|-1.25976|1.45396|-1.43827|-2.90424|0.576909|-4.13159|16.8903|
|Tz|7188|0.244777|0.194299|0.261771|-0.0795077|0.527597|-0.603723|0.920118|

### Raw Sim visual replay (`sim-data`) statistics

|axis|n|mean|std|median|p05|p95|min|max|
|---|---|---|---|---|---|---|---|---|
|Fx|7188|-2.54811|1.65681|-2.52251|-5.52536|0.0639928|-7.66918|4.81951|
|Fy|7188|0.43718|1.2733|0.697489|-2.17731|2.11007|-3.9835|3.79861|
|Fz|7188|-1.01719|2.82249|-0.879913|-4.30358|2.39908|-29.8733|5.40804|
|Tx|7188|-1.12196|1.58937|-1.48467|-3.24933|2.18373|-9.10618|3.32992|
|Ty|7188|-1.25976|1.45396|-1.43827|-2.90424|0.576909|-4.13159|16.8903|
|Tz|7188|0.244777|0.194299|0.261771|-0.0795077|0.527597|-0.603723|0.920118|

These values are identical to Real for every reported statistic because the underlying arrays are identical.

### Sim contact-force (`sim_force`, current local Sim dataset) statistics

|axis|n|mean|std|median|p05|p95|min|max|
|---|---|---|---|---|---|---|---|---|
|Fx|7188|6.74647|108.366|0|-27.5348|38.3445|-211.152|4800.25|
|Fy|7188|-17.2225|53.1954|0|-76.27|3.10359|-2013.52|478.808|
|Fz|7188|0.663847|19.0915|0|-11.5881|11.417|-132.375|852.345|
|Tx|7188|0|0|0|0|0|0|0|
|Ty|7188|0|0|0|0|0|0|0|
|Tz|7188|0|0|0|0|0|0|0|

For `sim_force`, the first three axes are simulator contact-force values; Tx/Ty/Tz are exactly constant zero in all 7,188 rows.

### First-difference statistics, ΔW_t = W_t - W_(t-1)

Real:

|axis|n|std|median_abs|p95_abs|
|---|---|---|---|---|
|Fx|7148|0.425104|0.113369|0.84518|
|Fy|7148|0.260007|0.0809522|0.571835|
|Fz|7148|1.46845|0.123014|1.33768|
|Tx|7148|0.376138|0.0663701|0.723905|
|Ty|7148|0.736966|0.0756593|0.700112|
|Tz|7148|0.0768226|0.0299202|0.168924|

Sim contact-force:

|axis|n|std|median_abs|p95_abs|
|---|---|---|---|---|
|Fx|7148|63.5171|2.57825e-07|12.2799|
|Fy|7148|29.4038|2.84438e-08|16.7644|
|Fz|7148|12.2408|2.59099e-07|5.07323|
|Tx|7148|0|0|0|
|Ty|7148|0|0|0|
|Tz|7148|0|0|0|

The `sim-data` first-difference table equals Real exactly. `sim_force` torque differences are zero because its torque columns are constant zero.

### NaN/Inf/constant checks

The complete machine-readable result is [`quality_checks.csv`](eval_outputs/force_data_audit/quality_checks.csv). No NaN or Inf occurs in any checked force axis. Real and `sim-data` have no constant axis and no zero samples. `sim_force` has no NaN/Inf; Fx/Fy/Fz are non-constant, while Tx/Ty/Tz are constant zero.

### Units and coordinates

- Real `obs/state/ee_wrench_base`: repository docs describe `[Fx,Fy,Fz]` in N and `[Tx,Ty,Tz]` in N·m, expressed in the robot base frame. Raw H5 attributes do not explicitly encode units.
- `sim-data` copies those Real values, so it inherits the same numerical convention rather than establishing an independent simulator convention.
- `sim_force`: the H5 metadata identifies `sim/force/contact_total_base` as the training source and says the training field is `obs/state/ee_wrench_base`; this supports the same robot-base coordinate naming. The H5 has no explicit unit attribute. The first three are contact force values (treated as N under the simulator force convention); the last three are zero and therefore do not represent simulated torque.
- No coordinate rotation, baseline subtraction, normalization, or force filtering is performed by the current H5-to-LeRobot converter or TAVLA transforms before `Normalize`.

## 3. Is Sim force really IsaacLab-generated?

### Direct array comparison

| comparison | result |
|---|---|
| Real vs `sim-data`, `obs/state/ee_wrench_base` | 40/40 exact, max abs diff=0 |
| Real vs `sim_force`, `obs/state/ee_wrench_base` | 0/40 exact, max abs diff=4803.81 |
| `sim_force` `obs/state/ee_wrench_base` vs `sim/force/contact_training_wrench` | 40/40 exact; max abs diff=0 |
| `sim_force` contact training wrench vs `[contact_total_base, 0,0,0]` | 40/40 exact |
| Local LeRobot Sim effort vs `sim_force` | 40/40 exact |

`sim-data` attrs on `traj_0` include `synthetic_visual_rollout=true` and `paired_real_source=/home/sutai/workspace/gujiawei/isaac_env/real_data/traj_0/data.h5`; all 40 Sim H5 files have the same pattern with matching trajectory IDs. Its state, actual action, timestamps, and selected force are all exact copies of Real. The front/wrist MP4 SHA-256 hashes differ from Real, so the visual stream is different while the low-dimensional stream is copied.

`sim_force` attrs on `traj_0` include `sim_force_saved=true`, `sim_force_training_field=obs/state/ee_wrench_base`, `sim_force_training_source=contact_total_base`, and `sim_force_proxy_fields_synchronized=true`. The local Sim LeRobot parquet effort/state/action exactly match this `sim_force` source for all 40 episodes; its sampled front frames also match `sim_force` videos exactly (see [`local_sim_image_source_checks.csv`](eval_outputs/force_data_audit/local_sim_image_source_checks.csv)).

The provenance level is therefore: **not Real-force reuse, strong stored-metadata/field evidence for a simulator contact-force proxy, but no independent IsaacLab generator proof inside TAVLA**. Also, `sim_force` retains the same Real-style `info`/`meta/env_kwargs` schema and paired Real state/action/timestamp path, so it should not be described as an independently replayed full simulator wrench. The reliable statement is limited to the saved `contact_total_base` force stream and its zero-filled six-dimensional compatibility field.

### What the current launcher can actually train on

[`scripts/run_pi0_sim_real_cotrain_30k.sh`](scripts/run_pi0_sim_real_cotrain_30k.sh):65-79 checks whether `local/tavla_single_arm_ee_wrench_sim_hand_noise_50/meta/info.json` exists. If it exists, the launcher reuses it and does not convert `sim-data`. That is the current situation. Therefore the current local Sim dataset is the `sim_force` contact-force set.

If that local dataset is deleted/rebuilt by the fallback in the same launcher, lines 69-74 convert `sim-data` with `--effort-key obs/state/ee_wrench_base`; that fallback would put the copied Real force into the Sim LeRobot dataset. This is a reproducible data-provenance hazard, not a force-alignment result.

## 4. Contact/free-space labels

An explicit annotation exists only at `data/traj_0/traj0_contact_phase_labels_v1.csv`; no corresponding reliable label file was found for the other Real trajectories or for Sim. The annotation covers all 267 frames, its `Fx..Tz` columns match Real H5 force values exactly, and `time_s` equals each H5 timestamp relative to the episode start exactly. `contact_label=0` is free-space, `contact_label=1` is contact, and `contact_label=-1` is `near_contact_uncertain`; the uncertain class is not used for the reliable distributions.

Counts: free-space 96, contact 116, uncertain 55. The explicit labels have high/medium confidence; no force threshold was created. The free/contact statistics are in [`contact_force_stats.csv`](eval_outputs/force_data_audit/contact_force_stats.csv), and the class-conditioned first-difference statistics are in [`contact_delta_stats.csv`](eval_outputs/force_data_audit/contact_delta_stats.csv). They are Real `traj_0` only, so they should not be interpreted as domain-wide or Sim contact statistics. [`contact_label_counts.csv`](eval_outputs/force_data_audit/contact_label_counts.csv) records the label provenance.

## 5. Requested figures

The figures use the raw six-axis arrays and relative episode time; no temporal alignment, resampling, or force normalization is applied.

- Real 3 trajectories: [`real_force_curves_3traj.png`](eval_outputs/force_data_audit/real_force_curves_3traj.png)
- Sim contact-force 3 trajectories (the current local Sim LeRobot source): [`sim_force_curves_3traj.png`](eval_outputs/force_data_audit/sim_force_curves_3traj.png)
- Real first differences: [`real_delta_force_curves_3traj.png`](eval_outputs/force_data_audit/real_delta_force_curves_3traj.png)
- Sim first differences: [`sim_delta_force_curves_3traj.png`](eval_outputs/force_data_audit/sim_delta_force_curves_3traj.png)
- Paired Real/Sim time overlay for `traj_25`, without numerical alignment: [`paired_real_sim_time_overlay_traj_25.png`](eval_outputs/force_data_audit/paired_real_sim_time_overlay_traj_25.png)
- Supplementary raw `sim-data` copied-Real force curves: [`sim_data_visual_replay_force_curves_3traj.png`](eval_outputs/force_data_audit/sim_data_visual_replay_force_curves_3traj.png)

## 6. Final answers to the requested yes/no questions

| question | answer |
|---|---|
| Real force source? | `data/traj_*/data.h5::obs/state/ee_wrench_base`, documented Real robot-base wrench. |
| Sim force source? | Two artifacts exist: raw `sim-data` is copied Real force; current local training Sim dataset is `sim_force::sim/force/contact_total_base` embedded as 3 force axes + zero torque axes. |
| Same coordinate system? | Real and `sim-data` share the copied Real base-frame convention. `sim_force` is labelled `contact_total_base`; no contradictory rotation is applied, but its torque part is missing/zero, so “same full 6D physical wrench” is false. |
| Same sampling frequency? | Nominally yes: 10 Hz; raw timestamp arrays are exactly equal across paired Real/Sim H5, with irregular per-step jitter/outliers. |
| Different force preprocessing? | Raw `data` and `sim-data` have no force preprocessing difference. `sim_force` changes the source and zero-fills torque axes. TAVLA then applies dataset-specific z-score normalization. |
| Was Sim force incorrectly reused from Real? | In `sim-data`: yes, confirmed 40/40 exact. In the current existing local Sim LeRobot dataset: no; it uses `sim_force`. However the launcher fallback can reintroduce the copied-Real `sim-data` force. |
| Does the model receive raw or processed force? | Raw H5/LeRobot effort is unfiltered and unnormalized at conversion; the model receives normalized effort after `Normalize`, shape `(B,1,6)` for the current `(0,)` history, then `EffortType.EXPERT` projects it to one model token. |

## 7. Machine-readable outputs

All outputs are under [`eval_outputs/force_data_audit`](eval_outputs/force_data_audit):

- `raw_force_stats.csv`: per-domain six-axis mean/std/median/p05/p95/min/max.
- `delta_force_stats.csv`: per-domain first-difference std/median-absolute/p95-absolute.
- `trajectory_summary.csv`: per-episode lengths and timestamp spacing.
- `quality_checks.csv`: NaN/Inf/constant/zero checks.
- `array_comparisons.csv`: all 40 paired array comparisons.
- `sim_force_field_relationships.csv`: contact-force field relationships.
- `local_lerobot_provenance.csv`: actual local training dataset vs raw H5 source.
- `model_input_effort_norm_stats.csv`: current 50/50 cotrain stored effort normalization stats.
- `contact_force_stats.csv`, `contact_delta_stats.csv`, `contact_label_counts.csv`: only the existing Real `traj_0` labels.
