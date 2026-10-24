# TAVLA Real-first force trend input

This is the first deployment-oriented force integration for TAVLA. It is an
ablation alongside the existing visual/state policy; it does not replace the
baseline and it does not use Sim force as quality supervision.

## What enters TAVLA

Each Real HDF5 episode uses `obs/state/ee_force_base[:,:3]`. For every
trajectory, the confirmed free-space interval is used to estimate its own
force baseline. The transform is causal after that calibration: smoothing and
the five-frame trend only use the current and past samples.

The per-frame `observation.effort` has ten values:

```text
force_dir_x/y/z          gated residual force direction
lateral_force_ratio      lateral residual / residual norm
axial_force_fraction     insertion-axis component / residual norm
trend_dir_x/y/z          gated five-frame causal trend direction
force_norm               smoothed residual force magnitude
trend_norm               five-frame smoothed trend magnitude
```

TAVLA receives the last ten frames (`-9 ... 0`, approximately one second at
10 Hz) through `EffortType.EXPERT_HIS_T`, so each force-history frame is
projected as its own action-expert token. No contact label, future pose,
future force, torque field, DANN, or Sim/Real hard alignment is used.

## Why Real is the first deployment path

The force-trend audit found consistent local trend information on Real but not
on the current Sim `contact_total_base` stream. Therefore the first policy
candidate is deliberately Real-primary. Sim can later be added as an
auxiliary visual/action dataset after a separate sensor-model audit, but its
sparse impulse signal must not be pooled into this force representation by
default.

## Dataset conversion and training

The conversion script is:

```text
scripts/convert_user_hdf5_to_lerobot.py
```

The reusable transform and deployment buffer are in:

```text
src/openpi/force_features.py
```

The configured training entry point is:

```text
pi0_lora_user_single_arm_real_force_trend_v1
```

The optional future-force auxiliary variant is:

```text
pi0_lora_user_single_arm_real_force_trend_v1_future
```

It uses the same 10-frame history and additionally loads the next 50 force
feature frames (the current 50-step action horizon). During training Pi0
predicts action and future force features jointly, with future-force MSE
weight `0.1`. During inference the predicted force trajectory is discarded
and only the action sequence is returned.

The robot-side caller should update `RealForceTrendBuffer` once per 10 Hz
force sample, call `history(10)`, and send that array as
`observation.effort`. The policy's normal statistics transform then handles
the feature scaling saved under the checkpoint assets.

## Required comparison

Run the existing Real EE-wrench baseline and this Real force-trend variant
with the same seed, steps, rollout protocol, and held-out episodes. The force
variant is useful only if it improves insertion behavior on Real rollouts; a
lower offline action loss alone is not enough.
