"""Measure full Pi0 LoRA training steps with the production input shapes."""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.training import config as configs, sharding, weight_loaders


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('raw6', 'latent'), required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-params', default='/workspace/gujiawei/.cache/openpi/openpi-assets/checkpoints/pi0_base/params')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    train.init_logging()
    name = ('pi0_lora_user_single_arm_ee_wrench_cotrain_aligned_50_50' if a.kind == 'raw6'
            else 'pi0_lora_user_single_arm_force_align_latent_cotrain_50_50')
    cfg = dataclasses.replace(configs.get_config(name), batch_size=a.batch_size, fsdp_devices=1,
                              weight_loader=weight_loaders.CheckpointWeightLoader(a.base_params), ema_decay=None)
    mesh = sharding.make_mesh(1)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rng = jax.random.key(42)
    state, state_sharding = train.init_train_state(cfg, rng, mesh, resume=False)
    obs_spec, act_spec = cfg.model.inputs_spec(batch_size=a.batch_size)
    def fill(spec):
        if np.issubdtype(spec.dtype, np.bool_): return jnp.ones(spec.shape, bool)
        return jnp.zeros(spec.shape, spec.dtype)
    obs = jax.tree.map(fill, obs_spec)
    obs = dataclasses.replace(obs, effort=jnp.zeros((a.batch_size, 1 if a.kind == 'raw6' else 10,
                                                    6 if a.kind == 'raw6' else 32), jnp.float32))
    batch = jax.device_put((obs, fill(act_spec)), data_sharding)
    step = jax.jit(functools.partial(train.train_step, cfg),
                   in_shardings=(replicated, state_sharding, data_sharding),
                   out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    result = dict(kind=a.kind, batch_size=a.batch_size, action_horizon=50, cameras=3,
                  image_size=[224,224], effort_shape=list(obs.effort.shape), steps=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for i in range(a.steps):
        tick = time.monotonic()
        with sharding.set_mesh(mesh): state, info = step(rng, state, batch)
        info = jax.device_get(info)
        if not all(np.isfinite(v).all() for v in info.values()): raise FloatingPointError(info)
        row = dict(step=i, seconds=time.monotonic()-tick, **{k:float(v) for k,v in info.items()})
        result['steps'].append(row)
        print(json.dumps(row), flush=True)
    result.update(completed=True, elapsed_s=time.monotonic()-started,
                  device_memory_stats=jax.devices()[0].memory_stats())
    a.output.write_text(json.dumps(result, indent=2))
    print('BENCHMARK_PASSED', str(a.output), flush=True)


if __name__ == '__main__':
    main()
