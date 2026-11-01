#!/usr/bin/env python3
"""Serve one of the four Host9 policies with the same six-axis robot input."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = '20261026_v1'
ENCODER_SHA256 = '06e5f3ee3ef69b3e6c9e2ef4bc0ebea249f6d3ef7da6da18b6171acfa2a6112e'
ENCODER_MANIFEST = ROOT / 'eval_outputs/force_predictive_v3_multitask/host9_student_rollouts_v1/manifest.json'
PORTS = {8000: ('dual_peg', 'raw6'), 8001: ('usb', 'raw6'),
         8002: ('dual_peg', 'encoder'), 8003: ('usb', 'encoder')}
PROMPTS = {
    'dual_peg': 'Insert both pegs of the pregrasped pink part into the two holes of the white fixture; keep holding it',
    'usb': 'Insert the pregrasped black USB flash drive into the middle USB-A port of the white power strip; keep the gripper closed',
}


def observation_inputs(obs, prompt):
    state = np.asarray(obs.get('state'), dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('state must be finite (8,): seven joint positions and the recorded gripper position')
    images = obs.get('images')
    if not isinstance(images, dict):
        raise ValueError('images must contain cam_high and cam_left_wrist')
    result = {}
    for name in ('cam_high', 'cam_left_wrist'):
        image = np.asarray(images.get(name))
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 4:
            raise ValueError(f'images/{name} must be RGB uint8 (H,W,3)')
        result[name] = image
    # Training used the same wrist frame for both wrist camera inputs.
    return {'state': state, 'images': result, 'prompt': prompt}


class Host9Session:
    """Keep policy randomness and the causal Real-domain encoder per connection."""

    def __init__(self, policy, task, encoder=None):
        self.policy = copy.copy(policy)
        self.task = task
        self.encoder = copy.deepcopy(encoder) if encoder is not None else None
        self.last_timestamp = None
        self.explicit_timestamps = None
        if self.encoder is not None:
            self.encoder.reset()

    def reset(self):
        if self.encoder is not None:
            self.encoder.reset()
        self.last_timestamp = None
        self.explicit_timestamps = None
        self.policy.reset()

    def infer(self, obs):
        if not isinstance(obs, dict):
            raise ValueError('Request must be a mapping')
        wrench = np.asarray(obs.get('effort'), dtype=np.float32)
        if wrench.shape == (6,):
            wrench = wrench[None, :]
        if wrench.ndim != 2 or wrench.shape[1] != 6 or not len(wrench) or not np.isfinite(wrench).all():
            raise ValueError('effort must be finite raw [Fx,Fy,Fz,Tx,Ty,Tz] with shape (6,) or (N,6)')
        if obs.get('force_source', 'ee_wrench_base') != 'ee_wrench_base':
            raise ValueError('Real input must use the training ee_wrench_base frame, sign and units')
        update_only = obs.get('update_only', False)
        if not isinstance(update_only, bool):
            raise ValueError('update_only must be a boolean')
        inputs = None if update_only else observation_inputs(obs, PROMPTS[self.task])

        scalar_time, vector_time = 'timestamp_s' in obs, 'timestamps_s' in obs
        if scalar_time and vector_time:
            raise ValueError('Use timestamp_s or timestamps_s, not both')
        explicit = scalar_time or vector_time
        if self.explicit_timestamps is not None and explicit != self.explicit_timestamps:
            raise ValueError('Keep one timestamp mode during an episode; reset to change it')
        if scalar_time:
            value = np.asarray(obs['timestamp_s'], dtype=np.float64)
            if value.ndim != 0:
                raise ValueError('timestamp_s must be a scalar')
            times = value.reshape(1)
        elif vector_time:
            times = np.asarray(obs['timestamps_s'], dtype=np.float64)
        else:
            start = 0.0 if self.last_timestamp is None else self.last_timestamp + 0.1
            times = start + np.arange(len(wrench), dtype=np.float64) * 0.1
        if times.shape != (len(wrench),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Provide one finite increasing timestamp per new force sample')
        if self.last_timestamp is not None and times[0] <= self.last_timestamp:
            raise ValueError('Only send new chronological samples; reset between episodes')

        self.explicit_timestamps = explicit
        self.last_timestamp = float(times[-1])
        if self.encoder is not None:
            for sample, stamp in zip(wrench, times, strict=True):
                self.encoder.push(sample[:3], stamp)
        if update_only:
            return {'ready': True, 'updated': len(wrench), 'timestamp_s': self.last_timestamp}
        inputs['effort'] = self.encoder.effort_history().astype(np.float32) if self.encoder else wrench[-1:]
        result = self.policy.infer(inputs)
        actions = np.asarray(result.get('actions'))
        if actions.shape != (50, 8) or not np.isfinite(actions).all():
            raise ValueError(f'Expected finite absolute actions (50,8), received {actions.shape}')
        return {**result, 'ready': True, 'timestamp_s': self.last_timestamp}


def validate_artifacts(port, config):
    task, kind = PORTS[port]
    name = f'pi0_lora_host9_{task}_{kind}_cotrain'
    checkpoint = ROOT / 'checkpoints' / name / VERSION / '29999'
    assets = checkpoint / 'assets'
    if not (checkpoint / 'params').is_dir() or not (assets / 'norm_stats.json').is_file():
        raise FileNotFoundError(f'Missing parameters or saved normalization statistics: {checkpoint}')
    repos = (assets / 'datasets.txt').read_text().splitlines()
    if repos != list(config.data.repo_id) or list(config.data.dataset_sampling_weights) != [0.5, 0.5]:
        raise ValueError('Checkpoint dataset IDs or Real/Sim weights do not match the configuration')
    stats = json.loads((assets / 'norm_stats.json').read_text())['norm_stats']
    for key, dim in (('state', 32), ('actions', 32), ('effort', 6 if kind == 'raw6' else 32)):
        values = stats[key]
        if len(values['mean']) != dim or any(not np.isfinite(np.asarray(v)).all() for v in values.values()):
            raise ValueError(f'Invalid checkpoint normalization for {key}')
    manifest = json.loads(ENCODER_MANIFEST.read_text())
    encoder_path = ENCODER_MANIFEST.parent / manifest['checkpoint']
    actual_sha = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    if manifest['encoder_sha256'] != ENCODER_SHA256 or actual_sha != ENCODER_SHA256:
        raise ValueError('Force encoder differs from the encoder used to build the policy datasets')
    if kind == 'encoder' and config.policy_metadata.get('encoder_sha256') != ENCODER_SHA256:
        raise ValueError('Policy configuration references a different encoder')
    lerobot = Path(os.environ.get('LEROBOT_HOME', ROOT.parent / '.cache/huggingface/lerobot'))
    for domain, repo in zip(('real', 'sim'), repos, strict=True):
        marker = json.loads((lerobot / repo / 'meta/host9_source.json').read_text())
        contract = marker['contract']
        if not marker['completed'] or (contract['task'], contract['kind'], contract['domain']) != (task, kind, domain):
            raise ValueError(f'Dataset contract does not match {repo}')
        if contract['prompt'] != PROMPTS[task]:
            raise ValueError(f'Training prompt differs for {repo}')
        if kind == 'encoder' and contract['encoder_sha256'] != ENCODER_SHA256:
            raise ValueError(f'Dataset encoder differs for {repo}')
    return checkpoint, repos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, choices=tuple(PORTS), required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    os.chdir(ROOT)
    task, kind = PORTS[args.port]
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as train_config

    config = train_config.get_config(f'pi0_lora_host9_{task}_{kind}_cotrain')
    checkpoint, repos = validate_artifacts(args.port, config)
    encoder = None
    if kind == 'encoder':
        from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
        encoder = ForcePredictiveEncoder(ENCODER_MANIFEST, domain='real', device='cpu', threads=1)
        encoder.model.requires_grad_(False)
    policy = policy_config.create_trained_policy(config, checkpoint, default_prompt=PROMPTS[task])
    metadata = {
        **policy.metadata, 'task': task, 'representation': kind, 'checkpoint': str(checkpoint),
        'training_datasets': repos, 'dataset_sampling_weights': [0.5, 0.5],
        'default_prompt': PROMPTS[task], 'prompt_is_fixed_to_training_task': True,
        'force_source': 'ee_wrench_base', 'force_frame': 'robot_base', 'force_units': 'N',
        'torque_units': 'Nm', 'effort_shapes': [[6], [1, 6], ['N', 6]],
        'encoder_location': 'server_cpu' if encoder is not None else None,
        'encoder_sha256': ENCODER_SHA256 if encoder is not None else None,
        'encoder_domain': 'real' if encoder is not None else None,
        'encoder_force_channels': ['Fx', 'Fy', 'Fz'] if encoder is not None else None,
        'encoder_history_frames': 32 if encoder is not None else 0,
        'policy_history_frames': 10 if encoder is not None else 1,
        'baseline_frames': 0, 'sample_period_s': 0.1, 'timestamp_required': False,
        'reset_gap_s': 0.5, 'history_isolated_per_connection': True,
        'update_only_supported': True, 'state_shape': [8], 'action_shape': [50, 8],
        'actions_are_absolute': True, 'image_channels': 'RGB', 'mirrored_wrist_camera': True,
    }

    class SessionServer(WebsocketPolicyServer):
        async def _handler(self, websocket):
            session = Host9Session(policy, task, encoder)
            await WebsocketPolicyServer(session, metadata=metadata)._handler(websocket)

    logging.info('Serving %s %s from %s on %s:%s', task, kind, checkpoint, args.host, args.port)
    SessionServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main()
