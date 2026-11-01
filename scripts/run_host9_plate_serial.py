"""Prepare plate datasets and train raw6 then encoder policies on GPU 0."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = '20261026_v1'
EXPERIMENT = '20261027_v1'
GPU_UUID = 'GPU-46d04482-9f93-1213-2cf3-f2a42f99f844'
ENCODER = ROOT / 'eval_outputs/force_predictive_v3_multitask/host9_student_rollouts_v1/manifest.json'
OUT = ROOT / 'eval_outputs/host9_plate_serial_20261027'
PYTHON = ROOT / '.venv/bin/python'


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'stage': 'preflight', 'models': {'raw6': 'pending', 'encoder': 'pending'}}

    def update(stage):
        state.update(stage=stage, updated_unix_s=time.time())
        tmp = OUT / 'status.json.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(OUT / 'status.json')
        print(stage, flush=True)

    env = {**os.environ, 'PYTHONPATH': f'{ROOT}/scripts:{ROOT}/src:{ROOT}/packages/openpi-client/src',
           'OPENPI_DATA_HOME': str(ROOT.parent / '.cache/openpi'),
           'LEROBOT_HOME': str(ROOT.parent / '.cache/huggingface/lerobot'),
           'HOST9_DATASET_VERSION': VERSION, 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    cpu = {**env, 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}
    gpu = {**env, 'CUDA_VISIBLE_DEVICES': GPU_UUID, 'JAX_PLATFORMS': 'cuda'}

    def run(cmd, target_env, log):
        print('RUN', ' '.join(map(str, cmd)), flush=True)
        with (ROOT / 'logs' / log).open('a', buffering=1) as output:
            subprocess.run([str(x) for x in cmd], env=target_env, stdout=output, stderr=subprocess.STDOUT, check=True)

    def prepare(kind):
        for domain in ('real', 'sim'):
            update(f'prepare_{kind}_{domain}')
            run([PYTHON, 'scripts/prepare_host9_policy_data.py', '--task', 'plate_sliding', '--kind', kind,
                 '--domain', domain, '--version', VERSION, '--encoder-manifest', ENCODER], cpu,
                f'host9_plate_prepare_{kind}_{domain}.log')
        update(f'norm_stats_{kind}')
        run([PYTHON, 'scripts/compute_norm_stats.py', '--config-name',
             f'pi0_lora_host9_plate_sliding_{kind}_cotrain', '--batch-size', '8', '--num-workers', '0'], cpu,
            f'host9_plate_{kind}_stats.log')

    def start_training(kind):
        name = f'pi0_lora_host9_plate_sliding_{kind}_cotrain'
        checkpoint = ROOT / 'checkpoints' / name / EXPERIMENT
        if (checkpoint / '29999').is_dir():
            print(f'Already complete: {checkpoint}', flush=True)
            return None
        cmd = [str(PYTHON), 'scripts/train.py', name, '--exp-name', EXPERIMENT, '--batch-size', '8',
               '--fsdp-devices', '1', '--num-workers', '0', '--num-train-steps', '30000',
               '--save-interval', '5000', '--no-wandb-enabled']
        if checkpoint.exists():
            if not any(x.name.isdigit() for x in checkpoint.iterdir()):
                raise RuntimeError(f'Existing run has no resumable checkpoint: {checkpoint}')
            cmd.append('--resume')
        # Never start a second policy while the first still owns the GPU.
        while True:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                              '--format=csv,noheader,nounits'], text=True)
            free = {a.strip(): int(b.strip()) for a, b in (line.split(',') for line in result.splitlines())}
            if free.get(GPU_UUID, 0) >= 50000:
                break
            update(f'waiting_for_gpu_{kind}')
            time.sleep(30)
        log = (ROOT / 'logs' / f'host9_plate_{kind}_train.log').open('a', buffering=1)
        process = subprocess.Popen(cmd, env=gpu, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        state['models'][kind] = 'running'
        state[f'{kind}_pid'] = process.pid
        state[f'{kind}_checkpoint'] = str(checkpoint)
        update(f'train_{kind}')
        return process

    def finish(kind, process):
        if process is not None and process.wait() != 0:
            state['models'][kind] = 'failed'
            raise RuntimeError(f'{kind} training failed; see logs/host9_plate_{kind}_train.log')
        final = ROOT / 'checkpoints' / f'pi0_lora_host9_plate_sliding_{kind}_cotrain' / EXPERIMENT / '29999'
        if not (final / 'params').is_dir() or not (final / 'train_state').is_dir():
            state['models'][kind] = 'failed'
            raise RuntimeError(f'Missing final checkpoint: {final}')
        state['models'][kind] = 'complete'
        update(f'{kind}_complete')

    raw = None
    try:
        manifest = json.loads(ENCODER.read_text())
        gates = manifest.get('quality_gate', {})
        if manifest.get('ready_domains') != ['real', 'sim'] or not all(gates.get(k) for k in
                ('frozen_real_verified', 'improved_over_initialized_student', 'future_prediction_improved')):
            raise RuntimeError('Encoder quality gates not passed')
        if hashlib.sha256((ENCODER.parent / manifest['checkpoint']).read_bytes()).hexdigest() != manifest['encoder_sha256']:
            raise RuntimeError('Encoder hash mismatch')
        state['encoder_sha256'] = manifest['encoder_sha256']
        prepare('raw6')
        raw = start_training('raw6')
        # CPU preparation can overlap with the first policy; the two policies never overlap.
        try:
            prepare('encoder')
            for domain in ('real', 'sim'):
                base = Path(env['LEROBOT_HOME']) / f'local/tavla_host9_{VERSION}_plate_sliding'
                a, b = [json.loads(Path(f'{base}_{kind}_{domain}/meta/host9_source.json').read_text())
                        for kind in ('raw6', 'encoder')]
                if a['contract']['sources'] != b['contract']['sources'] or a['segments'] != b['segments']:
                    raise RuntimeError(f'Raw and encoder {domain} data differ')
                if b['contract']['encoder_sha256'] != manifest['encoder_sha256']:
                    raise RuntimeError('Encoder dataset hash mismatch')
        except Exception:
            if raw is not None:
                finish('raw6', raw)
                raw = None
            raise
        finish('raw6', raw)
        raw = None
        encoded = start_training('encoder')
        finish('encoder', encoded)
        update('complete')
    except Exception as exc:
        state['error'] = str(exc)
        update('failed')
        raise


if __name__ == '__main__':
    main()
