#!/usr/bin/env python3
"""Start the four Host9 real-robot websocket services on physical GPU 0."""
import os
from pathlib import Path
import shlex
import socket
import subprocess

from serve_host9_real import PORTS


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    for port in PORTS:
        name = f'host9_server_{port}'
        if subprocess.run(['tmux', 'has-session', '-t', name], stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError(f'{name} already exists; inspect the session before starting another')
        with socket.socket() as sock:
            try:
                sock.bind(('0.0.0.0', port))
            except OSError as error:
                raise RuntimeError(f'Port {port} is already in use') from error
    gpu_uuid = subprocess.check_output(
        ['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(
        ['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < 50000:
        raise RuntimeError(f'GPU 0 has only {free} MiB free; four servers require more headroom')
    env = {
        'CUDA_VISIBLE_DEVICES': gpu_uuid, 'JAX_PLATFORMS': 'cuda',
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'PYTHONUNBUFFERED': '1',
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
        'OPENPI_DATA_HOME': os.environ.get('OPENPI_DATA_HOME', str(root.parent / '.cache/openpi')),
        'LEROBOT_HOME': os.environ.get('LEROBOT_HOME', str(root.parent / '.cache/huggingface/lerobot')),
        'HOST9_DATASET_VERSION': os.environ.get('HOST9_DATASET_VERSION', '20261026_v1'),
        'PYTHONPATH': f'{root}/scripts:{root}/src:{root}/packages/openpi-client/src',
    }
    (root / 'logs').mkdir(exist_ok=True)
    for port, (task, kind) in PORTS.items():
        name = f'host9_server_{port}'
        command = ['env', *[f'{key}={value}' for key, value in env.items()],
                   str(root / '.venv/bin/python'), 'scripts/serve_host9_real.py', '--port', str(port)]
        log = root / f'logs/{name}.log'
        shell = shlex.join(command) + f' 2>&1 | tee -a {shlex.quote(str(log))}'
        subprocess.run(['tmux', 'new-session', '-d', '-s', name, '-c', str(root),
                        'bash -o pipefail -c ' + shlex.quote(shell)], check=True)
        subprocess.run(['tmux', 'set-window-option', '-t', f'{name}:0', 'remain-on-exit', 'on'], check=True)
        print(f'STARTED {name}: GPU 0, {task}, {kind}, log {log}', flush=True)


if __name__ == '__main__':
    main()
