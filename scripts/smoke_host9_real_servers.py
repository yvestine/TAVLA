#!/usr/bin/env python3
"""Warm and check all four servers with recorded robot observations, without robot motion."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import cv2
import h5py
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from serve_host9_real import ENCODER_SHA256, PORTS, PROMPTS

ROOT = Path(__file__).resolve().parents[1]
LEROBOT = ROOT.parent / '.cache/huggingface/lerobot/local'


def recorded_observation(task):
    repo = LEROBOT / f'tavla_host9_20261026_v1_{task}_encoder_real'
    meta = json.loads((repo / 'meta/host9_source.json').read_text())
    segment = meta['segments'][0]
    source = next(x for x in meta['contract']['sources'] if x['trajectory'] == segment['trajectory'])
    frame = min(44, segment['end'] - segment['start'] - 1)
    if segment['start'] != 0 or frame < 10:
        raise ValueError('Expected an initial continuous recording with at least 11 frames')
    path = Path(source['path'])
    with h5py.File(path) as f:
        wrench = f['obs/state/ee_wrench_base'][:frame+1].astype(np.float32)
        stamps = f['timestamps'][:frame+1].astype(np.float64)
        state = np.r_[f['obs/state/joint_pos'][frame], f['obs/state/gripper_pos'][frame].reshape(-1)].astype(np.float32)
    if np.any(np.diff(stamps) <= 0) or np.any(np.diff(stamps) > 0.5):
        raise ValueError('The selected recording contains a gap')
    images = {}
    for name, video in (('cam_high', 'front_camera.mp4'), ('cam_left_wrist', 'wrist_camera.mp4')):
        capture = cv2.VideoCapture(str(path.parent / video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f'Cannot decode {path.parent / video}, frame {frame}')
        images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return path, frame, wrench, stamps, state, images


def connect_when_ready(port, deadline):
    while time.monotonic() < deadline:
        try:
            return connect(f'ws://127.0.0.1:{port}', compression=None, max_size=None, open_timeout=3)
        except (ConnectionRefusedError, OSError, TimeoutError):
            pane = subprocess.run(['tmux', 'display-message', '-p', '-t', f'host9_server_{port}',
                                   '#{pane_dead} #{pane_dead_status}'], text=True, capture_output=True)
            if pane.returncode or pane.stdout.strip().startswith('1'):
                raise RuntimeError(f'Server {port} exited during startup; inspect its log')
            time.sleep(2)
    raise TimeoutError(f'Server {port} did not listen before the startup deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--startup-timeout', type=int, default=300)
    args = parser.parse_args()
    recordings = {task: recorded_observation(task) for task in ('dual_peg', 'usb')}
    packer = msgpack_numpy.Packer()
    report = {'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'servers': []}

    def receive(ws):
        reply = ws.recv(timeout=240)
        if isinstance(reply, str):
            raise RuntimeError(reply)
        return msgpack_numpy.unpackb(reply)

    def request(ws, observation):
        start = time.perf_counter()
        ws.send(packer.pack(observation))
        return receive(ws), time.perf_counter() - start

    for port, (task, kind) in PORTS.items():
        path, frame, wrench, stamps, state, images = recordings[task]
        with connect_when_ready(port, time.monotonic() + args.startup_timeout) as ws:
            metadata = receive(ws)
            assert metadata['task'] == task and metadata['representation'] == kind
            assert metadata['checkpoint'].endswith(f'pi0_lora_host9_{task}_{kind}_cotrain/20261026_v1/29999')
            assert metadata['default_prompt'] == PROMPTS[task]
            assert metadata['timestamp_required'] is False and metadata['history_isolated_per_connection']
            if kind == 'encoder':
                assert metadata['encoder_sha256'] == ENCODER_SHA256 and metadata['encoder_domain'] == 'real'
            result, _ = request(ws, {'reset': True})
            assert result == {'reset': True}
            ack, _ = request(ws, {'effort': wrench[:-1], 'timestamps_s': stamps[:-1], 'update_only': True})
            assert ack['ready'] and ack['updated'] == len(wrench) - 1
            obs = {'state': state, 'images': images, 'effort': wrench[-1],
                   'timestamp_s': float(stamps[-1]), 'prompt': 'peg-in-hole'}
            first, cold = request(ws, obs)
            assert first['ready'] and first['actions'].shape == (50, 8) and np.isfinite(first['actions']).all()
            request(ws, {'reset': True})
            batch = {**obs, 'effort': wrench, 'timestamps_s': stamps}
            batch.pop('timestamp_s')
            repeated, warm = request(ws, batch)
            np.testing.assert_allclose(first['actions'], repeated['actions'], rtol=1e-6, atol=1e-6)
            # A second connection starts with its own timestamp and force history.
            with connect_when_ready(port, time.monotonic() + args.startup_timeout) as other:
                receive(other)
                independent, _ = request(other, {'effort': wrench[0], 'timestamp_s': float(stamps[0]),
                                                  'update_only': True})
                assert independent['ready'] and independent['updated'] == 1
            # Compatibility with clients that send six-axis force and no timestamps.
            request(ws, {'reset': True})
            untimed, untimed_s = request(ws, {'state': state, 'images': images, 'effort': wrench[-1]})
            assert untimed['ready'] and untimed['actions'].shape == (50, 8) and np.isfinite(untimed['actions']).all()
            report['servers'].append({'port': port, 'task': task, 'representation': kind,
                                      'checkpoint': metadata['checkpoint'], 'encoder_sha256': metadata['encoder_sha256'],
                                      'source': str(path), 'frame': frame, 'cold_s': cold, 'warm_s': warm,
                                      'untimed_s': untimed_s, 'actions_shape': [50, 8],
                                      'max_first_action_joint_delta_rad': float(np.max(np.abs(first['actions'][0, :7] - state[:7]))),
                                      'batch_stream_parity': True, 'independent_connection': True})
            print(json.dumps(report['servers'][-1]), flush=True)
    report['passed'] = True
    output = ROOT / 'logs/host9_real_servers_smoke_20261027.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('All four server checks passed:', output, flush=True)


if __name__ == '__main__':
    main()
