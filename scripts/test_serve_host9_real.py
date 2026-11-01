"""Deployment contract checks, including parity with actual prepared Real data."""
import json
from pathlib import Path
import unittest

from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
import h5py
import numpy as np
import pyarrow.parquet as pq

from serve_host9_real import ENCODER_MANIFEST, Host9Session, PORTS, PROMPTS, validate_artifacts

ROOT = Path(__file__).resolve().parents[1]
LEROBOT = ROOT.parent / '.cache/huggingface/lerobot/local'


class RecordingPolicy:
    def __init__(self):
        self.inputs = []
        self.resets = 0

    def infer(self, inputs):
        self.inputs.append(inputs)
        return {'actions': np.zeros((50, 8), dtype=np.float32)}

    def reset(self):
        self.resets += 1


def observation(effort, **extra):
    return {'state': np.zeros(8, np.float32),
            'images': {'cam_high': np.zeros((16, 20, 3), np.uint8),
                       'cam_left_wrist': np.zeros((16, 20, 3), np.uint8),
                       'cam_right_wrist': np.full((16, 20, 3), 255, np.uint8)},
            'effort': effort, 'prompt': 'peg-in-hole', **extra}


class Host9DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = ForcePredictiveEncoder(ENCODER_MANIFEST, domain='real', device='cpu')

    def test_real_stream_matches_training_parquet_for_both_tasks(self):
        for task in ('dual_peg', 'usb'):
            with self.subTest(task=task):
                repo = LEROBOT / f'tavla_host9_20261026_v1_{task}_encoder_real'
                meta = json.loads((repo / 'meta/host9_source.json').read_text())
                segment = meta['segments'][0]
                source = next(x for x in meta['contract']['sources'] if x['trajectory'] == segment['trajectory'])
                self.assertEqual(segment['start'], 0)
                count = min(45, segment['end'])
                self.assertGreaterEqual(count, 10)
                with h5py.File(source['path'], 'r') as f:
                    wrench = f['obs/state/ee_wrench_base'][:count].astype(np.float32)
                    stamps = f['timestamps'][:count].astype(np.float64)
                parquet = pq.read_table(repo / 'data/chunk-000/episode_000000.parquet',
                                        columns=['observation.effort'])
                expected = np.asarray(parquet['observation.effort'].to_pylist(), np.float32)
                session = Host9Session(RecordingPolicy(), task, self.encoder)
                session.infer({'effort': wrench[:count-1], 'timestamps_s': stamps[:count-1], 'update_only': True})
                result = session.infer(observation(wrench[count-1], timestamp_s=float(stamps[count-1])))
                self.assertEqual(result['actions'].shape, (50, 8))
                np.testing.assert_allclose(session.policy.inputs[-1]['effort'], expected[count-10:count], atol=1e-6)
                self.assertEqual(session.policy.inputs[-1]['prompt'], PROMPTS[task])

    def test_six_axis_without_timestamps_and_no_baseline_warmup(self):
        rng = np.random.default_rng(7)
        wrench = rng.normal(size=(14, 6)).astype(np.float32)
        session = Host9Session(RecordingPolicy(), 'usb', self.encoder)
        self.assertTrue(session.infer(observation(wrench[0]))['ready'])
        session.infer({'effort': wrench[1:13], 'update_only': True})
        session.infer(observation(wrench[13]))
        offline = ForcePredictiveEncoder(ENCODER_MANIFEST, domain='real', device='cpu')
        latents = offline.batch(wrench[:, :3], np.arange(len(wrench)) * 0.1)
        np.testing.assert_allclose(session.policy.inputs[-1]['effort'], latents[-10:], atol=1e-6)
        self.assertAlmostEqual(session.last_timestamp, 1.3)

    def test_raw_policy_receives_all_six_axes_and_training_prompt(self):
        wrench = np.arange(18, dtype=np.float32).reshape(3, 6)
        session = Host9Session(RecordingPolicy(), 'usb')
        result = session.infer(observation(wrench))
        supplied = session.policy.inputs[-1]
        np.testing.assert_array_equal(supplied['effort'], wrench[-1:])
        self.assertEqual(supplied['prompt'], PROMPTS['usb'])
        self.assertNotIn('cam_right_wrist', supplied['images'])
        self.assertTrue(result['ready'])

    def test_history_is_connection_local_and_gap_matches_reset(self):
        a = Host9Session(RecordingPolicy(), 'dual_peg', self.encoder)
        b = Host9Session(RecordingPolicy(), 'dual_peg', self.encoder)
        a.infer(observation(np.arange(6, dtype=np.float32), timestamp_s=1.0))
        b.infer(observation(np.arange(6, dtype=np.float32) * 2, timestamp_s=1.0))
        after_gap = np.ones(6, dtype=np.float32)
        a.infer(observation(after_gap, timestamp_s=3.0))
        history = a.policy.inputs[-1]['effort'].copy()
        a.reset()
        a.infer(observation(after_gap, timestamp_s=3.0))
        np.testing.assert_allclose(a.policy.inputs[-1]['effort'], history, atol=1e-6)
        self.assertEqual(len(b.encoder.time), 1)

    def test_bad_requests_do_not_advance_history(self):
        session = Host9Session(RecordingPolicy(), 'usb', self.encoder)
        session.infer(observation(np.zeros(6, np.float32), timestamp_s=1.0))
        for request in (observation(np.zeros((10, 32), np.float32), timestamp_s=1.1),
                        observation(np.zeros(6, np.float32), timestamp_s=1.0),
                        observation(np.full(6, np.nan, np.float32), timestamp_s=1.1),
                        observation(np.zeros(6, np.float32), timestamp_s=1.1, force_source='other')):
            with self.assertRaises(ValueError):
                session.infer(request)
            self.assertEqual(session.last_timestamp, 1.0)
            self.assertEqual(len(session.encoder.time), 1)

    def test_four_checkpoint_contracts(self):
        from openpi.training import config as train_config
        for port, (task, kind) in PORTS.items():
            with self.subTest(port=port):
                config = train_config.get_config(f'pi0_lora_host9_{task}_{kind}_cotrain')
                checkpoint, repos = validate_artifacts(port, config)
                self.assertEqual(checkpoint.name, '29999')
                self.assertEqual(len(repos), 2)


if __name__ == '__main__':
    unittest.main()
