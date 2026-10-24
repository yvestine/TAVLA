"""Check raw-wrench serving against the encoder used for offline training conversion."""

from pathlib import Path

from force_predictive_encoder_v3.runtime import ForcePredictiveEncoder
import numpy as np
import pytest
from serve_force_predictive_v3 import DEFAULT_CHECKPOINT
from serve_force_predictive_v3 import RawForcePolicy


class RecordingPolicy:
    def __init__(self):
        self.inputs = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def infer(self, obs):
        self.inputs.append(obs)
        return {"actions": np.zeros((50, 8), np.float32)}


@pytest.fixture(scope="module")
def encoder():
    root = Path(__file__).resolve().parents[1]
    manifest = root / DEFAULT_CHECKPOINT / "assets/force_predictive_v3_encoder_manifest.json"
    if not manifest.is_file():
        pytest.skip("Deployment checkpoint is not present")
    return ForcePredictiveEncoder(manifest, domain="sim", device="cpu")


def observation(wrench, times, **extra):
    return dict(
        images={name: np.zeros((16, 16, 3), np.uint8) for name in ("cam_high", "cam_left_wrist")},
        state=np.zeros(8, np.float32),
        effort=wrench,
        timestamps_s=times,
        **extra,
    )


def samples():
    times = np.arange(45, dtype=np.float64) * 0.1
    wrench = np.column_stack((np.sin(times), np.cos(times), times, times * 2, times * 3, times * 4)).astype(np.float32)
    return wrench, times


def test_chunked_requests_match_offline_training_encoder(encoder):
    wrench, times = samples()
    offline = ForcePredictiveEncoder(
        Path(__file__).resolve().parents[1] / DEFAULT_CHECKPOINT / "assets/force_predictive_v3_encoder_manifest.json"
    )
    latents = offline.batch(wrench[:, :3], times)
    policy = RecordingPolicy()
    session = RawForcePolicy(policy, encoder)
    result = session.infer(observation(wrench[:1], times[:1]))
    np.testing.assert_allclose(policy.inputs[-1]["effort"], np.repeat(latents[:1], 10, axis=0), atol=1e-6)
    for lo, hi in ((1, 6), (6, 35), (35, 45)):
        result = session.infer(observation(wrench[lo:hi], times[lo:hi]))
        history = latents[max(0, hi - 10) : hi]
        expected = np.pad(history, ((10 - len(history), 0), (0, 0)), mode="edge")
        np.testing.assert_allclose(policy.inputs[-1]["effort"], expected, atol=1e-6)
    np.testing.assert_allclose(result["predicted_force"], offline.predict_future(), atol=1e-6)
    assert result["actions"].shape == (50, 8)
    assert result["timestamp_s"] == times[-1]


def test_calibrated_link7_sign_and_unused_torque(encoder):
    wrench, times = samples()
    first, second = RecordingPolicy(), RecordingPolicy()
    aligned = RawForcePolicy(first, encoder)
    calibrated = RawForcePolicy(second, encoder)
    raw = wrench.copy()
    raw[:, :3] *= -1
    raw[:, 3:] = 9876
    a = aligned.infer(observation(wrench, times))
    b = calibrated.infer(observation(raw, times, force_source="panda_link7_wrench_calibrated"))
    np.testing.assert_array_equal(first.inputs[-1]["effort"], second.inputs[-1]["effort"])
    np.testing.assert_array_equal(a["predicted_force"], b["predicted_force"])


def test_update_only_matches_full_batch(encoder):
    wrench, times = samples()
    first, second = RecordingPolicy(), RecordingPolicy()
    incremental, batch = RawForcePolicy(first, encoder), RawForcePolicy(second, encoder)
    response = incremental.infer(dict(effort=wrench[:-1], timestamps_s=times[:-1], update_only=True))
    assert response["updated"] == len(wrench) - 1
    assert not first.inputs
    incremental.infer(observation(wrench[-1:], times[-1:]))
    batch.infer(observation(wrench, times))
    np.testing.assert_array_equal(first.inputs[-1]["effort"], second.inputs[-1]["effort"])


def test_gap_and_explicit_reset_clear_both_histories(encoder):
    wrench, times = samples()
    policy = RecordingPolicy()
    session = RawForcePolicy(policy, encoder)
    session.infer(observation(wrench, times))
    after_gap = session.infer(observation(wrench[:1], times[:1] + 10))
    gap_history = policy.inputs[-1]["effort"].copy()
    session.reset()
    after_reset = session.infer(observation(wrench[:1], times[:1] + 10))
    np.testing.assert_array_equal(policy.inputs[-1]["effort"], gap_history)
    np.testing.assert_array_equal(after_gap["predicted_force"], after_reset["predicted_force"])
    assert policy.resets == 1


def test_connections_have_independent_encoder_histories(encoder):
    wrench, times = samples()
    shared = RecordingPolicy()
    a, b = RawForcePolicy(shared, encoder), RawForcePolicy(shared, encoder)
    a.infer(observation(wrench, times))
    b.infer(observation(wrench[:1] * 2, times[:1]))
    a.infer(observation(wrench[-1:], times[-1:] + 0.1))
    assert len(a._encoder.time) == 32
    assert len(b._encoder.time) == 1
    a.reset()
    assert len(b._encoder.time) == 1


@pytest.mark.parametrize(
    "case", ("duplicate", "reversed", "missing_time", "nan", "latent", "source_change", "bad_image")
)
def test_invalid_requests_do_not_advance_encoder(encoder, case):
    wrench, times = samples()
    session = RawForcePolicy(RecordingPolicy(), encoder)
    session.infer(observation(wrench[:1], times[:1]))
    request = observation(wrench[1:3].copy(), times[1:3].copy())
    if case == "duplicate":
        request["timestamps_s"] = times[:2]
    elif case == "reversed":
        request["timestamps_s"] = times[1:3][::-1]
    elif case == "missing_time":
        del request["timestamps_s"]
    elif case == "nan":
        request["effort"][1, 0] = np.nan
    elif case == "latent":
        request["effort"] = np.zeros((10, 32), np.float32)
    elif case == "source_change":
        request["force_source"] = "panda_link7_wrench_calibrated"
    else:
        request["images"]["cam_high"] = np.zeros((16, 16, 3), np.float32)
    with pytest.raises(ValueError):
        session.infer(request)
    assert list(session._encoder.time) == [times[0]]


def test_single_raw_wrench_and_scalar_simulation_time(encoder):
    wrench, times = samples()
    policy = RecordingPolicy()
    session = RawForcePolicy(policy, encoder)
    obs = observation(wrench[0], times[:1])
    del obs["timestamps_s"]
    obs["timestamp_s"] = float(times[0])
    response = session.infer(obs)
    assert response["predicted_force"].shape == (5, 3)
    assert policy.inputs[-1]["effort"].dtype == np.float32
