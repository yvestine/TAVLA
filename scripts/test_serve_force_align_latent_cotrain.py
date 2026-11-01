"""Regression checks for the deployed force-alignment encoder's raw wrench API."""

from pathlib import Path

from insertion_alignment_encoder.deploy_infer import ForceAlignmentLatentEncoder
import numpy as np
import pytest
from serve_force_align_latent_cotrain import RawWrenchPolicy, ValidatedPolicy
import torch


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
    path = Path(__file__).resolve().parents[1] / "eval_outputs/insertion_alignment_encoder/encoder.pt"
    if not path.is_file():
        pytest.skip("The matching frozen deployment encoder is not installed")
    torch.set_num_threads(1)
    return ForceAlignmentLatentEncoder(path, device="cpu")


def samples(n=70):
    rng = np.random.default_rng(184)
    times = np.cumsum(rng.uniform(0.08, 0.12, n))
    force = rng.normal(size=(n, 6)).astype(np.float32)
    force[:, :3] += np.array([-2, 0.4, -1], dtype=np.float32)
    return force, times


def observation(force, times=None, **kwargs):
    result = dict(
        images={name: np.zeros((16, 16, 3), np.uint8) for name in ("cam_high", "cam_left_wrist")},
        state=np.zeros(8, np.float32),
        effort=force,
        prompt="peg-in-hole",
        **kwargs,
    )
    if times is not None:
        result["timestamps_s"] = times
    return result


def make_session(encoder):
    policy = RecordingPolicy()
    return RawWrenchPolicy(ValidatedPolicy(policy), encoder), policy


def test_streaming_matches_training_conversion_including_sliding_derivative(encoder):
    force, times = samples()
    session, recorded = make_session(encoder)
    previous = 0
    for stop in [20, 21, 33, 55, 70]:
        result = session.infer(observation(force[previous:stop], times[previous:stop]))
        expected = encoder.latents(force[:stop, :3], None, times[:stop])[-10:]
        np.testing.assert_allclose(recorded.inputs[-1]["effort"], expected, atol=2e-6, rtol=1e-5)
        assert recorded.inputs[-1]["effort"].dtype == np.float32
        assert result["ready"] and result["actions"].shape == (50, 8)
        previous = stop


def test_warmup_returns_status_without_fabricating_actions(encoder):
    force, times = samples()
    session, recorded = make_session(encoder)
    result = session.infer(observation(force[:19], times[:19]))
    assert result["ready"] is False and result["samples_needed"] == 1
    assert "actions" not in result and not recorded.inputs
    result = session.infer(observation(force[19:20], times[19:20]))
    assert result["ready"] and result["samples_needed"] == 0
    assert len(recorded.inputs) == 1


def test_update_only_continues_sampling_between_policy_requests(encoder):
    force, times = samples()
    a, rec_a = make_session(encoder)
    b, rec_b = make_session(encoder)
    result = a.infer(dict(effort=force[:69], timestamps_s=times[:69], update_only=True))
    assert result["ready"] and result["updated"] == 69 and not rec_a.inputs
    a.infer(observation(force[69:], times[69:]))
    b.infer(observation(force, times))
    np.testing.assert_allclose(rec_a.inputs[-1]["effort"], rec_b.inputs[-1]["effort"], atol=2e-6)


def test_torque_is_unused_for_this_three_axis_encoder(encoder):
    force, times = samples()
    changed = force.copy()
    changed[:, 3:] = 1234
    a, rec_a = make_session(encoder)
    b, rec_b = make_session(encoder)
    a.infer(observation(force, times))
    b.infer(observation(changed, times))
    np.testing.assert_array_equal(rec_a.inputs[-1]["effort"], rec_b.inputs[-1]["effort"])


def test_connection_histories_and_episode_resets_are_independent(encoder):
    force, times = samples()
    a, rec_a = make_session(encoder)
    b, rec_b = make_session(encoder)
    a.infer(observation(force, times))
    first_history = rec_a.inputs[-1]["effort"].copy()
    b.infer(dict(effort=force[:5], timestamps_s=times[:5], update_only=True))
    a.reset()
    assert a._stream.samples_received == 0 and b._stream.samples_received == 5
    assert not a._stream.ready and not b._stream.ready
    assert rec_a.resets == 1 and rec_b.resets == 0
    a.infer(observation(force, times))
    np.testing.assert_array_equal(first_history, rec_a.inputs[-1]["effort"])


def test_single_wrench_without_time_uses_fixed_training_sample_period(encoder):
    force, _ = samples()
    session, recorded = make_session(encoder)
    for sample in force[:20]:
        result = session.infer(dict(effort=sample, update_only=True))
    assert result["ready"] and result["timestamp_s"] == pytest.approx(1.9)
    session.infer(observation(force[20:21]))
    expected = encoder.latents(force[:21, :3], None, np.arange(21) * 0.1)[-10:]
    np.testing.assert_allclose(recorded.inputs[-1]["effort"], expected, atol=2e-6)


@pytest.mark.parametrize("case", ["latent", "nan", "duplicate", "reversed", "bad_image", "time_mode", "source"])
def test_bad_request_does_not_advance_force_state(encoder, case):
    force, times = samples()
    session, _ = make_session(encoder)
    session.infer(dict(effort=force[:20], timestamps_s=times[:20], update_only=True))
    request = observation(force[20:22].copy(), times[20:22].copy())
    if case == "latent":
        request["effort"] = np.zeros((10, 32), np.float32)
    elif case == "nan":
        request["effort"][1, 0] = np.nan
    elif case == "duplicate":
        request["timestamps_s"] = times[19:21]
    elif case == "reversed":
        request["timestamps_s"] = times[20:22][::-1]
    elif case == "bad_image":
        request["images"]["cam_high"] = np.zeros((16, 16, 3), np.float32)
    elif case == "time_mode":
        del request["timestamps_s"]
    else:
        request["force_source"] = "panda_link7_wrench_calibrated"
    with pytest.raises(ValueError):
        session.infer(request)
    assert session._stream.samples_received == 20
    assert session._stream.last_timestamp == times[19]


def test_scalar_timestamp_and_legacy_latent_input(encoder):
    force, times = samples()
    session, recorded = make_session(encoder)
    for sample, stamp in zip(force[:20], times[:20], strict=True):
        session.infer(dict(effort=sample, timestamp_s=float(stamp), update_only=True))
    result = session.infer(observation(force[20:21], times[20:21]))
    assert result["ready"]
    legacy = ValidatedPolicy(RecordingPolicy())
    result = legacy.infer(observation(recorded.inputs[-1]["effort"]))
    assert result["actions"].shape == (50, 8)
