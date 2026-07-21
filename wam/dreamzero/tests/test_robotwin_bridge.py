from types import SimpleNamespace

import numpy as np
import pytest

from wam.dreamzero.evaluation.robotwin.bridge import DreamZeroRoboTwinBridge
from wam.dreamzero.evaluation.robotwin.schema import ROBOTWIN_CAMERA_ORDER, RoboTwinSchema


def _schema(**overrides):
    data = {
        "embodiment_tag": "robotwin_dual_arm",
        "camera_order": list(ROBOTWIN_CAMERA_ORDER),
        "video_keys": ["video.high", "video.left", "video.right"],
        "state_fields": [
            {"key": "state.left", "start": 0, "stop": 2},
            {"key": "state.right", "start": 2, "stop": 4},
        ],
        "action_fields": [
            {"key": "action.left", "start": 0, "stop": 2},
            {"key": "action.right", "start": 2, "stop": 4},
        ],
        "state_dim": 4,
        "action_dim": 4,
        "action_horizon": 4,
        "execute_steps": 2,
        "frames_per_chunk": 2,
        "normalization_metadata": "relative_stats_dreamzero.json",
        "normalization_sha256": "a" * 64,
        "action_representation": "robotwin_native_joint_delta",
    }
    data.update(overrides)
    return RoboTwinSchema.from_dict(data)


class FakePolicy:
    def __init__(self):
        self.mode = None
        self.execute_steps = None
        self.reset_count = 0
        self.observations = []

    def set_fbfm_mode(self, mode):
        self.mode = mode

    def set_fbfm_execution_steps(self, execute_steps):
        self.execute_steps = execute_steps

    def reset_inference_session(self):
        self.reset_count += 1

    def lazy_joint_forward_causal(self, batch):
        self.observations.append(batch.obs)
        action = {
            "action.left": np.arange(8, dtype=np.float32).reshape(4, 2),
            "action.right": np.arange(8, 16, dtype=np.float32).reshape(4, 2),
        }
        return SimpleNamespace(act=action), np.zeros((1, 1), dtype=np.float32)


def _observation(value=1):
    observation = {
        "observation.state": np.arange(4, dtype=np.float32),
        "task": "stack two bowls",
    }
    for camera in ROBOTWIN_CAMERA_ORDER:
        observation[camera] = np.full((4, 6, 3), value, dtype=np.uint8)
    return observation


def test_schema_rejects_agibot_fallback_and_incomplete_action_layout():
    with pytest.raises(ValueError, match="not AgiBot"):
        _schema(embodiment_tag="agibot")
    with pytest.raises(ValueError, match="cover every dimension"):
        _schema(action_fields=[{"key": "action.left", "start": 0, "stop": 2}])


def test_bridge_buffers_real_feedback_and_uses_native_action_tail():
    policy = FakePolicy()
    bridge = DreamZeroRoboTwinBridge(
        policy=policy,
        schema=_schema(),
        mode="Feedback",
        batch_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    assert policy.mode == "Feedback"
    assert policy.execute_steps == 2

    bridge.handle({"reset": True, "prompt": "stack two bowls"})
    first = bridge.handle({"obs": _observation(1)})
    assert first["action"].shape == (4, 1, 2)
    assert policy.observations[-1]["video.high"].shape[0] == 1

    bridge.handle({"obs": _observation(2), "feedback": True})
    ack = bridge.handle({"obs": [_observation(3)], "compute_kv_cache": True})
    assert ack["kv_update"] == "deferred_to_next_joint_forward"
    bridge.handle({"obs": _observation(4)})
    encoded_video = policy.observations[-1]["video.high"]
    assert encoded_video.shape[0] == 2
    assert encoded_video[0, 0, 0, 0] == 1
    assert encoded_video[1, 0, 0, 0] == 2


def test_reset_clears_feedback_frames():
    policy = FakePolicy()
    bridge = DreamZeroRoboTwinBridge(
        policy=policy,
        schema=_schema(),
        mode="RTC",
        batch_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    bridge.handle({"obs": _observation(1), "feedback": True})
    bridge.handle({"reset": True})
    bridge.handle({"obs": _observation(9)})
    assert policy.reset_count == 1
    assert policy.observations[-1]["video.high"].shape[0] == 1
    assert policy.observations[-1]["video.high"][0, 0, 0, 0] == 9
