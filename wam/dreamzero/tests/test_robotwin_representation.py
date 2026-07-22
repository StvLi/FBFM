import json

import numpy as np
import pytest

from wam.dreamzero.evaluation.robotwin.validate_dataset import validate_dataset
from wam.dreamzero.evaluation.robotwin.representation import (
    eef14_to_simulator_action,
    format_simulator_observation,
    pack_eef14,
    state_from_endpose,
)


def _endpose():
    return {
        "left_endpose": [-0.3, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0],
        "left_gripper": 1.0,
        "right_endpose": [0.3, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0],
        "right_gripper": 0.0,
    }


def test_pack_eef14_uses_xyz_euler_gripper_layout():
    state = state_from_endpose(_endpose())
    assert state.shape == (14,)
    np.testing.assert_allclose(state[:7], [-0.3, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(state[7:], [0.3, -0.2, 0.9, 0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_pack_eef14_vectorizes_trajectory_and_normalizes_quaternion():
    pose = np.asarray([[0, 0, 0, 0, 0, 0, 2], [1, 2, 3, 0, 0, 0, 3]], dtype=np.float64)
    packed = pack_eef14(pose, np.ones(2), pose, np.zeros(2))
    assert packed.shape == (2, 14)
    np.testing.assert_allclose(packed[:, 3:6], 0.0, atol=1e-6)


def test_format_simulator_observation_uses_endpose_not_joint_vector():
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    observation = {
        "observation": {
            "head_camera": {"rgb": frame},
            "left_camera": {"rgb": frame},
            "right_camera": {"rgb": frame},
        },
        "endpose": _endpose(),
        "joint_action": {"vector": np.full(14, 99.0)},
    }
    formatted = format_simulator_observation(observation, "pick the bottle")
    assert formatted["task"] == "pick the bottle"
    assert formatted["observation.state"].shape == (14,)
    assert not np.equal(formatted["observation.state"], 99.0).any()


def test_zero_quaternion_is_rejected():
    endpose = _endpose()
    endpose["left_endpose"][-4:] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="zero quaternion"):
        state_from_endpose(endpose)


def test_eef14_action_round_trips_through_simulator_quaternions():
    action = np.array(
        [0.1, -0.2, 0.3, 0.2, -0.3, 0.4, 0.5, -0.1, 0.2, 0.4, -0.4, 0.1, 0.3, 0.7],
        dtype=np.float32,
    )
    simulator = eef14_to_simulator_action(action)
    assert simulator.shape == (16,)
    recovered = pack_eef14(simulator[:7], simulator[7], simulator[8:15], simulator[15])
    np.testing.assert_allclose(recovered, action, atol=1e-6)


def _write_valid_meta(root):
    meta = root / "meta"
    meta.mkdir()
    (meta / "embodiment.json").write_text(json.dumps({"embodiment_tag": "robotwin"}))
    modality = {
        "video": {key: {} for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
        "state": {
            key: {}
            for key in (
                "left_eef_position", "left_eef_rotation", "left_gripper",
                "right_eef_position", "right_eef_rotation", "right_gripper",
            )
        },
        "action": {
            key: {}
            for key in (
                "left_eef_position", "left_eef_rotation", "left_gripper",
                "right_eef_position", "right_eef_rotation", "right_gripper",
            )
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality))
    vector_stats = {key: [0.0] * 14 for key in ("mean", "std", "min", "max", "q01", "q99")}
    (meta / "stats.json").write_text(json.dumps({"observation.state": vector_stats, "action": vector_stats}))
    position_stats = {key: [0.0] * 3 for key in ("mean", "std", "min", "max", "q01", "q99")}
    (meta / "relative_stats_dreamzero.json").write_text(
        json.dumps({"left_eef_position": position_stats, "right_eef_position": position_stats})
    )
    (meta / "tasks.jsonl").write_text("{}\n")
    (meta / "episodes.jsonl").write_text("{}\n")
    return meta


def test_dataset_validation_accepts_position_only_relative_stats(tmp_path):
    _write_valid_meta(tmp_path)
    validate_dataset(tmp_path)


def test_dataset_validation_rejects_linear_relative_euler_stats(tmp_path):
    meta = _write_valid_meta(tmp_path)
    path = meta / "relative_stats_dreamzero.json"
    stats = json.loads(path.read_text())
    stats["left_eef_rotation"] = {key: [0.0] * 3 for key in ("mean", "std", "min", "max", "q01", "q99")}
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="non-positional relative keys"):
        validate_dataset(tmp_path)
