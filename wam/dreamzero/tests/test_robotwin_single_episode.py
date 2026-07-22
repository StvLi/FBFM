import numpy as np
import pytest

from wam.dreamzero.evaluation.robotwin.single_episode import (
    _validate_action_chunk,
    _validate_server_metadata,
)


def test_single_episode_rejects_mode_or_checkpoint_drift():
    metadata = {
        "constraint_mode": "None",
        "checkpoint_sha256": "a" * 64,
        "execute_steps": 8,
        "frames_per_chunk": 4,
    }
    assert _validate_server_metadata(metadata, "None", "a" * 64) == (8, 4)
    with pytest.raises(RuntimeError, match="mode"):
        _validate_server_metadata(metadata, "RTC", "a" * 64)
    with pytest.raises(RuntimeError, match="checkpoint"):
        _validate_server_metadata(metadata, "None", "b" * 64)


def test_single_episode_requires_finite_eef14_chunk():
    action = np.zeros((14, 1, 8), dtype=np.float32)
    assert _validate_action_chunk(action, 8).shape == (14, 1, 8)
    with pytest.raises(ValueError, match="shape"):
        _validate_action_chunk(np.zeros((14, 8)), 8)
    action[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _validate_action_chunk(action, 8)
