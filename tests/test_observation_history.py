import numpy as np
import pytest

from dreamzero_fbfm.observation_history import CausalObservationHistory


def _frame(value: int) -> np.ndarray:
    return np.full((256, 256, 3), value, dtype=np.uint8)


def test_causal_history_uses_one_frame_then_padded_four_frame_requests():
    history = CausalObservationHistory()
    state = np.zeros(8, dtype=np.float32)

    first, first_count = history.prepare(_frame(1), _frame(11), state, "task")
    second, second_count = history.prepare(_frame(2), _frame(12), state, "task")
    history.prepare(_frame(3), _frame(13), state, "task")
    fourth, fourth_count = history.prepare(_frame(4), _frame(14), state, "task")
    fifth, fifth_count = history.prepare(_frame(5), _frame(15), state, "task")

    assert first_count == 1
    assert first["main_images"].shape == (1, 1, 256, 256, 3)
    assert second_count == fourth_count == fifth_count == 4
    assert second["main_images"][0, :, 0, 0, 0].tolist() == [1, 1, 1, 2]
    assert fourth["main_images"][0, :, 0, 0, 0].tolist() == [1, 2, 3, 4]
    assert fifth["main_images"][0, :, 0, 0, 0].tolist() == [2, 3, 4, 5]
    assert fifth["wrist_images"][0, :, 0, 0, 0].tolist() == [12, 13, 14, 15]
    assert fifth["states"].shape == (1, 8)
    assert fifth["task_descriptions"] == ["task"]


def test_causal_history_reset_restores_single_frame_warmup():
    history = CausalObservationHistory()
    state = np.zeros(8, dtype=np.float32)
    history.prepare(_frame(1), _frame(11), state, "task")
    history.prepare(_frame(2), _frame(12), state, "task")
    history.reset()

    prepared, count = history.prepare(_frame(9), _frame(19), state, "new task")

    assert count == 1
    assert prepared["main_images"][0, :, 0, 0, 0].tolist() == [9]


def test_causal_history_rejects_invalid_shapes():
    history = CausalObservationHistory()
    with pytest.raises(ValueError, match="video history"):
        history.prepare(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros(8, dtype=np.float32),
            "task",
        )
