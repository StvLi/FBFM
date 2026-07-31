"""DreamZero's native one-frame warm-up and four-frame causal history."""

from __future__ import annotations

from collections import deque

import numpy as np


class CausalObservationHistory:
    """Build the video input expected by DreamZero's causal inference head.

    The upstream DreamZero service warms the causal cache with one frame. Every
    later inference request uses the most recent four request frames, padding
    missing history with the oldest available frame.
    """

    def __init__(self, frames_per_request: int = 4) -> None:
        if frames_per_request <= 0:
            raise ValueError("frames_per_request must be positive")
        self.frames_per_request = int(frames_per_request)
        self._main: deque[np.ndarray] = deque(maxlen=self.frames_per_request)
        self._wrist: deque[np.ndarray] = deque(maxlen=self.frames_per_request)

    def reset(self) -> None:
        self._main.clear()
        self._wrist.clear()

    def prepare(
        self, main: np.ndarray, wrist: np.ndarray, state: np.ndarray, task: str
    ) -> tuple[dict[str, object], int]:
        main_frame = np.asarray(main, dtype=np.uint8)
        wrist_frame = np.asarray(wrist, dtype=np.uint8)
        state_value = np.asarray(state, dtype=np.float32)
        if main_frame.shape != (256, 256, 3) or wrist_frame.shape != main_frame.shape:
            raise ValueError(
                f"invalid DreamZero video history shapes {main_frame.shape}, "
                f"{wrist_frame.shape}"
            )
        if state_value.shape != (8,):
            raise ValueError(f"invalid DreamZero state history shape {state_value.shape}")

        self._main.append(main_frame.copy())
        self._wrist.append(wrist_frame.copy())
        if len(self._main) == 1:
            main_frames = [self._main[0]]
            wrist_frames = [self._wrist[0]]
        else:
            main_frames = list(self._main)
            wrist_frames = list(self._wrist)
            while len(main_frames) < self.frames_per_request:
                main_frames.insert(0, main_frames[0])
                wrist_frames.insert(0, wrist_frames[0])

        frame_count = len(main_frames)
        return (
            {
                "main_images": np.stack(main_frames, axis=0)[None],
                "wrist_images": np.stack(wrist_frames, axis=0)[None],
                "states": state_value[None],
                "task_descriptions": [task],
            },
            frame_count,
        )
