"""LIBERO-side client for pseudo-asynchronous DreamZero FBFM inference."""

from __future__ import annotations

import socket
from typing import Any

import numpy as np

from .transport import decode_array, encode_array, receive_message, send_message


class FBFMClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 18766, timeout: float = 300.0) -> None:
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("FBFM transport is localhost-only")
        self.connection = socket.create_connection((host, port), timeout=10)
        self.connection.settimeout(timeout)

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        send_message(self.connection, message)
        response = receive_message(self.connection)
        if response is None:
            raise ConnectionError("model server closed without a response")
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", str(response)))
        return response

    @staticmethod
    def _observation(main: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> dict[str, Any]:
        return {
            "main_image": encode_array(np.asarray(main, dtype=np.uint8)),
            "wrist_image": encode_array(np.asarray(wrist, dtype=np.uint8)),
            "state": encode_array(np.asarray(state, dtype=np.float32)),
        }

    def reset(
        self,
        task_description: str,
        seed: int,
        *,
        mode: str | None = None,
        state_weight: float | None = None,
        state_feedback_kp: float | None = None,
    ) -> None:
        request: dict[str, Any] = {
            "type": "reset",
            "task_description": task_description,
            "seed": int(seed),
        }
        if mode is not None:
            request["expected_mode"] = mode
        if state_weight is not None:
            request["expected_state_weight"] = float(state_weight)
        if state_feedback_kp is not None:
            request["expected_state_feedback_kp"] = float(state_feedback_kp)
        self._request(request)

    def predict_sync(self, main: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> np.ndarray:
        response = self._request({"type": "predict_sync", **self._observation(main, wrist, state)})
        return decode_array(response["actions"], dtype="float32")

    def start_predict(
        self,
        main: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        committed_actions: np.ndarray,
    ) -> None:
        self._request(
            {
                "type": "predict_start",
                **self._observation(main, wrist, state),
                "committed_actions": encode_array(
                    np.asarray(committed_actions, dtype=np.float32)
                ),
            }
        )

    def feedback(
        self,
        action_offset: int,
        main: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
    ) -> None:
        self._request(
            {
                "type": "feedback",
                "action_offset": int(action_offset),
                **self._observation(main, wrist, state),
            }
        )

    def grant(self, count: int) -> dict[str, Any]:
        return self._request({"type": "grant", "count": int(count)})

    def result(self) -> np.ndarray:
        response = self._request({"type": "result"})
        return decode_array(response["actions"], dtype="float32")

    def cancel(self) -> None:
        self._request({"type": "cancel"})

    def close(self) -> None:
        try:
            self._request({"type": "close"})
        finally:
            self.connection.close()
