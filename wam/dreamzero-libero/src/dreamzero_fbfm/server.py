"""Single-GPU model server with a deterministic pseudo-asynchronous job loop."""

from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audit import JsonlAudit
from .constraints import ActionNormalizer, ConstraintMode
from .observation_history import CausalObservationHistory
from .runtime import DreamZeroFBFMRuntime, FeedbackObservation, InferenceCancelled
from .settings import DEFAULT_STATE_FEEDBACK_KP, DEFAULT_STATE_WEIGHT
from .transport import decode_array, encode_array, receive_message, send_message


@dataclass
class InferenceJob:
    event: threading.Event
    thread: threading.Thread | None = None
    actions: np.ndarray | None = None
    error: BaseException | None = None
    started_at: float = 0.0


class ModelServer:
    def __init__(
        self,
        policy: Any,
        normalizer: ActionNormalizer,
        reset_policy_state: Any,
        *,
        mode: ConstraintMode | str,
        host: str,
        port: int,
        audit_path: Path,
        beta: float = 10.0,
        state_weight: float = DEFAULT_STATE_WEIGHT,
        state_feedback_kp: float = DEFAULT_STATE_FEEDBACK_KP,
        diagnostic_vjp: bool = False,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("model server must bind to 127.0.0.1")
        self.policy = policy
        self.reset_policy_state = reset_policy_state
        self.host = host
        self.port = port
        self.audit = JsonlAudit(audit_path)
        self.runtime = DreamZeroFBFMRuntime(
            policy,
            normalizer,
            mode=mode,
            beta=beta,
            state_weight=state_weight,
            state_feedback_kp=state_feedback_kp,
            diagnostic_vjp=diagnostic_vjp,
            audit_path=str(audit_path),
        )
        self.task_description: str | None = None
        self.job: InferenceJob | None = None
        self.observation_history = CausalObservationHistory()

    def _validate_client_protocol(self, request: dict[str, Any]) -> None:
        expected_mode = request.get("expected_mode")
        if expected_mode is not None and expected_mode != self.runtime.mode.value:
            raise ValueError(
                f"client/server mode mismatch: client={expected_mode} "
                f"server={self.runtime.mode.value}"
            )
        for name in ("state_weight", "state_feedback_kp"):
            expected_key = f"expected_{name}"
            if expected_key not in request:
                continue
            expected = request[expected_key]
            actual = getattr(self.runtime, name)
            if isinstance(expected, bool) or not isinstance(expected, (int, float)):
                raise TypeError(f"expected_{name} must be numeric")
            if not math.isclose(float(expected), actual, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    f"client/server {name} mismatch: client={expected} server={actual}"
                )

    def _decode_observation(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.task_description is None:
            raise RuntimeError("reset must be called before inference")
        main = decode_array(request.get("main_image"), dtype="uint8")
        wrist = decode_array(request.get("wrist_image"), dtype="uint8")
        state = decode_array(request.get("state"), dtype="float32")
        if main.shape != (256, 256, 3) or wrist.shape != main.shape or state.shape != (8,):
            raise ValueError(f"invalid LIBERO observation shapes {main.shape}, {wrist.shape}, {state.shape}")
        return {"main": main, "wrist": wrist, "state": state}

    def _prepare_model_observation(
        self, observation: dict[str, Any], request_type: str
    ) -> dict[str, Any]:
        prepared, frame_count = self.observation_history.prepare(
            observation["main"],
            observation["wrist"],
            observation["state"],
            self.task_description or "",
        )
        self.audit.write(
            "model_input",
            request_type=request_type,
            video_frames=frame_count,
            causal_start_frame=int(self.policy.action_head.current_start_frame),
        )
        return prepared

    def _predict(self, observation: dict[str, Any]) -> np.ndarray:
        actions, _ = self.policy.predict_action_batch(observation, mode="eval")
        result = np.asarray(actions, dtype=np.float32)
        if result.shape != (1, 16, 7) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid DreamZero action output {result.shape}")
        return result[0]

    def _start_job(self, observation: dict[str, Any]) -> None:
        if self.job is not None and not self.job.event.is_set():
            raise RuntimeError("an inference job is already active")
        job = InferenceJob(event=threading.Event(), started_at=time.perf_counter())

        def run() -> None:
            try:
                job.actions = self._predict(observation)
            except BaseException as error:
                job.error = error
                self.runtime.cancel()
            finally:
                job.event.set()

        job.thread = threading.Thread(target=run, name="dreamzero-fbfm-inference", daemon=True)
        self.job = job
        job.thread.start()

    def _job_result(self, timeout: float = 300.0) -> np.ndarray:
        if self.job is None:
            raise RuntimeError("no inference job exists")
        if not self.job.event.wait(timeout):
            raise TimeoutError("inference job did not finish")
        if self.job.error is not None:
            raise self.job.error
        assert self.job.actions is not None
        return self.job.actions

    def serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        self.audit.write("server_ready", host=self.host, port=self.port)
        try:
            while True:
                connection, peer = server.accept()
                if peer[0] != "127.0.0.1":
                    connection.close()
                    self.audit.write("client_rejected", peer=str(peer))
                    continue
                self.audit.write("client_connected", peer=str(peer))
                try:
                    with connection:
                        connection.settimeout(360)
                        while True:
                            request = receive_message(connection)
                            if request is None:
                                break
                            request_type = request.get("type")
                            try:
                                response = self._handle(request_type, request)
                            except BaseException as error:
                                response = {
                                    "status": "error",
                                    "type": request_type,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                                self.audit.write(
                                    "server_error",
                                    request_type=request_type,
                                    error=response["error"],
                                )
                            send_message(connection, response)
                            if request_type == "close":
                                break
                except (OSError, TimeoutError) as error:
                    self.audit.write(
                        "client_connection_error",
                        peer=str(peer),
                        error=f"{type(error).__name__}: {error}",
                    )
                finally:
                    self.runtime.cancel()
                    self.job = None
                    self.audit.write("client_disconnected", peer=str(peer))
        finally:
            self.runtime.cancel()
            server.close()

    def _handle(self, request_type: str, request: dict[str, Any]) -> dict[str, Any]:
        if request_type == "reset":
            if self.job is not None and not self.job.event.is_set():
                raise RuntimeError("cannot reset during inference")
            task = request.get("task_description")
            seed = request.get("seed")
            if not isinstance(task, str) or not task or not isinstance(seed, int):
                raise ValueError("reset requires task_description and integer seed")
            self._validate_client_protocol(request)
            self.task_description = task
            self.reset_policy_state(self.policy, seed)
            self.observation_history.reset()
            self.job = None
            return {
                "status": "ok",
                "type": "reset",
                "mode": self.runtime.mode.value,
                "state_weight": self.runtime.state_weight,
                "state_feedback_kp": self.runtime.state_feedback_kp,
            }
        if request_type == "predict_sync":
            observation = self._prepare_model_observation(
                self._decode_observation(request), request_type
            )
            self.runtime.begin_chunk(None, pseudo_async=False)
            actions = self._predict(observation)
            return {"status": "ok", "type": request_type, "actions": encode_array(actions)}
        if request_type == "predict_start":
            raw_observation = self._decode_observation(request)
            observation = self._prepare_model_observation(raw_observation, request_type)
            committed = decode_array(request.get("committed_actions"), dtype="float32")
            self.runtime.begin_chunk(
                committed,
                pseudo_async=True,
                anchor_feedback=FeedbackObservation(
                    action_offset=0,
                    main_image=raw_observation["main"],
                    wrist_image=raw_observation["wrist"],
                    state=raw_observation["state"],
                    task_description=self.task_description or "",
                ),
            )
            self._start_job(observation)
            return {"status": "ok", "type": request_type}
        if request_type == "feedback":
            observation = self._decode_observation(request)
            offset = request.get("action_offset")
            if not isinstance(offset, int):
                raise ValueError("feedback requires integer action_offset")
            self.runtime.submit_feedback(
                FeedbackObservation(
                    action_offset=offset,
                    main_image=observation["main"],
                    wrist_image=observation["wrist"],
                    state=observation["state"],
                    task_description=self.task_description or "",
                )
            )
            return {"status": "ok", "type": request_type}
        if request_type == "grant":
            count = request.get("count")
            if not isinstance(count, int):
                raise ValueError("grant requires integer count")
            snapshot = self.runtime.clock.grant_and_wait(count)
            if not snapshot["accepted"] and self.job is not None and self.job.error is not None:
                raise self.job.error
            return {"status": "ok", "type": request_type, **snapshot}
        if request_type == "result":
            actions = self._job_result()
            return {"status": "ok", "type": request_type, "actions": encode_array(actions)}
        if request_type == "cancel":
            self.runtime.cancel()
            if self.job is not None:
                self.job.event.wait(30)
                if self.job.error is not None and not isinstance(self.job.error, InferenceCancelled):
                    raise self.job.error
            return {"status": "ok", "type": request_type}
        if request_type == "close":
            self.runtime.cancel()
            return {"status": "ok", "type": request_type}
        raise ValueError(f"unknown request type {request_type!r}")
