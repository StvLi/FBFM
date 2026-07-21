"""RoboTwin request protocol adapter for a native DreamZero checkpoint."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from .schema import RoboTwinSchema


class DreamZeroRoboTwinBridge:
    """Translate RoboTwin observations without borrowing AgiBot normalization.

    Feedback requests only buffer real RGB frames.  DreamZero's next joint
    causal forward performs its normal VAE encoding and true-observation KV
    replacement, so the model's native temporal compression remains the source
    of truth.  No LingBot 4-to-1 latent assumption is present here.
    """

    def __init__(
        self,
        *,
        policy: Any,
        schema: RoboTwinSchema,
        mode: str,
        batch_factory: Callable[..., Any] | None = None,
        forward_fn: Callable[[dict[str, Any]], tuple[Any, Any]] | None = None,
    ) -> None:
        if mode not in {"None", "RTC", "Feedback"}:
            raise ValueError(f"unknown FBFM mode {mode!r}")
        self.policy = policy
        self.schema = schema
        self.mode = mode
        self._batch_factory = batch_factory
        self._forward_fn = forward_fn
        self._frames: dict[str, list[np.ndarray]] = defaultdict(list)
        self._instruction: str | None = None
        self._latest_observation: dict[str, Any] | None = None
        self._feedback_since_prediction = False
        self._prediction_count = 0
        self.policy.set_fbfm_mode(mode)
        self.policy.set_fbfm_execution_steps(schema.execute_steps)

    def reset(self, instruction: str | None = None) -> dict[str, Any]:
        self._frames.clear()
        self._instruction = instruction
        self._latest_observation = None
        self._feedback_since_prediction = False
        self._prediction_count = 0
        self.policy.reset_inference_session()
        return {"ok": True, "event": "reset", "mode": self.mode}

    def _ingest(self, observation: dict[str, Any]) -> None:
        for source_key, target_key in zip(self.schema.camera_order, self.schema.video_keys):
            if source_key not in observation:
                raise KeyError(f"RoboTwin observation missing camera {source_key!r}")
            frame = np.asarray(observation[source_key])
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"{source_key} must have shape (H,W,3), got {frame.shape}")
            self._frames[target_key].append(frame.astype(np.uint8, copy=False))
            self._frames[target_key] = self._frames[target_key][-self.schema.frames_per_chunk :]
        if "task" in observation:
            self._instruction = str(observation["task"])
        self._latest_observation = observation

    def _encoded_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if "observation.state" not in observation:
            raise KeyError("RoboTwin observation missing 'observation.state'")
        encoded = self.schema.encode_state(observation["observation.state"])
        frame_count = 1 if self._prediction_count == 0 else self.schema.frames_per_chunk
        for target_key in self.schema.video_keys:
            frames = list(self._frames[target_key])
            if not frames:
                raise RuntimeError(f"no frames buffered for {target_key}")
            while len(frames) < frame_count:
                frames.insert(0, frames[0])
            encoded[target_key] = np.stack(frames[-frame_count:], axis=0)
        instruction = str(observation.get("task", self._instruction or "")).strip()
        if not instruction:
            raise ValueError("RoboTwin instruction is empty")
        encoded["annotation.language.action_text"] = instruction
        return encoded

    def _make_batch(self, encoded: dict[str, Any]) -> Any:
        if self._batch_factory is not None:
            return self._batch_factory(obs=encoded)
        from tianshou.data import Batch

        return Batch(obs=encoded)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("reset"):
            return self.reset(request.get("prompt") or request.get("task"))

        if request.get("feedback"):
            observation = request.get("obs")
            if not isinstance(observation, dict):
                raise ValueError("feedback request must contain one observation dict")
            self._ingest(observation)
            self._feedback_since_prediction = True
            return {"ok": True, "event": "feedback_buffered"}

        if request.get("compute_kv_cache"):
            observations = request.get("obs", [])
            if isinstance(observations, dict):
                observations = [observations]
            # LingBot-compatible clients resend the same keyframes here after
            # already sending each one via `feedback=True`. Avoid duplicating
            # those frames; accept the list as a fallback for simpler clients.
            if not self._feedback_since_prediction:
                for observation in observations:
                    self._ingest(observation)
                self._feedback_since_prediction = bool(observations)
            return {
                "ok": True,
                "event": "causal_context_buffered",
                "kv_update": "deferred_to_next_joint_forward",
            }

        observation = request.get("obs", request)
        if not isinstance(observation, dict):
            raise ValueError("inference request must contain an observation dict")
        # The reference RoboTwin client keeps passing its first observation on
        # normal inference calls and sends fresh observations separately via
        # the feedback endpoint. Prefer the latest feedback in that case.
        if self._prediction_count == 0 or not self._feedback_since_prediction:
            self._ingest(observation)
        active_observation = self._latest_observation
        if active_observation is None:
            raise RuntimeError("no observation available for DreamZero inference")
        encoded = self._encoded_observation(active_observation)
        if self._forward_fn is None:
            result, video = self.policy.lazy_joint_forward_causal(self._make_batch(encoded))
        else:
            result, video = self._forward_fn(encoded)
        self._prediction_count += 1
        self._feedback_since_prediction = False
        response = {
            "action": self.schema.decode_action(result.act),
            "mode": self.mode,
        }
        if request.get("save_visualization"):
            if hasattr(video, "detach"):
                video = video.detach().cpu().numpy()
            response["video"] = video
        return response
