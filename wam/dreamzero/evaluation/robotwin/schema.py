"""Checkpoint-native RoboTwin schema used by the DreamZero bridge.

There is deliberately no AgiBot/DROID fallback.  A RoboTwin post-training
checkpoint must ship this schema and normalization hash, otherwise serving is
rejected before the robot receives an action.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROBOTWIN_CAMERA_ORDER = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass(frozen=True)
class FieldSlice:
    key: str
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class RoboTwinSchema:
    embodiment_tag: str
    camera_order: tuple[str, ...]
    video_keys: tuple[str, ...]
    state_fields: tuple[FieldSlice, ...]
    action_fields: tuple[FieldSlice, ...]
    state_dim: int
    action_dim: int
    action_horizon: int
    execute_steps: int
    frames_per_chunk: int
    normalization_metadata: str
    normalization_sha256: str
    action_representation: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoboTwinSchema":
        schema = cls(
            embodiment_tag=str(data["embodiment_tag"]),
            camera_order=tuple(data["camera_order"]),
            video_keys=tuple(data["video_keys"]),
            state_fields=tuple(FieldSlice(**item) for item in data["state_fields"]),
            action_fields=tuple(FieldSlice(**item) for item in data["action_fields"]),
            state_dim=int(data["state_dim"]),
            action_dim=int(data["action_dim"]),
            action_horizon=int(data["action_horizon"]),
            execute_steps=int(data["execute_steps"]),
            frames_per_chunk=int(data["frames_per_chunk"]),
            normalization_metadata=str(data["normalization_metadata"]),
            normalization_sha256=str(data["normalization_sha256"]),
            action_representation=str(data["action_representation"]),
        )
        schema.validate()
        return schema

    @classmethod
    def load(cls, path: str | Path, *, verify_files: bool = True) -> "RoboTwinSchema":
        path = Path(path)
        schema = cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if verify_files:
            metadata = Path(schema.normalization_metadata)
            if not metadata.is_absolute():
                metadata = path.parent / metadata
            if not metadata.is_file():
                raise FileNotFoundError(f"normalization metadata not found: {metadata}")
            digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
            if digest != schema.normalization_sha256:
                raise ValueError(f"normalization metadata hash mismatch: expected {schema.normalization_sha256}, got {digest}")
        return schema

    def validate(self) -> None:
        if self.embodiment_tag.lower() in {"agibot", "oxe_droid", "droid"}:
            raise ValueError("RoboTwin serving requires a native post-trained embodiment, not AgiBot/DROID normalization")
        if self.camera_order != ROBOTWIN_CAMERA_ORDER:
            raise ValueError(f"camera_order must be {ROBOTWIN_CAMERA_ORDER}, got {self.camera_order}")
        if len(self.video_keys) != len(self.camera_order) or len(set(self.video_keys)) != len(self.video_keys):
            raise ValueError("video_keys must contain one distinct checkpoint key per camera")
        self._validate_cover(self.state_fields, self.state_dim, "state")
        self._validate_cover(self.action_fields, self.action_dim, "action")
        if self.action_horizon <= 0 or not 0 < self.execute_steps <= self.action_horizon:
            raise ValueError("execute_steps must be in [1, action_horizon]")
        if self.frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be positive")
        if self.execute_steps % self.frames_per_chunk:
            raise ValueError("execute_steps must be divisible by frames_per_chunk")
        if len(self.normalization_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.normalization_sha256.lower()):
            raise ValueError("normalization_sha256 must be a SHA-256 hex digest")
        if not self.action_representation.startswith("robotwin_native"):
            raise ValueError("action_representation must explicitly be robotwin_native*")

    @staticmethod
    def _validate_cover(fields: tuple[FieldSlice, ...], dimension: int, label: str) -> None:
        used = []
        for field in fields:
            if not 0 <= field.start < field.stop <= dimension:
                raise ValueError(f"invalid {label} slice {field}")
            used.extend(range(field.start, field.stop))
        if sorted(used) != list(range(dimension)):
            raise ValueError(f"{label} fields must cover every dimension exactly once")

    @property
    def action_tail_steps(self) -> int:
        return self.action_horizon - self.execute_steps

    def encode_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(f"RoboTwin state dim {state.shape[0]} != schema state_dim {self.state_dim}")
        return {field.key: state[field.start : field.stop].reshape(1, field.width) for field in self.state_fields}

    def decode_action(self, action: dict[str, Any]) -> np.ndarray:
        packed = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        for field in self.action_fields:
            if field.key not in action:
                raise KeyError(f"DreamZero output is missing native RoboTwin field {field.key!r}")
            value = action[field.key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float32).reshape(-1, field.width)
            if value.shape[0] < self.action_horizon:
                raise ValueError(f"{field.key} horizon {value.shape[0]} < required {self.action_horizon}")
            packed[:, field.start : field.stop] = value[: self.action_horizon]
        # RoboTwin evaluator convention: (action_dim, chunks, steps).
        return packed[: self.execute_steps].T[:, None, :]
