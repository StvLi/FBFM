"""LIBERO observation conversion shared by snapshot and rollout processes."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def quaternion_xyzw_to_axis_angle(quaternion: Any) -> np.ndarray:
    """Convert a LIBERO xyzw quaternion to a three-dimensional axis angle."""
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    if not np.isfinite(quat).all():
        raise ValueError("Quaternion contains a non-finite value")

    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0, abs_tol=1e-8):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(quat[3]) / denominator)).astype(
        np.float32
    )


def extract_libero_observation(observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract the two rotated camera images and 8D state used by DreamZero."""
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"LIBERO observation is missing required keys: {missing}")

    main_image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    for name, image in (("main_image", main_image), ("wrist_image", wrist_image)):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [H,W,3], got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {image.dtype}")

    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            quaternion_xyzw_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    if state.shape != (8,):
        raise ValueError(f"DreamZero LIBERO state must have shape (8,), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("DreamZero LIBERO state contains a non-finite value")

    return {
        "main_image": main_image,
        "wrist_image": wrist_image,
        "state": state,
    }


def as_model_batch(
    observation: Dict[str, np.ndarray], task_description: str
) -> Dict[str, Any]:
    """Add a batch dimension and RLinf rollout field names."""
    return {
        "main_images": observation["main_image"][None, ...],
        "wrist_images": observation["wrist_image"][None, ...],
        "states": observation["state"][None, ...],
        "task_descriptions": [str(task_description)],
    }
