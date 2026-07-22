"""Canonical RoboTwin observation/action representation used by DreamZero.

RoboTwin exposes joint positions and end-effector poses at the same time.  The
DreamZero policy in this integration predicts absolute end-effector targets,
so feeding the 14-D joint vector as if it were an EEF vector silently corrupts
relative-action normalization.  This module is the single conversion point
shared by dataset conversion and simulator evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


CAMERA_SOURCES = (
    "head_camera",
    "left_camera",
    "right_camera",
)
CAMERA_TARGETS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _pose_array(value: Any, *, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape[-1] != 7:
        raise ValueError(f"{name} must end in XYZ+quaternion(xyzw), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} contains non-finite values")
    quaternion = pose[..., 3:7]
    norm = np.linalg.norm(quaternion, axis=-1)
    if np.any(norm < 1e-8):
        raise ValueError(f"{name} contains a zero quaternion")
    pose = pose.copy()
    pose[..., 3:7] = quaternion / norm[..., None]
    return pose


def pack_eef14(
    left_pose: Any,
    left_gripper: Any,
    right_pose: Any,
    right_gripper: Any,
) -> np.ndarray:
    """Pack RoboTwin XYZ+quaternion poses as XYZ+Euler-XYZ+gripper (14-D)."""

    left = _pose_array(left_pose, name="left_pose")
    right = _pose_array(right_pose, name="right_pose")
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError(f"left/right pose shape mismatch: {left.shape} vs {right.shape}")
    leading = left.shape[:-1]
    left_grip = np.asarray(left_gripper, dtype=np.float64)
    right_grip = np.asarray(right_gripper, dtype=np.float64)
    if left_grip.shape != leading or right_grip.shape != leading:
        raise ValueError(
            "gripper shape must match pose leading dimensions: "
            f"poses={leading}, left={left_grip.shape}, right={right_grip.shape}"
        )
    left_euler = quaternion_xyzw_to_euler_xyz(left[..., 3:7])
    right_euler = quaternion_xyzw_to_euler_xyz(right[..., 3:7])
    packed = np.concatenate(
        (
            left[..., :3],
            left_euler,
            left_grip[..., None],
            right[..., :3],
            right_euler,
            right_grip[..., None],
        ),
        axis=-1,
    )
    return packed.astype(np.float32)


def quaternion_xyzw_to_euler_xyz(quaternion: Any) -> np.ndarray:
    """Convert normalized or unnormalized XYZW quaternions to XYZ Euler angles."""

    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError(f"quaternion must end in four values, got {values.shape}")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("quaternion contains a zero quaternion")
    x, y, z, w = np.moveaxis(values / norm, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def euler_xyz_to_quaternion_xyzw(euler: Any) -> np.ndarray:
    """Convert XYZ Euler angles to normalized XYZW quaternions."""

    values = np.asarray(euler, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(f"euler angles must end in three values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("euler angles contain non-finite values")
    roll, pitch, yaw = np.moveaxis(values * 0.5, -1, 0)
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    quaternion = np.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


def eef14_to_simulator_action(action: Any) -> np.ndarray:
    """Convert absolute dual EEF XYZ/Euler/gripper targets to RoboTwin EE16."""

    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (14,) or not np.isfinite(values).all():
        raise ValueError("DreamZero RoboTwin action must be one finite 14-D vector")
    return np.concatenate(
        (
            values[:3],
            euler_xyz_to_quaternion_xyzw(values[3:6]),
            values[6:7],
            values[7:10],
            euler_xyz_to_quaternion_xyzw(values[10:13]),
            values[13:14],
        )
    )


def state_from_endpose(endpose: Mapping[str, Any]) -> np.ndarray:
    """Convert one simulator ``observation['endpose']`` mapping to 14-D EEF state."""

    required = ("left_endpose", "left_gripper", "right_endpose", "right_gripper")
    missing = [key for key in required if key not in endpose]
    if missing:
        raise KeyError(f"RoboTwin endpose is missing {missing}")
    return pack_eef14(
        endpose["left_endpose"],
        endpose["left_gripper"],
        endpose["right_endpose"],
        endpose["right_gripper"],
    )


def format_simulator_observation(observation: Mapping[str, Any], instruction: str) -> dict[str, Any]:
    """Build the exact websocket observation expected by DreamZero's bridge."""

    if "observation" not in observation or "endpose" not in observation:
        raise KeyError("RoboTwin observation must contain camera observations and endpose")
    cameras = observation["observation"]
    result: dict[str, Any] = {
        "observation.state": state_from_endpose(observation["endpose"]),
        "task": str(instruction),
    }
    for source, target in zip(CAMERA_SOURCES, CAMERA_TARGETS):
        if source not in cameras or "rgb" not in cameras[source]:
            raise KeyError(f"RoboTwin observation is missing {source}.rgb")
        frame = np.asarray(cameras[source]["rgb"])
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"{source}.rgb must have shape (H,W,3), got {frame.shape}")
        result[target] = frame
    return result
