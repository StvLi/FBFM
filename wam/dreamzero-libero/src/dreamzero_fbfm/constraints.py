"""Aligned state/action constraint storage for one DreamZero chunk."""

from __future__ import annotations

import json
import threading
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class ConstraintMode(str, Enum):
    """Modes share one rollout and solver path and differ only by masks."""

    NONE = "NONE"
    RTC = "RTC"
    FBFM = "FBFM"

    @classmethod
    def parse(cls, value: str | "ConstraintMode") -> "ConstraintMode":
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().upper())


class ActionNormalizer:
    """Map physical LIBERO actions to DreamZero's padded normalized coordinates."""

    def __init__(self, q01: Tensor, q99: Tensor, model_dim: int = 32) -> None:
        if q01.ndim != 1 or q01.shape != q99.shape:
            raise ValueError("q01 and q99 must be equal-length vectors")
        if not q01.numel() <= model_dim:
            raise ValueError("environment action dimension exceeds model dimension")
        self.q01 = q01.detach().float().cpu()
        self.q99 = q99.detach().float().cpu()
        self.environment_dim = int(q01.numel())
        self.model_dim = int(model_dim)

    @classmethod
    def from_metadata(
        cls, metadata_path: str | Path, *, embodiment: str = "libero_sim", model_dim: int = 32
    ) -> "ActionNormalizer":
        with Path(metadata_path).open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        try:
            stats = metadata[embodiment]["statistics"]["action"]["actions"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing {embodiment} action statistics") from error
        return cls(
            torch.as_tensor(stats["q01"], dtype=torch.float32),
            torch.as_tensor(stats["q99"], dtype=torch.float32),
            model_dim=model_dim,
        )

    def normalize(self, actions: np.ndarray | Tensor) -> Tensor:
        value = torch.as_tensor(actions, dtype=torch.float32)
        if value.shape[-1] != self.environment_dim:
            raise ValueError(
                f"Expected physical action width {self.environment_dim}, got {value.shape}"
            )
        q01 = self.q01.to(value.device)
        q99 = self.q99.to(value.device)
        span = q99 - q01
        safe_span = torch.where(span == 0, torch.ones_like(span), span)
        normalized = 2 * (value - q01) / safe_span - 1
        normalized = torch.where(span == 0, value, normalized).clamp(-1, 1)
        padded = torch.zeros(*value.shape[:-1], self.model_dim, dtype=value.dtype, device=value.device)
        padded[..., : self.environment_dim] = normalized
        return padded


class ChunkConstraints:
    """Thread-safe targets and monotonically versioned state-mask updates."""

    def __init__(
        self,
        *,
        mode: ConstraintMode | str,
        action_targets: Tensor,
        action_mask: Tensor,
        state_targets: Tensor,
        state_mask: Tensor,
    ) -> None:
        if action_targets.shape != action_mask.shape:
            raise ValueError("action target/mask shape mismatch")
        if state_targets.ndim != 5 or state_mask.shape != (
            state_targets.shape[0],
            1,
            state_targets.shape[2],
            1,
            1,
        ):
            raise ValueError("state mask must be (B,1,F,1,1)")
        self.mode = ConstraintMode.parse(mode)
        self._lock = threading.RLock()
        self._closed = False
        self._version = 0
        self._action_targets = action_targets.detach().clone()
        self._action_mask = action_mask.detach().clone()
        self._state_targets = state_targets.detach().clone()
        self._state_mask = state_mask.detach().clone()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def update_state_slot(self, slot: int, latent: Tensor) -> bool:
        with self._lock:
            if self._closed or not 0 <= slot < self._state_targets.shape[2]:
                return False
            if self._state_mask[:, :, slot].gt(0).any():
                return False
            target = latent.detach().to(
                device=self._state_targets.device, dtype=self._state_targets.dtype
            )
            target = target.reshape_as(self._state_targets[:, :, slot])
            self._state_targets[:, :, slot].copy_(target)
            self._state_mask[:, :, slot] = 1
            self._version += 1
            return True

    def snapshot(self) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        with self._lock:
            state_mask = self._state_mask.clone()
            action_mask = self._action_mask.clone()
            if self.mode is not ConstraintMode.FBFM:
                state_mask.zero_()
            if self.mode is ConstraintMode.NONE:
                action_mask.zero_()
            return (
                self._state_targets.clone(),
                state_mask,
                self._action_targets.clone(),
                action_mask,
                self._version,
            )
