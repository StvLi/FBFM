"""Slot-aligned visual state feedback for Wan2.2 flow matching."""

from __future__ import annotations

import heapq
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch
from torch import Tensor
from wan.modules.vae2_2 import count_conv3d, patchify

# This pipeline has only visual-state residuals. The 56/9600 coefficient used
# by DreamZero balances state and action dimensions and does not apply here.
DEFAULT_STATE_WEIGHT = 1.0
DEFAULT_KP = 1.0


class FeedbackMode(str, Enum):
    """Paired inference modes that share one video denoising loop."""

    DIRECT = "DIRECT"
    FBFM = "FBFM"

    @classmethod
    def parse(cls, value: str | FeedbackMode) -> FeedbackMode:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().upper())


@dataclass(frozen=True)
class FeedbackEvent:
    """Four causal raw frames released for one future latent slot."""

    release_step: int
    slot: int
    frames: Tensor

    def __post_init__(self) -> None:
        if self.release_step < 0:
            raise ValueError("release_step must be non-negative")
        if self.slot <= 0:
            raise ValueError("feedback slots must follow the native I2V anchor")
        if self.frames.ndim != 4:
            raise ValueError("feedback frames must have shape (C,T,H,W)")


class FeedbackEncoder(Protocol):
    observations_per_slot: int

    def prime(self, anchor: Tensor) -> Tensor: ...

    def encode(self, frames: Tensor) -> Tensor: ...


class NativeWanStreamingEncoder:
    """Run the native Wan2.2 VAE causally with an independent feature cache.

    The official VAE clears its cache around each complete-video encode. This
    wrapper keeps a separate encoder cache, so solver-time feedback neither
    resets nor reuses the native I2V conditioning cache.
    """

    observations_per_slot = 4

    def __init__(self, vae: Any) -> None:
        self.vae = vae
        self.model = vae.model
        self._lock = threading.RLock()
        self.clear()

    def clear(self) -> None:
        with self._lock:
            self._feature_cache = [None] * count_conv3d(self.model.encoder)
            self._primed = False

    def prime(self, anchor: Tensor) -> Tensor:
        if anchor.ndim != 4 or anchor.shape[1] != 1:
            raise ValueError("feedback anchor must have shape (C,1,H,W)")
        with self._lock:
            if self._primed:
                raise RuntimeError("feedback encoder is already primed")
            latent = self._encode_chunk(anchor)
            self._primed = True
            return latent

    def encode(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4 or frames.shape[1] != self.observations_per_slot:
            raise ValueError(
                f"feedback chunk must have shape (C,{self.observations_per_slot},H,W)"
            )
        with self._lock:
            if not self._primed:
                raise RuntimeError("prime the feedback encoder before encoding frames")
            latent = self._encode_chunk(frames)
        if latent.shape[1] != 1:
            raise RuntimeError(
                "one feedback frame group must produce exactly one latent slot, "
                f"got {tuple(latent.shape)}"
            )
        return latent

    @torch.no_grad()
    def _encode_chunk(self, video: Tensor) -> Tensor:
        value = video.unsqueeze(0).to(
            device=self.vae.device, dtype=self.model.encoder.conv1.weight.dtype
        )
        device_type = value.device.type
        with torch.amp.autocast(device_type, enabled=False):
            value = patchify(value, patch_size=2)
            feature_index = [0]
            encoded = self.model.encoder(
                value,
                feat_cache=self._feature_cache,
                feat_idx=feature_index,
            )
            mean, _ = self.model.conv1(encoded).chunk(2, dim=1)
            scale_mean, scale_inverse_std = self.vae.scale
            mean = mean - scale_mean.view(1, self.model.z_dim, 1, 1, 1)
            mean = mean * scale_inverse_std.view(1, self.model.z_dim, 1, 1, 1)
        return mean.float().squeeze(0)


class StateSlotConstraints:
    """Thread-safe visual latent targets for one generation chunk."""

    def __init__(self, template: Tensor) -> None:
        if template.ndim != 4:
            raise ValueError("latent template must have shape (C,F,H,W)")
        self._lock = threading.RLock()
        self._targets = torch.zeros_like(template)
        self._mask = torch.zeros(
            1,
            template.shape[1],
            1,
            1,
            device=template.device,
            dtype=template.dtype,
        )
        self._version = 0
        self._closed = False

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def update(self, slot: int, latent: Tensor) -> bool:
        with self._lock:
            if self._closed or not 0 <= slot < self._targets.shape[1]:
                return False
            if self._mask[:, slot].gt(0).any():
                return False
            target = latent.detach().to(
                device=self._targets.device, dtype=self._targets.dtype
            )
            target = target.reshape_as(self._targets[:, slot : slot + 1])
            self._targets[:, slot : slot + 1].copy_(target)
            self._mask[:, slot] = 1
            self._version += 1
            return True

    def snapshot(self, mode: FeedbackMode | str) -> tuple[Tensor, Tensor, int]:
        with self._lock:
            mask = self._mask.clone()
            if FeedbackMode.parse(mode) is FeedbackMode.DIRECT:
                mask.zero_()
            return self._targets.clone(), mask, self._version


@dataclass(frozen=True)
class StateFeedbackResult:
    velocity: Tensor
    diagnostics: dict[str, float | int | bool]


def guidance_weight(sigma: Tensor | float, beta: float) -> Tensor:
    """LingBot/FBFM few-step guidance schedule in flow sigma coordinates."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    sigma_value = torch.as_tensor(sigma, dtype=torch.float32)
    tau = 1 - sigma_value
    r_squared = sigma_value.square() / (tau.square() + sigma_value.square())
    raw = sigma_value / (tau * r_squared)
    return torch.nan_to_num(raw, nan=0.0, posinf=beta, neginf=0.0).clamp(0, beta)


def endpoint_state_guidance(
    *,
    sample: Tensor,
    velocity: Tensor,
    target: Tensor,
    mask: Tensor,
    sigma: Tensor | float,
    beta: float = 10.0,
    state_weight: float = 1.0,
    kp: float = DEFAULT_KP,
) -> StateFeedbackResult:
    """Apply a visual-only endpoint VJP without any action-flow variables."""
    if sample.shape != velocity.shape or sample.shape != target.shape:
        raise ValueError("sample, velocity, and target shapes must match")
    if mask.shape != (1, sample.shape[1], 1, 1):
        raise ValueError("state mask must have shape (1,F,1,1)")
    if not 0 < state_weight <= 1:
        raise ValueError("state_weight must be in (0,1]")
    if kp <= 0:
        raise ValueError("kp must be positive")
    if not sample.requires_grad:
        raise ValueError("guided sample must require gradients")

    mask_nonzero = int(torch.count_nonzero(mask).item())
    if mask_nonzero == 0:
        return StateFeedbackResult(
            velocity.detach(),
            {
                "guided": False,
                "mask_nonzero": 0,
                "error_norm": 0.0,
                "correction_norm": 0.0,
                "guidance_weight": 0.0,
            },
        )

    sigma_value = torch.as_tensor(sigma, device=sample.device, dtype=sample.dtype)
    endpoint = sample - sigma_value * velocity
    error = (target - endpoint) * mask * state_weight
    correction = torch.autograd.grad(
        endpoint,
        sample,
        grad_outputs=error.detach(),
        retain_graph=False,
        create_graph=False,
    )[0]
    weight = guidance_weight(sigma, beta).to(device=sample.device, dtype=sample.dtype)
    guided_velocity = (velocity - kp * weight * correction).detach()
    if not torch.isfinite(guided_velocity).all():
        raise FloatingPointError("FBFM produced a non-finite video velocity")
    return StateFeedbackResult(
        guided_velocity,
        {
            "guided": True,
            "mask_nonzero": mask_nonzero,
            "error_norm": float(error.detach().float().norm().item()),
            "correction_norm": float(correction.detach().float().norm().item()),
            "guidance_weight": float(weight.detach().float().item()),
            "kp": float(kp),
            "base_velocity_norm": float(velocity.detach().float().norm().item()),
            "guided_velocity_norm": float(guided_velocity.float().norm().item()),
        },
    )


def masked_endpoint_error(
    *,
    sample: Tensor,
    velocity: Tensor,
    target: Tensor,
    mask: Tensor,
    sigma: Tensor | float,
    state_weight: float = 1.0,
) -> Tensor:
    """Build the detached endpoint residual used as the VJP cotangent."""
    if sample.shape != velocity.shape or sample.shape != target.shape:
        raise ValueError("sample, velocity, and target shapes must match")
    if mask.shape != (1, sample.shape[1], 1, 1):
        raise ValueError("state mask must have shape (1,F,1,1)")
    if not 0 < state_weight <= 1:
        raise ValueError("state_weight must be in (0,1]")

    sigma_value = torch.as_tensor(sigma, device=sample.device, dtype=sample.dtype)
    endpoint = sample.detach() - sigma_value * velocity.detach()
    return ((target - endpoint) * mask * state_weight).detach()


def endpoint_state_guidance_from_cfg_vjps(
    *,
    velocity: Tensor,
    error: Tensor,
    mask: Tensor,
    conditional_vjp: Tensor,
    unconditional_vjp: Tensor,
    sigma: Tensor | float,
    guide_scale: float,
    beta: float = 10.0,
    kp: float = DEFAULT_KP,
) -> StateFeedbackResult:
    """Apply endpoint guidance from separately evaluated CFG branch VJPs.

    Splitting the two VJPs is algebraically equivalent to differentiating the
    combined CFG velocity, but only one DiT backward graph needs to be resident
    at a time.
    """
    if not (
        velocity.shape
        == error.shape
        == conditional_vjp.shape
        == unconditional_vjp.shape
    ):
        raise ValueError("velocity, error, and CFG VJP shapes must match")
    if mask.shape != (1, velocity.shape[1], 1, 1):
        raise ValueError("state mask must have shape (1,F,1,1)")
    if kp <= 0:
        raise ValueError("kp must be positive")

    sigma_value = torch.as_tensor(sigma, device=velocity.device, dtype=velocity.dtype)
    cfg_vjp = guide_scale * conditional_vjp + (1.0 - guide_scale) * unconditional_vjp
    correction = error - sigma_value * cfg_vjp
    weight = guidance_weight(sigma, beta).to(
        device=velocity.device, dtype=velocity.dtype
    )
    guided_velocity = (velocity.detach() - kp * weight * correction).detach()
    if not torch.isfinite(guided_velocity).all():
        raise FloatingPointError("FBFM produced a non-finite video velocity")
    return StateFeedbackResult(
        guided_velocity,
        {
            "guided": True,
            "mask_nonzero": int(torch.count_nonzero(mask).item()),
            "error_norm": float(error.float().norm().item()),
            "correction_norm": float(correction.detach().float().norm().item()),
            "guidance_weight": float(weight.detach().float().item()),
            "kp": float(kp),
            "base_velocity_norm": float(velocity.detach().float().norm().item()),
            "guided_velocity_norm": float(guided_velocity.float().norm().item()),
        },
    )


class StateFeedbackController:
    """Release raw observations and refresh state constraints at solver boundaries."""

    def __init__(
        self,
        *,
        mode: FeedbackMode | str,
        encoder: FeedbackEncoder,
        anchor: Tensor,
        latent_template: Tensor,
        events: Sequence[FeedbackEvent],
        beta: float = 10.0,
        state_weight: float = DEFAULT_STATE_WEIGHT,
        kp: float = DEFAULT_KP,
    ) -> None:
        self.mode = FeedbackMode.parse(mode)
        self.encoder = encoder
        self.constraints = StateSlotConstraints(latent_template)
        self.beta = float(beta)
        self.state_weight = float(state_weight)
        if kp <= 0:
            raise ValueError("kp must be positive")
        self.kp = float(kp)
        self.audit: list[dict[str, Any]] = []
        self._queue: list[tuple[int, int, FeedbackEvent]] = []
        seen_slots: set[int] = set()
        for sequence, event in enumerate(events):
            if event.slot in seen_slots:
                raise ValueError(f"duplicate feedback slot {event.slot}")
            if event.slot >= latent_template.shape[1]:
                raise ValueError(
                    f"feedback slot {event.slot} exceeds latent horizon "
                    f"{latent_template.shape[1]}"
                )
            if event.frames.shape[1] != encoder.observations_per_slot:
                raise ValueError(
                    f"slot {event.slot} requires {encoder.observations_per_slot} frames"
                )
            seen_slots.add(event.slot)
            heapq.heappush(self._queue, (event.release_step, sequence, event))
        if self.mode is FeedbackMode.FBFM:
            anchor_latent = self.encoder.prime(anchor)
            self.audit.append(
                {
                    "event": "feedback_prime",
                    "anchor_latent_shape": list(anchor_latent.shape),
                }
            )

    def advance(self, solver_step: int) -> int:
        """Encode every observation event visible at this solver boundary."""
        if self.mode is FeedbackMode.DIRECT:
            return 0
        updated = 0
        while self._queue and self._queue[0][0] <= solver_step:
            _, _, event = heapq.heappop(self._queue)
            latent = self.encoder.encode(event.frames)
            accepted = self.constraints.update(event.slot, latent)
            if not accepted:
                raise RuntimeError(f"feedback slot {event.slot} was rejected")
            updated += 1
            self.audit.append(
                {
                    "event": "feedback_update",
                    "solver_step": solver_step,
                    "release_step": event.release_step,
                    "slot": event.slot,
                    "constraint_version": self.constraints.version,
                }
            )
        return updated

    def guide(
        self,
        *,
        solver_step: int,
        sample: Tensor,
        velocity: Tensor,
        sigma: Tensor | float,
    ) -> StateFeedbackResult:
        target, mask, version = self.constraints.snapshot(self.mode)
        result = endpoint_state_guidance(
            sample=sample,
            velocity=velocity,
            target=target,
            mask=mask,
            sigma=sigma,
            beta=self.beta,
            state_weight=self.state_weight,
            kp=self.kp,
        )
        self.audit.append(
            {
                "event": "solver_guidance",
                "solver_step": solver_step,
                "sigma": float(torch.as_tensor(sigma).item()),
                "constraint_version": version,
                **result.diagnostics,
            }
        )
        return result


def evenly_spaced_release_steps(slot_count: int, solver_steps: int) -> list[int]:
    """Build a deterministic pseudo-clock schedule strictly inside the solve."""
    if slot_count < 0:
        raise ValueError("slot_count must be non-negative")
    if solver_steps <= 1 and slot_count:
        raise ValueError("at least two solver steps are required for feedback")
    if slot_count >= solver_steps:
        raise ValueError("feedback slots must be fewer than solver steps")
    return [
        max(1, math.floor((index + 1) * solver_steps / (slot_count + 1)))
        for index in range(slot_count)
    ]
