"""Inference-time FBFM guidance for DreamZero's joint video/action flow.

The module intentionally does not reimplement UniPC.  It adjusts the flow
prediction that is passed to the unmodified scheduler, preserving all solver
history and corrector bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch
import torch.distributed as dist
from torch import Tensor


class ConstraintMode(str, Enum):
    NONE = "None"
    RTC = "RTC"
    FEEDBACK = "Feedback"

    @classmethod
    def parse(cls, value: str | "ConstraintMode" | None) -> "ConstraintMode":
        if isinstance(value, cls):
            return value
        normalized = "None" if value is None else str(value).strip()
        aliases = {member.value.lower(): member for member in cls}
        try:
            return aliases[normalized.lower()]
        except KeyError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown FBFM constraint mode {value!r}; expected one of {choices}") from exc


@dataclass(frozen=True)
class FBFMGuidanceConfig:
    mode: ConstraintMode = ConstraintMode.NONE
    max_guidance_weight: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ConstraintMode.parse(self.mode))
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")


@dataclass
class JointConstraints:
    """Targets and broadcastable masks in the scheduler sample layouts."""

    video_target: Tensor | None = None
    video_mask: Tensor | None = None
    action_target: Tensor | None = None
    action_mask: Tensor | None = None

    def has_video(self) -> bool:
        return self.video_target is not None and self.video_mask is not None

    def has_action(self) -> bool:
        return self.action_target is not None and self.action_mask is not None

    def empty(self) -> bool:
        return not self.has_video() and not self.has_action()


def prepare_guided_inputs(video_sample: Tensor, action_sample: Tensor) -> tuple[Tensor, Tensor]:
    """Create leaf inputs before the denoiser forward so its Jacobian is retained."""

    return (
        video_sample.detach().clone().requires_grad_(True),
        action_sample.detach().clone().requires_grad_(True),
    )


def prefix_constraint(sample: Tensor, prefix: Tensor | None, *, time_dim: int) -> tuple[Tensor, Tensor] | None:
    """Place a temporal prefix into a full-sized target and return its mask."""

    if prefix is None or prefix.numel() == 0:
        return None
    if sample.ndim != prefix.ndim:
        raise ValueError(f"sample/prefix rank mismatch: {sample.shape} vs {prefix.shape}")
    time_dim = time_dim % sample.ndim
    for dim, (sample_size, prefix_size) in enumerate(zip(sample.shape, prefix.shape)):
        if dim != time_dim and sample_size != prefix_size:
            raise ValueError(
                f"sample/prefix shape mismatch outside time dim {time_dim}: {sample.shape} vs {prefix.shape}"
            )
    count = min(sample.shape[time_dim], prefix.shape[time_dim])
    if count <= 0:
        return None
    target = torch.zeros_like(sample)
    mask = torch.zeros_like(sample)
    sample_slice = [slice(None)] * sample.ndim
    prefix_slice = [slice(None)] * prefix.ndim
    sample_slice[time_dim] = slice(0, count)
    prefix_slice[time_dim] = slice(prefix.shape[time_dim] - count, prefix.shape[time_dim])
    target[tuple(sample_slice)] = prefix.to(device=sample.device, dtype=sample.dtype)[tuple(prefix_slice)]
    mask[tuple(sample_slice)] = 1
    return target, mask


def clean_prediction(sample: Tensor, flow: Tensor, sigma: Tensor | float) -> Tensor:
    """FlowUniPC's flow-prediction convention: x_clean = x_sigma - sigma * v."""

    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=sample.dtype)
    return sample - sigma_tensor * flow


def guidance_weight(sigma: Tensor | float, max_guidance_weight: float, *, like: Tensor) -> Tensor:
    """RTC/FBFM gain expressed directly on DreamZero's sigma time axis."""

    sigma_tensor = torch.as_tensor(sigma, device=like.device, dtype=torch.float32)
    tau = 1.0 - sigma_tensor
    sigma_sq = sigma_tensor.square()
    tau_sq = tau.square()
    c = torch.nan_to_num(sigma_tensor / tau, nan=0.0, posinf=max_guidance_weight)
    inv_r2 = torch.nan_to_num((sigma_sq + tau_sq) / sigma_sq, nan=0.0, posinf=max_guidance_weight)
    weight = torch.nan_to_num(c * inv_r2, nan=0.0, posinf=max_guidance_weight)
    return weight.clamp(max=max_guidance_weight).to(dtype=like.dtype)


def _validated_error(target: Tensor, prediction: Tensor, mask: Tensor) -> Tensor:
    if target.shape != prediction.shape:
        raise ValueError(f"constraint target shape {target.shape} != clean prediction shape {prediction.shape}")
    try:
        broadcast_mask = torch.broadcast_to(mask, prediction.shape)
    except RuntimeError as exc:
        raise ValueError(f"constraint mask {mask.shape} is not broadcastable to {prediction.shape}") from exc
    return ((target.to(prediction) - prediction) * broadcast_mask.to(prediction)).detach()


def apply_joint_guidance(
    *,
    video_sample: Tensor,
    action_sample: Tensor,
    video_flow: Tensor,
    action_flow: Tensor,
    video_local_flow: Tensor,
    action_local_flow: Tensor,
    video_sigma: Tensor | float,
    action_sigma: Tensor | float,
    constraints: JointConstraints,
    config: FBFMGuidanceConfig,
    world_size: int = 1,
    reduce_correction: Callable[[Tensor], None] | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply a joint VJP correction while preserving the original UniPC solver.

    ``video_flow`` and ``action_flow`` are the globally assembled numerical CFG
    outputs.  ``*_local_flow`` contains only the differentiable contribution of
    the current rank.  For two-rank CFG each rank owns one branch; splitting the
    identity term by ``world_size`` and summing the corrections reconstructs the
    VJP of the global clean prediction without differentiating through P2P.
    """

    if config.mode is ConstraintMode.NONE or constraints.empty():
        return video_flow, action_flow
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not video_sample.requires_grad or not action_sample.requires_grad:
        raise RuntimeError("guided inputs must require gradients before the denoiser forward")

    global_video_clean = clean_prediction(video_sample, video_flow, video_sigma)
    global_action_clean = clean_prediction(action_sample, action_flow, action_sigma)
    local_video_clean = video_sample / world_size - torch.as_tensor(
        video_sigma, device=video_sample.device, dtype=video_sample.dtype
    ) * video_local_flow
    local_action_clean = action_sample / world_size - torch.as_tensor(
        action_sigma, device=action_sample.device, dtype=action_sample.dtype
    ) * action_local_flow

    outputs: list[Tensor] = []
    grad_outputs: list[Tensor] = []
    if config.mode is ConstraintMode.FEEDBACK and constraints.has_video():
        video_error = _validated_error(constraints.video_target, global_video_clean, constraints.video_mask)
        outputs.append(local_video_clean)
        grad_outputs.append(video_error * guidance_weight(video_sigma, config.max_guidance_weight, like=video_error))
    if constraints.has_action():
        action_error = _validated_error(constraints.action_target, global_action_clean, constraints.action_mask)
        outputs.append(local_action_clean)
        grad_outputs.append(action_error * guidance_weight(action_sigma, config.max_guidance_weight, like=action_error))
    if not outputs:
        return video_flow, action_flow

    correction_video, correction_action = torch.autograd.grad(
        outputs=outputs,
        inputs=(video_sample, action_sample),
        grad_outputs=grad_outputs,
        retain_graph=False,
        allow_unused=True,
    )
    correction_video = torch.zeros_like(video_sample) if correction_video is None else correction_video
    correction_action = torch.zeros_like(action_sample) if correction_action is None else correction_action

    if world_size > 1:
        reducer = reduce_correction
        if reducer is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed FBFM requires an initialized process group or reduce_correction")
            reducer = lambda tensor: dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reducer(correction_video)
        reducer(correction_action)

    return (
        video_flow - correction_video.to(video_flow),
        action_flow - correction_action.to(action_flow),
    )
