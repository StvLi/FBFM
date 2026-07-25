"""Joint endpoint-VJP guidance from the FBFM parallel-WAM formulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EndpointLinearization:
    """Autograd graph for one DiT clean-endpoint Jacobian."""

    video_sample: Tensor
    action_sample: Tensor
    video_endpoint: Tensor
    action_endpoint: Tensor


@dataclass(frozen=True)
class GuidanceResult:
    video_velocity: Tensor
    action_velocity: Tensor
    diagnostics: dict[str, float | int | bool]
    linearization: EndpointLinearization | None = None


def guidance_weight(sigma: Tensor | float, beta: float) -> Tensor:
    """Return the clipped few-step schedule in sigma coordinates."""
    sigma_value = torch.as_tensor(sigma, dtype=torch.float32)
    tau = 1 - sigma_value
    r_squared = sigma_value.square() / (tau.square() + sigma_value.square())
    raw = sigma_value / (tau * r_squared)
    return torch.nan_to_num(raw, nan=0.0, posinf=beta, neginf=0.0).clamp(0, beta)


def joint_fbfm_guidance(
    *,
    video_sample: Tensor,
    action_sample: Tensor,
    video_velocity: Tensor,
    action_velocity: Tensor,
    video_target: Tensor,
    video_mask: Tensor,
    action_target: Tensor,
    action_mask: Tensor,
    sigma: Tensor | float,
    beta: float = 10.0,
    decompose_vjp: bool = False,
    linearization: EndpointLinearization | None = None,
    cache_linearization: bool = False,
) -> GuidanceResult:
    """Apply guidance with a fresh or cached clean-endpoint Jacobian.

    When ``linearization`` is supplied, the residual is recomputed from the
    current sample, velocity, sigma, target, and mask. Only the VJP Jacobian is
    reused from the earlier DiT evaluation.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if video_sample.shape != video_velocity.shape or action_sample.shape != action_velocity.shape:
        raise ValueError("sample/velocity shape mismatch")
    if video_target.shape != video_sample.shape or action_target.shape != action_sample.shape:
        raise ValueError("constraint target shape mismatch")

    has_state = bool(torch.any(video_mask).item())
    has_action = bool(torch.any(action_mask).item())
    if not has_state and not has_action:
        return GuidanceResult(
            video_velocity.detach(),
            action_velocity.detach(),
            {
                "guided": False,
                "state_mask_nonzero": 0,
                "action_mask_nonzero": 0,
                "state_error_norm": 0.0,
                "action_error_norm": 0.0,
                "video_correction_norm": 0.0,
                "action_correction_norm": 0.0,
                "guidance_weight": 0.0,
            },
        )

    sigma_value = torch.as_tensor(
        sigma, device=video_sample.device, dtype=video_sample.dtype
    )
    video_endpoint = video_sample - sigma_value * video_velocity
    action_endpoint = action_sample - sigma_value * action_velocity
    state_error = (video_target - video_endpoint) * video_mask
    action_error = (action_target - action_endpoint) * action_mask
    if linearization is None:
        active_linearization = EndpointLinearization(
            video_sample=video_sample,
            action_sample=action_sample,
            video_endpoint=video_endpoint,
            action_endpoint=action_endpoint,
        )
    else:
        if (
            linearization.video_sample.shape != video_sample.shape
            or linearization.action_sample.shape != action_sample.shape
        ):
            raise ValueError("cached endpoint linearization shape mismatch")
        active_linearization = linearization
    keep_graph = cache_linearization or linearization is not None
    component_diagnostics: dict[str, float] = {}
    if decompose_vjp:
        state_corrections = torch.autograd.grad(
            outputs=active_linearization.video_endpoint,
            inputs=(
                active_linearization.video_sample,
                active_linearization.action_sample,
            ),
            grad_outputs=state_error.detach(),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        action_corrections = torch.autograd.grad(
            outputs=active_linearization.action_endpoint,
            inputs=(
                active_linearization.video_sample,
                active_linearization.action_sample,
            ),
            grad_outputs=action_error.detach(),
            retain_graph=keep_graph,
            create_graph=False,
            allow_unused=True,
        )
        state_video = (
            torch.zeros_like(video_sample)
            if state_corrections[0] is None
            else state_corrections[0]
        )
        state_action = (
            torch.zeros_like(action_sample)
            if state_corrections[1] is None
            else state_corrections[1]
        )
        action_video = (
            torch.zeros_like(video_sample)
            if action_corrections[0] is None
            else action_corrections[0]
        )
        action_action = (
            torch.zeros_like(action_sample)
            if action_corrections[1] is None
            else action_corrections[1]
        )
        video_correction = state_video + action_video
        action_correction = state_action + action_action
        component_diagnostics = {
            "state_to_video_correction_norm": float(
                state_video.detach().float().norm().item()
            ),
            "state_to_action_correction_norm": float(
                state_action.detach().float().norm().item()
            ),
            "action_to_video_correction_norm": float(
                action_video.detach().float().norm().item()
            ),
            "action_to_action_correction_norm": float(
                action_action.detach().float().norm().item()
            ),
        }
    else:
        video_correction, action_correction = torch.autograd.grad(
            outputs=(
                active_linearization.video_endpoint,
                active_linearization.action_endpoint,
            ),
            inputs=(
                active_linearization.video_sample,
                active_linearization.action_sample,
            ),
            grad_outputs=(state_error.detach(), action_error.detach()),
            retain_graph=keep_graph,
            create_graph=False,
            allow_unused=False,
        )
    weight = guidance_weight(sigma, beta).to(video_sample.device)
    guided_video = (video_velocity - weight * video_correction).detach()
    guided_action = (action_velocity - weight * action_correction).detach()
    if not torch.isfinite(guided_video).all() or not torch.isfinite(guided_action).all():
        raise FloatingPointError("FBFM produced a non-finite guided velocity")
    state_coordinates = int(
        torch.count_nonzero(video_mask.expand_as(video_sample)).item()
    )
    action_coordinates = int(
        torch.count_nonzero(action_mask.expand_as(action_sample)).item()
    )
    state_scale = state_coordinates**0.5 if state_coordinates else 1.0
    action_scale = action_coordinates**0.5 if action_coordinates else 1.0
    return GuidanceResult(
        guided_video,
        guided_action,
        {
            "guided": True,
            "state_mask_nonzero": int(torch.count_nonzero(video_mask).item()),
            "action_mask_nonzero": int(torch.count_nonzero(action_mask).item()),
            "state_mask_coordinate_count": state_coordinates,
            "action_mask_coordinate_count": action_coordinates,
            "state_error_norm": float(state_error.detach().float().norm().item()),
            "action_error_norm": float(action_error.detach().float().norm().item()),
            "state_error_rms": float(state_error.detach().float().norm().item() / state_scale),
            "action_error_rms": float(action_error.detach().float().norm().item() / action_scale),
            "base_video_velocity_norm": float(video_velocity.detach().float().norm().item()),
            "base_action_velocity_norm": float(action_velocity.detach().float().norm().item()),
            "guided_video_velocity_norm": float(guided_video.float().norm().item()),
            "guided_action_velocity_norm": float(guided_action.float().norm().item()),
            "video_correction_norm": float(video_correction.detach().float().norm().item()),
            "action_correction_norm": float(action_correction.detach().float().norm().item()),
            "guidance_weight": float(weight.item()),
            "vjp_decomposed": decompose_vjp,
            "jacobian_reused": linearization is not None,
            **component_diagnostics,
        },
        active_linearization if keep_graph else None,
    )
