"""Real-Time Chunking (RTC) utilities for action chunking.

This is adapted from LeRobot's RTC implementation and provides a lightweight
processor for diffusion-based action prediction. It is intentionally self-
contained to keep Lingbot-VA dependencies minimal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class RTCAttentionSchedule(str, Enum):
    """Supported prefix attention schedules for RTC guidance."""

    ZEROS = "zeros"
    ONES = "ones"
    LINEAR = "linear"
    EXP = "exp"

    @classmethod
    def from_value(cls, value: str | "RTCAttentionSchedule") -> "RTCAttentionSchedule":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


@dataclass
class RTCConfig:
    """Configuration for RTC guidance in action diffusion."""

    enabled: bool = False
    execution_horizon: int = 10
    max_guidance_weight: float = 5.0
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies."""

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int,
        time: float | Tensor,
        original_denoise_step_partial,
        execution_horizon: int | None = None,
    ) -> Tensor:
        """Apply RTC prefix guidance around an existing denoiser.

        Args:
            x_t: Current noisy actions with shape ``(B, T, A)``.
            prev_chunk_left_over: Unexecuted prefix from previous chunk,
                also shape ``(B, T, A)``. If ``None``, guidance is skipped.
            inference_delay: Number of timesteps from the prefix to use.
            time: Normalized diffusion time in [0, 1].
            original_denoise_step_partial: Callable that returns the base
                denoised velocity given ``x_t``.
            execution_horizon: Horizon used to build prefix weights.
        """

        # In the original implementation, time runs from 0 -> 1.
        # In our diffusion loop, we typically step from 1 -> 0.
        tau = 1 - time

        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)

        x_t = x_t.clone().detach()

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        if execution_horizon > prev_chunk_left_over.shape[1]:
            execution_horizon = prev_chunk_left_over.shape[1]

        batch_size, action_chunk_size, action_dim = x_t.shape

        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = torch.zeros(batch_size, action_chunk_size, action_dim, device=x_t.device)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
            .to(x_t.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)

            x1_t = x_t - time * v_t
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
        tau_tensor = torch.as_tensor(tau)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / squared_one_minus_tau
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        return v_t - guidance_weight * correction

    def get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = min(start, end)
        schedule = RTCAttentionSchedule.from_value(self.rtc_config.prefix_attention_schedule)

        if schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif schedule == RTCAttentionSchedule.LINEAR:
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif schedule == RTCAttentionSchedule.EXP:
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        else:
            raise ValueError(f"Unsupported RTC attention schedule: {schedule}")

        return weights

    def _linweights(self, start: int, end: int, total: int) -> Tensor:
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights: Tensor, total: int, end: int) -> Tensor:
        zeros_len = total - end

        if zeros_len <= 0:
            return weights

        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights: Tensor, start: int, total: int) -> Tensor:
        ones_len = min(start, total)

        if ones_len <= 0:
            return weights

        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])
