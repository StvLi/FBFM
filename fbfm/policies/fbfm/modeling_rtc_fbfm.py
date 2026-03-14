#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real-Time Chunking (RTC) implementation for LeRobot.

Based on Physical Intelligence's Kinetix implementation:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214

Based on LeRobot's RTC implementation:
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from fbfm.configs.types import RTCAttentionSchedule
from fbfm.policies.fbfm.configuration_rtc import RTCConfig
from fbfm.policies.fbfm.debug_tracker import Tracker

logger = logging.getLogger(__name__)


@dataclass
class RTCPrevChunk:
    """Container for previous chunk leftovers used by RTC.

    This structure is intended for the extended RTC variant where both:
    - action leftovers (from the previous action chunk), and
    - observed state latents (VAE-encoded images during execution)

    are passed together to guide the next chunk generation.

    It does not change the existing RTC behavior: callers may keep passing
    a plain Tensor as ``prev_chunk_left_over`` to ``denoise_step``.  This
    class is a typed way to bundle future state+action information once the
    call sites are updated.
    """

    action: Tensor | None = None
    """Unexecuted actions from the previous chunk."""

    state: Tensor | None = None
    """Observed state latents (e.g. VAE-encoded image features) during execution."""

    inference_delay: int = 0
    """Number of timesteps of inference delay associated with this leftover."""

    execution_horizon: int | None = None
    """Optional horizon for action prefix weights (falls back to config if None)."""

    state_execution_horizon: int | None = None
    """Optional horizon for state prefix weights (falls back to execution_horizon if None)."""

    def append_state_latent(self, new_latent: Tensor) -> None:
        """Append a newly observed state latent along the time dimension.

        This is intended to be called incrementally while executing a chunk:
        each time a new image is observed and encoded by the VAE encoder, the
        resulting latent can be appended here. Internally we store state
        latents as a 2D tensor of shape ``(T, state_dim)`` where ``T`` is the
        number of collected steps (no batch dimension).

        Args:
            new_latent: Single-step state latent with shape ``(state_dim,)`` or
                ``(1, state_dim)``. A small sequence ``(T_new, state_dim)`` is
                also accepted and concatenated in order.
        """
        if new_latent is None:
            return

        if not isinstance(new_latent, Tensor):
            raise TypeError(f"new_latent must be a Tensor, got {type(new_latent)}")

        # Normalize to 2D (T_new, D)
        if new_latent.dim() == 1:
            new_latent = new_latent.unsqueeze(0)
        elif new_latent.dim() == 2:
            pass
        else:
            raise ValueError(
                f"append_state_latent expects a 1D or 2D tensor, got shape {tuple(new_latent.shape)}"
            )

        if self.state is None:
            # First observation
            self.state = new_latent
            return

        if self.state.dim() != 2:
            raise ValueError(
                f"RTCPrevChunk.state is expected to be 2D (T, D), got shape {tuple(self.state.shape)}"
            )

        if self.state.shape[1] != new_latent.shape[1]:
            raise ValueError(
                "State latent dimension mismatch when appending: "
                f"existing D={self.state.shape[1]}, new D={new_latent.shape[1]}"
            )

        self.state = torch.cat([self.state, new_latent], dim=0)


def prepare_prev_chunk_left_over(
    action_left_over: Tensor | None = None,
    observed_state_latents: Tensor | None = None,
    *,
    inference_delay: int = 0,
    execution_horizon: int | None = None,
    state_execution_horizon: int | None = None,
) -> RTCPrevChunk | None:
    """Helper to build an ``RTCPrevChunk`` instance for extended RTC.

    This function is meant to be called on the *execution* side, where we know:
    - which actions from the previous chunk are still unexecuted, and
    - which state latents (e.g. VAE-encoded image features) were observed while
      executing that leftover.

    It does not affect existing RTC usage. Callers that only use action RTC can
    keep passing a plain Tensor as ``prev_chunk_left_over`` to ``denoise_step``.

    Args:
        action_left_over: Unexecuted actions from the previous chunk
            (or ``None`` if no action prefix is used).
        observed_state_latents: State latents observed during execution of the
            previous chunk's leftover (or ``None`` if no state feedback is used).
        inference_delay: Number of timesteps of inference delay associated with
            this leftover. This will be stored in the resulting ``RTCPrevChunk``.
        execution_horizon: Optional action horizon override for this leftover.
        state_execution_horizon: Optional state horizon override. If ``None``,
            the extended RTC logic may fall back to ``execution_horizon`` or the
            config default.

    Returns:
        ``RTCPrevChunk`` if at least one of ``action_left_over`` or
        ``observed_state_latents`` is provided; otherwise ``None``.
    """
    if action_left_over is None and observed_state_latents is None:
        return None

    # Initialise with action leftover; state latents are typically collected
    # incrementally via ``RTCPrevChunk.append_state_latent`` while executing.
    prev = RTCPrevChunk(
        action=action_left_over,
        state=None,
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
        state_execution_horizon=state_execution_horizon,
    )

    # If a pre-collected sequence of state latents is already available, we
    # append it once so that the internal layout is consistent with the
    # incremental update path.
    if observed_state_latents is not None:
        prev.append_state_latent(observed_state_latents)

    return prev


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    This class implements RTC techniques including velocity calculation,
    prefix attention, and adaptive chunk processing.
    """

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config

        self.tracker = None

        if rtc_config.debug:
            self.tracker = Tracker(
                enabled=rtc_config.debug,
                maxlen=rtc_config.debug_maxlen,
            )

    # ====================== Tracker Proxy Methods ======================
    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        """Proxy method to track debug information.

        If tracker is None or disabled, this method does nothing.
        Otherwise, it forwards the call to tracker.track().
        """
        if self.tracker is not None:
            self.tracker.track(
                time=time,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                **metadata,
            )

    def get_all_debug_steps(self) -> list:
        """Get all debug steps from tracker.

        Returns empty list if tracker is disabled or None.
        """
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        """Check if debug tracking is enabled.

        Returns True if tracker exists and is enabled.
        """
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        """Reset the tracker, clearing all recorded steps.

        Does nothing if tracker is None.
        """
        if self.tracker is not None:
            self.tracker.reset()

    # ====================== End Tracker Proxy Methods ======================

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ) -> Tensor:
        """RTC guidance wrapper around an existing denoiser.

        This method wraps an original denoising callable that only takes ``x_t`` and
        returns a base denoised velocity ``v_t``. It then applies Real-Time Chunking
        (RTC) and feedback flow-matching prefix guidance using the leftover prefix 
        actions and observed latented states from the previous chunk.

        Args:
            x_t (Tensor): Current latent/state to denoise. Shape ``(B, T, A)`` or ``(T, A)``.
            prev_chunk_left_over (Tensor | RTCPrevChunk | None): Unexecuted prefix
                from the previous chunk. For backward compatibility this can be:

                - a Tensor of shape ``(B, T_prev, A)`` or ``(T_prev, A)`` (original
                  RTC behavior, action-only guidance); or
                - an ``RTCPrevChunk`` instance, where only the ``action`` field is
                  currently used for guidance (state feedback will be handled in a
                  later extension).

                If ``None`` or if an ``RTCPrevChunk`` is provided without an
                ``action`` tensor, no guidance is applied and the method returns
                ``v_t`` from the original denoiser.
            inference_delay (int): Number of timesteps from the prefix to use for guidance.
            time (float | Tensor): Scalar in [0, 1] indicating normalized time. Must be
                broadcastable with ``x_t``.
            original_denoise_step_partial (Callable[[Tensor], Tensor]): Callable that
                computes the base denoised velocity given only ``x_t``.
            execution_horizon (int | None): Horizon used to build prefix weights. If
                ``None``, defaults to ``self.rtc_config.execution_horizon``.

        Returns:
            Tensor: Guided velocity with the same shape as ``v_t``.

        Notes:
            - If inputs are 2D, a batch dimension is temporarily added and removed at the end.
            - If ``prev_chunk_left_over`` is shorter than the current chunk length ``T``, it is
              right-padded with zeros to match ``T``.
            - Prefix weights are constructed via ``get_prefix_weights(inference_delay, execution_horizon, T)``
              and broadcast to ``(B, T, A)``.
            - Guidance correction is computed via autograd using ``x1_t = x_t + time * v_t`` and
              ``error = (prev_chunk_left_over - x1_t) * weights``.
            - The final guidance weight is clamped by ``max_guidance_weight`` from the config.

        Reference:
            https://www.physicalintelligence.company/download/real_time_chunking.pdf
        """

        # In the original implementation, the time goes from 0 to 1 and
        # In our implementation, the time goes from 1 to 0
        # So we need to invert the time
        tau = 1 - time

        # Normalize prev_chunk_left_over into a Tensor (action-only guidance).
        # For now we ignore state latents in RTCPrevChunk; only the action field
        # participates in the same way as the original Tensor-based API.
        if isinstance(prev_chunk_left_over, RTCPrevChunk):
            # If an execution_horizon override was set on the struct, prefer it.
            if prev_chunk_left_over.execution_horizon is not None:
                execution_horizon = prev_chunk_left_over.execution_horizon
            # Use the delay stored in the struct if present.
            inference_delay = prev_chunk_left_over.inference_delay
            prev_action = prev_chunk_left_over.action
            if prev_action is None:
                v_t = original_denoise_step_partial(x_t)
                return v_t
            prev_chunk_tensor = prev_action
        else:
            if prev_chunk_left_over is None:
                # First step, no guidance - return v_t
                v_t = original_denoise_step_partial(x_t)
                return v_t
            prev_chunk_tensor = prev_chunk_left_over

        x_t = x_t.clone().detach()

        squeezed = False
        if len(x_t.shape) < 3:
            # Add batch dimension
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if len(prev_chunk_tensor.shape) < 3:
            # Add batch dimension
            prev_chunk_tensor = prev_chunk_tensor.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        # If the previous action chunk is to short then it doesn't make sense to use long execution horizon
        # because there is nothing to merge
        if execution_horizon > prev_chunk_tensor.shape[1]:
            execution_horizon = prev_chunk_tensor.shape[1]

        batch_size = x_t.shape[0]           # B(B) Batch数量
        chunk_size = x_t.shape[1]           # T(H) Chunk内step数
        output_dim = x_t.shape[2]           # D(n) step维数(状态+动作)

        # Optionally build a full [state; action] prefix when state feedback is
        # enabled and state latents have been collected in RTCPrevChunk. The
        # guidance is still applied to the *entire* x (state + action together)
        # by using this combined prefix tensor in the original RTC formula.
        use_state_feedback = (
            isinstance(prev_chunk_left_over, RTCPrevChunk)      # FBFM需要RTCPrevChunk输入状态反馈
            and self.rtc_config.state_feedback_enabled          # 确认状态反馈是能
            and self.rtc_config.chunk_state_dim is not None     # 确认状态维数
            and self.rtc_config.chunk_action_dim is not None    # 确认动作维数
            and prev_chunk_left_over.state is not None          # 确认存在至少一帧状态输入
        )

        if use_state_feedback:
            state_dim = self.rtc_config.chunk_state_dim    # 配置导入的全部状态维数
            action_dim = self.rtc_config.chunk_action_dim  # 配置导入的全部动作维数
            if state_dim + action_dim != output_dim:       # 检查配置导入的维数合法性
                raise ValueError(
                    f"chunk_state_dim + chunk_action_dim must equal x_t last dim: "
                    f"got {state_dim} + {action_dim} != {output_dim}"
                )

            # Start from zeros and fill state and (optionally) action segments.
            combined_prefix = torch.zeros(                  
                batch_size,
                chunk_size,
                output_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )

            # --- Fill state segment [ :state_dim ] --- (B,T,D_state)
            state_src = prev_chunk_left_over.state
            if state_src is not None:
                if state_src.dim() == 1:
                    state_src = state_src.unsqueeze(0).unsqueeze(0)
                elif state_src.dim() == 2:
                    state_src = state_src.unsqueeze(0)
                elif state_src.dim() != 3:
                    raise ValueError(
                        f"RTCPrevChunk.state must be 1D, 2D or 3D tensor, got shape {tuple(state_src.shape)}"
                    )
                # 准备公式中的 ‘Y’ 的状态部分
                t_state = min(state_src.shape[1], chunk_size)   # 有效状态反馈的步数
                d_state = min(state_src.shape[2], state_dim)    # 有效状态反馈的维数 （应当为全部） but OK
                combined_prefix[:, :t_state, :d_state] = state_src[:, :t_state, :d_state]

            # --- Fill action segment [state_dim:] if available --- (B,T,D_action])
            action_src = prev_chunk_left_over.action
            if action_src is not None:
                if action_src.dim() == 1:
                    action_src = action_src.unsqueeze(0).unsqueeze(0)
                elif action_src.dim() == 2:
                    action_src = action_src.unsqueeze(0)
                elif action_src.dim() != 3:
                    raise ValueError(
                        f"RTCPrevChunk.action must be 1D, 2D or 3D tensor, got shape {tuple(action_src.shape)}"
                    )
                # 准备公式中的 ‘Y’ 的动作部分
                t_action = min(action_src.shape[1], chunk_size)    # 有效动作反馈的步数
                d_action = min(action_src.shape[2], action_dim)    # 有效动作反馈的维数 （应当为全部） but OK
                combined_prefix[:, :t_action, state_dim : state_dim + d_action] = action_src[:, :t_action, :d_action]

            prev_chunk_tensor = combined_prefix

            # Weights diag(W): chunk layout is *per-timestep* [state; action].
            #   x_t[t] = [ state(t) of dim state_dim ; action(t) of dim action_dim ]
            # So over the chunk we have:
            #   step 0: [s0_1..s0_state_dim, a0_1..a0_action_dim]
            #   step 1: [s1_1..s1_state_dim, a1_1..a1_action_dim]
            #   ...
            # W is (1, T, D): W[t, 0:state_dim]=1 for t in "当前已有状态" else 0;
            # W[t, state_dim:]=1 for t in "上个chunk遗留动作" else 0.
            state_exec_horizon = (
                prev_chunk_left_over.state_execution_horizon
                if prev_chunk_left_over.state_execution_horizon is not None
                else self.rtc_config.state_execution_horizon    # weired but OK
            )
            t_state_cap = min(t_state, state_exec_horizon, chunk_size)
            weights = torch.zeros(
                1,                  # B = 1
                chunk_size,         # T
                output_dim,         # D
                device=x_t.device,
                dtype=x_t.dtype,
            )
            weights[:, :t_state_cap, :state_dim] = 1.0
            weights[:, :t_action, state_dim:] = 1.0
        else:   # 不应该走进这个分支
            weights = (
                self.get_prefix_weights(inference_delay, execution_horizon, chunk_size)
                .to(x_t.device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )

        if prev_chunk_tensor.shape[1] < chunk_size or prev_chunk_tensor.shape[2] < output_dim:
            padded = torch.zeros(batch_size, chunk_size, output_dim).to(x_t.device)
            padded[:, : prev_chunk_tensor.shape[1], : prev_chunk_tensor.shape[2]] = prev_chunk_tensor
            prev_chunk_tensor = padded

        assert prev_chunk_tensor.shape == x_t.shape, (
            "The padded previous chunk must be the same size as the input tensor"
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)    # v()
            x_t.requires_grad_(True)    
            x1_t = x_t - time * v_t  # noqa: N806
            err = (prev_chunk_tensor - x1_t) * weights
            grad_outputs = err.clone().detach()
            # 自动求梯度（雅可比矩阵）
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        # Tau-dependent base guidance (same formula for both state and action). # 系数: k_p
        tau_tensor = torch.as_tensor(tau, device=x_t.device)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (
            squared_one_minus_tau + 1e-8
        )
        c = torch.nan_to_num(
            (1 - tau_tensor) / (tau_tensor + 1e-8), posinf=1e6
        )
        base_guidance_weight = torch.nan_to_num(c * inv_r2, posinf=1e6)

        # Symmetric caps: each branch truncated by its own max (for maintenance).
        action_max_guidance_weight = torch.as_tensor(
            self.rtc_config.max_guidance_weight, device=x_t.device
        )
        state_max_guidance_weight = torch.as_tensor(
            self.rtc_config.state_max_guidance_weight, device=x_t.device
        )
        action_guidance_weight = torch.minimum(
            base_guidance_weight, action_max_guidance_weight
        )
        state_guidance_weight = torch.minimum(
            base_guidance_weight, state_max_guidance_weight
        )

        if use_state_feedback and state_dim is not None:
            correction_state = correction[:, :, :state_dim]
            correction_action = correction[:, :, state_dim:]
            result = v_t - torch.cat(
                [
                    state_guidance_weight * correction_state,
                    action_guidance_weight * correction_action,
                ],
                dim=-1,
            )
        else:
            result = v_t - action_guidance_weight * correction

        # Remove the batch dimension if it was added
        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        self.track(
            time=time,
            x1_t=x1_t,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=action_guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

        return result

    def get_prefix_weights(self, start, end, total):
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR:
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)

        return weights

    def _linweights(self, start, end, total):
        skip_steps_at_end = max(total - end, 0)

        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        zeros_len = total - end

        if zeros_len <= 0:
            return weights

        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights, start, total):
        ones_len = min(start, total)

        if ones_len <= 0:
            return weights

        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])
