"""
Standalone FBFM (State Feedback Flow-Matching) Inference Engine.

This module reimplements the core algorithm from modeling_rtc.py
(RTCProcessor.denoise_step) in a self-contained form that does not depend on
the `lerobot` package. It is used for the 1D pre-experiment validation.

RELATIONSHIP TO modeling_rtc.py:
    The two implementations are mathematically equivalent under their
    respective time conventions:
      - modeling_rtc.py: time ∈ [1, 0] (noise at t=1, data at t=0)
        x̂₁ = x_t − time · v_t  ;  result = v_t − k_p · correction
      - this file:       τ ∈ [0, 1] (noise at τ=0, data at τ=1)
        x̂₁ = x_τ + (1−τ) · v_τ ;  result = v_τ + k_p · correction
    Both give identical next-step x values since the sign flips cancel.
    The guidance weight formula k_p is identical in both.

FBFM EXTENSION (Full Jacobian):
    In modeling_rtc.py, v_t is computed BEFORE requires_grad_(True),
    giving ∂x̂₁/∂x = I (identity Jacobian). This is fine for action-only
    guidance (original RTC).

    For FBFM, we compute v_t AFTER requires_grad_(True) so that
    ∂x̂₁/∂x = I + (1−τ)·∂model/∂x (full Jacobian). This matches the
    formula in core_thoughts.md Algorithm 1 line 34 and is essential for
    cross-dimensional coupling: state errors propagate to action corrections.

The key formulas (from core_thoughts.md):

    v_ΠGDM(X^τ, o, τ) = v(X^τ, o, τ) + k_p · (Y − X̂¹)ᵀ diag(W) · ∂X̂¹/∂X^τ

    where:
        X^τ = [Z^τ, A^τ]  (state latent + action, concatenated per timestep)
        X̂¹  = X^τ + (1 - τ) · v(X^τ, o, τ)    (one-step prediction to clean data)
        k_p  = min(β, (1 - τ)/τ · 1/r²_τ)
        r²_τ = (1 - τ)² / (τ² + (1 - τ)²)
        Y    = [observed_states, leftover_actions]  (feedback signal)
        W    = binary mask (1 where feedback is available, 0 elsewhere)

Three operational modes are supported:
    1. Vanilla:  no guidance (standard flow-matching sampling)
    2. RTC:      action-only guidance from previous chunk leftovers
                 (identity Jacobian, matching original modeling_rtc.py)
    3. FBFM:     state + action feedback guidance (our proposed method)
                 (full model Jacobian for cross-dimensional coupling)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch
from torch import Tensor


class GuidanceMode(Enum):
    """Inference guidance mode."""
    VANILLA = "vanilla"         # No guidance at all
    RTC = "rtc"                 # Action-only prefix guidance (baseline)
    FBFM = "fbfm"               # Full state + action feedback guidance (ours)
    FBFM_IDENTITY = "fbfm_id"   # State+action feedback but identity Jacobian (ablation)


@dataclass
class FBFMConfig:
    """Configuration for FBFM inference.

    Maps to RTCConfig fields in the original codebase.
    """
    # Mode
    mode: GuidanceMode = GuidanceMode.FBFM

    # Dimensions (for the 1D experiment: state=2, action=1, chunk_dim=3)
    state_dim: int = 2
    action_dim: int = 1
    horizon: int = 16

    # Guidance parameters
    max_guidance_weight: float = 10.0       # β for action
    state_max_guidance_weight: float = 10.0  # β for state
    execution_horizon: int = 10             # for prefix weight schedule
    state_execution_horizon: int = 10       # for state prefix

    # Denoising
    num_denoise_steps: int = 20

    # Prefix attention schedule: 'linear' or 'ones'
    prefix_schedule: str = "linear"


@dataclass
class PrevChunkInfo:
    """Container for previous chunk feedback information.

    Analogous to RTCPrevChunk in modeling_rtc.py.
    """
    action_leftover: Tensor | None = None
    """Unexecuted actions from the previous chunk: (T_left, action_dim)."""

    observed_states: Tensor | None = None
    """State observations collected during execution: (T_obs, state_dim)."""

    inference_delay: int = 0
    """Number of timesteps the inference was delayed."""

    def append_state(self, new_state: Tensor) -> None:
        """Append a newly observed state.

        Args:
            new_state: (state_dim,) or (1, state_dim) or (T, state_dim).
        """
        if new_state is None:
            return
        if new_state.dim() == 1:
            new_state = new_state.unsqueeze(0)
        if self.observed_states is None:
            self.observed_states = new_state
        else:
            self.observed_states = torch.cat([self.observed_states, new_state], dim=0)


# ======================================================================
# Core FBFM Inference
# ======================================================================

class FBFMProcessor:
    """State Feedback Flow-Matching processor.

    This class implements the guided denoising loop described in Algorithm 1
    of core_thoughts.md, faithfully mirroring the RTCProcessor.denoise_step
    logic from modeling_rtc.py.
    """

    def __init__(self, cfg: FBFMConfig):
        self.cfg = cfg
        self.debug_steps: list[dict] = []

    def reset_debug(self):
        self.debug_steps.clear()

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def guided_inference(
        self,
        model_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        observation: Tensor,
        prev_chunk: PrevChunkInfo | None = None,
        device: str = "cpu",
        target: Tensor | None = None,
    ) -> Tensor:
        """Run guided flow-matching inference (Algorithm 1, GUIDEDINFERENCE).

        Args:
            model_fn: callable(x_tau, observation, tau) → v_pred.
                      Takes (B, H, D), (B, state_dim), (B,) → (B, H, D).
            observation: (state_dim,) or (B, state_dim) – current observation o_t.
            prev_chunk: previous chunk feedback info (None for first chunk).
            device: torch device.
            target: (target_dim,) or (B, target_dim) – normalized target for model conditioning.

        Returns:
            Tensor of shape (H, chunk_dim) – generated chunk X^1.
        """
        cfg = self.cfg
        H = cfg.horizon
        D = cfg.state_dim + cfg.action_dim

        # Ensure observation is batched
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        B = observation.shape[0]

        # Ensure target is batched
        if target is not None and target.dim() == 1:
            target = target.unsqueeze(0)

        # Initialize from noise: X^0 ~ N(0, I)
        x = torch.randn(B, H, D, device=device)
        dt_step = 1.0 / cfg.num_denoise_steps

        # Determine guidance mode
        use_guidance = (
            cfg.mode != GuidanceMode.VANILLA
            and prev_chunk is not None
        )

        for step_idx in range(cfg.num_denoise_steps):
            # Flow time τ ∈ [0, 1], step from 0 toward 1
            tau_val = step_idx * dt_step
            tau = torch.full((B,), tau_val, device=device)

            if use_guidance:
                v, debug_info = self._guided_denoise_step(
                    x, observation, tau, tau_val, model_fn, prev_chunk, step_idx,
                    target=target,
                )
            else:
                v = model_fn(x, observation, tau, target=target)
                debug_info = None

            x = x.detach() + dt_step * v.detach()  # detach to avoid graph accumulation

            if debug_info is not None:
                self.debug_steps.append(debug_info)

        # Return unbatched if B=1
        if B == 1:
            return x.squeeze(0)  # (H, D)
        return x

    # ------------------------------------------------------------------
    # Guided denoising step (mirrors RTCProcessor.denoise_step)
    # ------------------------------------------------------------------

    def _guided_denoise_step(
        self,
        x_t: Tensor,
        observation: Tensor,
        tau: Tensor,
        tau_val: float,
        model_fn: Callable,
        prev_chunk: PrevChunkInfo,
        step_idx: int,
        target: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Single guided denoising step.

        Implements the PiGDM guidance from core_thoughts.md Algorithm 1:
        - Build combined prefix Y (state + action) or action-only prefix
        - Compute weight mask W (binary, 1 where feedback available)
        - Compute one-step prediction X̂¹ = X^τ + (1-τ)·v(X^τ, o, τ)
        - Compute gradient correction via autograd: g = (Y-X̂¹)·W · ∂X̂¹/∂X^τ
        - Apply guidance: v_guided = v + min(β, k_p) · g

        TIME CONVENTION EQUIVALENCE:
            This uses τ ∈ [0,1] (flow convention: 0=noise, 1=data).
            modeling_rtc.py uses time ∈ [1,0] (1=noise, 0=data).
            The formulas differ in sign but produce identical results:
              modeling_rtc:  x̂₁ = x - time·v;  result = v - k·corr
              here:          x̂₁ = x + (1-τ)·v; result = v + k·corr
            Since time = 1-τ, both signs cancel and yield the same update.

        JACOBIAN:
            RTC mode: identity Jacobian (v_t detached from x_input's grad),
                matching modeling_rtc.py's original behavior.
            FBFM mode: full model Jacobian (v_t computed WITH grad),
                matching core_thoughts.md ∂f_X̂¹/∂X' formula.
        """
        cfg = self.cfg
        B, H, D = x_t.shape
        state_dim = cfg.state_dim
        action_dim = cfg.action_dim

        # Time convention mapping:
        #   Our flow convention:  τ_flow ∈ [0, 1], 0=noise, 1=data
        #   modeling_rtc.py:      time ∈ [1, 0],   1=noise, 0=data
        #   Relationship:         time = 1 - τ_flow
        #   modeling_rtc.py also defines: tau_rtc = 1 - time = τ_flow
        # So our tau_val IS the same as tau_rtc in modeling_rtc.py.
        time_rtc = 1.0 - tau_val  # This is 'time' in modeling_rtc.py

        # Build prefix & weights
        if cfg.mode in (GuidanceMode.FBFM, GuidanceMode.FBFM_IDENTITY):
            combined_prefix, weights = self._build_fbfm_prefix_and_weights(
                prev_chunk, B, H, D, x_t.device
            )
        else:  # RTC (action-only)
            combined_prefix, weights = self._build_rtc_prefix_and_weights(
                prev_chunk, B, H, D, x_t.device
            )

        # Clone x_t for autograd.
        #
        # KEY DESIGN CHOICE: Jacobian mode
        #
        # Original RTC (action-only guidance, modeling_rtc.py):
        #   v_t computed BEFORE requires_grad_(True) → ∂x1/∂x = I (identity)
        #   This is correct for action-only guidance where state err should not
        #   change action corrections.
        #
        # FBFM (state+action guidance, our extension):
        #   v_t computed AFTER requires_grad_(True) → ∂x1/∂x includes model Jacobian
        #   This allows state error to propagate through the model and influence
        #   action corrections. This cross-dimensional coupling is the CORE of
        #   state feedback: observed state drift should inform action adjustments.
        #
        #   Without full Jacobian, state feedback can only modify state predictions
        #   but NOT actions, making it essentially useless.
        #
        # FBFM_IDENTITY (ablation control):
        #   Same prefix (state+action) as FBFM, but identity Jacobian.
        #   This isolates the Jacobian's contribution: if FBFM >> FBFM_IDENTITY,
        #   it proves the Jacobian is the key innovation, not just "having state feedback".
        x_input = x_t.clone().detach()

        with torch.enable_grad():
            if cfg.mode == GuidanceMode.FBFM:
                # FBFM: full Jacobian — model participates in grad graph
                # so that state error correction propagates to action dimensions.
                x_input.requires_grad_(True)
                v_t = model_fn(x_input, observation, tau, target=target)
                # x1_hat = x_input + (1-τ) * model(x_input, ...)
                # ∂x1_hat/∂x_input = I + (1-τ) * ∂model/∂x_input  (includes Jacobian)
                x1_hat = x_input + (1 - tau_val) * v_t
            else:
                # RTC: detached Jacobian — original behavior (identity Jacobian)
                v_t = model_fn(x_input, observation, tau, target=target)
                x_input.requires_grad_(True)
                x1_hat = x_input + (1 - tau_val) * v_t

            # Error: (Y - X̂¹) * W
            err = (combined_prefix - x1_hat) * weights

            # Gradient: ∂X̂¹/∂X^τ applied to error
            # For RTC: Jacobian = I, so correction = err.
            # For FBFM: Jacobian ≠ I, so state errors propagate to action dims.
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(
                x1_hat, x_input, grad_outputs, retain_graph=False
            )[0]

        # Guidance weight: k_p = min(β, (1-τ)/τ · 1/r²_τ)
        # where r²_τ = (1-τ)² / (τ² + (1-τ)²)
        # So 1/r²_τ = (τ² + (1-τ)²) / (1-τ)²
        # k_p = min(β, (1-τ)/τ · (τ² + (1-τ)²) / (1-τ)²)
        #     = min(β, (τ² + (1-τ)²) / (τ · (1-τ)))
        #     ≈ min(β, 1/τ + τ/(1-τ)) for τ not near 0 or 1
        tau_t = torch.tensor(tau_val, device=x_t.device).clamp(min=1e-8)
        one_minus_tau = (1 - tau_t).clamp(min=1e-8)

        squared_one_minus_tau = one_minus_tau ** 2
        # 1/r²_τ = (τ² + (1-τ)²) / (1-τ)²  — exact formula from core_thoughts.md.
        # one_minus_tau is already clamped to ≥ 1e-8, so no +1e-8 needed here.
        inv_r2 = (squared_one_minus_tau + tau_t ** 2) / squared_one_minus_tau
        c = torch.nan_to_num(one_minus_tau / (tau_t + 1e-8), posinf=1e6)
        base_guidance_weight = torch.nan_to_num(c * inv_r2, posinf=1e6)

        # Separate caps for state and action
        action_gw = torch.minimum(
            base_guidance_weight,
            torch.tensor(cfg.max_guidance_weight, device=x_t.device),
        )
        state_gw = torch.minimum(
            base_guidance_weight,
            torch.tensor(cfg.state_max_guidance_weight, device=x_t.device),
        )

        # Apply guided velocity with correction clipping to prevent spikes
        if cfg.mode in (GuidanceMode.FBFM, GuidanceMode.FBFM_IDENTITY):
            # Separate guidance for state and action dimensions
            correction_state = correction[:, :, :state_dim]
            correction_action = correction[:, :, state_dim:]

            # FBFM: Use tighter clip for smoother action (reduces jitter)
            # Apply temporal smoothing to action correction for continuity
            action_corr_raw = action_gw * correction_action

            # Simple temporal smoothing: weighted average with neighbors
            if action_corr_raw.shape[1] > 2:
                # Pad for boundary handling
                padded = torch.nn.functional.pad(action_corr_raw, (0, 0, 1, 1), mode='replicate')
                # Smooth: 0.25*prev + 0.5*curr + 0.25*next
                smoothed = (padded[:, :-2, :] * 0.25 +
                           padded[:, 1:-1, :] * 0.5 +
                           padded[:, 2:, :] * 0.25)
                action_corr_smooth = smoothed
            else:
                action_corr_smooth = action_corr_raw

            # Tighter clip for FBFM to reduce jitter (±1.0 instead of ±2.0)
            action_correction_clipped = torch.clamp(
                action_corr_smooth,
                min=-1.0,
                max=1.0
            )

            guided_v = v_t + torch.cat(
                [
                    state_gw * correction_state,
                    action_correction_clipped,
                ],
                dim=-1,
            )
        else:
            # RTC & Vanilla: Looser clip (±2.0) - allows faster response but more jitter
            action_correction_clipped = torch.clamp(
                action_gw * correction,
                min=-2.0,
                max=2.0
            )
            guided_v = v_t + action_correction_clipped

        # Debug info
        debug_info = {
            "step_idx": step_idx,
            "tau": tau_val,
            "guidance_weight_action": action_gw.item(),
            "guidance_weight_state": state_gw.item(),
            "correction_norm": correction.norm().item(),
            "err_norm": err.norm().item(),
            "x1_hat_mean": x1_hat.detach().mean().item(),
        }

        return guided_v, debug_info

    # ------------------------------------------------------------------
    # Prefix & weight builders
    # ------------------------------------------------------------------

    def _build_fbfm_prefix_and_weights(
        self,
        prev_chunk: PrevChunkInfo,
        B: int,
        H: int,
        D: int,
        device,
    ) -> tuple[Tensor, Tensor]:
        """Build combined [state; action] prefix and weight mask for FBFM.

        Mirrors modeling_rtc.py lines 388-470 (use_state_feedback=True branch).
        """
        state_dim = self.cfg.state_dim
        action_dim = self.cfg.action_dim

        combined_prefix = torch.zeros(B, H, D, device=device)
        weights = torch.zeros(B, H, D, device=device)

        # Fill state segment
        t_state = 0
        if prev_chunk.observed_states is not None:
            states = prev_chunk.observed_states
            if states.dim() == 1:
                states = states.unsqueeze(0)
            t_state = min(states.shape[0], H)
            d_state = min(states.shape[-1], state_dim)
            combined_prefix[:, :t_state, :d_state] = states[:t_state, :d_state].unsqueeze(0)

        # Fill action segment
        t_act = 0
        if prev_chunk.action_leftover is not None:
            actions = prev_chunk.action_leftover
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            t_act = min(actions.shape[0], H)
            d_act = min(actions.shape[-1], action_dim)
            combined_prefix[:, :t_act, state_dim : state_dim + d_act] = actions[:t_act, :d_act].unsqueeze(0)

        # Weight mask W
        # State: capped by state_execution_horizon (matching modeling_rtc.py line 457)
        state_exec_h = min(t_state, self.cfg.state_execution_horizon, H)
        weights[:, :state_exec_h, :state_dim] = 1.0

        # Action: NOT capped by execution_horizon (matching modeling_rtc.py line 465)
        # In the original code: weights[:, :t_act, state_dim:] = 1.0
        # t_act is simply min(action_src.shape[1], action_chunk_size)
        weights[:, :t_act, state_dim:] = 1.0

        return combined_prefix, weights

    def _build_rtc_prefix_and_weights(
        self,
        prev_chunk: PrevChunkInfo,
        B: int,
        H: int,
        D: int,
        device,
    ) -> tuple[Tensor, Tensor]:
        """Build action-only prefix and linear schedule weights for RTC.

        Mirrors modeling_rtc.py lines 472-485 (use_state_feedback=False branch).
        In the original code, x_t is a pure action chunk. In our experiment,
        x_t is [state, action], so we place the leftover action into the
        action segment [state_dim:] and apply weights only to that segment.
        """
        state_dim = self.cfg.state_dim
        action_dim = self.cfg.action_dim
        combined_prefix = torch.zeros(B, H, D, device=device)

        t_act = 0
        if prev_chunk.action_leftover is not None:
            actions = prev_chunk.action_leftover
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            t_act = min(actions.shape[0], H)
            d_act = min(actions.shape[-1], action_dim)
            # Place action in the action segment of the chunk [state_dim:]
            combined_prefix[:, :t_act, state_dim : state_dim + d_act] = (
                actions[:t_act, :d_act].unsqueeze(0)
            )

        # Build linear schedule weights, applied only to action dims
        inference_delay = prev_chunk.inference_delay
        execution_horizon = self.cfg.execution_horizon
        start = min(inference_delay, execution_horizon)

        weights_1d = self._get_prefix_weights(start, execution_horizon, H).to(device)
        # Expand to (B, H, D) but zero out state dims
        weights = torch.zeros(B, H, D, device=device)
        weights[:, :, state_dim:] = weights_1d.unsqueeze(0).unsqueeze(-1).expand(
            B, H, action_dim
        )

        return combined_prefix, weights

    def _get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        """Generate prefix attention weights (linear schedule).

        Mirrors RTCProcessor.get_prefix_weights from modeling_rtc.py.
        """
        start = min(start, end)

        if self.cfg.prefix_schedule == "ones":
            weights = torch.ones(total)
            weights[end:] = 0.0
            return weights

        # Linear schedule (default)
        skip = max(total - end, 0)
        linspace_steps = total - skip - start

        if end <= start or linspace_steps <= 0:
            weights = torch.zeros(total)
            weights[:start] = 1.0
            return weights

        lin = torch.linspace(1, 0, linspace_steps + 2)[1:-1]

        # Add trailing zeros
        if total - end > 0:
            lin = torch.cat([lin, torch.zeros(total - end)])

        # Add leading ones
        if start > 0:
            lin = torch.cat([torch.ones(min(start, total)), lin])

        # Ensure correct length
        if lin.shape[0] < total:
            lin = torch.cat([lin, torch.zeros(total - lin.shape[0])])
        elif lin.shape[0] > total:
            lin = lin[:total]

        return lin


# ======================================================================
# High-level sampling with normalization
# ======================================================================

def fbfm_sample(
    model,
    observation: Tensor,
    norm_stats: dict,
    cfg: FBFMConfig,
    prev_chunk: PrevChunkInfo | None = None,
    device: str = "cpu",
    target: Tensor | None = None,
) -> tuple[Tensor, Tensor, list]:
    """Sample a chunk using FBFM-guided flow matching.

    This is the high-level entry point that handles normalization.

    Args:
        model: trained FlowMatchingDiT.
        observation: (state_dim,) – raw (unnormalized) observation.
        norm_stats: dict with chunk_mean, chunk_std, obs_mean, obs_std,
                    target_mean, target_std.
        cfg: FBFM configuration.
        prev_chunk: previous chunk info with raw (unnormalized) tensors.
            If provided, state/action will be normalized internally.
        device: torch device.
        target: (target_dim,) – raw (unnormalized) target state (e.g., [target_pos, 0.0]).
                If None, uses zeros (backward compatible).

    Returns:
        (states, actions, debug_steps) where:
            states:  (H, state_dim) – predicted states (denormalized)
            actions: (H, action_dim) – predicted actions (denormalized)
            debug_steps: list of debug dicts from the inference
    """
    state_dim = cfg.state_dim
    chunk_mean = norm_stats["chunk_mean"].to(device)
    chunk_std = norm_stats["chunk_std"].to(device)
    obs_mean = norm_stats["obs_mean"].to(device)
    obs_std = norm_stats["obs_std"].to(device)

    # Normalize observation
    obs_norm = (observation.to(device) - obs_mean.squeeze(0)) / obs_std.squeeze(0)

    # Normalize target
    target_norm = None
    if target is not None:
        target_mean = norm_stats.get("target_mean")
        target_std = norm_stats.get("target_std")
        if target_mean is not None and target_std is not None:
            target_norm = (target.to(device) - target_mean.squeeze(0).to(device)) / target_std.squeeze(0).to(device)
        else:
            target_norm = target.to(device)

    # Normalize prev_chunk if provided
    normalized_prev = None
    if prev_chunk is not None:
        normalized_prev = PrevChunkInfo(
            inference_delay=prev_chunk.inference_delay,
        )
        if prev_chunk.action_leftover is not None:
            # Action is the last action_dim dims of chunk
            a_mean = chunk_mean[0, 0, state_dim:]  # (action_dim,)
            a_std = chunk_std[0, 0, state_dim:]
            normalized_prev.action_leftover = (
                (prev_chunk.action_leftover.to(device) - a_mean) / a_std
            )
        if prev_chunk.observed_states is not None:
            # State is the first state_dim dims of chunk
            s_mean = chunk_mean[0, 0, :state_dim]  # (state_dim,)
            s_std = chunk_std[0, 0, :state_dim]
            normalized_prev.observed_states = (
                (prev_chunk.observed_states.to(device) - s_mean) / s_std
            )

    # Build model callable
    model.eval()

    def model_fn(x_tau, obs, tau, target=None):
        return model(x_tau, obs, tau, target=target)

    # Run guided inference
    processor = FBFMProcessor(cfg)
    processor.reset_debug()

    chunk = processor.guided_inference(
        model_fn=model_fn,
        observation=obs_norm,
        prev_chunk=normalized_prev,
        device=device,
        target=target_norm,
    )

    # Denormalize output chunk
    # chunk: (H, chunk_dim)
    if chunk.dim() == 2:
        chunk = chunk.unsqueeze(0)  # (1, H, D)
    chunk_denorm = chunk.cpu() * chunk_std.cpu() + chunk_mean.cpu()
    chunk_denorm = chunk_denorm.squeeze(0)  # (H, D)

    states = chunk_denorm[:, :state_dim]    # (H, state_dim)
    actions = chunk_denorm[:, state_dim:]   # (H, action_dim)

    return states, actions, processor.debug_steps
