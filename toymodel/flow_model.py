"""
Lightweight Flow-Matching Model for the 1D FBFM Pre-experiment.

Architecture: small Transformer (DiT-like) conditioned on:
  - observation o_t:  (state_dim,) = (2,)
  - flow time τ:      scalar in [0, 1]
  - noisy chunk X^τ:  (H, chunk_dim) where chunk_dim = state_dim + action_dim = 3

Output: predicted flow velocity v_π of shape (H, chunk_dim).

The model uses sinusoidal time embedding + cross-attention from the observation,
closely matching the architecture described in guidance.md §2.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
# Building blocks
# ======================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional / timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) or scalar – flow time in [0, 1].
        Returns:
            (B, dim) embedding.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class AdaLN(nn.Module):
    """Adaptive Layer Norm: shift & scale from conditioning vector."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x:    (B, T, D)
            cond: (B, cond_dim)
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)  # (B, D) each
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Single Transformer block with AdaLN conditioning."""

    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.adaln2 = AdaLN(hidden_dim, cond_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # Self-attention with AdaLN
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        # FFN with AdaLN
        h = self.adaln2(x, cond)
        h = self.mlp(h)
        x = x + h
        return x


# ======================================================================
# Main Model
# ======================================================================

class FlowMatchingDiT(nn.Module):
    """Lightweight DiT for 1D flow-matching.

    Predicts the flow velocity v(X^τ, o, τ) used in the ODE:
        dX/dτ = v(X^τ, o, τ)

    With linear interpolation training:
        X^τ = τ * X^1 + (1 - τ) * ε,  ε ~ N(0, I)
        loss = || v_pred - (X^1 - ε) ||^2

    NOTE: In the codebase convention (modeling_rtc.py), time goes from 1→0
    during inference (i.e. τ_rtc = 1 - τ_flow). This model uses the
    standard flow-matching convention τ ∈ [0, 1] where τ=0 is noise and
    τ=1 is data. The conversion is handled at the call site.

    The model is conditioned on:
      - observation o_t (current state)
      - target position (goal signal), enabling target-dependent policies
      - flow time τ
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        horizon: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        time_embed_dim: int = 64,
        target_dim: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_dim = state_dim + action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        # Embeddings
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        # Condition = [obs_embed; target_embed; time_embed]
        cond_dim = hidden_dim
        # Split hidden_dim into 3 parts for obs, target, time
        obs_dim = hidden_dim // 3
        tgt_dim = hidden_dim // 3
        time_dim = hidden_dim - obs_dim - tgt_dim  # remainder
        self.obs_proj = nn.Linear(state_dim, obs_dim)
        self.target_proj = nn.Linear(target_dim, tgt_dim)
        self.time_proj = nn.Linear(time_embed_dim, time_dim)
        self.cond_proj = nn.Linear(hidden_dim, cond_dim)

        # Input projection: noisy chunk → hidden
        self.input_proj = nn.Linear(self.chunk_dim, hidden_dim)

        # Learnable positional embedding for horizon steps
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim) * 0.02)

        # Transformer layers
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, cond_dim, num_heads) for _ in range(num_layers)]
        )

        # Output projection → velocity in chunk space
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.chunk_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training dynamics."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init last layer for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_tau: Tensor,
        observation: Tensor,
        tau: Tensor,
        target: Tensor | None = None,
    ) -> Tensor:
        """Predict flow velocity.

        Args:
            x_tau:       (B, H, chunk_dim) – noisy chunk at flow time τ.
            observation: (B, state_dim)    – current observation o_t.
            tau:         (B,) or scalar    – flow time ∈ [0, 1].
            target:      (B, target_dim)   – target state (optional, zeros if None).

        Returns:
            v_pred: (B, H, chunk_dim) – predicted velocity.
        """
        B = x_tau.shape[0]

        # Build condition vector
        if tau.dim() == 0:
            tau = tau.expand(B)
        time_emb = self.time_embed(tau)                       # (B, time_embed_dim)
        obs_emb = self.obs_proj(observation)                  # (B, obs_dim)
        time_emb = self.time_proj(time_emb)                   # (B, time_dim)

        # Target conditioning
        if target is None:
            target = torch.zeros(B, self.target_dim, device=x_tau.device)
        tgt_emb = self.target_proj(target)                    # (B, tgt_dim)

        cond = self.cond_proj(torch.cat([obs_emb, tgt_emb, time_emb], dim=-1))  # (B, cond_dim)

        # Input projection + positional embedding
        h = self.input_proj(x_tau) + self.pos_embed[:, : x_tau.shape[1], :]  # (B, H, hidden)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cond)

        # Output
        h = self.output_norm(h)
        v_pred = self.output_proj(h)  # (B, H, chunk_dim)
        return v_pred


# ======================================================================
# Flow-Matching Training Utilities
# ======================================================================

def flow_matching_loss(
    model: FlowMatchingDiT,
    x1: Tensor,
    observation: Tensor,
    sigma_min: float = 1e-4,
    target: Tensor | None = None,
) -> Tensor:
    """Compute the conditional flow-matching loss.

    Uses the optimal transport (OT) path:
        X^τ = τ * X^1 + (1 - τ) * ε,   ε ~ N(0, I)
        target velocity = X^1 - ε

    Args:
        model: the flow-matching model.
        x1: (B, H, chunk_dim) – clean data chunks.
        observation: (B, state_dim) – current observations.
        sigma_min: minimum noise scale.
        target: (B, target_dim) – target state for each sample (optional).

    Returns:
        Scalar MSE loss.
    """
    B = x1.shape[0]
    device = x1.device

    # Sample random flow time τ ~ U(sigma_min, 1)
    tau = torch.rand(B, device=device) * (1 - sigma_min) + sigma_min

    # Sample noise
    eps = torch.randn_like(x1)

    # Interpolate
    tau_expanded = tau[:, None, None]  # (B, 1, 1)
    x_tau = tau_expanded * x1 + (1 - tau_expanded) * eps

    # Target velocity (OT)
    v_target = x1 - eps

    # Predict
    v_pred = model(x_tau, observation, tau, target=target)

    # MSE loss
    loss = ((v_pred - v_target) ** 2).mean()
    return loss


@torch.no_grad()
def flow_matching_sample(
    model: FlowMatchingDiT,
    observation: Tensor,
    num_steps: int = 20,
    horizon: int | None = None,
    target: Tensor | None = None,
) -> Tensor:
    """Sample from the flow-matching model via Euler integration.

    Integrates from τ=0 (noise) to τ=1 (data):
        X^{τ+Δτ} = X^τ + Δτ * v(X^τ, o, τ)

    Args:
        model: trained flow-matching model.
        observation: (B, state_dim) – current observation.
        num_steps: number of Euler integration steps.
        horizon: chunk length (default: model.horizon).
        target: (B, target_dim) – target state (optional).

    Returns:
        (B, H, chunk_dim) – sampled clean chunk X^1.
    """
    device = observation.device
    B = observation.shape[0]
    H = horizon or model.horizon
    D = model.chunk_dim

    x = torch.randn(B, H, D, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        tau = torch.full((B,), i * dt, device=device)
        v = model(x, observation, tau, target=target)
        x = x + dt * v

    return x
