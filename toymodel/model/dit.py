"""
model/dit.py — Lightweight DiT (Diffusion Transformer) for Flow Matching

Architecture overview:
    Input:  X^tau (noisy token chunk) + obs (conditioning state) + tau (noise level)
    Output: v (velocity field)  — same shape as X^tau

    X^tau: (B, H, token_dim) = (B, 16, 3)  — each token is [x, x_dot, u]
    obs:   (B, obs_dim)      = (B, 2)
    tau:   (B,)              scalar in [0, 1]

    v:     (B, H, token_dim) = (B, 16, 3)

Design choices (lightweight for MSD toy problem):
    - Each timestep token (token_dim,) projected to (d_model,) as one sequence token
    - Sinusoidal tau embedding → projected to d_model
    - obs linearly projected to d_model
    - N_layers transformer blocks with cross-attention (obs as key/value)
    - Final linear head → (B, H, token_dim)

This is intentionally small (d_model=128, N_layers=4) to train fast on CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Sinusoidal timestep embedding (standard in diffusion models)
# -----------------------------------------------------------------------

def sinusoidal_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding for scalar tau ∈ [0, 1].

    Args:
        tau: (B,) — noise level
        dim: int  — embedding dimension (must be even)

    Returns:
        emb: (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    # frequencies: (half,)
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=tau.device) / half
    )
    # args: (B, half)
    args = tau[:, None] * freqs[None, :]
    # emb: (B, dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


# -----------------------------------------------------------------------
# Single DiT block: self-attention on X + cross-attention with obs+tau
# -----------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    One DiT block:
        1. LayerNorm + Self-attention on sequence tokens
        2. LayerNorm + Cross-attention (query=tokens, key/value=context)
        3. LayerNorm + FFN

    Args:
        d_model:   token dimension
        n_heads:   number of attention heads
        d_context: dimension of cross-attention context (obs + tau embedding)
        ffn_mult:  FFN hidden dim = ffn_mult * d_model
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_context: int = 128,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, kdim=d_context, vdim=d_context, batch_first=True
        )

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,        # (B, seq_len, d_model)
        context: torch.Tensor,  # (B, ctx_len, d_context)
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Cross-attention: query from x, key/value from context
        x = x + self.cross_attn(self.norm2(x), context, context)[0]
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


# -----------------------------------------------------------------------
# Full DiT policy network
# -----------------------------------------------------------------------

class FlowMatchingDiT(nn.Module):
    """
    Lightweight DiT policy for Flow Matching on MSD token chunks.

    Each token represents [x, x_dot, u] (state after action + action).

    Forward pass:
        v = model(X_tau, obs, tau)

    Shapes:
        X_tau: (B, H, token_dim) = (B, 16, 3)  — noisy token chunk
        obs:   (B, obs_dim)      = (B, 2)       — conditioning state
        tau:   (B,)              scalar ∈ [0,1]  — noise level

        v:     (B, H, token_dim) = (B, 16, 3)  — predicted velocity field
    """

    def __init__(
        self,
        H: int         = 16,
        token_dim: int = 3,
        obs_dim: int   = 2,
        d_model: int   = 128,
        n_heads: int   = 4,
        n_layers: int  = 4,
        ffn_mult: int  = 4,
    ):
        super().__init__()
        self.H         = H
        self.token_dim = token_dim
        self.obs_dim   = obs_dim
        self.d_model   = d_model

        # --- Input projection: each token (token_dim,) → (d_model,) ---
        self.token_proj = nn.Linear(token_dim, d_model)

        # --- Positional embedding for the H tokens ---
        # Learnable, shape: (1, H, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, H, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # --- Context: tau embedding + obs projection → (B, 2, d_model) ---
        # tau sinusoidal embedding → linear → (B, d_model)
        self.tau_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # obs linear projection → (B, d_model)
        self.obs_proj = nn.Linear(obs_dim, d_model)

        # --- DiT blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(d_model=d_model, n_heads=n_heads,
                     d_context=d_model, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])

        # --- Output head: (B, H, d_model) → (B, H, token_dim) ---
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, token_dim)

        # Zero-init output projection (standard for diffusion models)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        X_tau: torch.Tensor,  # (B, H, token_dim)
        obs:   torch.Tensor,  # (B, obs_dim)
        tau:   torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Returns:
            v: (B, H, token_dim) — predicted velocity field
        """
        B = X_tau.shape[0]

        # tokens: (B, H, d_model)
        tokens = self.token_proj(X_tau) + self.pos_emb

        # tau_emb: (B, d_model)
        tau_emb = self.tau_proj(sinusoidal_embedding(tau, self.d_model))
        # obs_emb: (B, d_model)
        obs_emb = self.obs_proj(obs)
        # context: (B, 2, d_model)
        context = torch.stack([tau_emb, obs_emb], dim=1)

        for block in self.blocks:
            tokens = block(tokens, context)

        # v: (B, H, token_dim)
        v = self.out_proj(self.out_norm(tokens))
        return v

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
