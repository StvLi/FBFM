"""
model/dit.py — Lightweight DiT (Diffusion Transformer) for Flow Matching

Architecture overview:
    Input:  X^tau (noisy token chunk) + obs (conditioning state) + tau (noise level)
    Output: v (velocity field)  — same shape as X^tau

    X^tau: (B, H, token_dim) = (B, 16, 3)  — each token is [x, x_dot, u]
    obs:   (B, obs_dim)      = (B, 3)       — [x, x_dot, x_ref]
    tau:   (B,) or (B, H)    scalar or per-position in [0, 1]

    v:     (B, H, token_dim) = (B, 16, 3)

Design choices (lightweight for MSD toy problem):
    - Each timestep token (token_dim,) projected to (d_model,) as one sequence token
    - Per-position sinusoidal tau embedding → projected to d_model, added to tokens
    - obs linearly projected to d_model, used as cross-attention context
    - N_layers transformer blocks with cross-attention (obs as key/value)
    - Final linear head → (B, H, token_dim)

    tau can be scalar (B,) or per-position (B, H) to support simulated_delay
    training and inpainting inference (frozen positions get tau=1.0).

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
    Sinusoidal embedding for tau ∈ [0, 1].

    Args:
        tau: (B,) or (B, H) — noise level (scalar or per-position)
        dim: int  — embedding dimension (must be even)

    Returns:
        emb: (B, dim) if tau is (B,), or (B, H, dim) if tau is (B, H)
    """
    assert dim % 2 == 0
    half = dim // 2
    # frequencies: (half,)
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=tau.device) / half
    )

    if tau.ndim == 1:
        # (B,) → (B, dim)
        args = tau[:, None] * freqs[None, :]          # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    elif tau.ndim == 2:
        # (B, H) → (B, H, dim)
        args = tau.unsqueeze(-1) * freqs[None, None, :]  # (B, H, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, H, dim)
    else:
        raise ValueError(f"tau must be 1D or 2D, got ndim={tau.ndim}")

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
        obs:   (B, obs_dim)      = (B, 3)       — conditioning [x, x_dot, x_ref]
        tau:   (B,)              scalar ∈ [0,1]  — noise level

        v:     (B, H, token_dim) = (B, 16, 3)  — predicted velocity field
    """

    def __init__(
        self,
        H: int         = 16,
        token_dim: int = 3,
        obs_dim: int   = 3,
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
        tau:   torch.Tensor,  # (B,) scalar or (B, H) per-position
    ) -> torch.Tensor:
        """
        Returns:
            v: (B, H, token_dim) — predicted velocity field
        """
        B = X_tau.shape[0]

        # tokens: (B, H, d_model)
        tokens = self.token_proj(X_tau) + self.pos_emb

        # Per-position tau embedding added directly to tokens
        if tau.ndim == 1:
            # (B,) → broadcast to (B, H)
            tau_pp = tau[:, None].expand(-1, self.H)  # (B, H)
        else:
            tau_pp = tau  # already (B, H)

        # tau_emb: (B, H, d_model) — per-position
        tau_emb = self.tau_proj(sinusoidal_embedding(tau_pp, self.d_model))
        tokens = tokens + tau_emb

        # obs_emb: (B, 1, d_model) — cross-attention context (obs only)
        obs_emb = self.obs_proj(obs).unsqueeze(1)  # (B, 1, d_model)

        for block in self.blocks:
            tokens = block(tokens, obs_emb)

        # v: (B, H, token_dim)
        v = self.out_proj(self.out_norm(tokens))
        return v

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
