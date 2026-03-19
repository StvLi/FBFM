"""
model/dataset.py — Flow Matching Dataset

Loads expert_dataset.npz and prepares (X0, X1, obs) triples for FM training.

Flow Matching training setup:
    - X1: (H, token_dim) = (16, 3)  — clean token chunk (target)
    - X0: (H, token_dim) = (16, 3)  — noise sample ~ N(0, I)
    - obs: (obs_dim,)    = (3,)     — conditioning observation [x, x_dot, x_ref]
    - X_tau = tau * X1 + (1 - tau) * X0   — linear interpolation
    - target velocity: v_target = X1 - X0

Token format:
    Token[h] = [state_{h+1}, action_h] = [x_{h+1}, x_dot_{h+1}, u_h]
    Constructed as: cat(states[:, 1:H+1, :], actions, dim=-1) → (N, H, 3)

Normalization:
    Per-dimension stats: mean and std are (token_dim,) = (3,) vectors.
    Each dimension (x, x_dot, u) normalized independently to zero-mean unit-std.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class FlowMatchingDataset(Dataset):
    """
    PyTorch Dataset wrapping the PID expert dataset.

    Each item returns:
        X1:  (H, token_dim)  — clean token chunk, normalized
        obs: (obs_dim,)      — conditioning state (first state of chunk)

    X0 (noise) and tau are sampled fresh each call in the training loop,
    not stored here, to avoid overfitting to fixed noise.

    Shapes:
        H         = 16   (prediction horizon)
        token_dim = 3    ([x, x_dot, u])
        obs_dim   = 3    ([x, x_dot, x_ref])
    """

    def __init__(
        self,
        npz_path: str,
        normalize: bool = True,
        stats: dict = None,
    ):
        """
        Args:
            npz_path:  path to expert_dataset.npz
            normalize: whether to normalize tokens per-dim to N(0,1)
            stats:     pre-computed {'mean': np.array(3,), 'std': np.array(3,)}
                       — if None, computed from data
        """
        data = np.load(npz_path)

        # states:  (N, H+1, 2)
        # actions: (N, H, 1)
        states  = torch.from_numpy(data["states"]).float()   # (N, H+1, 2)
        actions = torch.from_numpy(data["actions"]).float()   # (N, H, 1)

        N, H_plus1, state_dim = states.shape
        H = H_plus1 - 1
        self.N   = N
        self.H   = H

        # Token[h] = [state_{h+1}, action_h] → cat along last dim
        # states[:, 1:H+1, :]: (N, H, 2) — state AFTER executing action h
        # actions:              (N, H, 1)
        self.tokens = torch.cat([states[:, 1:H+1, :], actions], dim=-1)  # (N, H, 3)
        self.token_dim = self.tokens.shape[-1]  # 3

        # obs = initial state + reference position → [x, x_dot, x_ref]
        refs = torch.from_numpy(data["refs"]).float()  # (N, H, 1)
        self.obs = torch.cat([states[:, 0, :], refs[:, 0, :]], dim=-1)  # (N, 3)
        self.obs_dim = self.obs.shape[-1]  # 3

        # Per-dimension normalization stats
        if normalize:
            if stats is None:
                flat = self.tokens.reshape(-1, self.token_dim)  # (N*H, 3)
                self.stats = {
                    "mean": flat.mean(dim=0).numpy(),            # (3,)
                    "std":  (flat.std(dim=0) + 1e-8).numpy(),    # (3,)
                }
            else:
                self.stats = stats

            mean_t = torch.from_numpy(self.stats["mean"]).float()  # (3,)
            std_t  = torch.from_numpy(self.stats["std"]).float()   # (3,)
            self.tokens = (self.tokens - mean_t) / std_t
        else:
            self.stats = {
                "mean": np.zeros(self.token_dim, dtype=np.float32),
                "std":  np.ones(self.token_dim, dtype=np.float32),
            }

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        """
        Returns:
            X1:  (H, token_dim) — clean token chunk (normalized)
            obs: (obs_dim,)     — first state of chunk as conditioning
        """
        return self.tokens[idx], self.obs[idx]

    def denormalize(self, X: torch.Tensor) -> torch.Tensor:
        """Undo normalization. X: (..., token_dim) → original scale."""
        mean = torch.tensor(self.stats["mean"], dtype=X.dtype, device=X.device)
        std  = torch.tensor(self.stats["std"],  dtype=X.dtype, device=X.device)
        return X * std + mean
