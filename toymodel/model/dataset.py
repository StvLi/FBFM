"""
model/dataset.py — Flow Matching Dataset

Loads expert_dataset.npz and prepares (X0, X1, obs) triples for FM training.

Flow Matching training setup:
    - X1: (H, action_dim) = (16, 1)  — clean action chunk (target)
    - X0: (H, action_dim) = (16, 1)  — noise sample ~ N(0, I)
    - obs: (obs_dim,)     = (2,)      — conditioning observation (first state of chunk)
    - X_tau = tau * X1 + (1 - tau) * X0   — linear interpolation
    - target velocity: v_target = X1 - X0

Normalization:
    Actions are normalized to zero mean, unit std across the training set.
    Stats are saved alongside the dataset for use at inference time.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class FlowMatchingDataset(Dataset):
    """
    PyTorch Dataset wrapping the PID expert dataset.

    Each item returns:
        X1:  (H, action_dim)  — clean action chunk, normalized
        obs: (obs_dim,)       — conditioning state (first state of chunk)

    X0 (noise) and tau are sampled fresh each call in the training loop,
    not stored here, to avoid overfitting to fixed noise.

    Shapes:
        H          = 16   (prediction horizon)
        action_dim = 1
        obs_dim    = 2    ([x, x_dot])
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
            normalize: whether to normalize actions to N(0,1)
            stats:     pre-computed {'mean': ..., 'std': ...} — if None, computed from data
        """
        data = np.load(npz_path)

        # states:  (N, H+1, 2)
        # actions: (N, H, 1)
        # refs:    (N, H, 1)
        self.states  = torch.from_numpy(data["states"]).float()   # (N, H+1, 2)
        self.actions = torch.from_numpy(data["actions"]).float()  # (N, H, 1)

        self.N, self.H, self.action_dim = self.actions.shape
        self.obs_dim = self.states.shape[-1]  # 2

        # Normalization stats (computed over all actions in dataset)
        if normalize:
            if stats is None:
                # actions: (N, H, 1) → flatten to (N*H,) for stats
                flat = self.actions.reshape(-1)
                self.stats = {
                    "mean": flat.mean().item(),
                    "std":  flat.std().item() + 1e-8,
                }
            else:
                self.stats = stats

            self.actions = (self.actions - self.stats["mean"]) / self.stats["std"]
        else:
            self.stats = {"mean": 0.0, "std": 1.0}

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        """
        Returns:
            X1:  (H, action_dim) — clean action chunk (normalized)
            obs: (obs_dim,)      — first state of chunk as conditioning
        """
        # X1: (H, action_dim) = (16, 1)
        X1 = self.actions[idx]

        # obs: (obs_dim,) = (2,) — use the first state of the chunk
        obs = self.states[idx, 0]  # (2,)

        return X1, obs

    def denormalize(self, X: torch.Tensor) -> torch.Tensor:
        """Undo normalization. X: (..., action_dim) → original scale."""
        return X * self.stats["std"] + self.stats["mean"]
