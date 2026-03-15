"""
Training script for the 1D Flow-Matching model.

Generates expert data → trains the lightweight DiT → saves checkpoint.

Usage:
    python -m pre_test_1.train [--epochs 200] [--batch_size 256] [--lr 3e-4]
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from pre_test_1.expert_data import generate_dataset, save_dataset, load_dataset, ExpertChunkDataset
from pre_test_1.flow_model import FlowMatchingDiT, flow_matching_loss, flow_matching_sample
from pre_test_1.physics_env import EnvConfig


def train(
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    horizon: int = 16,
    num_trajectories: int = 500,
    steps_per_traj: int = 300,
    hidden_dim: int = 128,
    num_layers: int = 4,
    save_dir: str = "pre_test_1/checkpoints",
    data_path: str = "pre_test_1/data/expert_dataset.pt",
    device: str = "auto",
    seed: int = 42,
):
    """Full training pipeline."""
    torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    # ----------------------------------------------------------------
    # 1. Generate or load data
    # ----------------------------------------------------------------
    if os.path.exists(data_path):
        print(f"[Train] Loading existing dataset from {data_path}")
        dataset = load_dataset(data_path)
    else:
        print(f"[Train] Generating new dataset ({num_trajectories} trajectories)...")
        env_cfg = EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02)
        dataset = generate_dataset(
            num_trajectories=num_trajectories,
            steps_per_traj=steps_per_traj,
            horizon=horizon,
            env_cfg=env_cfg,
            seed=seed,
        )
        save_dataset(dataset, data_path)

    # Compute normalization statistics for better training
    chunk_mean = dataset.chunks.mean(dim=0, keepdim=True)
    chunk_std = dataset.chunks.std(dim=0, keepdim=True).clamp(min=1e-6)
    obs_mean = dataset.observations.mean(dim=0, keepdim=True)
    obs_std = dataset.observations.std(dim=0, keepdim=True).clamp(min=1e-6)

    # Target normalization: use the first timestep of target as the goal signal
    # target shape is (N, H, state_dim) — we use target[:, 0, :] as the goal
    target_goals = dataset.targets[:, 0, :]  # (N, state_dim)
    target_mean = target_goals.mean(dim=0, keepdim=True)
    target_std = target_goals.std(dim=0, keepdim=True).clamp(min=1e-6)

    # Normalize
    normalized_chunks = (dataset.chunks - chunk_mean) / chunk_std
    normalized_obs = (dataset.observations - obs_mean) / obs_std
    normalized_targets = (target_goals - target_mean) / target_std
    norm_dataset = ExpertChunkDataset(normalized_obs, normalized_chunks, dataset.targets)
    # Store normalized targets as a separate attribute for training
    norm_dataset._normalized_targets = normalized_targets

    loader = DataLoader(norm_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[Train] Dataset: {len(norm_dataset)} samples, {len(loader)} batches/epoch")

    # ----------------------------------------------------------------
    # 2. Build model
    # ----------------------------------------------------------------
    state_dim = 2
    action_dim = 1

    model = FlowMatchingDiT(
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        target_dim=state_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    best_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Train] Starting training for {epochs} epochs...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in loader:
            obs = batch["observation"].to(device)      # (B, state_dim)
            chunks = batch["chunk"].to(device)          # (B, H, chunk_dim)
            target_goal = batch["target_goal"].to(device)  # (B, state_dim)

            loss = flow_matching_loss(model, chunks, obs, target=target_goal)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:4d}/{epochs} | loss: {avg_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e} | time: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "horizon": horizon,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "target_dim": state_dim,
                    },
                    "normalization": {
                        "chunk_mean": chunk_mean,
                        "chunk_std": chunk_std,
                        "obs_mean": obs_mean,
                        "obs_std": obs_std,
                        "target_mean": target_mean,
                        "target_std": target_std,
                    },
                    "epoch": epoch,
                    "loss": best_loss,
                },
                os.path.join(save_dir, "best_model.pt"),
            )

    total_time = time.time() - t_start
    print(f"\n[Train] Training complete in {total_time:.1f}s. Best loss: {best_loss:.6f}")
    print(f"[Train] Best model saved to {os.path.join(save_dir, 'best_model.pt')}")

    # ----------------------------------------------------------------
    # 4. Quick sanity check: sample from model
    # ----------------------------------------------------------------
    model.eval()
    test_obs = normalized_obs[:4].to(device)
    test_targets = normalized_targets[:4].to(device)
    with torch.no_grad():
        samples = flow_matching_sample(model, test_obs, num_steps=20, horizon=horizon, target=test_targets)
    # Denormalize
    samples_denorm = samples.cpu() * chunk_std + chunk_mean
    print(f"\n[Train] Sanity check - sampled chunk shape: {samples_denorm.shape}")
    print(f"  Sample[0] first step: state={samples_denorm[0, 0, :2].tolist()}, "
          f"action={samples_denorm[0, 0, 2:].tolist()}")

    return model


def load_trained_model(
    checkpoint_path: str = "pre_test_1/checkpoints/best_model.pt",
    device: str = "auto",
) -> tuple:
    """Load a trained model and its normalization stats.

    Returns:
        (model, norm_stats) where norm_stats is a dict with chunk_mean/std, obs_mean/std.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = FlowMatchingDiT(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        horizon=cfg["horizon"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        target_dim=cfg.get("target_dim", cfg["state_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm_stats = ckpt["normalization"]
    # Move normalization tensors to device
    for k, v in norm_stats.items():
        if isinstance(v, torch.Tensor):
            norm_stats[k] = v.to(device)

    print(f"[Load] Model loaded from {checkpoint_path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f})")
    return model, norm_stats


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D Flow-Matching model for FBFM pre-experiment")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_trajectories", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))
