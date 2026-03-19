"""
train/train_fm.py — Flow Matching training loop

Loss:
    L = E_{tau, X0, X1} [ || v_theta(X^tau, obs, tau) - (X1 - X0) ||^2 ]

    where:
        X0    ~ N(0, I)                          (B, H, token_dim)
        X1    = clean token chunk from dataset   (B, H, token_dim)
        tau   ~ Uniform(0, 1)                    (B,)
        X^tau = tau * X1 + (1 - tau) * X0        (B, H, token_dim)  — linear interpolation
        target velocity = X1 - X0               (B, H, token_dim)

    token_dim = 3: each token is [x, x_dot, u]

Usage:
    python train/train_fm.py
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# All output paths are anchored to the toymodel/ root, regardless of CWD
TOYMODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOYMODEL_ROOT)

from model.dit import FlowMatchingDiT
from model.dataset import FlowMatchingDataset


# -----------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------

CFG = {
    # Data
    "data_path":    os.path.join(TOYMODEL_ROOT, "data", "expert_dataset.npz"),
    "val_fraction": 0.1,

    # Model
    "H":         16,
    "token_dim": 3,
    "obs_dim":   3,
    "d_model":    128,
    "n_heads":    4,
    "n_layers":   4,

    # Training
    "epochs":     800,
    "batch_size": 512,
    "lr":         3e-4,
    "weight_decay": 1e-5,
    "lr_warmup_epochs": 50,

    # Checkpointing
    "ckpt_dir":   os.path.join(TOYMODEL_ROOT, "checkpoints"),
    "ckpt_name":  "fm_best.pt",
    "save_every": 400,

    # Misc
    "seed":   42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def fm_loss(
    policy: FlowMatchingDiT,
    X1: torch.Tensor,    # (B, H, token_dim) — clean chunk
    obs: torch.Tensor,   # (B, obs_dim)
    device: str,
) -> torch.Tensor:
    """
    Compute one-step FM loss for a batch.

    Returns:
        loss: scalar tensor
    """
    B = X1.shape[0]

    # X0 ~ N(0, I),  shape: (B, H, token_dim)
    X0 = torch.randn_like(X1)

    # tau ~ Uniform(0, 1),  shape: (B,)
    tau = torch.rand(B, device=device)

    # Linear interpolation: X^tau = tau * X1 + (1 - tau) * X0
    tau_bc = tau[:, None, None]  # (B, 1, 1) for broadcasting
    X_tau = tau_bc * X1 + (1.0 - tau_bc) * X0  # (B, H, token_dim)

    # Target velocity: v_target = X1 - X0
    v_target = X1 - X0  # (B, H, token_dim)

    # Predicted velocity
    v_pred = policy(X_tau, obs, tau)  # (B, H, token_dim)

    loss = nn.functional.mse_loss(v_pred, v_target)
    return loss


def get_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_lambda


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def train(cfg: dict = CFG):
    set_seed(cfg["seed"])
    device = cfg["device"]
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    print(f"Device: {device}")

    # --- Dataset ---
    dataset = FlowMatchingDataset(cfg["data_path"], normalize=True)
    s = dataset.stats
    print(f"Dataset: {len(dataset)} chunks  |  "
          f"token stats: mean={s['mean']}, std={s['std']}")

    n_val  = max(1, int(len(dataset) * cfg["val_fraction"]))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, drop_last=False)

    # --- Model ---
    policy = FlowMatchingDiT(
        H=cfg["H"], token_dim=cfg["token_dim"], obs_dim=cfg["obs_dim"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
    ).to(device)
    print(f"Model params: {policy.count_params():,}")

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(cfg["lr_warmup_epochs"], cfg["epochs"])
    )

    # --- Training ---
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        policy.train()
        train_losses = []
        for X1, obs in train_loader:
            # X1:  (B, H, token_dim)
            # obs: (B, obs_dim)
            X1  = X1.to(device)
            obs = obs.to(device)

            optimizer.zero_grad()
            loss = fm_loss(policy, X1, obs, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        policy.eval()
        val_losses = []
        with torch.no_grad():
            for X1, obs in val_loader:
                X1  = X1.to(device)
                obs = obs.to(device)
                loss = fm_loss(policy, X1, obs, device)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg['epochs']}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(cfg["ckpt_dir"], cfg["ckpt_name"])
            torch.save({
                "epoch":        epoch,
                "model_state":  policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":     val_loss,
                "cfg":          cfg,
                "token_stats":  dataset.stats,
            }, ckpt_path)

        # Periodic checkpoint
        if epoch % cfg["save_every"] == 0:
            periodic_path = os.path.join(cfg["ckpt_dir"], f"fm_epoch{epoch}.pt")
            torch.save({"epoch": epoch, "model_state": policy.state_dict(),
                        "token_stats": dataset.stats}, periodic_path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")
    print(f"Best checkpoint saved to: {os.path.join(cfg['ckpt_dir'], cfg['ckpt_name'])}")

    # Save loss history
    history_path = os.path.join(cfg["ckpt_dir"], "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return policy, history, dataset.stats


# -----------------------------------------------------------------------
# Loss curve plot
# -----------------------------------------------------------------------

def plot_loss_curve(history: dict, save_path: str = None):
    if save_path is None:
        save_path = os.path.join(TOYMODEL_ROOT, "checkpoints", "loss_curve.png")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
    })

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="Val loss",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"FM Loss $\|\hat{v} - v^*\|^2$")
    ax.set_title("Flow Matching Training Convergence")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to {save_path}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # First collect full dataset if not present
    if not os.path.exists(CFG["data_path"]):
        print(f"Dataset not found at {CFG['data_path']}, collecting now...")
        sys.path.insert(0, ".")
        from data.collect_dataset import collect_dataset
        collect_dataset(n_trajs=200, T_per_traj=200, H=16,
                        save_path=CFG["data_path"])

    policy, history, token_stats = train()
    plot_loss_curve(history)
