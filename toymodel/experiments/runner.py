"""
experiments/runner.py — Unified experiment runner

Loads a trained checkpoint, runs all three algorithms (FM / RTC / FBFM)
on a given MSD environment + reference trajectory, and returns results.

Usage:
    from experiments.runner import load_policy, run_three_algos

Shapes:
    obs:      (obs_dim,)       = (3,)
    chunk:    (H, token_dim)   = (16, 3)  — each token [x, x_dot, u]
    result:   dict with xs_true / xs_obs / actions / times / ref_seq  — all (T,)
"""

import sys
import os
import numpy as np
import torch

TOYMODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOYMODEL_ROOT)

from model.dit import FlowMatchingDiT
from sim.msd_env import MSDEnv
from algo.fm   import rollout_fm
from algo.rtc  import rollout_rtc
from algo.fbfm import rollout_fbfm


# -----------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------

def load_policy(
    ckpt_path: str = "checkpoints/fm_best.pt",
    device: str = "cpu",
) -> tuple:
    """
    Load trained FlowMatchingDiT from checkpoint.

    Returns:
        policy:      FlowMatchingDiT (eval mode)
        token_stats: {'mean': np.array(3,), 'std': np.array(3,)}
        cfg:         training config dict
    """
    # PyTorch 2.6+ defaults to weights_only=True, which can fail for
    # checkpoints storing numpy scalars in cfg/stats. This project saves
    # trusted local checkpoints with metadata, so we explicitly disable it.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})

    policy = FlowMatchingDiT(
        H         = cfg.get("H",         16),
        token_dim = cfg.get("token_dim",  3),
        obs_dim   = cfg.get("obs_dim",    3),
        d_model   = cfg.get("d_model",  128),
        n_heads   = cfg.get("n_heads",    4),
        n_layers  = cfg.get("n_layers",   4),
    ).to(device)

    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    token_stats = ckpt.get("token_stats", {"mean": np.zeros(3), "std": np.ones(3)})
    return policy, token_stats, cfg


# -----------------------------------------------------------------------
# Run all three algorithms on one scenario
# -----------------------------------------------------------------------

def run_three_algos(
    policy: FlowMatchingDiT,
    token_stats: dict,
    env_cfg: dict,              # {'m', 'k', 'c', 'dt'} — test-time env params
    ref_seq: np.ndarray,        # (T,) reference trajectory
    disturbance_fn=None,        # Callable: (step: int) -> float  external force
    algo_cfg: dict = None,      # override default algo hyperparams
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """
    Run FM, RTC, FBFM on the same scenario and return all results.

    Args:
        policy:          trained policy
        token_stats:     per-dim normalization stats
        env_cfg:         test-time MSD parameters (may differ from training)
        ref_seq:         (T,) reference positions
        disturbance_fn:  optional Callable(step) -> float for Exp-B
        algo_cfg:        optional overrides for {s_chunk, n_steps, n_inner, beta}
        seed:            random seed for env noise
        device:          torch device

    Returns:
        dict with keys 'fm', 'rtc', 'fbfm', each containing the rollout dict
    """
    cfg = {
        "s_chunk": 5,
        "n_steps": 20,
        "n_inner": 4,
        "beta":    3.0,
    }
    if algo_cfg:
        cfg.update(algo_cfg)

    results = {}

    def _set_rollout_seed(s: int):
        """Set deterministic seeds so cross-algorithm comparisons are reproducible."""
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    for algo_idx, algo_name in enumerate(["fm", "rtc", "fbfm"]):
        # Keep each algorithm rollout reproducible across repeated experiment runs.
        _set_rollout_seed(seed + algo_idx)

        env = MSDEnv(
            m=env_cfg.get("m", 1.0),
            k=env_cfg.get("k", 2.0),
            c=env_cfg.get("c", 0.5),
            dt=env_cfg.get("dt", 0.05),
            seed=seed,
        )

        if disturbance_fn is not None:
            _original_step = env.step
            _step_counter  = [0]

            def _step_with_disturbance(u, disturbance=0.0, add_obs_noise=True):
                d = disturbance_fn(_step_counter[0])
                _step_counter[0] += 1
                return _original_step(u, disturbance=d, add_obs_noise=add_obs_noise)

            env.step = _step_with_disturbance

        env.reset()

        if algo_name == "fm":
            result = rollout_fm(
                policy, env, ref_seq,
                s_chunk=cfg["s_chunk"],
                n_steps=cfg["n_steps"],
                token_stats=token_stats,
                device=device,
            )
        elif algo_name == "rtc":
            result = rollout_rtc(
                policy, env, ref_seq,
                s_chunk=cfg["s_chunk"],
                n_steps=cfg["n_steps"],
                beta=cfg["beta"],
                token_stats=token_stats,
                device=device,
            )
        else:  # fbfm
            result = rollout_fbfm(
                policy, env, ref_seq,
                n_steps=cfg["n_steps"],
                n_inner=cfg["n_inner"],
                beta=cfg["beta"],
                token_stats=token_stats,
                device=device,
            )

        err = ref_seq - result["xs_true"]
        result["rmse"]    = float(np.sqrt((err**2).mean()))
        result["mae"]     = float(np.abs(err).mean())
        result["max_err"] = float(np.abs(err).max())

        results[algo_name] = result
        print(f"  [{algo_name.upper():4s}]  RMSE={result['rmse']:.4f}  "
              f"MAE={result['mae']:.4f}  max={result['max_err']:.4f}")

    return results


# -----------------------------------------------------------------------
# Reference trajectory helpers (test-time)
# -----------------------------------------------------------------------

def make_test_refs(T: int = 200, dt: float = 0.05) -> dict:
    """
    Build the two standard test reference trajectories.

    Returns:
        dict with 'step' and 'sinusoidal' keys, each (T,)
    """
    t = np.arange(T) * dt
    return {
        "step":       np.ones(T, dtype=np.float32),
        "sinusoidal": np.sin(2 * np.pi * 0.15 * t).astype(np.float32),
    }
