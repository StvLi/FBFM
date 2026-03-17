"""
experiments/exp_a_mismatch.py — Exp-A: Parameter Mismatch

Tests all three algorithms under model-plant mismatch:
    Training env: m=1.0, k=2.0, c=0.5
    Test envs:    m×1.5, m×2, m×3, k×3  (one at a time)

Expected result: FBFM's within-chunk feedback allows it to detect and
correct for the changed dynamics, yielding lower tracking error than
FM and RTC which are blind during execution.

Saves results to results/exp_a_mismatch.npz
"""

import sys
import os
import json
import numpy as np

TOYMODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOYMODEL_ROOT)

from experiments.runner import load_policy, run_three_algos, make_test_refs


# -----------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------

MISMATCH_CONDITIONS = {
    "nominal":    {"m": 1.0, "k": 2.0, "c": 0.5},
    "mass×1.5":   {"m": 1.5, "k": 2.0, "c": 0.5},
    "mass×2":     {"m": 2.0, "k": 2.0, "c": 0.5},
    "mass×3":     {"m": 3.0, "k": 2.0, "c": 0.5},
    "stiff×3":    {"m": 1.0, "k": 6.0, "c": 0.5},
}

T_TEST  = 200   # test trajectory length (steps)
DT      = 0.05  # must match training


def run_exp_a(
    ckpt_path: str = None,
    device: str = "cpu",
    save_dir: str = None,
    seed: int = 42,
):
    if ckpt_path is None:
        ckpt_path = os.path.join(TOYMODEL_ROOT, "checkpoints", "fm_best.pt")
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results", "exp_a_mismatch")
    os.makedirs(save_dir, exist_ok=True)

    policy, token_stats, _ = load_policy(ckpt_path, device)
    refs = make_test_refs(T=T_TEST, dt=DT)

    all_metrics = {}

    for ref_name, ref_seq in refs.items():
        print(f"\n=== Exp-A | ref={ref_name} ===")
        all_metrics[ref_name] = {}

        for cond_name, env_params in MISMATCH_CONDITIONS.items():
            print(f"  Condition: {cond_name}  params={env_params}")
            env_cfg = {**env_params, "dt": DT}

            results = run_three_algos(
                policy, token_stats, env_cfg, ref_seq,
                seed=seed, device=device,
            )

            # Store metrics
            all_metrics[ref_name][cond_name] = {
                algo: {
                    "rmse":    results[algo]["rmse"],
                    "mae":     results[algo]["mae"],
                    "max_err": results[algo]["max_err"],
                }
                for algo in ["fm", "rtc", "fbfm"]
            }

            # Save trajectories for plotting
            np.savez(
                os.path.join(save_dir, f"traj_{ref_name}_{cond_name.replace('×','x')}.npz"),
                **{f"{algo}_{k}": v
                   for algo in ["fm", "rtc", "fbfm"]
                   for k, v in results[algo].items()
                   if isinstance(v, np.ndarray)},
                ref_seq=ref_seq,
            )

    # Save metrics summary
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nExp-A complete. Results saved to {save_dir}/")
    return all_metrics


if __name__ == "__main__":
    run_exp_a()
