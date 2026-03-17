"""
experiments/exp_d_sensitivity.py — Exp-D: Feedback Frequency Sensitivity

Sweeps n_inner (denoising steps per interleave block) to find the optimal
interleave granularity for FBFM.

n_inner values: 1, 2, 4, 8, 16
    n_inner=1:  execute 1 action per denoising step (maximum feedback rate)
    n_inner=16: execute 16 actions before any denoising (= no interleaving)

Saves results to results/exp_d_sensitivity/
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.runner import load_policy, run_three_algos, make_test_refs
from algo.fbfm import rollout_fbfm
from sim.msd_env import MSDEnv


T_TEST   = 200
DT       = 0.05
N_INNER_VALUES = [1, 2, 4, 8, 16]

TEST_CONDITIONS = {
    "nominal": {"m": 1.0, "k": 2.0, "c": 0.5, "dt": DT},
    "mass×2":  {"m": 2.0, "k": 2.0, "c": 0.5, "dt": DT},
}


def run_exp_d(
    ckpt_path: str = "checkpoints/fm_best.pt",
    device: str = "cpu",
    save_dir: str = "results/exp_d_sensitivity",
    seed: int = 42,
):
    os.makedirs(save_dir, exist_ok=True)

    policy, action_stats, _ = load_policy(ckpt_path, device)
    refs = make_test_refs(T=T_TEST, dt=DT)

    all_metrics = {}

    for ref_name, ref_seq in refs.items():
        print(f"\n=== Exp-D | ref={ref_name} ===")
        all_metrics[ref_name] = {}

        for cond_name, env_cfg in TEST_CONDITIONS.items():
            print(f"  Condition: {cond_name}")
            all_metrics[ref_name][cond_name] = {}

            for n_inner in N_INNER_VALUES:
                env = MSDEnv(
                    m=env_cfg["m"], k=env_cfg["k"], c=env_cfg["c"],
                    dt=env_cfg["dt"], seed=seed,
                )
                env.reset()

                result = rollout_fbfm(
                    policy, env, ref_seq,
                    n_steps=16, n_inner=n_inner,
                    beta=10.0,
                    action_stats=action_stats,
                    device=device,
                )

                err = ref_seq - result["xs_true"]
                rmse = float(np.sqrt((err**2).mean()))
                mae  = float(np.abs(err).mean())

                all_metrics[ref_name][cond_name][f"n_inner={n_inner}"] = {
                    "rmse": rmse, "mae": mae,
                }
                print(f"    n_inner={n_inner:2d}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nExp-D complete. Results saved to {save_dir}/")
    return all_metrics


if __name__ == "__main__":
    run_exp_d()
