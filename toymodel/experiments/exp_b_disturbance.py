"""
experiments/exp_b_disturbance.py — Exp-B: In-Chunk Impulse Disturbance

Applies a sudden external force at step 3 within each execution chunk.
Tests whether FBFM can respond within the same chunk, while FM and RTC
must wait until the next inference cycle.

Disturbance schedule:
    Every s_chunk steps, at the 3rd execution step, apply F_impulse for 1 step.

Saves results to results/exp_b_disturbance/
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.runner import load_policy, run_three_algos, make_test_refs


F_IMPULSE  = 5.0   # N — impulse force magnitude
IMPULSE_AT = 2     # step index within chunk (0-indexed) when impulse fires
S_CHUNK    = 5     # must match algo_cfg s_chunk
T_TEST     = 200
DT         = 0.05


def make_disturbance_fn(f_impulse: float, impulse_at: int, s_chunk: int):
    """
    Returns a disturbance_fn(step) -> float that fires an impulse at
    position `impulse_at` within every chunk of length `s_chunk`.
    """
    def disturbance_fn(step: int) -> float:
        pos_in_chunk = step % s_chunk
        return f_impulse if pos_in_chunk == impulse_at else 0.0
    return disturbance_fn


def run_exp_b(
    ckpt_path: str = "checkpoints/fm_best.pt",
    device: str = "cpu",
    save_dir: str = "results/exp_b_disturbance",
    seed: int = 42,
):
    os.makedirs(save_dir, exist_ok=True)

    policy, action_stats, _ = load_policy(ckpt_path, device)
    refs = make_test_refs(T=T_TEST, dt=DT)

    disturbance_fn = make_disturbance_fn(F_IMPULSE, IMPULSE_AT, S_CHUNK)

    # Test under nominal params and one mismatch condition
    test_conditions = {
        "nominal": {"m": 1.0, "k": 2.0, "c": 0.5, "dt": DT},
        "mass×2":  {"m": 2.0, "k": 2.0, "c": 0.5, "dt": DT},
    }

    all_metrics = {}

    for ref_name, ref_seq in refs.items():
        print(f"\n=== Exp-B | ref={ref_name} ===")
        all_metrics[ref_name] = {}

        for cond_name, env_cfg in test_conditions.items():
            print(f"  Condition: {cond_name}")

            results = run_three_algos(
                policy, action_stats, env_cfg, ref_seq,
                disturbance_fn=disturbance_fn,
                seed=seed, device=device,
            )

            all_metrics[ref_name][cond_name] = {
                algo: {
                    "rmse":    results[algo]["rmse"],
                    "mae":     results[algo]["mae"],
                    "max_err": results[algo]["max_err"],
                }
                for algo in ["fm", "rtc", "fbfm"]
            }

            np.savez(
                os.path.join(save_dir, f"traj_{ref_name}_{cond_name.replace('×','x')}.npz"),
                **{f"{algo}_{k}": v
                   for algo in ["fm", "rtc", "fbfm"]
                   for k, v in results[algo].items()
                   if isinstance(v, np.ndarray)},
                ref_seq=ref_seq,
                impulse_steps=np.array([t for t in range(T_TEST)
                                        if t % S_CHUNK == IMPULSE_AT]),
            )

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nExp-B complete. Results saved to {save_dir}/")
    return all_metrics


if __name__ == "__main__":
    run_exp_b()
