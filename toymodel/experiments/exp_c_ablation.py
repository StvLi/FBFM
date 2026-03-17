"""
experiments/exp_c_ablation.py — Exp-C: FBFM Ablation (k_p = 0)

Compares full FBFM vs FBFM with feedback term disabled (k_p = 0).
Isolates the contribution of the real-time guidance term.

To disable feedback: pass beta=0 to guided_inference_fbfm, which forces
k_p = min(0, ...) = 0, effectively making FBFM identical to vanilla FM
but with the interleaved execution rhythm.

Saves results to results/exp_c_ablation/
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.runner import load_policy, run_three_algos, make_test_refs


T_TEST = 200
DT     = 0.05

TEST_CONDITIONS = {
    "nominal": {"m": 1.0, "k": 2.0, "c": 0.5, "dt": DT},
    "mass×2":  {"m": 2.0, "k": 2.0, "c": 0.5, "dt": DT},
    "mass×3":  {"m": 3.0, "k": 2.0, "c": 0.5, "dt": DT},
}


def run_exp_c(
    ckpt_path: str = "checkpoints/fm_best.pt",
    device: str = "cpu",
    save_dir: str = "results/exp_c_ablation",
    seed: int = 42,
):
    os.makedirs(save_dir, exist_ok=True)

    policy, action_stats, _ = load_policy(ckpt_path, device)
    refs = make_test_refs(T=T_TEST, dt=DT)

    all_metrics = {}

    for ref_name, ref_seq in refs.items():
        print(f"\n=== Exp-C | ref={ref_name} ===")
        all_metrics[ref_name] = {}

        for cond_name, env_cfg in TEST_CONDITIONS.items():
            print(f"  Condition: {cond_name}")

            # Full FBFM (beta=10)
            results_full = run_three_algos(
                policy, action_stats, env_cfg, ref_seq,
                algo_cfg={"beta": 10.0},
                seed=seed, device=device,
            )

            # Ablated FBFM (beta=0 → k_p=0, no feedback)
            results_ablated = run_three_algos(
                policy, action_stats, env_cfg, ref_seq,
                algo_cfg={"beta": 0.0},
                seed=seed, device=device,
            )

            all_metrics[ref_name][cond_name] = {
                "fbfm_full": {
                    "rmse":    results_full["fbfm"]["rmse"],
                    "mae":     results_full["fbfm"]["mae"],
                    "max_err": results_full["fbfm"]["max_err"],
                },
                "fbfm_no_feedback": {
                    "rmse":    results_ablated["fbfm"]["rmse"],
                    "mae":     results_ablated["fbfm"]["mae"],
                    "max_err": results_ablated["fbfm"]["max_err"],
                },
                "rtc": {
                    "rmse":    results_full["rtc"]["rmse"],
                    "mae":     results_full["rtc"]["mae"],
                    "max_err": results_full["rtc"]["max_err"],
                },
            }

            np.savez(
                os.path.join(save_dir, f"traj_{ref_name}_{cond_name.replace('×','x')}.npz"),
                fbfm_full_xs_true=results_full["fbfm"]["xs_true"],
                fbfm_ablated_xs_true=results_ablated["fbfm"]["xs_true"],
                rtc_xs_true=results_full["rtc"]["xs_true"],
                ref_seq=ref_seq,
            )

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nExp-C complete. Results saved to {save_dir}/")
    return all_metrics


if __name__ == "__main__":
    run_exp_c()
