"""
experiments/run_all.py — Run all four experiments sequentially

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --ckpt checkpoints/fm_best.pt --device cuda
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp_a_mismatch    import run_exp_a
from experiments.exp_b_disturbance import run_exp_b
from experiments.exp_c_ablation    import run_exp_c
from experiments.exp_d_sensitivity import run_exp_d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="checkpoints/fm_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    kwargs = dict(ckpt_path=args.ckpt, device=args.device, seed=args.seed)

    print("=" * 60)
    print("Running Exp-A: Parameter Mismatch")
    print("=" * 60)
    run_exp_a(**kwargs)

    print("\n" + "=" * 60)
    print("Running Exp-B: In-Chunk Impulse Disturbance")
    print("=" * 60)
    run_exp_b(**kwargs)

    print("\n" + "=" * 60)
    print("Running Exp-C: FBFM Ablation (k_p=0)")
    print("=" * 60)
    run_exp_c(**kwargs)

    print("\n" + "=" * 60)
    print("Running Exp-D: Feedback Frequency Sensitivity")
    print("=" * 60)
    run_exp_d(**kwargs)

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("Results saved under results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
