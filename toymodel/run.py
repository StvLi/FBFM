"""
CLI entry point for the FBFM pre-experiment suite.

Usage:
    python -m pre_test_1.run [--exp a|b|c|d|all] [--seeds 5]
"""

import argparse
import time as time_mod

import torch

from pre_test_1.config import CHECKPOINT, RESULTS_DIR
from pre_test_1.train import load_trained_model
from pre_test_1.experiments import experiment_a, experiment_b, experiment_c, experiment_d


def main():
    parser = argparse.ArgumentParser(description="FBFM Final Experiments (Publication Quality)")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["a", "b", "c", "d", "all"])
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Device: {device}")
    print(f"[Main] Seeds: {seeds}")

    model, norm_stats = load_trained_model(CHECKPOINT, device)

    t0 = time_mod.time()

    if args.exp in ("a", "all"):
        experiment_a(model, norm_stats, device, seeds)

    if args.exp in ("b", "all"):
        experiment_b(model, norm_stats, device, seeds)

    if args.exp in ("c", "all"):
        experiment_c(model, norm_stats, device, seeds)

    if args.exp in ("d", "all"):
        experiment_d(model, norm_stats, device, seeds)

    elapsed = time_mod.time() - t0
    print(f"\n[Main] Completed in {elapsed:.1f}s")
    print(f"[Main] Results → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
