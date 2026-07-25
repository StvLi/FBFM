#!/usr/bin/env python3
"""Run every LIBERO task sequentially and refresh resumable result tables."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from dreamzero_fbfm.experiment_ledger import (
    TaskSpec,
    load_jsonl,
    task_directory,
    write_tables,
)

DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
DEFAULT_MAX_STEPS = {
    "libero_spatial": 480,
    "libero_object": 480,
    "libero_goal": 480,
    "libero_10": 480,
    "libero_90": 480,
}


def discover_tasks(suite_names: list[str]) -> list[TaskSpec]:
    from libero.libero import benchmark

    registry = benchmark.get_benchmark_dict()
    specs: list[TaskSpec] = []
    for suite_name in suite_names:
        suite = registry[suite_name]()
        for task_id in range(int(suite.n_tasks)):
            task = suite.get_task(task_id)
            specs.append(TaskSpec(suite_name, task_id, task.language))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", action="append", choices=DEFAULT_SUITES)
    parser.add_argument("--mode", choices=("NONE", "RTC", "FBFM"), default="FBFM")
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override the suite-specific episode horizons for every selected suite",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver-release-policy",
        choices=("uniform", "after_feedback"),
        default="uniform",
    )
    parser.add_argument(
        "--model-seed-rule", choices=("fixed", "trial_offset"), default="fixed"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")

    suites = args.suite or list(DEFAULT_SUITES)
    max_steps_by_suite = {
        suite: args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS[suite]
        for suite in suites
    }
    specs = discover_tasks(suites)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = {
        "benchmark": "LIBERO",
        "mode": args.mode,
        "suites": suites,
        "tasks": len(specs),
        "trials_per_task": args.trials,
        "expected_episodes": len(specs) * args.trials,
        "max_steps_by_suite": max_steps_by_suite,
        "horizon_policy": "uniform_480_for_cross_suite_comparison",
        "environment_seed": args.seed,
        "model_seed_rule": args.model_seed_rule,
        "solver_release_policy": args.solver_release_policy,
        "feedback_observation_stride": 1,
        "feedback_encoding": "causal_rolling_hold",
        "state_weight": args.state_weight,
        "code_commit": args.code_commit,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        compared = (
            "mode", "suites", "trials_per_task", "max_steps_by_suite",
            "model_seed_rule", "solver_release_policy", "state_weight", "code_commit",
        )
        if any(existing.get(key) != manifest[key] for key in compared):
            raise ValueError("existing benchmark manifest does not match requested protocol")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_tables(args.output, specs, args.trials, mode=args.mode, code_commit=args.code_commit)
    if args.tables_only:
        return

    for spec in specs:
        directory = task_directory(args.output, spec)
        records = load_jsonl(directory / "episodes.jsonl")
        completed = sorted(int(record["trial_id"]) for record in records)
        if completed != list(range(len(completed))):
            raise ValueError(f"task output is not a resumable prefix: {directory}: {completed}")
        remaining = args.trials - len(completed)
        if remaining <= 0:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPOSITORY / "scripts" / "libero_experiment.py"),
            "--base-workspace", str(args.base_workspace),
            "--mode", args.mode,
            "--state-weight", str(args.state_weight),
            "--suite", spec.suite,
            "--task-id", str(spec.task_id),
            "--trial-start", str(len(completed)),
            "--trials", str(remaining),
            "--seed", str(args.seed),
            "--max-steps", str(max_steps_by_suite[spec.suite]),
            "--model-seed-rule", args.model_seed_rule,
            "--solver-release-policy", args.solver_release_policy,
            "--host", args.host,
            "--port", str(args.port),
            "--output", str(directory),
        ]
        try:
            subprocess.run(command, check=True)
        finally:
            write_tables(
                args.output,
                specs,
                args.trials,
                mode=args.mode,
                code_commit=args.code_commit,
            )


if __name__ == "__main__":
    main()
