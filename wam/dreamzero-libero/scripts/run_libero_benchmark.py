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
from dreamzero_fbfm.settings import DEFAULT_STATE_FEEDBACK_KP, DEFAULT_STATE_WEIGHT

AVAILABLE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
DEFAULT_SUITES = AVAILABLE_SUITES[:4]
DEFAULT_MAX_STEPS = {
    "libero_spatial": 480,
    "libero_object": 480,
    "libero_goal": 480,
    "libero_10": 480,
    "libero_90": 480,
}


def parse_task_selectors(values: list[str]) -> list[tuple[str, int]]:
    selectors: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for value in values:
        try:
            suite, task_id_text = value.rsplit(":", 1)
            task_id = int(task_id_text)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"invalid task selector {value!r}; expected SUITE:TASK_ID"
            ) from error
        selector = (suite, task_id)
        if suite not in AVAILABLE_SUITES or task_id < 0:
            raise ValueError(f"invalid task selector: {value!r}")
        if selector in seen:
            raise ValueError(f"duplicate task selector: {value!r}")
        selectors.append(selector)
        seen.add(selector)
    return selectors


def discover_tasks(
    suite_names: list[str], selectors: list[tuple[str, int]] | None = None
) -> list[TaskSpec]:
    from libero.libero import benchmark

    registry = benchmark.get_benchmark_dict()
    if selectors is not None:
        suites = {suite_name: registry[suite_name]() for suite_name in suite_names}
        specs: list[TaskSpec] = []
        for suite_name, task_id in selectors:
            suite = suites[suite_name]
            if task_id >= int(suite.n_tasks):
                raise ValueError(
                    f"task selector out of range: {suite_name}:{task_id}"
                )
            task = suite.get_task(task_id)
            specs.append(TaskSpec(suite_name, task_id, task.language))
        return specs

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
    parser.add_argument("--suite", action="append", choices=AVAILABLE_SUITES)
    parser.add_argument(
        "--task",
        action="append",
        metavar="SUITE:TASK_ID",
        help="Run only this task; repeat for a fixed screening subset",
    )
    parser.add_argument("--mode", choices=("NONE", "RTC", "FBFM"), default="FBFM")
    parser.add_argument("--state-weight", type=float, default=DEFAULT_STATE_WEIGHT)
    parser.add_argument(
        "--state-feedback-kp", type=float, default=DEFAULT_STATE_FEEDBACK_KP
    )
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

    selectors = parse_task_selectors(args.task) if args.task else None
    if selectors is not None and args.suite:
        parser.error("--task cannot be combined with --suite")
    suites = (
        list(dict.fromkeys(suite for suite, _ in selectors))
        if selectors is not None
        else (args.suite or list(DEFAULT_SUITES))
    )
    max_steps_by_suite = {
        suite: args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS[suite]
        for suite in suites
    }
    specs = discover_tasks(suites, selectors)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = {
        "benchmark": "LIBERO",
        "mode": args.mode,
        "suites": suites,
        "task_ids_by_suite": {
            suite: [spec.task_id for spec in specs if spec.suite == suite]
            for suite in suites
        },
        "tasks": len(specs),
        "trials_per_task": args.trials,
        "expected_episodes": len(specs) * args.trials,
        "max_steps_by_suite": max_steps_by_suite,
        "horizon_policy": "uniform_480_for_cross_suite_comparison",
        "environment_seed": args.seed,
        "model_seed_rule": args.model_seed_rule,
        "solver_release_policy": args.solver_release_policy,
        "feedback_observation_stride": 1,
        "feedback_encoding": "causal_rolling_past",
        "state_weight": args.state_weight,
        "state_feedback_kp": args.state_feedback_kp,
        "effective_state_weight": args.state_weight * args.state_feedback_kp,
        "code_commit": args.code_commit,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        compared = (
            "mode", "suites", "task_ids_by_suite", "trials_per_task", "max_steps_by_suite",
            "model_seed_rule", "solver_release_policy", "state_weight",
            "state_feedback_kp", "code_commit",
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
            "--state-feedback-kp", str(args.state_feedback_kp),
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
