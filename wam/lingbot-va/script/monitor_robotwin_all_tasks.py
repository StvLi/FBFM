#!/usr/bin/env python3
"""Persist periodic progress/GPU snapshots for a detached RoboTwin run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--launcher-script", type=Path)
    parser.add_argument("--max-relaunches", type=int, default=3)
    parser.add_argument("--aggregate-script", type=Path, required=True)
    parser.add_argument("--import-adjust", type=Path)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--mode", default="FBFM")
    parser.add_argument("--paper-prefix")
    parser.add_argument("--experiment-date")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(exist_ok=True)
    launcher_pid = args.launcher_pid
    launcher_process: subprocess.Popen | None = None
    relaunches = 0
    launcher_env = os.environ.copy()
    launcher_env.update(
        {
            "LINGBOT_VA_ALL_TASKS_ROOT": str(args.root.resolve()),
            "LINGBOT_VA_TASK_SET_ROOT": str(args.root.resolve()),
            "ROBOTWIN_EPISODES_PER_TASK": str(args.episodes_per_task),
            "FBFM_PAPER_EXPERIMENT_DIR": str(args.paper_dir.resolve()),
            "LINGBOT_VA_CONSTRAINT_VARIANT": (
                "FBFM-static"
                if args.mode.strip().upper() == "FBFM-STATIC"
                else args.mode.strip().upper()
            ),
        }
    )
    if args.import_adjust:
        launcher_env["LINGBOT_VA_ADJUST_BOTTLE_AGGREGATE"] = str(args.import_adjust.resolve())
    if args.paper_prefix:
        launcher_env["LINGBOT_VA_PAPER_PREFIX"] = args.paper_prefix
    if args.experiment_date:
        launcher_env["LINGBOT_VA_EXPERIMENT_DATE"] = args.experiment_date
    if args.tasks_file:
        launcher_env["LINGBOT_VA_TASKS_FILE"] = str(args.tasks_file.resolve())

    while True:
        aggregate_command = [
            "flock",
            str(args.root / ".aggregate.lock"),
            sys.executable,
            str(args.aggregate_script),
            str(args.root),
            "--episodes-per-task",
            str(args.episodes_per_task),
            "--paper-dir",
            str(args.paper_dir),
            "--mode",
            args.mode,
        ]
        if args.import_adjust:
            aggregate_command.extend(["--import-adjust", str(args.import_adjust)])
        if args.paper_prefix:
            aggregate_command.extend(["--paper-prefix", args.paper_prefix])
        if args.experiment_date:
            aggregate_command.extend(["--experiment-date", args.experiment_date])
        if args.tasks_file:
            aggregate_command.extend(["--tasks-file", str(args.tasks_file)])
        subprocess.run(
            aggregate_command,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        aggregate = json.loads((args.root / "aggregate.json").read_text(encoding="utf-8"))
        launcher_is_alive = (
            launcher_process.poll() is None
            if launcher_process is not None and launcher_process.pid == launcher_pid
            else alive(launcher_pid)
        )
        try:
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            gpu_csv = gpu.stdout.strip() if gpu.returncode == 0 else None
        except FileNotFoundError:
            gpu_csv = None
        append_jsonl(
            args.root / "status_reports.jsonl",
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "launcher_alive": launcher_is_alive,
                "launcher_pid": launcher_pid,
                "relaunches": relaunches,
                "status": aggregate["status"],
                "tasks_complete": aggregate["tasks_complete"],
                "tasks_requested": aggregate["tasks_requested"],
                "episodes_completed": aggregate["episodes_completed"],
                "episodes_requested": aggregate["episodes_requested"],
                "successes": aggregate["successes"],
                "micro_success_rate": aggregate["micro_success_rate"],
                "gpu_csv": gpu_csv,
            },
        )
        if aggregate["status"] == "complete":
            break
        if not launcher_is_alive:
            if args.launcher_script and relaunches < args.max_relaunches:
                relaunches += 1
                launcher_log = (args.root / "logs" / "launcher.log").open(
                    "a", encoding="utf-8"
                )
                launcher_process = subprocess.Popen(
                    [str(args.launcher_script)],
                    stdout=launcher_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=launcher_env,
                )
                launcher_log.close()
                launcher_pid = launcher_process.pid
                append_jsonl(
                    args.root / "status_reports.jsonl",
                    {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "event": "launcher_restarted",
                        "launcher_pid": launcher_pid,
                        "relaunches": relaunches,
                    },
                )
            else:
                break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
