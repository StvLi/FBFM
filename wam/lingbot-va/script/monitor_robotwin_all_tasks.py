#!/usr/bin/env python3
"""Persist 30-minute progress/GPU snapshots for a detached RoboTwin run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    parser.add_argument("--aggregate-script", type=Path, required=True)
    parser.add_argument("--import-adjust", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    args = parser.parse_args()

    while True:
        subprocess.run(
            [
                "flock",
                str(args.root / ".aggregate.lock"),
                sys.executable,
                str(args.aggregate_script),
                str(args.root),
                "--episodes-per-task",
                str(args.episodes_per_task),
                "--import-adjust",
                str(args.import_adjust),
                "--paper-dir",
                str(args.paper_dir),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        aggregate = json.loads((args.root / "aggregate.json").read_text(encoding="utf-8"))
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
        append_jsonl(
            args.root / "status_reports.jsonl",
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "launcher_alive": alive(args.launcher_pid),
                "status": aggregate["status"],
                "tasks_complete": aggregate["tasks_complete"],
                "tasks_requested": aggregate["tasks_requested"],
                "episodes_completed": aggregate["episodes_completed"],
                "episodes_requested": aggregate["episodes_requested"],
                "successes": aggregate["successes"],
                "micro_success_rate": aggregate["micro_success_rate"],
                "gpu_csv": gpu.stdout.strip() if gpu.returncode == 0 else None,
            },
        )
        if aggregate["status"] == "complete" or not alive(args.launcher_pid):
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    import sys

    main()
