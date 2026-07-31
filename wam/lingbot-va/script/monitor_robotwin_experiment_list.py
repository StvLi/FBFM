#!/usr/bin/env python3
"""Write 20-minute progress snapshots for a detached experiment-list run."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


FATAL_RE = re.compile(
    r"CUDA out of memory|non-finite|Fatal Python error", re.IGNORECASE
)
TRACEBACK_RE = re.compile(r"Traceback", re.IGNORECASE)
SETUP_REJECTION_RE = re.compile(
    r"error occurs ! target_pose cannot be None for move action"
)


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


def anomaly_counts(root: Path) -> dict[str, int]:
    counts = {
        "fatal_records": 0,
        "traceback_records": 0,
        "recoverable_setup_rejections": 0,
    }
    for path in (root / "jobs").glob("*/logs/*.log"):
        try:
            value = path.read_text(encoding="utf-8", errors="replace")
            counts["fatal_records"] += len(FATAL_RE.findall(value))
            counts["traceback_records"] += len(TRACEBACK_RE.findall(value))
            counts["recoverable_setup_rejections"] += len(
                SETUP_REJECTION_RE.findall(value)
            )
        except OSError:
            pass
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--aggregate-script", type=Path, required=True)
    parser.add_argument("--experiment-list", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--paper-prefix", required=True)
    parser.add_argument("--interval-seconds", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    while True:
        subprocess.run(
            [
                "flock",
                str(args.root / ".aggregate.lock"),
                sys.executable,
                str(args.aggregate_script),
                str(args.root),
                "--experiment-list",
                str(args.experiment_list),
                "--paper-dir",
                str(args.paper_dir),
                "--paper-prefix",
                args.paper_prefix,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        aggregate = json.loads((args.root / "aggregate.json").read_text(encoding="utf-8"))
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
        snapshot = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "launcher_alive": alive(args.launcher_pid),
            "launcher_pid": args.launcher_pid,
            "status": aggregate["status"],
            "jobs_complete": aggregate["jobs_complete"],
            "jobs_requested": aggregate["jobs_requested"],
            "jobs_running": aggregate["jobs_running"],
            "jobs_failed": aggregate["jobs_failed"],
            "episodes_completed": aggregate["episodes_completed"],
            "episodes_requested": aggregate["episodes_requested"],
            "successes": aggregate["successes"],
            "gpu_csv": gpu_csv,
        }
        snapshot.update(anomaly_counts(args.root))
        append_jsonl(args.root / "status_reports.jsonl", snapshot)
        if args.once or aggregate["status"] == "complete":
            break
        if not snapshot["launcher_alive"]:
            append_jsonl(
                args.root / "status_reports.jsonl",
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "event": "launcher_exited_before_completion",
                },
            )
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
