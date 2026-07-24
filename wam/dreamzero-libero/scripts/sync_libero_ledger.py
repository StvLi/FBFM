#!/usr/bin/env python3
"""Periodically mirror the remote LIBERO ledger into the local paper repository."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path


REMOTE_FILES = ("live_status.md", "manifest.json", "task_summary.csv", "trials.csv")


def synchronize(
    host: str,
    remote_output: Path,
    local_experiments: Path,
    local_prefix: str,
) -> tuple[int, int, int, int]:
    local_experiments.mkdir(parents=True, exist_ok=True)
    for remote_name in REMOTE_FILES:
        local_name = f"{local_prefix}_{remote_name}"
        subprocess.run(
            [
                "rsync",
                "-a",
                f"{host}:{remote_output}/{remote_name}",
                str(local_experiments / local_name),
            ],
            check=True,
        )
    with (local_experiments / f"{local_prefix}_task_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    complete = sum(row["status"] == "complete" for row in rows)
    episodes = sum(int(row["trials"]) for row in rows)
    successes = sum(int(row["successes"]) for row in rows)
    history = local_experiments / f"{local_prefix}_poll_history.csv"
    write_header = not history.exists()
    with history.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "checked_at", "complete_tasks", "total_tasks", "episodes",
                "successes", "micro_success_rate",
            ),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "complete_tasks": complete,
                "total_tasks": len(rows),
                "episodes": episodes,
                "successes": successes,
                "micro_success_rate": successes / episodes if episodes else "",
            }
        )
    return complete, len(rows), episodes, successes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-output", type=Path, required=True)
    parser.add_argument("--local-experiments", type=Path, required=True)
    parser.add_argument("--local-prefix", default="dreamzero_fbfm")
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval-seconds must be positive")

    while True:
        try:
            complete, total, episodes, successes = synchronize(
                args.host, args.remote_output, args.local_experiments, args.local_prefix
            )
            print(
                datetime.now().astimezone().isoformat(timespec="seconds"),
                f"synchronized {complete}/{total} complete tasks, "
                f"{successes}/{episodes} successful episodes",
                flush=True,
            )
            if args.once or complete == total:
                return
        except (OSError, subprocess.CalledProcessError, KeyError, ValueError) as error:
            print(
                datetime.now().astimezone().isoformat(timespec="seconds"),
                f"synchronization failed: {type(error).__name__}: {error}",
                flush=True,
            )
            if args.once:
                raise
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
