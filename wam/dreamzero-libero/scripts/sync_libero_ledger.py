#!/usr/bin/env python3
"""Periodically mirror the remote LIBERO ledger into the local paper repository."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path


REMOTE_FILES = {
    "live_status.md": "dreamzero_fbfm_live_status.md",
    "manifest.json": "dreamzero_fbfm_manifest.json",
    "task_summary.csv": "dreamzero_fbfm_task_summary.csv",
    "trials.csv": "dreamzero_fbfm_trials.csv",
}


def synchronize(host: str, remote_output: Path, local_experiments: Path) -> tuple[int, int]:
    local_experiments.mkdir(parents=True, exist_ok=True)
    for remote_name, local_name in REMOTE_FILES.items():
        subprocess.run(
            [
                "rsync",
                "-a",
                f"{host}:{remote_output}/{remote_name}",
                str(local_experiments / local_name),
            ],
            check=True,
        )
    with (local_experiments / REMOTE_FILES["task_summary.csv"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    return sum(row["status"] == "complete" for row in rows), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-output", type=Path, required=True)
    parser.add_argument("--local-experiments", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval-seconds must be positive")

    while True:
        try:
            complete, total = synchronize(args.host, args.remote_output, args.local_experiments)
            print(
                datetime.now().astimezone().isoformat(timespec="seconds"),
                f"synchronized {complete}/{total} complete tasks",
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
