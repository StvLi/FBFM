#!/usr/bin/env python3
"""Periodically mirror the remote LIBERO ledger into the local paper repository."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path


REMOTE_FILES = ("live_status.md", "manifest.json", "task_summary.csv", "trials.csv")
SNAPSHOT_ATTEMPTS = 5
LIVE_TOTALS_PATTERN = re.compile(
    r"\|\s*(\d+)/(\d+)\s*\|\s*(\d+)/(\d+)\s*\|\s*(\d+)\s*\|"
)
INDEX_START = "<!-- dreamzero-fbfm-causal-full-live:start -->"
INDEX_END = "<!-- dreamzero-fbfm-causal-full-live:end -->"


def _parse_success(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _validate_snapshot(staging: Path) -> tuple[int, int, int, int]:
    with (staging / "task_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (staging / "trials.csv").open(encoding="utf-8", newline="") as handle:
        trials = list(csv.DictReader(handle))

    complete = sum(row["status"] == "complete" for row in rows)
    episodes = sum(int(row["trials"]) for row in rows)
    successes = sum(int(row["successes"]) for row in rows)
    trial_successes = sum(_parse_success(row["success"]) for row in trials)
    live_status = (staging / "live_status.md").read_text(encoding="utf-8")
    match = LIVE_TOTALS_PATTERN.search(live_status)
    if match is None:
        raise ValueError("live status summary row is missing")
    live_complete, live_total, live_episodes, _, live_successes = map(int, match.groups())

    expected = (complete, len(rows), episodes, successes)
    observed = (live_complete, live_total, live_episodes, live_successes)
    if observed != expected:
        raise ValueError(
            f"mixed ledger snapshot: live={observed}, task_summary={expected}"
        )
    if len(trials) != episodes or trial_successes != successes:
        raise ValueError(
            "mixed ledger snapshot: "
            f"trials=({len(trials)}, {trial_successes}), "
            f"task_summary=({episodes}, {successes})"
        )
    return expected


def _install_snapshot(staging: Path, local_experiments: Path, local_prefix: str) -> None:
    for remote_name in REMOTE_FILES:
        destination = local_experiments / f"{local_prefix}_{remote_name}"
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copy2(staging / remote_name, temporary)
        temporary.replace(destination)


def _update_experiment_index(
    staging: Path,
    local_experiments: Path,
    local_prefix: str,
    complete: int,
    total: int,
    episodes: int,
    successes: int,
) -> None:
    index = local_experiments / "README.md"
    if not index.is_file():
        return
    with (staging / "task_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target_episodes = sum(int(row["target_trials"]) for row in rows)
    active = next((row for row in rows if row["status"] == "running"), None)
    if active is None:
        active = next((row for row in rows if row["status"] == "pending"), None)
    if active is None:
        current = "complete"
    else:
        suite = active["suite"]
        task_id = int(active["task_id"])
        current = (
            f"`{suite}/task_{task_id:03d}`: "
            f"{active['successes']}/{active['trials']} successful episodes"
        )
    rate = successes / episodes if episodes else 0.0
    updated_match = re.search(
        r"Updated:\s*`([^`]+)`",
        (staging / "live_status.md").read_text(encoding="utf-8"),
    )
    updated = updated_match.group(1) if updated_match else "unknown"
    live_name = f"{local_prefix}_live_status.md"
    task_name = f"{local_prefix}_task_summary.csv"
    trials_name = f"{local_prefix}_trials.csv"
    block = "\n".join(
        (
            INDEX_START,
            "## Active DreamZero causal-FBFM run",
            "",
            "| Item | Live value |",
            "| --- | --- |",
            "| Scope | `libero_spatial` + `libero_object`; "
            f"{total} tasks x 20 trials = {target_episodes} episodes |",
            f"| Snapshot | `{updated}` |",
            f"| Progress | {complete}/{total} complete tasks; "
            f"{episodes}/{target_episodes} episodes |",
            f"| Success | {successes}/{episodes}; **{rate:.1%} Micro (partial)** |",
            f"| Current task | {current} |",
            "| Protocol | causal rolling `FBFM`; pseudo-async overlap; "
            "uniform 480-step horizon |",
            f"| Records | [`live`]({live_name}), [`tasks`]({task_name}), "
            f"[`trials`]({trials_name}) |",
            "| Interpretation | Running rows are progress indicators; only "
            "20-trial task rows are final |",
            INDEX_END,
        )
    )
    content = index.read_text(encoding="utf-8")
    if INDEX_START in content and INDEX_END in content:
        before, remainder = content.split(INDEX_START, 1)
        _, after = remainder.split(INDEX_END, 1)
        updated_content = before + block + after
    else:
        anchor = "## Main experiment matrix"
        if anchor not in content:
            raise ValueError("experiment README is missing the main matrix anchor")
        updated_content = content.replace(anchor, f"{block}\n\n{anchor}", 1)
    temporary = index.with_name(f".{index.name}.dreamzero.tmp")
    temporary.write_text(updated_content, encoding="utf-8")
    temporary.replace(index)


def synchronize(
    host: str,
    remote_output: Path,
    local_experiments: Path,
    local_prefix: str,
) -> tuple[int, int, int, int]:
    local_experiments.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".libero-ledger-", dir=local_experiments
    ) as temporary_directory:
        staging = Path(temporary_directory)
        for attempt in range(1, SNAPSHOT_ATTEMPTS + 1):
            for remote_name in REMOTE_FILES:
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        f"{host}:{remote_output}/{remote_name}",
                        str(staging / remote_name),
                    ],
                    check=True,
                )
            try:
                complete, total, episodes, successes = _validate_snapshot(staging)
                break
            except ValueError:
                if attempt == SNAPSHOT_ATTEMPTS:
                    raise
                time.sleep(1)
        _install_snapshot(staging, local_experiments, local_prefix)
        _update_experiment_index(
            staging,
            local_experiments,
            local_prefix,
            complete,
            total,
            episodes,
            successes,
        )

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
                "total_tasks": total,
                "episodes": episodes,
                "successes": successes,
                "micro_success_rate": successes / episodes if episodes else "",
            }
        )
    return complete, total, episodes, successes


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
