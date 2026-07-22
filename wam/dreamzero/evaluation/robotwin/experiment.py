"""Canonical episode manifests and result aggregation for DreamZero RoboTwin.

The manifest is generated once from simulator-validated candidates and reused
unchanged by None, RTC, and Feedback.  This prevents each mode from silently
accepting a different sequence of initialization seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TASKS = (
    "adjust_bottle",
    "click_alarmclock",
    "click_bell",
    "grab_roller",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "pick_dual_bottles",
    "place_container_plate",
    "place_empty_cup",
    "place_object_stand",
    "press_stapler",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_bowls_two",
    "turn_switch",
)
CONFIGS = ("demo_clean", "demo_randomized")
MODES = ("None", "RTC", "Feedback")
MODE_LABELS = {"None": "Baseline", "RTC": "RTC", "Feedback": "Ours"}
EPISODES_PER_CELL = 20
INSTRUCTION_POOL_SIZE = 20


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def model_noise_seed(episode: dict[str, Any], chunk_index: int) -> int:
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return stable_seed(episode["model_noise_seed_base"], chunk_index)


def freeze_manifest(
    candidates: Iterable[dict[str, Any]],
    *,
    episodes_per_cell: int = EPISODES_PER_CELL,
    instruction_pool_size: int = INSTRUCTION_POOL_SIZE,
    tasks: Iterable[str] = TASKS,
    configs: Iterable[str] = CONFIGS,
) -> list[dict[str, Any]]:
    """Freeze the first validated candidates for every task/config cell.

    Candidate records are emitted by a simulator initialization pass.  They
    must already contain the exact chosen instruction and randomization data.
    Randomized records must also pin the background texture and its checksum.
    """

    tasks = tuple(tasks)
    configs = tuple(configs)
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must be a non-empty subset of {TASKS}")
    if not configs or any(config not in CONFIGS for config in configs):
        raise ValueError(f"configs must be a non-empty subset of {CONFIGS}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if not candidate.get("accepted", False):
            continue
        task = candidate.get("task")
        config = candidate.get("config")
        if task not in TASKS or config not in CONFIGS:
            continue
        grouped[(task, config)].append(dict(candidate))

    frozen = []
    for task in tasks:
        for config in configs:
            records = grouped[(task, config)]
            records.sort(key=lambda item: int(item["seed"]))
            if len(records) < episodes_per_cell:
                raise ValueError(
                    f"{task}/{config}: need {episodes_per_cell} accepted candidates, found {len(records)}"
                )
            seen_seeds: set[int] = set()
            for index, candidate in enumerate(records[:episodes_per_cell]):
                seed = int(candidate["seed"])
                if seed in seen_seeds:
                    raise ValueError(f"{task}/{config}: duplicate accepted seed {seed}")
                seen_seeds.add(seed)
                instruction = str(candidate.get("instruction", "")).strip()
                if not instruction:
                    raise ValueError(f"{task}/{config}/{seed}: missing frozen instruction")
                instruction_index = int(candidate.get("instruction_index", -1))
                if not 0 <= instruction_index < instruction_pool_size:
                    raise ValueError(
                        f"{task}/{config}/{seed}: instruction_index must be in [0, {instruction_pool_size})"
                    )

                texture_path = candidate.get("background_texture")
                texture_sha = candidate.get("background_texture_sha256")
                if config == "demo_randomized" and (not texture_path or not texture_sha):
                    raise ValueError(f"{task}/{config}/{seed}: randomized episode must freeze texture path/checksum")
                episode_id = f"{task}:{config}:{index:02d}:{seed}"
                frozen.append(
                    {
                        "episode_id": episode_id,
                        "task": task,
                        "config": config,
                        "seed": seed,
                        "instruction": instruction,
                        "instruction_index": instruction_index,
                        "randomization": candidate.get("randomization", {}),
                        "background_texture": texture_path,
                        "background_texture_sha256": texture_sha,
                        "model_noise_seed_base": stable_seed(episode_id, "dreamzero"),
                    }
                )
    validate_manifest(frozen, episodes_per_cell=episodes_per_cell, tasks=tasks, configs=configs)
    return frozen


def validate_manifest(
    records: Iterable[dict[str, Any]],
    *,
    episodes_per_cell: int = EPISODES_PER_CELL,
    tasks: Iterable[str] = TASKS,
    configs: Iterable[str] = CONFIGS,
) -> None:
    records = list(records)
    tasks = tuple(tasks)
    configs = tuple(configs)
    expected = len(tasks) * len(configs) * episodes_per_cell
    if len(records) != expected:
        raise ValueError(f"manifest must contain {expected} episodes, found {len(records)}")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    episode_ids: set[str] = set()
    for record in records:
        episode_id = str(record["episode_id"])
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id {episode_id}")
        episode_ids.add(episode_id)
        cell = (record["task"], record["config"])
        if cell[0] not in tasks or cell[1] not in configs:
            raise ValueError(f"unknown manifest cell {cell}")
        counts[cell] += 1
        if cell[1] == "demo_randomized" and not record.get("background_texture_sha256"):
            raise ValueError(f"{episode_id}: randomized texture checksum missing")
    for task in tasks:
        for config in configs:
            if counts[(task, config)] != episodes_per_cell:
                raise ValueError(f"{task}/{config}: expected {episodes_per_cell}, found {counts[(task, config)]}")


def validate_results(
    manifest: Iterable[dict[str, Any]],
    results: Iterable[dict[str, Any]],
    *,
    allow_partial: bool = False,
) -> dict[tuple[str, str], dict[str, Any]]:
    manifest = list(manifest)
    valid_ids = {record["episode_id"] for record in manifest}
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoint_hashes: set[str] = set()
    for result in results:
        mode = result.get("mode")
        episode_id = result.get("episode_id")
        if mode not in MODES:
            raise ValueError(f"unknown result mode {mode!r}")
        if episode_id not in valid_ids:
            raise ValueError(f"result references unknown episode {episode_id!r}")
        key = (mode, episode_id)
        if key in indexed:
            raise ValueError(f"duplicate result {mode}/{episode_id}")
        if not isinstance(result.get("success"), bool):
            raise ValueError(f"{mode}/{episode_id}: success must be JSON boolean")
        checkpoint_hash = str(result.get("checkpoint_sha256", "")).strip()
        if not checkpoint_hash:
            raise ValueError(f"{mode}/{episode_id}: checkpoint_sha256 missing")
        checkpoint_hashes.add(checkpoint_hash)
        indexed[key] = result
    if len(checkpoint_hashes) > 1:
        raise ValueError("modes did not use the same checkpoint hash")
    expected = len(valid_ids) * len(MODES)
    if not allow_partial and len(indexed) != expected:
        raise ValueError(f"complete evaluation needs {expected} results, found {len(indexed)}")
    return indexed


def _cell(records: list[dict[str, Any]], indexed: dict[tuple[str, str], dict[str, Any]], mode: str, task: str, config: str) -> tuple[int, int]:
    ids = [record["episode_id"] for record in records if record["task"] == task and record["config"] == config]
    values = [indexed[(mode, episode_id)]["success"] for episode_id in ids if (mode, episode_id) in indexed]
    return sum(values), len(values)


def _format_cell(successes: int, total: int) -> str:
    return "N/A" if total == 0 else f"{successes}/{total} ({100.0 * successes / total:.1f}%)"


def bootstrap_ci(values: list[bool], *, samples: int = 10_000, seed: int = 20_260_221) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return estimates[int(samples * 0.025)], estimates[min(samples - 1, int(samples * 0.975))]


def aggregate(manifest: list[dict[str, Any]], results: list[dict[str, Any]], *, allow_partial: bool = False) -> tuple[str, dict[str, Any]]:
    validate_manifest(manifest)
    indexed = validate_results(manifest, results, allow_partial=allow_partial)
    headers = ["Task"] + [f"{MODE_LABELS[mode]} {label}" for label in ("Clean", "Random") for mode in MODES]
    rows = []
    for task in TASKS:
        row = [task]
        for config in CONFIGS:
            for mode in MODES:
                row.append(_format_cell(*_cell(manifest, indexed, mode, task, config)))
        rows.append(row)

    summary: dict[str, Any] = {}
    for mode in MODES:
        summary[mode] = {}
        overall_values = []
        for config, label in zip(CONFIGS, ("Clean", "Random")):
            values = [
                indexed[(mode, record["episode_id"])]["success"]
                for record in manifest
                if record["config"] == config and (mode, record["episode_id"]) in indexed
            ]
            overall_values.extend(values)
            low, high = bootstrap_ci(values, seed=stable_seed(mode, config))
            summary[mode][label] = {
                "success": sum(values),
                "total": len(values),
                "rate": sum(values) / len(values) if values else None,
                "bootstrap_95_ci": [low, high],
            }
        low, high = bootstrap_ci(overall_values, seed=stable_seed(mode, "overall"))
        summary[mode]["Overall"] = {
            "success": sum(overall_values),
            "total": len(overall_values),
            "rate": sum(overall_values) / len(overall_values) if overall_values else None,
            "bootstrap_95_ci": [low, high],
        }

    separator = ["---"] * len(headers)
    markdown = "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )
    return markdown, summary


def _freeze_cli(args: argparse.Namespace) -> None:
    records = freeze_manifest(
        _read_jsonl(args.candidates),
        episodes_per_cell=args.episodes_per_cell,
        instruction_pool_size=args.instruction_pool_size,
        tasks=args.tasks,
        configs=args.configs,
    )
    _write_jsonl(args.output, records)
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")


def _aggregate_cli(args: argparse.Namespace) -> None:
    manifest = _read_jsonl(args.manifest)
    results = _read_jsonl(args.results)
    markdown, summary = aggregate(manifest, results, allow_partial=args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n", encoding="utf-8", newline="\n")
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--candidates", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--episodes-per-cell", type=int, default=EPISODES_PER_CELL)
    freeze.add_argument("--instruction-pool-size", type=int, default=INSTRUCTION_POOL_SIZE)
    freeze.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    freeze.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    freeze.set_defaults(handler=_freeze_cli)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--manifest", type=Path, required=True)
    aggregate_parser.add_argument("--results", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--allow-partial", action="store_true")
    aggregate_parser.set_defaults(handler=_aggregate_cli)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
