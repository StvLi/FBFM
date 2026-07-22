"""Resumable launcher that binds every mode to the same canonical episodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .experiment import CONFIGS, MODES, TASKS, _read_jsonl, model_noise_seed, validate_manifest


def select_shard(episodes: list[dict], *, shard_index: int, num_shards: int) -> list[dict]:
    """Deterministically partition an already validated canonical manifest."""

    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return episodes[shard_index::num_shards]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--manifest-tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--manifest-configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide the single-episode evaluator command after --")

    episodes = _read_jsonl(args.manifest)
    validate_manifest(episodes, tasks=args.manifest_tasks, configs=args.manifest_configs)
    checkpoint = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
    checkpoint_hash = checkpoint["checkpoint_sha256"]
    selected = [
        episode
        for episode in episodes
        if (args.task is None or episode["task"] == args.task)
        and (args.config is None or episode["config"] == args.config)
    ]
    selected = select_shard(
        selected,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(
        f"SHARD {args.shard_index}/{args.num_shards}: "
        f"{len(selected)} canonical episodes"
    )
    for episode in selected:
        result_path = args.output_dir / args.mode / f"{episode['episode_id'].replace(':', '__')}.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError(f"resume result uses another checkpoint: {result_path}")
            print(f"SKIP {episode['episode_id']}")
            continue
        result_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "FBFM_CONSTRAINT_MODE": args.mode,
                "ROBOTWIN_EPISODE_JSON": json.dumps(episode, ensure_ascii=False, sort_keys=True),
                "ROBOTWIN_RESULT_PATH": os.fspath(result_path),
                "DREAMZERO_MODEL_NOISE_SEED_BASE": str(episode["model_noise_seed_base"]),
                "DREAMZERO_FIRST_CHUNK_NOISE_SEED": str(model_noise_seed(episode, 0)),
                "DREAMZERO_CHECKPOINT_SHA256": checkpoint_hash,
            }
        )
        print(f"RUN {args.mode} {episode['episode_id']}")
        if args.dry_run:
            continue
        subprocess.run(args.command, check=True, env=env)
        if not result_path.is_file():
            raise RuntimeError(f"single-episode evaluator did not write {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result.get("success"), bool):
            raise ValueError(f"result must contain boolean success: {result_path}")
        result.update(
            {
                "mode": args.mode,
                "episode_id": episode["episode_id"],
                "checkpoint_sha256": checkpoint_hash,
            }
        )
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
