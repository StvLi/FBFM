#!/usr/bin/env python3
"""Validate the method invariants in one solver JSONL record."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--mode", choices=("NONE", "RTC", "FBFM"), required=True)
    parser.add_argument("--minimum-async-chunks", type=int, default=1)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.audit.read_text(encoding="utf-8").splitlines()]
    evaluations_per_chunk = 8
    chunks: list[dict] = []
    current_chunk: dict | None = None
    for record in records:
        if record.get("event") == "chunk_begin":
            current_chunk = {"begin": record, "steps": []}
            chunks.append(current_chunk)
        elif record.get("event") == "solver_step" and current_chunk is not None:
            current_chunk["steps"].append(record)

    steps = [step for chunk in chunks for step in chunk["steps"]]
    async_chunks = [chunk for chunk in chunks if chunk["begin"].get("pseudo_async")]
    complete_async_chunks = [
        chunk for chunk in async_chunks if len(chunk["steps"]) == evaluations_per_chunk
    ]
    if len(complete_async_chunks) < args.minimum_async_chunks:
        raise AssertionError(
            f"too few complete async chunks: {len(complete_async_chunks)}"
        )
    if any(len(chunk["steps"]) > evaluations_per_chunk for chunk in chunks):
        raise AssertionError("a chunk contains more than eight native DiT evaluations")
    for record in steps:
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise AssertionError(f"non-finite {key}: {record}")

    async_steps = [step for chunk in async_chunks for step in chunk["steps"]]
    if args.mode == "NONE":
        if any(
            record["guided"]
            or record["state_mask_nonzero"]
            or record["action_mask_nonzero"]
            for record in async_steps
        ):
            raise AssertionError("NONE must have zero constraints and guidance")
    else:
        if any(record["action_mask_nonzero"] != 56 for record in async_steps):
            raise AssertionError("RTC/FBFM action mask must be exactly 8x7")
        if not all(record["guided"] for record in async_steps):
            raise AssertionError("constrained solver evaluations must be guided")
    if args.mode == "RTC" and any(record["state_mask_nonzero"] for record in async_steps):
        raise AssertionError("RTC must not expose state constraints")
    if args.mode == "FBFM":
        state_steps = [record for record in async_steps if record["state_mask_nonzero"]]
        if not state_steps:
            raise AssertionError("FBFM did not expose any state feedback")
        for chunk_record in async_chunks:
            chunk = chunk_record["steps"]
            if not chunk:
                continue
            expected = list(range(1, len(chunk) + 1))
            if [record.get("context_version") for record in chunk] != expected:
                raise AssertionError("FBFM must refresh its state target once per evaluation")
            if [record.get("feedback_action_offsets") for record in chunk] != [
                [offset] for offset in expected
            ]:
                raise AssertionError("FBFM feedback offsets must progress from 1 in each chunk")
            if any(record.get("feedback_state_slots") != [0] for record in chunk):
                raise AssertionError("the active overlap wave must revise latent slot 0")
        if not any(record["action_correction_norm"] > 0 for record in state_steps):
            raise AssertionError("state feedback produced no action-coordinate correction")

    summary = {
        "status": "ok",
        "mode": args.mode,
        "chunks": len(chunks),
        "async_chunks": len(async_chunks),
        "complete_async_chunks": len(complete_async_chunks),
        "solver_steps": len(steps),
        "guided_steps": sum(int(record["guided"]) for record in steps),
        "state_guided_steps": sum(int(record["state_mask_nonzero"] > 0) for record in steps),
        "max_gpu_allocated_bytes": max(record["gpu_allocated_bytes"] for record in steps),
        "max_gpu_peak_allocated_bytes": max(
            record.get("gpu_peak_allocated_bytes", record["gpu_allocated_bytes"])
            for record in steps
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
