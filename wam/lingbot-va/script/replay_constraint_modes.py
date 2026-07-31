#!/usr/bin/env python3
"""Deterministic, model-free replay of LingBot-VA constraint semantics."""

import argparse
import json
from pathlib import Path
import sys

import torch

LINGBOT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LINGBOT_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(LINGBOT_ROOT), str(LINGBOT_ROOT / "wan_va")]

from lingbot_va_bridge import (  # noqa: E402
    ChunkConstraintContext,
    ConstraintMode,
    build_rtc_action_mask,
)


def _solver_step_replay(mode: ConstraintMode) -> list[dict]:
    """Inject a state between solver steps and record what later steps see."""
    context = ChunkConstraintContext(
        mode=mode,
        chunk_id=9,
        target_frame_st_id=20,
        action_targets=torch.zeros(1, 16, 4, 1, 1),
        action_mask=torch.ones(1, 1, 4, 1, 1),
        state_targets=torch.zeros(1, 3, 2, 1, 1),
        state_mask=torch.zeros(1, 1, 2, 1, 1),
    )
    trace = []
    for solver_step in range(4):
        if solver_step == 2:
            # This is the same boundary ordering used by the video FM loop:
            # drain feedback, update the versioned context, then snapshot.
            context.update_state_slot(
                global_slot_id=20, state=torch.full((1, 3, 1, 1), 7.0)
            )
        target, mask, version = context.snapshot_state_constraints()
        trace.append(
            {
                "solver_step": solver_step,
                "context_version": version,
                "visible_mask_sum": float(mask.sum()),
                "visible_target_sum": float((target * mask).sum()),
            }
        )
    return trace


def replay(total: int, delay: int, execution_horizon: int, schedule: str) -> dict:
    rtc_mask = build_rtc_action_mask(
        total=total,
        inference_delay=delay,
        execution_horizon=execution_horizon,
        schedule=schedule,
    )
    action_target = torch.arange(total, dtype=torch.float32).reshape(1, 1, total, 1, 1)
    state_target = torch.zeros(1, 1, 2, 1, 1)
    state_mask = torch.zeros_like(state_target)

    results = {}
    for mode in ConstraintMode:
        context = ChunkConstraintContext(
            mode=mode,
            chunk_id=7,
            target_frame_st_id=11,
            action_targets=action_target,
            action_mask=rtc_mask.reshape(1, 1, total, 1, 1),
            state_targets=state_target,
            state_mask=state_mask,
        )
        before = context.snapshot_state_constraints()
        accepted = context.update_state_slot(
            global_slot_id=11, state=torch.tensor([[[[42.0]]]])
        )
        after = context.snapshot_state_constraints()
        _, action_mask, _ = context.snapshot_action_constraints()
        results[mode.value] = {
            "feedback_accepted": accepted,
            "version_before": before[2],
            "version_after": after[2],
            "visible_state_mask_sum": float(after[1].sum()),
            "visible_state_target_slot0": float(after[0][0, 0, 0, 0, 0]),
            "action_mask": action_mask.flatten().tolist(),
            "solver_step_trace": _solver_step_replay(mode),
        }

    assert results["NONE"]["visible_state_mask_sum"] == 0
    assert sum(results["NONE"]["action_mask"]) == 0
    assert results["RTC"]["visible_state_mask_sum"] == 0
    assert results["RTC"]["action_mask"] == results["FBFM"]["action_mask"]
    assert results["FBFM"]["visible_state_mask_sum"] == 1
    assert results["NONE"]["solver_step_trace"][2]["visible_mask_sum"] == 0
    assert results["RTC"]["solver_step_trace"][2]["visible_mask_sum"] == 0
    assert results["FBFM"]["solver_step_trace"][1]["context_version"] == 0
    assert results["FBFM"]["solver_step_trace"][2]["context_version"] == 1
    assert results["FBFM"]["solver_step_trace"][2]["visible_target_sum"] == 21
    return {
        "coordinates": {
            "H": total,
            "d": delay,
            "s": execution_horizon,
            "schedule": schedule.upper(),
            "frozen": [0, delay],
            "soft": [delay, total - execution_horizon],
            "free": [total - execution_horizon, total],
        },
        "modes": results,
        "action_vector_semantics": {
            "supported_external_dimensions": [14, 16],
            "model_target_layout": "(B,D,F,N,1)",
            "mask_layout": "(B,1,F,N,1)",
            "broadcast_axis": "D (the complete action vector)",
        },
        "checks": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--s", type=int, default=12)
    parser.add_argument("--schedule", choices=("LINEAR", "EXP"), default="EXP")
    args = parser.parse_args()
    print(json.dumps(replay(args.H, args.d, args.s, args.schedule), indent=2))


if __name__ == "__main__":
    main()
