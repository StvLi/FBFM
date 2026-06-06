# Robotwin FBFM success eval record

This document records the code changes that led to the first successful
Robotwin eval with FBFM enabled on June 6, 2026.

## Result

- Task: `adjust_bottle`
- Mode: `demo_randomized`
- Checkpoint:
  `/mnt/dataset/datasets/cjt_personal/pretrained_models/lingbot-va-posttrain-robotwin`
- Guidance scales: `video_guidance_scale=5`, `action_guidance_scale=1`
- Episode count: `1`
- Result: success
- Metrics: `succ_num=1.0`, `total_num=1.0`, `succ_rate=1.0`

Output files from the successful run:

- Metrics:
  `wam/lingbot-va/robotwin_outputs/adjust_bottle_randomized_20260606_afterfix_eval/client/stseed-10000/metrics/adjust_bottle/res.json`
- Video:
  `wam/lingbot-va/robotwin_outputs/adjust_bottle_randomized_20260606_afterfix_eval/client/stseed-10000/visualization/adjust_bottle/0_Raise_the_handheld_green_bottle_from_the_table_using_the_correct_arm,_the_left_arm._True.mp4`

The successful run showed feedback windows advancing through
`target_frame_st_id=2 -> 3 -> 4 -> 5 -> 6 -> 7`, with the feedback mask reset
for each new window.

## Preceding commit: 3081b19

Commit:
`3081b192b17559b3c01386c39768ee70adaa2527`

Message:
`fix: preserve FBFM feedback states across chunks and restore correct scheduler/action-state constraint alignment`

This commit established the basic FBFM semantics needed before the later
Robotwin-specific temporal fix.

Main changes:

- Added `docs/fbfm_fix_plan.md`, documenting the core integration problems:
  broken scheduler Jacobian path, feedback state loss across adapter rebuilds,
  action prefix misalignment, state latent slot misalignment, and ambiguous
  feedback encoder streaming semantics.
- Added `latent_to_state_vectors` in `wan_va/lingbot_va_bridge.py` to convert
  `(B, C, F, H, W)` video latents into per-frame flattened state vectors.
- Added persistent feedback buffer helpers:
  `FeedbackStateBuffer`, `SlotAlignedStateBuffer`, and `FeedbackSlotTracker`.
- Extended `VA_PrevChunkAdapter` so it can receive explicit previous states,
  explicit state masks, and explicit state constrained counts.
- Changed action constraints from "use the whole previous action chunk" to
  "use the inference-delay tail as the prefix constraint".
- Fixed the scheduler guidance autograd path by setting
  `x_t.requires_grad_(True)` before running the denoiser, so the correction term
  includes the `x_t -> v_t -> x1_t` Jacobian path.
- Added separate feedback streaming VAE wrappers on the server side.
- Added `_make_prev_chunk_adapter()` so adapter construction consumes persistent
  feedback state instead of resetting it at inference time.
- Added CPU tests covering no-guidance scheduler equivalence, target/weight
  sensitivity, feedback buffers, slot trackers, action tail prefixing, and
  explicit state masks.

## Follow-up commit: 14dc6fd

Commit:
`14dc6fd1a303e16ef16be95635675909ebad0d0b`

Message:
`修复: 对齐 Robotwin 反馈 latent 时序引导`

This commit tightened Robotwin feedback alignment.

Main changes:

- Initial observations prime the feedback VAE cache but do not become
  constraints.
- Feedback received before `compute_kv_cache` only advances feedback context.
- After `compute_kv_cache`, the server opens a feedback accumulation window and
  records `feedback_target_frame_st_id`.
- In the current Robotwin setup, every four sampled observations map to one GT
  latent state slot.
- The server accumulates pending feedback observations and encodes them in
  four-observation groups.
- The server requires each four-observation group to produce exactly one latent
  slot. Any latent frame count mismatch raises an error.
- State feedback is appended with an explicit local slot and global frame id.
- Once the current chunk's state slots are full, extra feedback is dropped
  instead of overfilling the state buffer.
- Tests were added for four-observation to one-slot alignment, latent frame
  mismatch rejection, and feedback window reopening.

## Latest client-side pseudo-async fix

Files:

- `evaluation/robotwin/pseudo_async.py`
- `evaluation/robotwin/eval_polict_client_openpi.py`
- `tests/test_fbfm_bridge.py`

Root cause:

- In non-first chunks, the client hit the pseudo-async wait branch and executed
  `continue`.
- That skipped the later `compute_kv_cache` trigger point.
- As a result, the server kept filling the first feedback window. Once that
  window was full, later feedback was dropped as overfill.

Fix:

- Added explicit helper functions:
  `should_wait_for_async_result`, `should_receive_async_result`, and
  `should_request_next_kv_cache`.
- Preserved the first chunk trigger at `step_count == 16`.
- Changed non-first chunks to request the next KV cache at `step_count == 32`.
- Added tests proving that the later chunk no longer triggers at 16 and does
  trigger at 32.

This fix is what allowed the successful run to show multiple feedback windows
instead of only filling the first one.

## Runtime compatibility changes

The latest patch also includes environment compatibility updates needed for the
current machine:

- `evaluation/robotwin/eval_polict_client_openpi.py` now reads `ROBOTWIN_ROOT`
  and defaults to `/mnt/dataset/projs/projects/RoboTwin`.
- `wan_va/configs/va_robotwin_cfg.py` now reads `LINGBOT_VA_MODEL` for the model
  path and `LINGBOT_VA_ENABLE_OFFLOAD` for the offload toggle.
- `wan_va/distributed/fsdp.py` falls back to returning the model unchanged when
  the current PyTorch build does not expose the newer `fully_shard` API.
- `wan_va/modules/model.py` guards `flex_attention` and `flash_attn` imports so
  unavailable optional attention backends fail only when selected.

## Unchanged items

- The official checkpoint was not modified.
- Guidance scales remained `video=5` and `action=1`.
- Feedback was not reverted to single-frame encoding.
- The imagined-video panel is still not restored. The successful eval saved
  real frames, but the imagined stream count remained zero.

## Source versus generated files

The source and documentation changes should be committed. Generated files such
as `__pycache__`, `robotwin_outputs/`, and `visualization/` should remain
uncommitted.
