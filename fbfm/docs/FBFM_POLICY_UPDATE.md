# FBFM Policy & Real Robot Demo Updates

## Summary

This update introduces a new FBFM policy wrapper for real-time chunking with VAE state feedback and updates the real-robot evaluation script to use it.

## Changes

### New Policy Wrapper (`fbfm/policies/fbfm/fbfm_policy.py`)

- Added `FBFMPolicy` class built on `torch.nn.Module`.
- Added RTC-related initialization:
  - `RTCProcessor` setup
  - shared state for current action chunk, step count, observation cache, and previous chunk buffer
  - delay queue for inference latency tracking
  - background inference thread for asynchronous chunk updates
- Implemented `get_action` to:
  - store the latest observation
  - encode observation into a latent via `vae_encoder.encode`
  - append state latents to `RTCPrevChunk`
  - return step-wise actions from the current chunk with thread-safe synchronization
- Implemented `_guided_inference` to:
  - run RTC-guided denoising steps over noise initialization
  - use `RTCProcessor.denoise_step` to incorporate previous chunk leftovers and latency
  - return a new action chunk tensor
- Implemented `reset` and `close` to properly reset shared state and cleanly stop the background thread.

### Real Robot Demo Update (`fbfm/examples/fbfm/eval_with_real_robot.py`)

- Imported the new `FBFMPolicy` and RTC helpers.
- Added a `VAEEncoder` wrapper stub with an `encode` method placeholder.
- Replaced the old RTC action queue + worker threads with a single main loop that:
  - pulls observations
  - calls `policy.get_action`
  - sends actions to the robot at the target FPS
- Added cleanup calls to `policy.reset()` and `policy.close()` after the loop.

## How to Use

Run the updated demo with RTC and state feedback enabled, e.g.:

```bash
uv run examples/rtc/eval_with_real_robot.py \
    --policy.path=<模型路径> \
    --policy.device=cuda \
    --rtc.enabled=true \
    --rtc.execution_horizon=20 \
    --rtc.chunk_state_dim=32 \
    --rtc.chunk_action_dim=7 \
    --rtc.s_chunk=5 \
    --rtc.state_feedback_enabled=true \
    --rtc.state_max_guidance_weight=1.0 \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodemxxxx \
    --task="Your task" \
    --duration=60
```

## Notes

- You must implement `VAEEncoder.encode` to match your model and ensure it returns a latent vector of size `chunk_state_dim`.
- The new policy expects RTC parameters under `cfg.rtc`, including `chunk_size`, `chunk_state_dim`, and `chunk_action_dim`.
