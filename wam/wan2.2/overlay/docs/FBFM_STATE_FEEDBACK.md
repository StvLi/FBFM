# Wan2.2 visual state feedback

This route applies FBFM only to Wan2.2's video flow. It intentionally has no
action flow, RTC mask, robot proprioception, or cross-modal action Jacobian.

## Temporal contract

Wan2.2-TI2V-5B uses a causal VAE with temporal stride 4. The launch image is
native I2V slot 0. Every following group of four measured frames produces one
future feedback slot:

```text
raw frame 0      -> native I2V anchor, latent slot 0
raw frames 1-4   -> feedback latent slot 1
raw frames 5-8   -> feedback latent slot 2
...
```

No current frame is copied into an unobserved future position. The feedback
video must include the launch frame, have the same scene timing as the target
prediction, and contain four future frames for every requested feedback slot.
Solver release steps are zero-based boundaries in `[0, sample_steps)`.

At each solver boundary the pipeline:

1. releases all raw feedback events scheduled for that boundary;
2. encodes them with an independent causal VAE feature cache;
3. updates the slot-aligned latent target and binary temporal mask;
4. recomputes the video endpoint and endpoint VJP;
5. sends the guided video velocity to the native UniPC scheduler.

The two FBFM VJP branches use per-block DiT gradient checkpointing by default.
This does not alter DIRECT or no-gradient forward evaluation, but avoids
retaining every 5B transformer activation at once for full 720P feedback.

The direct baseline uses the same frame loop, prompt, noise seed, CFG, and
solver. Its feedback mask is always zero.

The CLI inherits the official TI2V-5B checkpoint defaults unless explicitly
overridden: 720P maximum area, 121 frames, 50 UniPC steps, shift 5, and CFG 5.
Shorter horizons and smaller areas are integration-test settings, not the
official deployment protocol.

Each run writes a JSON audit beside the video, including feedback slot/version
updates, per-step VJP norms, and peak CUDA allocation during the solver loop.

The default visual-state coefficient is `1.0`. The `56/9600` coefficient used
by DreamZero compensates for the different state and action dimensions in its
joint feedback objective. This pipeline has no action variables, so applying
that coefficient would incorrectly attenuate the only residual. The current
implementation does not clip the correction norm; use the per-step audit norms
to diagnose unstable full-weight runs without silently changing the FBFM
objective.

The independent proportional gain `--kp` scales the complete feedback velocity
correction after the state residual and VJP have been computed:

```text
guided_velocity = base_velocity - kp * guidance_weight * correction
```

Its default is `1.0` for backward compatibility. Gain ablations must keep
`--state-weight 1.0` and vary `--kp`, so the visual-state objective itself is
not confused with controller damping.

The CLI enables deterministic CUDA algorithms by default and forwards that
setting to FlashAttention's deterministic backward path. Keep determinism
enabled for paired comparisons; use `--no-deterministic` only for
backend/performance diagnosis.

The Blackwell environment provides `flash_attn==2.8.3.post1`, built for
`sm_120` against the installed PyTorch CUDA 12.9 runtime. A BF16 forward and
backward smoke test passes. The attention dispatcher retains a PyTorch SDPA
fallback with key masking and output-dtype restoration, so DIRECT and FBFM
always share the same attention backend.

The streaming implementation can be checked against the native full-video VAE
without loading the text encoder or DiT:

```bash
python scripts/validate_fbfm_vae.py --ckpt-dir /path/to/Wan2.2-TI2V-5B
```

## Commands

Set the external checkpoint once for the generation examples:

```bash
export WAN22_CKPT_DIR=/path/to/Wan2.2-TI2V-5B
```

Direct prediction:

```bash
python generate_fbfm.py \
  --mode DIRECT \
  --image examples/i2v_input.JPG \
  --prompt "The scene evolves with rapid object motion." \
  --output results/direct.mp4
```

FBFM with two observed latent slots released at solver steps 10 and 20:

```bash
python generate_fbfm.py \
  --mode FBFM \
  --feedback-video /path/to/measured_sequence.mp4 \
  --feedback-release-steps 10,20 \
  --kp 0.05 \
  --prompt "The scene evolves with rapid object motion." \
  --output results/fbfm.mp4
```

For a cheap integration smoke test, use `--frame-num 9 --sample-steps 2
--max-area 114688` and release one slot at step 1. `--max-area` is only a debug
override; paper runs should use an official `--size`. Paper comparisons should
use the same explicit release schedule in every paired trial and evaluate only
unconstrained future slots after each feedback point.
