# Parallel-WAM implementation record

## Mathematical mapping

DreamZero transports predicted video latents `Z` and a 16x32 padded action tensor
`A` in one DiT call. At solver evaluation `k`, the runtime reconstructs

`Z1 = Z_sigma - sigma * v_Z` and `A1 = A_sigma - sigma * v_A`.

It builds the masked discrepancy from native-latent visual targets and normalized
action targets, then evaluates one joint VJP with respect to both noisy inputs.
The returned gradients retain the cross-modal Jacobian blocks. Since DreamZero's
solver runs from decreasing `sigma` rather than increasing paper time `tau`, the
guided model outputs are `v - lambda * g`. The unmodified UniPC schedulers consume
those outputs. Every returned tensor is detached before the next step.

## DreamZero hook boundary

The runtime wraps one loaded `WANPolicyHead` instance at
`_run_diffusion_steps`. Cache-writing calls pass through under `no_grad`. Joint
denoising calls wait for a pseudo-clock grant, encode newly queued feedback, take a
versioned constraint snapshot, run the original conditional/unconditional DiT,
and apply the joint endpoint VJP. Equal guided conditional/unconditional video
outputs make the upstream CFG expression an identity without changing its code.
The conditional action output remains DreamZero's native action flow.

All modes force all 16 `dit_step_mask` entries on. Cached velocity reuse cannot
represent a newly arrived constraint or its current endpoint Jacobian, so it is not
used in the matched method comparison.

## Temporal alignment

The initial chunk is generated synchronously. For each following wave, inference
starts from the current observation while 8 committed actions execute. Those
physical 7D actions are q01/q99-normalized and padded to the model's 32D action
coordinates; only the first 8x7 coordinates are masked. The generated prefix is
therefore aligned to actions executed during the virtual delay, and slots 8:16 are
the next executable suffix.

Two solver evaluations are completed after each simulated action. Feedback is
sampled after action offsets 2, 4, 6 and 8. The four transformed multi-view frames
are encoded together by the checkpoint's frozen WAN VAE and assigned once to the
first of the two predicted latent slots. The second slot remains governed by the
pretrained prior. A state target never enters solver-start history in that same
generation.

## Preserved contracts

- RLinf DreamZero transforms and metadata define image composition and action scale.
- The checkpoint remains 1,828 BF16 tensors with strict key/shape loading.
- DreamZero's text/image encoders, VAE, causal cache and CFG remain unchanged.
- Video and action each retain their original `FlowUniPCMultistepScheduler`.
- LIBERO receives finite `(16,7)` chunks with binary gripper postprocessing.
- `NONE` is exactly zero guidance on the same pseudo-asynchronous execution path.

## Audit contract

Each numerical solver record contains mode, solver index, sigma, feedback count,
constraint version, state/action mask populations, endpoint-error norms, both VJP
correction norms, guidance weight and allocated CUDA bytes. Exceptions are written
as server error records. Episode records include task/init/seed, success, executed
steps, wave count, timing and the full deterministic grant schedule.
