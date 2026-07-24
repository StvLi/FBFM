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

All modes preserve the checkpoint's native 16-step UniPC schedule and 8-evaluation
`dit_step_mask`. A newly arrived constraint is consumed at the next native DiT
evaluation; skipped scheduler steps reuse the most recent (possibly guided) velocity
exactly as in upstream DreamZero. This is required for `NONE` to preserve the
released policy's numerical trajectory.

## Temporal alignment

The initial chunk is generated synchronously. For each following wave, inference
starts from the current observation while 8 committed actions execute. Those
physical 7D actions are q01/q99-normalized and padded to the model's 32D action
coordinates; only the first 8x7 coordinates are masked. The generated prefix is
therefore aligned to actions executed during the virtual delay, and slots 8:16 are
the next executable suffix.

One native DiT evaluation is completed after each simulated action. Every action
produces a feedback observation. DreamZero's frozen causal WAN VAE maps an anchor
plus four sampled frames to one future latent, while one predicted latent spans
eight actions. At intermediate actions, the latest real observation is held into
the not-yet-observed sample positions and the five-frame causal window is
re-encoded. The first predicted latent slot is therefore refreshed eight times;
at offsets 2, 4, 6 and 8 another real sample replaces the held value, and the
offset-8 target exactly equals the complete anchor-plus-four-frame encoding. The
second slot remains governed by the pretrained prior. A state target never enters
solver-start history in that same generation.

The active action overlap contains `8x7=56` physical coordinates, whereas one
state latent contains `48x10x20=9600` coordinates. The fixed state block weight is
therefore `56/9600`; this gives the two active modality blocks equal aggregate
coordinate weight before the joint VJP. The unweighted/binary state-mask choice is
retained as a documented ablation, not used as the default matched run.

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
