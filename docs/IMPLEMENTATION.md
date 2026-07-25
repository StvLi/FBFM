# Parallel-WAM implementation record

## Mathematical mapping

DreamZero transports predicted video latents `Z` and a 16x32 padded action tensor
`A` in one DiT call. Let native DiT evaluation `k` produce base velocity `v_k`
and endpoint Jacobian `J_k`. At each UniPC scheduler index `j` served by that
cached DiT result, the runtime reconstructs

`Xhat_j = X_j - sigma_j * v_k` and `e_j = W * (Y_j - Xhat_j)`.

It then evaluates `g_j = J_k^T e_j`. The DiT velocity and Jacobian are refreshed
only at the next native evaluation, but `Xhat_j`, `e_j`, the guidance schedule,
and guided velocity `v_k - lambda(sigma_j) * g_j` are recomputed at every UniPC
index. The joint VJP retains the state-to-action and action-to-state Jacobian
blocks. Since DreamZero runs from decreasing `sigma` rather than increasing
paper time `tau`, the correction has the minus sign shown above.

## DreamZero hook boundary

The runtime wraps one loaded `WANPolicyHead` instance at
`_run_diffusion_steps`. Cache-writing calls pass through under `no_grad`. Joint
denoising calls wait for a pseudo-clock grant, encode newly queued feedback, take a
versioned constraint snapshot, and run the original conditional/unconditional
DiT. A small callback at the scheduler boundary applies guidance only to the
current update. `prev_predictions` therefore stores the detached native DiT
velocity rather than a guided velocity.

All modes preserve the checkpoint's native 16-step UniPC schedule and 8-evaluation
`dit_step_mask`. A newly arrived constraint is consumed at the next scheduler
update. At skipped DiT indices, only the last endpoint Jacobian and native
velocity are reused; the residual is always current. `NONE` returns the native
velocity unchanged and preserves the released policy's numerical trajectory.

## Temporal alignment

The initial chunk is generated synchronously. For each following wave, inference
starts from the current observation while 8 committed actions execute. Those
physical 7D actions are q01/q99-normalized and padded to the model's 32D action
coordinates; only the first 8x7 coordinates are masked. The generated prefix is
therefore aligned to actions executed during the virtual delay, and slots 8:16 are
the next executable suffix.

One native DiT cache block is completed after each simulated action. Every action
produces a feedback observation. DreamZero's frozen causal WAN VAE maps an anchor
plus four training-stride frames to one future latent; LIBERO SFT sampled those
frames three environment steps apart. The encoder keeps all observations in
causal order but refreshes its hard target only at offsets 3, 6, 9, and 12. Its
windows progress as `[0,0,0,0,3]`, `[0,0,0,3,6]`,
`[0,0,3,6,9]`, and `[0,3,6,9,12]`. With the eight-action overlap used here, the
first two refreshes can occur in one generation wave. A state target never enters
solver-start history in that same generation.

The active action overlap contains `8x7=56` physical coordinates, whereas one
state latent contains `48x10x20=9600` coordinates. The action block remains a
binary hard-overlap mask. The DreamZero route applies the state preconditioner
`sqrt(56/9600)=0.0763762616`, which equalizes expected Euclidean mask energy
under the diagnostic assumption of independent, equal-variance coordinates and
an identity Jacobian. The older `56/9600` coefficient equalizes only the L1 sum
of mask entries and makes state correction norms about 13.1 times weaker than
this RMS-balanced value. A binary state coefficient of `1.0` and the L1-mass
coefficient remain explicit ablations; neither is mixed into the default result.

## Preserved contracts

- RLinf DreamZero transforms and metadata define image composition and action scale.
- The checkpoint remains 1,828 BF16 tensors with strict key/shape loading.
- DreamZero's text/image encoders, VAE, causal cache and CFG remain unchanged.
- Video and action each retain their original `FlowUniPCMultistepScheduler`.
- LIBERO receives finite `(16,7)` chunks with binary gripper postprocessing.
- `NONE` is exactly zero guidance on the same pseudo-asynchronous execution path.

## Audit contract

Each native `solver_step` record contains mode, DiT index, sigma, feedback count,
constraint version, mask populations, endpoint errors, VJP norms, guidance weight,
and allocated CUDA bytes. Each `scheduler_guidance_step` additionally records its
UniPC index and whether the Jacobian was refreshed or reused. Exceptions are
written as server error records. Episode records include task/init/seed, success,
executed steps, wave count, timing and the deterministic grant schedule.
