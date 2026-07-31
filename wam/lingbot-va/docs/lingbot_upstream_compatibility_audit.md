# Lingbot-VA / FBFM Compatibility Audit

> This is a historical compatibility audit. Absolute paths and branch names
> below document the original validation machine; they are not reproduction
> entry points. Use [`../README.md`](../README.md) and the repository-root
> `README.md` for the current portable setup and launch commands.

## Scope and baselines

This audit separates Lingbot-VA inference semantics from the FBFM integration.
The three code baselines are:

- upstream Lingbot-VA: `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`
- FBFM repository comparison point: `e244b158fe44c8a524f1cebe8940b228a1311f5f`
- active FBFM branch before local fixes: `a78902d` (`origin/Pseudo-Asynchronous`)

The primary rule is that `NONE` must preserve upstream model calls, solver
updates, cache writes, and action selection. FBFM-specific behavior may only be
active when its corresponding constraint mask or runtime mode is enabled.

## Upstream semantics that must be preserved

1. The video flow runs before the action flow.
2. Both flows append one `t=0` transformer call with `update_cache=1`.
   This call writes the prediction cache; its numerical output is intentionally
   ignored and must not be integrated by the scheduler.
3. Real observations and executed actions update history through
   `_compute_kv_cache` with `update_cache=2` for video and then action.
4. The main streaming VAE cache is the history encoder state. FBFM feedback
   encoding must not clear, advance, or otherwise mutate that cache.
5. The native `FlowMatchScheduler` is unchanged. The upstream and embedded
   copies have SHA-256
   `9833300936d1001e5dfb0db76d5e0e131429685980de7b3c9716288ec28da7f1`.
6. Transformer, KV-cache, and VAE mathematics are not modified for FBFM. The
   current `modules/model.py` differences only guard optional Flash/Flex
   Attention imports; the checkpoint selects `attn_mode="torch"`.

## Confirmed FBFM integration defects

### Autograd graph retained across solver steps

`WrapperedFlowMatchScheduler.step` globally enabled gradients and returned a
sample retaining the denoiser/VJP graph. The next solver step then used that
sample as its input, chaining entire transformer graphs across the chunk.
Asynchronous saves further extended graph lifetime. This explains monotonic GPU
memory growth and increasing per-step latency.

Local fix: create a fresh input VJP graph per guided step, detach the solver
result, run zero-mask/NONE paths under `no_grad`, and copy detached tensors to
CPU before queuing a save.

### Lingbot cache-only call integrated as an extra solver step

The wrapper treated the padded cache-write call as `to_final=True` and applied
another Euler update. This differs from upstream for both video and action; the
action difference directly changes the returned command chunk.

Local fix: add an explicit `cache_only` path that calls the transformer once to
preserve `update_cache=1`, ignores its output, and returns the detached sample
unchanged.

### Feedback encoder initialized in NONE

The first observation was unconditionally encoded into the separate feedback
VAE stream, even in `NONE`. This did not belong to upstream inference and added
unnecessary activation cache and latency.

Local fix: allocate, prime, and enqueue the feedback stream only in `FBFM`
mode. `NONE` and `RTC` acknowledge client feedback messages as ignored without
advancing any feedback VAE state.

### RoboTwin protocol changed from pseudo-asynchronous to wall-clock concurrent

Commit `e777e86` explicitly replaced the pseudo-asynchronous rollout with a
background inference thread and measured wall-clock completion. This conflicts
with the paper's controlled RoboTwin protocol in `aaai_paper/docs/TODO.md`: the
experiment must define simulation-step to solver-step progression, feedback
delivery, and result handoff independently of host timing.

This must not be fixed by reverting to `e777e86^`: that implementation skips
environment actions while simulating delay and cannot guarantee that feedback
enters the active solver. A deterministic virtual-clock gate is required.

### Real observations were written into history twice

The wall-clock rewrite first used the four observations collected from an
executed suffix to update the real KV cache and generate a chunk. At the next
loop iteration it immediately submitted the same four observations in another
combined cache/generation request. This advanced `frame_st_id` twice for one real
transition and made the observation and action time dimensions disagree.

Local fix: keep one consumable pending-history segment. The initial segment is
Lingbot's initial latent plus its conditioned first action frame. Every later
segment is exactly the real observations and action-frame suffix produced while
the preceding solver was active. A segment is consumed once at the following
launch; feedback produced after launch remains in the dynamic FBFM set until
then. `_compute_kv_cache` now rejects observation/action frame-count mismatches
before writing attention memory.

### Previous-action targets used physical rather than model coordinates

`_infer` returns actions after Lingbot's quantile denormalization, and
`self.last_action` therefore lives in the public robot-action coordinates. The
FBFM adapter passed those values directly to a solver whose action sample is in
the normalized 30-channel model coordinates. The resulting discrepancy mixed
two coordinate systems and also risked guiding unused action channels.

Local fix: pass `self.last_action` through Lingbot's unchanged
`preprocess_action`, zero the same unused channels as `_prepare_latent_input`,
and only then align the normalized previous suffix with the new chunk prefix.
The real-history KV path continues to use the same upstream preprocessing. The
temporal RTC mask is also intersected with Lingbot's native action-channel mask,
so unused transformer outputs cannot contribute a VJP correction.

## FBFM extensions that should remain isolated

- A feedback-specific streaming VAE wrapper is necessary so feedback encoding
  does not corrupt Lingbot's main history stream.
- A versioned `ChunkConstraintContext` is necessary so feedback received before
  a video solver evaluation is visible at that evaluation.
- Previous-action targets and masks remain fixed for one generated chunk. They
  describe cross-chunk overlap, not the moving execution pointer.
- FBFM state feedback applies to the video flow. The corrected final video
  latent is then written to Lingbot's native prediction cache and conditions the
  subsequent action flow, preserving the upstream state-first/action-second
  factorization.

## Verification completed

CPU regression tests currently prove:

- wrapper and native scheduler equality with no constraint or a zero mask;
- no returned sample has `requires_grad` or `grad_fn`;
- repeated guided steps do not chain solver graphs or populate parameter grads;
- the sigma-parameterized clean endpoint, endpoint VJP, guidance schedule, and
  Euler update exactly match the paper equation in an analytic test;
- `NONE` video/action trajectories match native scheduler results;
- transformer cache flags match upstream (`0 ... 0, 1`);
- the cache-only transformer input is the final native solver sample;
- `NONE` does not initialize the feedback VAE stream;
- pseudo-asynchronous history segments are consumed exactly once and preserve
  observation/action time alignment;
- previous-action targets round-trip through Lingbot's native normalization and
  match the action solver's coordinate system.

Run:

```bash
source /path/to/activate-lingbot-va.sh
cd /path/to/FBFM
pytest -q -p no:cacheprovider \
  wam/lingbot-va/tests/test_fbfm_bridge.py \
  wam/lingbot-va/tests/test_async_transport.py
```

Current result: `39 passed`.

The downloaded Robotwin checkpoint was also parsed and constructed on CPU:

- custom `WanTransformer3DModel`: 5,088,872,670 parameters, 841 tensors in
  three shards, `in_channels=48`, `attn_mode=torch`;
- `UMT5EncoderModel`: 5,680,910,336 parameters, 242 tensors in three shards;
- `AutoencoderKLWan`: 704,688,668 parameters, 196 tensors;
- `T5TokenizerFast`: vocabulary size 256,300.

The transformer loader reports the legacy `patch_embedding.weight/bias` as
unused. This is consistent with the unmodified upstream Lingbot class: both the
embedded and official code use the checkpoint's `patch_embedding_mlp` and do
not define the older `patch_embedding` module. No model-owned parameter is
reported missing. The VAE's ignored `clip_output=false` config field is likewise
a no-op under the installed Diffusers version.

## Remaining gates before benchmark execution

1. Do not use wall-clock completion to decide solver progress or model real
   deployment latency in this mathematical-method experiment.
2. Record per-chunk solver step, feedback version, masks, correction norms, and
   allocated/reserved/peak GPU memory. Wall-clock timing is outside the present
   mathematical-method validation.
3. Verify the checkpoint structure and load it in `NONE` before enabling RTC or
   FBFM.
4. Pause DreamZero x LIBERO before any Lingbot GPU process is started.
5. Run one identical RoboTwin seed in `NONE`, `RTC`, and `FBFM`, then expand only
   after numerical finiteness and bounded memory are confirmed.
