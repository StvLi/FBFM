# DreamZero x FBFM x LIBERO

Training-free Feedback Flow Matching for the joint state-action flow in the
RLinf DreamZero WAN2.2 5B LIBERO policy.

This is the DreamZero/LIBERO route inside the FBFM monorepo. The independent
LingBot-VA/RoboTwin route is at [`../lingbot-va`](../lingbot-va/README.md).

This repository is an integration layer. It does not fork or retrain DreamZero,
change its checkpoint, replace UniPC, alter LIBERO actions, or redefine model
time using measured latency. It adds:

- one block-diagonal state/action constraint at each UniPC scheduler update;
- endpoint VJP guidance with the paper's clipped few-step schedule, refreshing
  the Jacobian only when DreamZero evaluates its DiT;
- native DreamZero VAE encoding for per-action rolling visual feedback;
- native DreamZero causal inference history: one-frame warm-up followed by
  four-frame inference-anchor requests with continuous KV-cache positions;
- a deterministic pseudo-clock linking 8 executed actions to the checkpoint's
  8 native DiT evaluations across 16 UniPC scheduler steps;
- a localhost protocol between the Python 3.11 model and Python 3.8 simulator;
- auditable JSONL records for masks, errors, corrections, versions and memory.

## Fixed validation protocol

| Variable | Value |
|---|---:|
| action horizon `H` | 16 |
| inference delay `d` | 8 |
| executed suffix `s` | 8 |
| UniPC scheduler steps | 16 |
| native DiT evaluations | 8 |
| causal model input | 1-frame warm-up, then latest 4 inference anchors |
| grants per simulator step | 1 |
| feedback cadence | every executed action |
| rolling VAE sample interval | 3 action steps (checkpoint training stride) |
| observations per latent | 4 |
| active state slots per wave | first of 2 predicted latent slots |
| state alignment weight | `56/9600 = 0.0058333333` |
| state feedback gain `kp` | `0.04869675251658631` |
| guidance clip `beta` | 10 |

`NONE`, `RTC`, and `FBFM` use the same model, native 8-DiT cache schedule, rollout,
noise seeds and pseudo-clock. Only masks differ: `NONE` uses zero masks, `RTC`
uses the normalized 7-channel action prefix, and `FBFM` adds observed latent
state slots. The action overlap remains a binary hard constraint. The state
alignment coefficient stays fixed at the interpretable L1-mass value `56/9600`.
`--state-feedback-kp` independently scales only the aligned state residual; it
does not scale action overlap. The public submission defaults to the selected paper gain `kp=0.04869675251658631`; `kp=1` remains a documented search point.

DreamZero reuses each native DiT prediction across one or more UniPC updates.
The integration deliberately keeps the cached prediction unguided. For every
skipped DiT update it reconstructs the clean endpoint from the current solver
sample and sigma, recomputes the masked residual, and applies a new VJP using
the most recent DiT Jacobian. This prevents one old guided velocity from being
integrated repeatedly at indices such as `2,3,4,5`.

## External dependencies

This integration does not vendor DreamZero, LIBERO, RLinf assets, environments,
or checkpoints. A deployment base workspace must provide:

| Item | Expected path |
| --- | --- |
| DreamZero source | `$DREAMZERO_BASE_WORKSPACE/dreamzero` |
| LIBERO source/install | `$DREAMZERO_BASE_WORKSPACE/LIBERO` |
| DreamZero model environment | `$FBFM_ENV_ROOT/fbfm-dreamzero` |
| LIBERO simulator environment | `$FBFM_ENV_ROOT/fbfm-libero` |
| RLinf SFT checkpoint | `$DREAMZERO_CHECKPOINT` |
| UMT5 tokenizer | `$DREAMZERO_TOKENIZER` |
| Wan2.2 diffusion/T5/VAE bundle | `$DREAMZERO_WAN_CHECKPOINT` |
| Wan CLIP image encoder | `$DREAMZERO_IMAGE_ENCODER` |

The model and simulator environments are intentionally separate.
`$DREAMZERO_WAN_CHECKPOINT` must contain
`models_t5_umt5-xxl-enc-bf16.pth` and `Wan2.2_VAE.pth`;
`$DREAMZERO_IMAGE_ENCODER` must name
`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`. The server checks
these files before importing the RLinf model, replaces the training-machine
paths at both levels of the checkpoint config, and refuses implicit downloads.

The repository-wide bootstrap applies the small scheduler-callback patch
automatically:

```bash
bash scripts/bootstrap/fetch_upstreams.sh --route dreamzero
bash scripts/bootstrap/create_envs.sh --route dreamzero
bash scripts/bootstrap/verify.sh --route dreamzero --strict
```

The following `git apply` command is only the manual alternative for a pristine
checkout at the pinned DreamZero revision; do not run it again after the
bootstrap has already applied the patch:

```bash
git -C "$BASE/dreamzero" apply \
  "$REPO/patches/dreamzero_external_step_guidance.patch"
```

The patch does not modify the DiT cache mask or either UniPC scheduler. It only
exposes the current scheduler sample and native cached velocity to this runtime.

## Source ownership

The complete route spans two deliberate source locations in this monorepo:

| Source | Responsibility | Import environment |
| --- | --- | --- |
| `fbfm/model_runtime.py` | RLinf config construction, strict single-GPU policy loading, episode cache reset | DreamZero Python 3.11 |
| `fbfm/libero_observation.py` | LIBERO images, quaternion conversion, 8D state and dummy action | LIBERO simulator |
| `wam/dreamzero-libero/src/dreamzero_fbfm/` | FBFM constraints, joint solver hook, pseudo-clock and transport | route-specific |

Both launchers bootstrap the monorepo root and load the checked-in `fbfm`
modules before any same-named directory under the deployment workspace. A
machine-local `$DREAMZERO_BASE_WORKSPACE/fbfm` copy is not required and must not
be treated as source of record.

## Unified launch commands

Set `DREAMZERO_BASE_WORKSPACE` to the deployment workspace containing the
external sources listed above and set every artifact path explicitly. Start
the model server with the isolated environment created by the root bootstrap:

```bash
FBFM_ROOT=$(git rev-parse --show-toplevel)
REPO=$FBFM_ROOT/wam/dreamzero-libero
BASE=${DREAMZERO_BASE_WORKSPACE:?Set DREAMZERO_BASE_WORKSPACE}
ENV_ROOT=${FBFM_ENV_ROOT:-$FBFM_ROOT/.venvs}
MODEL_PY=${FBFM_DREAMZERO_PYTHON:-$ENV_ROOT/fbfm-dreamzero/bin/python}
DREAMZERO_CHECKPOINT=${DREAMZERO_CHECKPOINT:?Set DREAMZERO_CHECKPOINT}
DREAMZERO_TOKENIZER=${DREAMZERO_TOKENIZER:?Set DREAMZERO_TOKENIZER}
DREAMZERO_WAN_CHECKPOINT=${DREAMZERO_WAN_CHECKPOINT:?Set DREAMZERO_WAN_CHECKPOINT}
DREAMZERO_IMAGE_ENCODER=${DREAMZERO_IMAGE_ENCODER:?Set DREAMZERO_IMAGE_ENCODER}

"$MODEL_PY" "$REPO/scripts/model_server.py" \
  --base-workspace $BASE \
  --checkpoint "$DREAMZERO_CHECKPOINT" \
  --tokenizer "$DREAMZERO_TOKENIZER" \
  --wan-checkpoint "$DREAMZERO_WAN_CHECKPOINT" \
  --image-encoder "$DREAMZERO_IMAGE_ENCODER" \
  --mode FBFM --state-weight 0.005833333333333334 \
  --state-feedback-kp 0.04869675251658631 --port 18766 \
  --audit "$REPO/results/smoke_fbfm/solver.jsonl" \
  --ready-file "$REPO/results/smoke_fbfm/ready.json"
```

Then run LIBERO in a second shell:

```bash
FBFM_ROOT=$(git rev-parse --show-toplevel)
REPO=$FBFM_ROOT/wam/dreamzero-libero
BASE=${DREAMZERO_BASE_WORKSPACE:?Set DREAMZERO_BASE_WORKSPACE}
ENV_ROOT=${FBFM_ENV_ROOT:-$FBFM_ROOT/.venvs}
SIM_PY=${FBFM_LIBERO_PYTHON:-$ENV_ROOT/fbfm-libero/bin/python}
PYTHONPATH="$REPO/src:$FBFM_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$SIM_PY" "$REPO/scripts/libero_experiment.py" \
  --base-workspace $BASE --mode FBFM --state-weight 0.005833333333333334 \
  --state-feedback-kp 0.04869675251658631 \
  --suite libero_spatial --task-id 0 \
  --trial-start 0 --trials 1 --max-steps 480 --port 18766 \
  --model-seed-rule fixed --solver-release-policy uniform \
  --output $REPO/results/smoke_fbfm
```

For a true upstream-style control, launch the server with `--mode NONE` and run
the client with `--mode NONE --rollout-protocol native_sync`. This replans from
the latest observation and executes `chunk[:8]` without calling the overlap,
feedback, or pseudo-clock APIs. The default `pseudo_async_overlap` protocol is
the matched `NONE` / `RTC` / `FBFM` ablation path and must not be reported as
the native DreamZero base.

Run CPU regressions with:

```bash
PYTHONPATH=src:../.. python -m pytest
```

Primary generated files are ignored under `results/`: `episodes.jsonl`,
`summary.json`, `solver.jsonl`, and the model-load ready report.

## State-feedback kp search

The screening search fixes `--state-weight 0.005833333333333334` and varies only
`--state-feedback-kp`. Its initial logarithmic points are `0.001`, `1`, and
`100`. The `kp=1` point reuses the matching rows from the completed L1-mass
record. Each new point runs four fixed tasks (`libero_spatial` 1 and 9;
`libero_object` 0 and 6), official init IDs 0-4, fixed model seed 0, and a
distinct result directory. Later points bisect a selected interval
geometrically:

```text
next_kp = sqrt(lower_kp * upper_kp)
```

The machine-readable protocol is in `config/kp_search.yaml`. The model server
and benchmark client must receive the same `--state-weight` and
`--state-feedback-kp`; the reset handshake rejects a mismatch before an episode
starts.

## Paper four-suite benchmark

The model server accepts sequential simulator clients without reloading the
checkpoint. Run the complete benchmark from the LIBERO environment after the
server reports ready:

```bash
CODE_COMMIT=$(git -C "$FBFM_ROOT" rev-parse --short HEAD)
PYTHONPATH=$REPO/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  $BASE/envs/miniconda3/envs/libero/bin/python $REPO/scripts/run_libero_benchmark.py \
  --base-workspace $BASE --mode FBFM --state-weight 0.005833333333333334 \
  --state-feedback-kp 0.04869675251658631 \
  --trials 20 --max-steps 480 \
  --model-seed-rule fixed --solver-release-policy uniform \
  --code-commit "$CODE_COMMIT" --port 18766 \
  --output $REPO/results/libero_all_fbfm_20_$CODE_COMMIT
```

By default the launcher covers `libero_spatial`, `libero_object`, `libero_goal`,
and `libero_10`: 40 tasks and 800 episodes. `libero_90` is available only with an explicit `--suite libero_90` and is excluded from the paper because the checkpoint was not trained on that suite. It resumes only
from a contiguous prefix of official trial IDs, refuses duplicate records, and
atomically refreshes `task_summary.csv`, `trials.csv`, and `live_status.md`
after every task.

The comparison protocol uses a uniform 480-step episode horizon for every
suite. This intentionally caps LIBERO-10 below RLinf's 520-step reference so
all suites share one limit; the manifest records this choice. The default
`uniform` pseudo-clock executes one committed overlap action and then releases
one native DreamZero DiT evaluation. Every real observation is retained, while
the rolling VAE target is refreshed at the checkpoint's three-action sampling
stride. Missing history at a slot boundary is left-padded with its measured
anchor; no unobserved future frame is inserted. This protocol is for deterministic
pseudo-asynchronous method evaluation, not wall-clock latency.

The four-frame model-input history is separate from rolling feedback. Only
`predict_sync` and `predict_start` anchors advance DreamZero's causal KV-cache
history; per-action `feedback` observations update FBFM targets without
entering that history. Sending one frame for every request triggers DreamZero's
upstream `videos.shape[2] == 1` reset path and invalidates overlap results.

`sync_libero_ledger.py` can mirror those four ledger files into a paper
repository at a fixed interval. It exits automatically after all 40 paper task rows
are complete.
