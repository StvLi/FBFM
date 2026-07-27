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
| state feedback gain `kp` | `1.0` |
| guidance clip `beta` | 10 |

`NONE`, `RTC`, and `FBFM` use the same model, native 8-DiT cache schedule, rollout,
noise seeds and pseudo-clock. Only masks differ: `NONE` uses zero masks, `RTC`
uses the normalized 7-channel action prefix, and `FBFM` adds observed latent
state slots. The action overlap remains a binary hard constraint. The state
alignment coefficient stays fixed at the interpretable L1-mass value `56/9600`.
`--state-feedback-kp` independently scales only the aligned state residual; it
does not scale action overlap. The default `kp=1` is numerically identical to the
completed L1-mass experiment.

DreamZero reuses each native DiT prediction across one or more UniPC updates.
The integration deliberately keeps the cached prediction unguided. For every
skipped DiT update it reconstructs the clean endpoint from the current solver
sample and sigma, recomputes the masked residual, and applies a new VJP using
the most recent DiT Jacobian. This prevents one old guided velocity from being
integrated repeatedly at indices such as `2,3,4,5`.

## External dependencies

This integration does not vendor DreamZero, LIBERO, RLinf assets, environments,
or checkpoints. A deployment base workspace must provide:

| Item | Expected location under `$DREAMZERO_BASE_WORKSPACE` |
| --- | --- |
| DreamZero source | `dreamzero/` |
| LIBERO source/install | `LIBERO/` and the LIBERO environment |
| DreamZero model environment | `envs/miniconda3/envs/dreamzero` |
| LIBERO simulator environment | `envs/miniconda3/envs/libero` |
| RLinf SFT checkpoint | `checkpoints/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000` |
| UMT5 tokenizer | `assets/tokenizers/umt5-xxl` |

The model and simulator environments are intentionally separate.

The DreamZero source needs the small scheduler-callback patch before launching
the server:

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

## A6000 commands

The prepared base workspace is `/home/deepcybo-lite/fbfm_ws`. Start the model
server in its DreamZero environment:

```bash
FBFM_ROOT=$(git rev-parse --show-toplevel)
REPO=$FBFM_ROOT/wam/dreamzero-libero
BASE=${DREAMZERO_BASE_WORKSPACE:-/home/deepcybo-lite/fbfm_ws}
$BASE/envs/miniconda3/envs/dreamzero/bin/python $REPO/scripts/model_server.py \
  --base-workspace $BASE \
  --checkpoint $BASE/checkpoints/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000 \
  --tokenizer $BASE/assets/tokenizers/umt5-xxl \
  --mode FBFM --state-weight 0.005833333333333334 \
  --state-feedback-kp 1.0 --port 18766 \
  --audit $REPO/results/smoke_fbfm/solver.jsonl \
  --ready-file $REPO/results/smoke_fbfm/ready.json
```

Then run LIBERO in a second shell:

```bash
FBFM_ROOT=$(git rev-parse --show-toplevel)
REPO=$FBFM_ROOT/wam/dreamzero-libero
BASE=${DREAMZERO_BASE_WORKSPACE:-/home/deepcybo-lite/fbfm_ws}
PYTHONPATH=$REPO/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  $BASE/envs/miniconda3/envs/libero/bin/python $REPO/scripts/libero_experiment.py \
  --base-workspace $BASE --mode FBFM --state-weight 0.005833333333333334 \
  --state-feedback-kp 1.0 \
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

## Full 130-task benchmark

The model server accepts sequential simulator clients without reloading the
checkpoint. Run the complete benchmark from the LIBERO environment after the
server reports ready:

```bash
CODE_COMMIT=$(git -C "$FBFM_ROOT" rev-parse --short HEAD)
PYTHONPATH=$REPO/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  $BASE/envs/miniconda3/envs/libero/bin/python $REPO/scripts/run_libero_benchmark.py \
  --base-workspace $BASE --mode FBFM --state-weight 0.005833333333333334 \
  --state-feedback-kp 1.0 \
  --trials 20 --max-steps 480 \
  --model-seed-rule fixed --solver-release-policy uniform \
  --code-commit "$CODE_COMMIT" --port 18766 \
  --output $REPO/results/libero_all_fbfm_20_$CODE_COMMIT
```

The launcher covers `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`, and `libero_90`: 130 tasks and 2,600 episodes. It resumes only
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
repository at a fixed interval. It exits automatically after all 130 task rows
are complete.
