# DreamZero x FBFM x LIBERO

Training-free Feedback Flow Matching for the joint state-action flow in the
RLinf DreamZero WAN2.2 5B LIBERO policy.

This is the DreamZero/LIBERO route inside the FBFM monorepo. The independent
LingBot-VA/RoboTwin route is at [`../lingbot-va`](../lingbot-va/README.md).

This repository is an integration layer. It does not fork or retrain DreamZero,
change its checkpoint, replace UniPC, alter LIBERO actions, or redefine model
time using measured latency. It adds:

- one block-diagonal state/action constraint at each joint solver evaluation;
- endpoint VJP guidance with the paper's clipped few-step schedule;
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
| feedback sample stride | 2 observations |
| observations per latent | 4 |
| active state slots per wave | first of 2 predicted latent slots |
| state modality weight | `56/9600` (active action/state coordinate balance) |
| guidance clip `beta` | 10 |

`NONE`, `RTC`, and `FBFM` use the same model, native 8-DiT cache schedule, rollout,
noise seeds and pseudo-clock. Only masks differ: `NONE` uses zero masks, `RTC`
uses the normalized 7-channel action prefix, and `FBFM` adds observed latent
state slots.

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
  --mode FBFM --port 18766 \
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
  --base-workspace $BASE --mode FBFM --suite libero_spatial --task-id 0 \
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

## Full 130-task benchmark

The model server accepts sequential simulator clients without reloading the
checkpoint. Run the complete benchmark from the LIBERO environment after the
server reports ready:

```bash
CODE_COMMIT=$(git -C "$FBFM_ROOT" rev-parse --short HEAD)
PYTHONPATH=$REPO/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  $BASE/envs/miniconda3/envs/libero/bin/python $REPO/scripts/run_libero_benchmark.py \
  --base-workspace $BASE --mode FBFM --trials 20 --max-steps 480 \
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
`uniform` pseudo-clock executes one committed overlap action, refreshes the
aligned latent target from the newest causal observation window, and then
releases one native DreamZero DiT evaluation. Incomplete four-frame VAE blocks
are completed by holding the latest real observation forward; the target is
re-encoded and versioned after every action, and equals the native complete
block encoding at the final sample. This protocol is for deterministic
pseudo-asynchronous method evaluation, not wall-clock latency.

The four-frame model-input history is separate from rolling feedback. Only
`predict_sync` and `predict_start` anchors advance DreamZero's causal KV-cache
history; per-action `feedback` observations update FBFM targets without
entering that history. Sending one frame for every request triggers DreamZero's
upstream `videos.shape[2] == 1` reset path and invalidates overlap results.

`sync_libero_ledger.py` can mirror those four ledger files into a paper
repository at a fixed interval. It exits automatically after all 130 task rows
are complete.
