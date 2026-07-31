# Feedback Flow Matching (FBFM)

This is the public reproduction repository for **Feedback Flow Matching
(FBFM)**, a training-free method that constrains a world-action model with
observations collected while the current action chunk is being executed. The
`submission` branch combines the paper's three independently reproducible
routes without mixing their model runtimes, simulator environments, weights,
or generated data.

| Route | Model / environment | Scope | Entry point |
| --- | --- | --- | --- |
| LingBot-VA | LingBot-VA + RoboTwin 2.0 | closed-loop robot evaluation; `NONE`, `RTC`, and `FBFM` | [`wam/lingbot-va`](wam/lingbot-va/README.md) |
| DreamZero | DreamZero + LIBERO | closed-loop robot evaluation; joint video/action feedback | [`wam/dreamzero-libero`](wam/dreamzero-libero/README.md) |
| Wan2.2 | Wan2.2-TI2V-5B + recorded video | visual-state auxiliary experiment only | [`wam/wan2.2`](wam/wan2.2/README.md) |

The Wan2.2 route is not a robot policy and its recorded RealSense experiment is
not an online-control or success-rate result. Results from the three routes use
different models and protocols and must not be compared as if they were one
benchmark.

## Quick start

Prerequisites are Linux, Git, Bash 4+, an NVIDIA driver/CUDA toolchain for GPU
launches, and Conda or Mamba (recommended). The CPU smoke tests do not load
model weights. Full environments install large CUDA packages, including
PyTorch and flash-attention, so verify that the chosen builds match the host
driver and GPU architecture.

```bash
git clone https://github.com/StvLi/FBFM.git
cd FBFM
git switch submission

export FBFM_ROOT="$PWD"
export FBFM_EXTERNAL_ROOT="$FBFM_ROOT/external"
export FBFM_ENV_ROOT="$FBFM_ROOT/.venvs"

# Clone immutable upstream revisions and apply the checked integration patches.
bash scripts/bootstrap/fetch_upstreams.sh --route all

# Build five isolated Python environments for the three routes.
bash scripts/bootstrap/create_envs.sh --route all

# Source/revision/license/patch checks; no model or episode is launched.
bash scripts/bootstrap/verify.sh --strict

# Bounded CPU/import tests; no complete simulator episode is run.
bash scripts/smoke.sh all
```

`fetch_upstreams.sh` never downloads checkpoints or datasets. It safely and
idempotently applies the RoboTwin raster, DreamZero scheduler-callback, and
Wan2.2 FBFM patches only after checking the pinned commits. It refuses a dirty
checkout that contains any other local change. Use `--dry-run` to inspect the
plan and `--offline` to use already available Git objects.

Environment creation can also be inspected without changing the machine:

```bash
bash scripts/bootstrap/create_envs.sh --route all --dry-run
```

Use `FBFM_EXTERNAL_ROOT` and `FBFM_ENV_ROOT` to keep sources and environments
outside this checkout. Checkpoints, environments, third-party source trees,
simulator outputs, and generated videos are deliberately excluded from Git.

## Immutable upstream sources

[`third_party/manifest.yaml`](third_party/manifest.yaml) is the machine-readable
source of record.

| Dependency | Repository | Revision | License |
| --- | --- | --- | --- |
| LingBot-VA | `https://github.com/robbyant/lingbot-va.git` | `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` | Apache-2.0 |
| RoboTwin | `https://github.com/RoboTwin-Platform/RoboTwin.git` | `2eeec322d95799f537cbfe5f291a8220d965ccb8` | MIT |
| DreamZero | `https://github.com/dreamzero0/dreamzero.git` | `ab790c198fbce33503358efbbd4187ce9a89adf3` | Apache-2.0 |
| LIBERO | `https://github.com/RLinf/LIBERO.git` | `0c5e40cc4ae63e09c14e7df6f74481e9ee8585f7` | MIT |
| RLinf | `https://github.com/RLinf/RLinf.git` | `26179807d701950cf2933554bfb9bb596e662b68` | Apache-2.0 |
| Wan2.2 | `https://github.com/Wan-Video/Wan2.2.git` | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` | Apache-2.0 |

The FBFM repository is Apache-2.0. Upstream code, model weights, datasets, and
simulator assets remain subject to their own licenses and terms. Do not
redistribute a checkpoint merely because its loader is included here.

## Repository and environment layout

```text
FBFM/
  fbfm/                       shared method/deployment adapters
  wam/lingbot-va/             LingBot-VA x RoboTwin integration
  wam/dreamzero-libero/       DreamZero x LIBERO integration
  wam/wan2.2/                 Wan2.2 overlay, patch, tests, and audit summaries
  scripts/bootstrap/          source, environment, and verification helpers
  scripts/smoke.sh            bounded route tests
  scripts/package_submission.sh
  third_party/manifest.yaml
  external/                   generated upstream checkouts; ignored
  .venvs/                     generated environments; ignored
```

Model and simulator packages are intentionally separated:

| Process | Default interpreter | Python | Main contents |
| --- | --- | ---: | --- |
| LingBot model server | `.venvs/fbfm-lingbot-va/bin/python` | 3.10 | LingBot-VA, PyTorch, FBFM bridge |
| RoboTwin client | `.venvs/fbfm-robotwin/bin/python` | 3.10 | SAPIEN, RoboTwin, CuRobo stack |
| DreamZero model server | `.venvs/fbfm-dreamzero/bin/python` | 3.11 | DreamZero, RLinf, joint FBFM runtime |
| LIBERO client | `.venvs/fbfm-libero/bin/python` | 3.8 | LIBERO, robosuite, MuJoCo stack |
| Wan2.2 inference | `.venvs/fbfm-wan2.2/bin/python` | 3.10 | patched Wan2.2 TI2V stack |

Build one route at a time when preferred:

```bash
bash scripts/bootstrap/create_envs.sh --route lingbot
bash scripts/bootstrap/create_envs.sh --route dreamzero
bash scripts/bootstrap/create_envs.sh --route wan
```

The script prefers Conda/Mamba and falls back to `venv` when the exact Python
interpreter is available. `--skip-install` only creates the environments;
`--skip-upstream` is useful for lightweight route tests. Existing prepared
interpreters can be used directly by setting the route-specific variables
listed by `bash scripts/smoke.sh --help`.

## Model artifacts

Weights are not downloaded by the bootstrap and must stay outside Git.

| Route | Required artifact | Publisher name |
| --- | --- | --- |
| LingBot | RoboTwin post-training checkpoint | `robbyant/lingbot-va-posttrain-robotwin` |
| DreamZero | policy checkpoint | `RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000` |
| DreamZero | tokenizer | `umt5-xxl` |
| Wan2.2 | TI2V checkpoint | `Wan-AI/Wan2.2-TI2V-5B` |

Set explicit paths and ask the verifier to check them:

```bash
export LINGBOT_VA_MODEL=/path/to/lingbot-va-posttrain-robotwin
export DREAMZERO_CHECKPOINT=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000
export DREAMZERO_TOKENIZER=/path/to/umt5-xxl
export WAN22_CKPT_DIR=/path/to/Wan2.2-TI2V-5B
export WAN_CHECKPOINT="$WAN22_CKPT_DIR"

bash scripts/bootstrap/verify.sh --strict --checkpoints
```

For LingBot inference, the checkpoint's `transformer/config.json` must use
`"attn_mode": "torch"` or `"flashattn"`; the training-only `"flex"` mode is
not valid for evaluation.

## Bounded verification

These commands validate integration contracts without waiting for a benchmark
episode or reporting a success rate:

```bash
bash scripts/bootstrap/verify.sh --strict
bash scripts/smoke.sh shared
bash scripts/smoke.sh lingbot
bash scripts/smoke.sh dreamzero
bash scripts/smoke.sh wan
```

The route smoke tests use `PYTHONDONTWRITEBYTECODE=1`, disable third-party
pytest plugin autoloading, and avoid the full checkpoint. To use pre-existing
environments outside `.venvs`, for example:

```bash
FBFM_LINGBOT_PYTHON=/path/to/lingbot/bin/python bash scripts/smoke.sh lingbot
FBFM_DREAMZERO_PYTHON=/path/to/dreamzero/bin/python bash scripts/smoke.sh dreamzero
FBFM_WAN_PYTHON=/path/to/wan/bin/python \
  FBFM_WAN_SOURCE_ROOT="$FBFM_EXTERNAL_ROOT/Wan2.2" \
  bash scripts/smoke.sh wan
```

## LingBot-VA + RoboTwin

The three launchers share the checkpoint, rollout path, seed, solver budget,
and pseudo-clock. Only the exposed constraints change.

| Mode | Previous-action constraint | Live state constraint |
| --- | ---: | ---: |
| `NONE` | no | no |
| `RTC` | yes | no |
| `FBFM` | yes | yes |

Configure the two processes and run a one-episode launch:

```bash
export ROBOTWIN_ROOT="$FBFM_EXTERNAL_ROOT/RoboTwin"
export LINGBOT_SERVER_PYTHON="$FBFM_ENV_ROOT/fbfm-lingbot-va/bin/python"
export ROBOTWIN_CLIENT_PYTHON="$FBFM_ENV_ROOT/fbfm-robotwin/bin/python"
export LINGBOT_VA_MODEL=/path/to/lingbot-va-posttrain-robotwin
export LINGBOT_SERVER_GPU=0
export ROBOTWIN_CLIENT_GPU=0
export LINGBOT_VA_PORT=29156
export LINGBOT_VA_MASTER_PORT=29161
export LINGBOT_VA_ENABLE_OFFLOAD=1

cd "$FBFM_ROOT/wam/lingbot-va"
bash script/run_robotwin_none.sh 1
bash script/run_robotwin_rtc.sh 1
bash script/run_robotwin_fbfm.sh 1
```

Run the modes separately so ports and GPU memory do not overlap. The optional
positional argument is the number of episodes. The launcher starts the model
server and simulator client, checks the resulting `res.json`, and always stops
its own server. Override `ROBOTWIN_TASK_NAME`, `ROBOTWIN_TASK_CONFIG`, and
`ROBOTWIN_EVAL_SEED` for another task/configuration. Outputs are written below
`wam/lingbot-va/robotwin_outputs/` and are ignored by Git.

The bootstrap applies the included RoboTwin raster-backend compatibility patch.
The default launch is headless NVIDIA Vulkan with SAPIEN ray tracing disabled;
it does not require Xorg or `DISPLAY`. The documented parallel launchers target
a high-memory GPU. Start with one server/client pair on smaller devices.

## DreamZero + LIBERO

DreamZero uses a Python 3.11 model server and a separate Python 3.8 LIBERO
client connected on localhost. The bootstrap applies the small external
scheduler-callback patch to pinned DreamZero source; it does not change the
checkpoint, the 16-step UniPC scheduler, or its native 8-DiT cache schedule.

The AAAI paper reports exactly these four suites:

```text
libero_spatial
libero_object
libero_goal
libero_10
```

`libero_90` is intentionally excluded from the submission protocol. The
paper-final FBFM coefficients must be supplied to both server and client:

```text
state_weight = 56/9600 = 0.005833333333333334
state_feedback_kp = 0.04869675251658631
```

Start the model server in shell A:

```bash
export DREAMZERO_BASE_WORKSPACE="$FBFM_EXTERNAL_ROOT"
export DREAMZERO_CHECKPOINT=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000
export DREAMZERO_TOKENIZER=/path/to/umt5-xxl

ROOT="$FBFM_ROOT"
ROUTE="$ROOT/wam/dreamzero-libero"
BASE="$DREAMZERO_BASE_WORKSPACE"
MODEL_PY="$FBFM_ENV_ROOT/fbfm-dreamzero/bin/python"
STATE_WEIGHT=0.005833333333333334
STATE_KP=0.04869675251658631
mkdir -p "$ROUTE/results/smoke_fbfm"

"$MODEL_PY" "$ROUTE/scripts/model_server.py" \
  --base-workspace "$BASE" \
  --checkpoint "$DREAMZERO_CHECKPOINT" \
  --tokenizer "$DREAMZERO_TOKENIZER" \
  --mode FBFM \
  --state-weight "$STATE_WEIGHT" \
  --state-feedback-kp "$STATE_KP" \
  --port 18766 \
  --audit "$ROUTE/results/smoke_fbfm/solver.jsonl" \
  --ready-file "$ROUTE/results/smoke_fbfm/ready.json"
```

After `ready.json` appears, launch one task in shell B. This is the bounded GPU
and simulator launch check; stop it after initialization if a complete episode
is not required.

```bash
ROOT="$FBFM_ROOT"
ROUTE="$ROOT/wam/dreamzero-libero"
BASE="$FBFM_EXTERNAL_ROOT"
SIM_PY="$FBFM_ENV_ROOT/fbfm-libero/bin/python"
STATE_WEIGHT=0.005833333333333334
STATE_KP=0.04869675251658631

PYTHONPATH="$ROUTE/src:$ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$SIM_PY" "$ROUTE/scripts/libero_experiment.py" \
  --base-workspace "$BASE" \
  --mode FBFM \
  --state-weight "$STATE_WEIGHT" \
  --state-feedback-kp "$STATE_KP" \
  --suite libero_spatial --task-id 0 \
  --trial-start 0 --trials 1 --max-steps 480 \
  --model-seed-rule fixed --solver-release-policy uniform \
  --port 18766 --output "$ROUTE/results/smoke_fbfm"
```

For the complete paper suite list, keep the server running and use explicit
repeated `--suite` arguments so `libero_90` cannot be selected by a default:

```bash
CODE_COMMIT=$(git -C "$FBFM_ROOT" rev-parse --short HEAD)
OUT="$FBFM_ROOT/wam/dreamzero-libero/results/libero_four_fbfm_20_$CODE_COMMIT"

PYTHONPATH="$FBFM_ROOT/wam/dreamzero-libero/src:$FBFM_ROOT" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$FBFM_ENV_ROOT/fbfm-libero/bin/python" \
  "$FBFM_ROOT/wam/dreamzero-libero/scripts/run_libero_benchmark.py" \
  --base-workspace "$FBFM_EXTERNAL_ROOT" \
  --mode FBFM \
  --state-weight 0.005833333333333334 \
  --state-feedback-kp 0.04869675251658631 \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --trials 20 --max-steps 480 \
  --model-seed-rule fixed --solver-release-policy uniform \
  --code-commit "$CODE_COMMIT" --port 18766 --output "$OUT"
```

The four suites contain 40 tasks, so 20 trials request 800 episodes. The
launcher resumes only a contiguous prefix of official trial IDs and records
the code commit and protocol. For matched ablations, pass the same parameters
and change `--mode` on both server and client to `NONE` or `RTC`. A native
upstream control uses `NONE` with `--rollout-protocol native_sync`; do not label
the matched pseudo-asynchronous `NONE` path as the native DreamZero baseline.

## Wan2.2 visual-state auxiliary route

The unified fetch creates a complete patched checkout at
`$FBFM_EXTERNAL_ROOT/Wan2.2`. The tracked directory `wam/wan2.2/` contains the
overlay, patch, tests, documentation, and sanitized audit summaries, not a
second copy of upstream.

```bash
export WAN_ROOT="$FBFM_EXTERNAL_ROOT/Wan2.2"
export WAN_PY="$FBFM_ENV_ROOT/fbfm-wan2.2/bin/python"
export WAN22_CKPT_DIR=/path/to/Wan2.2-TI2V-5B

cd "$WAN_ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  "$WAN_PY" -m pytest -p no:cacheprovider -q \
  tests/test_fbfm_state_feedback.py
"$WAN_PY" generate_fbfm.py --help
```

The unit tests run on CPU and the help check only validates entrypoint import.
The native streaming-VAE equivalence check requires the checkpoint and a GPU:

```bash
"$WAN_PY" scripts/validate_fbfm_vae.py \
  --ckpt-dir "$WAN22_CKPT_DIR" --device cuda:0 \
  --height 64 --width 64
```

Run an official-size direct prediction:

```bash
mkdir -p "$FBFM_ROOT/wam/wan2.2/results/manual"
"$WAN_PY" generate_fbfm.py \
  --mode DIRECT \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --image "$WAN_ROOT/examples/i2v_input.JPG" \
  --prompt "The scene evolves with rapid object motion." \
  --size '1280*704' --frame-num 121 \
  --sample-steps 50 --sample-shift 5 --guide-scale 5 --seed 0 \
  --output "$FBFM_ROOT/wam/wan2.2/results/manual/direct.mp4"
```

Run the paired FBFM prediction with a recorded reference sequence. The video
must contain the launch frame plus four measured future frames for every
released latent slot; the two release steps below therefore require at least
nine frames.

```bash
"$WAN_PY" generate_fbfm.py \
  --mode FBFM \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --image "$WAN_ROOT/examples/i2v_input.JPG" \
  --feedback-video /path/to/reference_sequence.mp4 \
  --feedback-release-steps 10,20 \
  --state-weight 1.0 --kp 1.0 \
  --prompt "The scene evolves with rapid object motion." \
  --size '1280*704' --frame-num 121 \
  --sample-steps 50 --sample-shift 5 --guide-scale 5 --seed 0 \
  --output "$FBFM_ROOT/wam/wan2.2/results/manual/fbfm.mp4" \
  --audit "$FBFM_ROOT/wam/wan2.2/results/manual/fbfm.json"
```

For a small-area GPU launch test, use a 9-frame horizon, two solver steps, CPU
T5, model offload, and gradient checkpointing. These reduced settings are only
an integration test and are not paper results:

```bash
"$WAN_PY" generate_fbfm.py \
  --mode DIRECT \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --image "$WAN_ROOT/examples/i2v_input.JPG" \
  --prompt "A short integration test." \
  --max-area 65536 --frame-num 9 --sample-steps 2 \
  --t5-cpu --offload-model --convert-model-dtype \
  --gradient-checkpointing \
  --output "$FBFM_ROOT/wam/wan2.2/results/manual/direct_tiny.mp4"
```

### Recorded real-video result and privacy boundary

The audited RealSense D435i experiment is a pre-recorded visual-prediction
diagnostic. Against its reference sequence, the checked sanitized summary is:

| Metric | Wan2.2 direct | FBFM | Direction |
| --- | ---: | ---: | --- |
| full-frame MAE | 9.6261 | 9.2731 | lower is better |
| PSNR (dB) | 20.0618 | 23.1021 | higher is better |
| temporal-gradient MAE | 2.8468 | 5.4817 | lower is better; **FBFM is worse** |

Thus full-frame MAE and PSNR improve while temporal-gradient error degrades.
This experiment is not evidence of uniformly improved video dynamics and is
not a closed-loop robot success-rate claim. The exact audit values are in
[`wam/wan2.2/artifacts/robot_arm_ball_stop/metrics.json`](wam/wan2.2/artifacts/robot_arm_ball_stop/metrics.json).

The raw ROS2 `.db3` bag is not distributed. It contains RealSense device and
ASIC serial numbers, USB physical-port and firmware metadata, calibration, and
a real laboratory recording. Do not publish raw bags, original preprocessing
metadata, camera streams, unreviewed frames, people/backgrounds, or generated
results without data-owner consent, identifier removal, visual privacy review,
and a separate data license. The source archive intentionally rejects `.db3`
files and checkpoint directories.

## Reproducibility rules

- Compare modes with the same checkpoint, task/init state, environment and
  model seeds, solver budget, execution horizon, and pseudo-clock.
- Pseudo-asynchronous time is defined by simulator steps and released solver
  evaluations. Measured wall-clock latency is a resource metric, not the method
  schedule.
- Treat partial task rows as progress only. Report a task success rate only
  after all requested official trials are complete.
- Preserve per-episode records, solver audits, manifest files, and the exact
  code revision. Do not silently change one route's upstream model behavior
  while evaluating another route.

## Build the AAAI source archive

Packaging uses `git archive`, so only committed files on the `submission`
branch are included. The script refuses a dirty worktree and scans the archive
for raw bags, checkpoints, external source, Python caches, and bytecode.

```bash
git switch submission
git status --short                 # must print nothing
bash scripts/bootstrap/verify.sh --strict
bash scripts/smoke.sh all
bash scripts/package_submission.sh /path/to/FBFM-submission.tar.gz
```

The packager writes both `FBFM-submission.tar.gz` and its `.sha256` file. Review
the archive listing and third-party licenses before uploading it to the AAAI
submission system.
