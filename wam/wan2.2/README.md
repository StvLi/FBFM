# Wan2.2 visual-state FBFM route

This directory reproduces the paper's Wan2.2-TI2V-5B auxiliary experiments.
It contains a reviewed overlay and patch, not a fork or a second vendored copy
of upstream Wan2.2. The code revision, model revision, environment profiles,
tests, and curated result checksums are all recorded here.

This route is **visual-state only**. It has no action flow, RTC action mask,
robot policy, or online control loop. The RealSense sequence is a pre-recorded
visual-prediction/state-tracking demonstration and must not be reported as a
robot success-rate result.

## Contents and ownership boundary

| Path | Purpose |
| --- | --- |
| `UPSTREAM.lock` | immutable upstream code and model revisions |
| `patches/wan2.2_fbfm.patch` | patch applied to the pinned upstream tree |
| `overlay/` | reviewable copy of every FBFM addition or modified source file |
| `environment/` | audited three-phase Python/CUDA dependency profiles |
| `scripts/fetch_upstream.sh` | standalone safe/idempotent fetch alternative |
| `artifacts/` | sanitized summaries and exactly three reviewed MP4 files |

Upstream source, Python environments, checkpoints, raw recordings, and normal
run outputs remain outside Git.

## 1. Fetch the pinned upstream

From the FBFM repository root, the preferred repository-wide command is:

```bash
export FBFM_ROOT="$PWD"
export FBFM_EXTERNAL_ROOT="$FBFM_ROOT/external"
export FBFM_ENV_ROOT="$FBFM_ROOT/.venvs"

bash scripts/bootstrap/fetch_upstreams.sh --route wan
export WAN_ROOT="$FBFM_EXTERNAL_ROOT/Wan2.2"
```

That command checks out Wan-Video/Wan2.2 at
`42bf4cfaa384bc21833865abc2f9e6c0e67233dc` and applies the reviewed
patch. It is safe to re-run and refuses unrelated local modifications.

As a standalone alternative, create the route-local vendor checkout:

```bash
bash wam/wan2.2/scripts/fetch_upstream.sh
export WAN_ROOT="$FBFM_ROOT/wam/wan2.2/vendor/Wan2.2"
```

Do not run both paths unless two checkouts are intentionally wanted. Pass an
explicit destination to the standalone helper when needed; `--offline` forbids
network access and requires the pinned object to exist locally:

```bash
bash wam/wan2.2/scripts/fetch_upstream.sh --offline /path/to/existing/Wan2.2
```

For a pristine pinned checkout, `git apply --check` verifies forward
applicability. For a checkout already produced by either helper, the reverse
check verifies that the patch is present:

```bash
PATCH="$FBFM_ROOT/wam/wan2.2/patches/wan2.2_fbfm.patch"
git -C "$WAN_ROOT" apply --reverse --check "$PATCH"
test "$(git -C "$WAN_ROOT" rev-parse HEAD)" = \
  42bf4cfaa384bc21833865abc2f9e6c0e67233dc
```

## 2. Build the Wan environment

The validated profile is Python 3.10 with PyTorch 2.9.0+cu129,
torchvision/torchaudio 0.24.0/2.9.0+cu129, diffusers 0.39.0,
transformers 4.51.3, and optional FlashAttention 2.8.3.post1. The installer
uses three phases because FlashAttention's build imports PyTorch:

1. `requirements-torch-cu129.txt`;
2. `requirements-runtime.txt`;
3. `requirements-flash-attn.txt` with `--no-build-isolation`.

Create the isolated environment with the unified bootstrap:

```bash
bash scripts/bootstrap/create_envs.sh --route wan
export WAN_PY="$FBFM_ENV_ROOT/fbfm-wan2.2/bin/python"
```

The CUDA 12.9 wheels require a sufficiently recent NVIDIA driver.
FlashAttention additionally requires a compatible CUDA toolkit/compiler.
It is an optimization, not a correctness dependency: the patched attention
module has a tested PyTorch SDPA fallback. Skip its build explicitly on an
incompatible host:

```bash
FBFM_WAN_INSTALL_FLASH_ATTN=0 \
  bash scripts/bootstrap/create_envs.sh --route wan
```

To reuse a preinstalled compatible torch stack, point `FBFM_ENV_ROOT` at a
parent containing `fbfm-wan2.2/bin/python` and validate rather than replace
torch:

```bash
FBFM_WAN_TORCH_REQUIREMENTS=existing \
FBFM_WAN_INSTALL_FLASH_ATTN=0 \
  bash scripts/bootstrap/create_envs.sh --route wan
```

A custom torch requirements file can replace `existing`. Build parallelism
for FlashAttention is controlled by `FBFM_WAN_MAX_JOBS` (default 4).
`environment/audited-freeze-cu129.txt` records the complete validated
Python package closure for provenance.

The equivalent manual installation, useful for debugging bootstrap failures,
is:

```bash
conda create -y --prefix "$FBFM_ENV_ROOT/fbfm-wan2.2" python=3.10 pip
WAN_PY="$FBFM_ENV_ROOT/fbfm-wan2.2/bin/python"

"$WAN_PY" -m pip install --upgrade pip setuptools wheel
"$WAN_PY" -m pip install \
  -r "$FBFM_ROOT/wam/wan2.2/environment/requirements-torch-cu129.txt"
"$WAN_PY" -m pip install \
  -r "$FBFM_ROOT/wam/wan2.2/environment/requirements-runtime.txt"
MAX_JOBS=4 "$WAN_PY" -m pip install --no-build-isolation \
  -r "$FBFM_ROOT/wam/wan2.2/environment/requirements-flash-attn.txt"
"$WAN_PY" -m pip install --no-deps -e "$WAN_ROOT"
```

Omit only the FlashAttention command to use SDPA. Do not install the upstream
all-in-one `requirements.txt` in a fresh environment: it asks pip to build
FlashAttention before a usable torch build is guaranteed.

## 3. Download and verify the checkpoint

Weights are external and are never packed into this repository. The audited
model source is `Wan-AI/Wan2.2-TI2V-5B` at Hugging Face revision
`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`. After accepting the
publisher's terms, download that immutable revision:

```bash
export WAN22_CKPT_DIR=/path/to/checkpoints/Wan2.2-TI2V-5B

hf download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir "$WAN22_CKPT_DIR"
```

A complete launch needs all of the following, not just the VAE:

```text
config.json
diffusion_pytorch_model.safetensors.index.json
diffusion_pytorch_model-{00001,00002,00003}-of-00003.safetensors
models_t5_umt5-xxl-enc-bf16.pth
Wan2.2_VAE.pth
google/umt5-xxl/{tokenizer.json,tokenizer_config.json,spiece.model,special_tokens_map.json}
```

The unified verifier checks the code revision, upstream license, patch, and
each checkpoint component without loading a model:

```bash
WAN_CHECKPOINT="$WAN22_CKPT_DIR" \
  bash scripts/bootstrap/verify.sh --route wan --strict --checkpoints
```

## 4. Bounded smoke and launch tests

The default smoke is CPU/import-only. It runs the ten FBFM contract tests and
imports the real CLI with `--help`; it does not download or load weights:

```bash
cd "$FBFM_ROOT"
FBFM_WAN_PYTHON="$WAN_PY" \
FBFM_WAN_SOURCE_ROOT="$WAN_ROOT" \
  bash scripts/smoke.sh wan
```

The optional streaming-VAE equivalence test loads only `Wan2.2_VAE.pth`
and uses a 64x64, nine-frame tensor:

```bash
FBFM_WAN_PYTHON="$WAN_PY" \
FBFM_WAN_SOURCE_ROOT="$WAN_ROOT" \
  bash scripts/smoke.sh wan --model "$WAN22_CKPT_DIR"
```

For a real end-to-end GPU launch without an official-size run, use five
frames, two solver steps, and a small spatial area:

```bash
mkdir -p "$FBFM_ROOT/wam/wan2.2/results/smoke"
cd "$WAN_ROOT"

"$WAN_PY" generate_fbfm.py \
  --mode DIRECT \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --image examples/i2v_input.JPG \
  --prompt "A short integration test." \
  --max-area 16384 --frame-num 5 --sample-steps 2 \
  --t5-cpu --offload-model --convert-model-dtype \
  --output "$FBFM_ROOT/wam/wan2.2/results/smoke/direct_5f.mp4"
```

A bounded FBFM-path launch can reuse the curated reference, take its frame 0
as the matching anchor, and release one four-frame observation slot at solver
boundary 1:

```bash
"$WAN_PY" generate_fbfm.py \
  --mode FBFM \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --feedback-video \
    "$FBFM_ROOT/wam/wan2.2/artifacts/robot_arm_ball_stop/reference_future_121f.mp4" \
  --feedback-release-steps 1 \
  --prompt "A short integration test." \
  --max-area 16384 --frame-num 5 --sample-steps 2 \
  --state-weight 1.0 --kp 1.0 --beta 10 \
  --t5-cpu --offload-model --convert-model-dtype --gradient-checkpointing \
  --output "$FBFM_ROOT/wam/wan2.2/results/smoke/fbfm_5f.mp4"
```

These reduced launches validate initialization, checkpoint loading, the
solver, and output writing. They are not paper-result settings.

## 5. Official-size paired inference

Use the same checkpoint, image, prompt, seed, scheduler budget, and output size
for DIRECT and FBFM. The official TI2V protocol is 121 frames, 50 UniPC steps,
1280x704, sample shift 5, guidance scale 5, and seed 0.

```bash
RESULT_DIR="$FBFM_ROOT/wam/wan2.2/results/manual"
mkdir -p "$RESULT_DIR"
cd "$WAN_ROOT"

"$WAN_PY" generate_fbfm.py \
  --mode DIRECT \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --image examples/i2v_input.JPG \
  --prompt "The scene evolves with rapid object motion." \
  --size '1280*704' --frame-num 121 --sample-steps 50 \
  --sample-shift 5 --guide-scale 5 --seed 0 \
  --output "$RESULT_DIR/direct.mp4" \
  --audit "$RESULT_DIR/direct.json"

"$WAN_PY" generate_fbfm.py \
  --mode FBFM \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --image examples/i2v_input.JPG \
  --feedback-video /path/to/reference_sequence.mp4 \
  --feedback-release-steps 10,20 \
  --state-weight 1.0 --kp 1.0 --beta 10 \
  --prompt "The scene evolves with rapid object motion." \
  --size '1280*704' --frame-num 121 --sample-steps 50 \
  --sample-shift 5 --guide-scale 5 --seed 0 \
  --output "$RESULT_DIR/fbfm.mp4" \
  --audit "$RESULT_DIR/fbfm.json"
```

The feedback sequence includes the launch frame followed by four measured
future frames per released latent slot. Release steps are zero-based solver
boundaries and must be explicit for a paper run. `--max-area` and short
horizons are integration-test settings only. Official-size FBFM VJPs are
memory intensive; the validated workstation used a 96 GB GPU.

## 6. RealSense robot-arm/ball auxiliary experiment

The raw ROS2 SQLite bag is intentionally not distributed. With a separately
authorized recording, preprocess directly into the path expected by the
comparison helper:

```bash
RESULT_DIR="$WAN_ROOT/results/robot_arm_ball_stop"
mkdir -p "$RESULT_DIR"
cd "$WAN_ROOT"

"$WAN_PY" scripts/extract_realsense_color_db3.py \
  /authorized/path/recording.db3 "$RESULT_DIR"

"$WAN_PY" generate_fbfm.py --mode DIRECT \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --image "$RESULT_DIR/anchor_frame_048.png" \
  --prompt "<task prompt>" \
  --size '1280*704' --frame-num 121 --sample-steps 50 \
  --sample-shift 5 --guide-scale 5 --seed 0 \
  --output "$RESULT_DIR/base_future.mp4" \
  --audit "$RESULT_DIR/base_future.json"

"$WAN_PY" generate_fbfm.py --mode FBFM \
  --ckpt-dir "$WAN22_CKPT_DIR" \
  --image "$RESULT_DIR/anchor_frame_048.png" \
  --feedback-video "$RESULT_DIR/reference_future_121f.mp4" \
  --feedback-release-steps 1,3,4,6,8,9,11,12,14,16,17,19,20,22,24,25,27,29,30,32,33,35,37,38,40,41,43,45,46,48 \
  --state-weight 1.0 --kp 1.0 --beta 10 \
  --prompt "<task prompt>" \
  --size '1280*704' --frame-num 121 --sample-steps 50 \
  --sample-shift 5 --guide-scale 5 --seed 0 \
  --output "$RESULT_DIR/fbfm_ours_future.mp4" \
  --audit "$RESULT_DIR/fbfm_ours_future.json"

"$WAN_PY" scripts/prepare_robot_arm_ball_stop_comparison.py
```

The checked summary reports improved full-frame MAE/PSNR and worse temporal
gradient MAE. This mixed diagnostic is not evidence of uniformly improved
video dynamics and is not an online-control claim.

## 7. Ball slot-count diagnostic

The second auxiliary helper evaluates 10, 20, and 30 released visual slots at
`kp=0.05`. The source clip is not redistributed. Prepare an authorized,
normalized set below `$WAN_ROOT/results/ball_meet_ball` containing
`anchor_frame_048.png`, `reference_future_121f.mp4`,
`direct_future.mp4`, and the matching direct audit, then run:

```bash
cd "$WAN_ROOT"
WAN22_PYTHON="$WAN_PY" WAN22_CKPT_DIR="$WAN22_CKPT_DIR" \
  bash scripts/run_ball_slot_ablation_kp_0p05.sh
"$WAN_PY" scripts/prepare_ball_kp_slot_ablation.py
```

The sanitized checked-in JSON records configuration/provenance only; it is not
a substitute for the non-distributed source data.

## Release, privacy, and artifact checks

Do not add ROS bags (`*.db3`/`*.bag`), camera streams, unreviewed
frame extracts, checkpoints, Hugging Face/ModelScope caches, normal
`results/` outputs, audit dumps, or Python caches to Git. The audited
D435i bag contains device identifiers, USB topology, calibration, and a real
laboratory recording. Public data release requires owner consent, identifier
removal, visual privacy review, and a separate data license.

Exactly three curated MP4s are allowlisted for the source release. Verify them
before packaging:

```bash
cd "$FBFM_ROOT/wam/wan2.2/artifacts"
sha256sum -c SHA256SUMS
```

The upstream Apache-2.0 license, model terms, dependency licenses, and
`NOTICE.md` remain in force. See
`overlay/docs/FBFM_STATE_FEEDBACK.md` for the temporal-slot contract,
VJP/endpoint details, and DIRECT-versus-FBFM semantics.
