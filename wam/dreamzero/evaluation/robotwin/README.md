# DreamZero + FBFM on RoboTwin

This directory contains the native RoboTwin bridge and the fairness tooling for
the three constraint modes.  It intentionally refuses to serve RoboTwin with
AgiBot or DROID normalization metadata.

## Constraint modes

- `None`: original DreamZero path. No constraint target, VJP, feedback VAE
  warm-up, or forced DiT recomputation is introduced.
- `RTC`: constrains only the unexecuted tail of the previous normalized action
  horizon. It does not construct a video target.
- `Feedback`: uses the same action tail plus the latent produced by DreamZero's
  own VAE from buffered real RoboTwin observations.

The correction is computed before each guided denoiser call. The unmodified
`FlowUniPCMultistepScheduler.step` receives the corrected video/action flow, so
its multistep history is not reimplemented. Guided steps bypass velocity reuse
because a cached flow has no Jacobian at the current sample; causal KV context
caching remains enabled. Two-rank CFG computes differentiable conditional and
unconditional contributions on their owning ranks and sums their input VJPs.

## Native checkpoint gate

A usable checkpoint must contain all of the following:

1. DreamZero model files and `experiment_cfg/conf.yaml`;
2. a post-training config whose embodiment is `robotwin`;
3. native RoboTwin `relative_stats_dreamzero.json`;
4. `robotwin_schema.json`, based on `robotwin_schema.example.json`, with the
   exact normalization checksum;
5. `checkpoint_manifest.json`, generated with:

```bash
python -m evaluation.robotwin.checkpoint_manifest "$MODEL_PATH" \
  --output "$MODEL_PATH/checkpoint_manifest.json"
```

The current remote host did not contain such a checkpoint or a native
RoboTwin LeRobot dataset at the time this integration was created. The three
provided conda environments also lacked a complete DreamZero dependency set,
and both direct and proxy Hugging Face access failed. Do not substitute the
FastWAM or LingBot checkpoints.

## Dataset and post-training

The native data contract is defined by
`groot/vla/configs/data/dreamzero/robotwin_relative.yaml`. Validate converted
data and launch LoRA post-training from the DreamZero directory:

```bash
python -m evaluation.robotwin.validate_dataset "$ROBOTWIN_DATA_ROOT"
ROBOTWIN_DATA_ROOT="$ROBOTWIN_DATA_ROOT" \
PRETRAINED_MODEL_PATH="$CHECKPOINT_ROOT/DreamZero-AgiBot" \
WAN_CKPT_DIR="$CHECKPOINT_ROOT/Wan2.1-I2V-14B-480P" \
TOKENIZER_DIR="$CHECKPOINT_ROOT/umt5-xxl" \
bash scripts/train/robotwin_training.sh
```

`download_assets.sh` provides resumable `hf download` commands once server
network access is restored.

## Server and episode protocol

```bash
FBFM_CONSTRAINT_MODE=Feedback \
MODEL_PATH=/path/to/dreamzero_robotwin_checkpoint \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/robotwin/launch_server.sh
```

The websocket bridge accepts the LingBot-style `reset`, `feedback`, and
`compute_kv_cache` messages. Feedback only buffers RGB observations. On the
next joint causal forward, DreamZero performs its native VAE encoding and true
observation KV replacement. Thus RGB-frame to latent-slot alignment is derived
from the actual VAE output rather than a hard-coded 4-to-1 rule.

## Frozen evaluation

First run a simulator initialization pass that writes candidate JSONL records
including accepted status, seed, chosen instruction, randomization parameters,
and background texture path/checksum. Freeze exactly 20 accepted episodes per
task/config:

```bash
python -m evaluation.robotwin.experiment freeze \
  --candidates accepted_candidates.jsonl \
  --output canonical_episodes.jsonl
```

All three modes must use that same file and checkpoint manifest. The resumable
launcher exports one frozen episode at a time to a single-episode evaluator:

```bash
python -m evaluation.robotwin.run_manifest \
  --manifest canonical_episodes.jsonl \
  --checkpoint-manifest checkpoint_manifest.json \
  --mode None \
  --output-dir results -- <single-episode evaluator command>
```

Aggregate completed JSONL results into the requested paper table and bootstrap
95% confidence intervals:

```bash
python -m evaluation.robotwin.experiment aggregate \
  --manifest canonical_episodes.jsonl \
  --results all_results.jsonl \
  --output dreamzero_fbfm_robotwin.md
```
