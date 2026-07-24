# DreamZero x FBFM x LIBERO

Training-free Feedback Flow Matching for the joint state-action flow in the
RLinf DreamZero WAN2.2 5B LIBERO policy.

This repository is an integration layer. It does not fork or retrain DreamZero,
change its checkpoint, replace UniPC, alter LIBERO actions, or redefine model
time using measured latency. It adds:

- one block-diagonal state/action constraint at each joint solver evaluation;
- endpoint VJP guidance with the paper's clipped few-step schedule;
- native DreamZero VAE encoding for live visual feedback;
- a deterministic pseudo-clock linking 8 executed actions to 16 solver steps;
- a localhost protocol between the Python 3.11 model and Python 3.8 simulator;
- auditable JSONL records for masks, errors, corrections, versions and memory.

## Fixed validation protocol

| Variable | Value |
|---|---:|
| action horizon `H` | 16 |
| inference delay `d` | 8 |
| executed suffix `s` | 8 |
| solver evaluations | 16 |
| grants per simulator step | 2 |
| feedback sample stride | 2 observations |
| observations per latent | 4 |
| active state slots per wave | first of 2 predicted latent slots |
| guidance clip `beta` | 10 |

`NONE`, `RTC`, and `FBFM` use the same model, 16 full DiT evaluations, rollout,
noise seeds and pseudo-clock. Only masks differ: `NONE` uses zero masks, `RTC`
uses the normalized 7-channel action prefix, and `FBFM` adds observed latent
state slots.

## A6000 commands

The prepared base workspace is `/home/deepcybo-lite/fbfm_ws`. Start the model
server in its DreamZero environment:

```bash
BASE=/home/deepcybo-lite/fbfm_ws
REPO=/home/deepcybo-lite/peize/DreamZero-FBFM-LIBERO
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
BASE=/home/deepcybo-lite/fbfm_ws
REPO=/home/deepcybo-lite/peize/DreamZero-FBFM-LIBERO
PYTHONPATH=$REPO/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  $BASE/envs/miniconda3/envs/libero/bin/python $REPO/scripts/libero_experiment.py \
  --base-workspace $BASE --mode FBFM --suite libero_spatial --task-id 0 \
  --trial-start 0 --trials 1 --max-steps 220 --port 18766 \
  --output $REPO/results/smoke_fbfm
```

Run CPU regressions with:

```bash
PYTHONPATH=src python -m pytest
```

Primary generated files are ignored under `results/`: `episodes.jsonl`,
`summary.json`, `solver.jsonl`, and the model-load ready report.
