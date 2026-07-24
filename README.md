# FBFM

Feedback Flow Matching (FBFM) is a training-free method for constraining a
world-action model with state observations collected while its current action
chunk is being executed. This repository keeps the method-level components and
two benchmark integrations in one place without mixing their model runtimes.

## Implementations

| Route | Model and benchmark | WAM structure | Status | Entry point |
| --- | --- | --- | --- | --- |
| LingBot-VA | LingBot-VA + RoboTwin | stage-wise video/action flow | runnable; full benchmark tooling included | [`wam/lingbot-va`](wam/lingbot-va/README.md) |
| DreamZero | DreamZero + LIBERO | joint video/action flow | runnable; 130-task benchmark active | [`wam/dreamzero-libero`](wam/dreamzero-libero/README.md) |

The two routes implement the same FBFM idea at different solver boundaries.
They do not share model checkpoints, simulator environments, caches, or server
processes.

## Repository layout

```text
FBFM/
  fbfm/                         # method-level and earlier policy components
  toymodel/                     # small mathematical experiments
  wam/
    README.md                   # route index and ownership boundaries
    lingbot-va/                 # LingBot-VA x FBFM x RoboTwin
    dreamzero-libero/           # DreamZero x FBFM x LIBERO
```

Generated trajectories, videos, checkpoints, third-party repositories, and
Conda environments are intentionally external to Git.

## LingBot-VA x RoboTwin

This route preserves LingBot-VA's separated video/action flow and inserts FBFM
at its pseudo-asynchronous solver boundary. It provides matched `NONE`, `RTC`,
and `FBFM` launchers, RoboTwin raster-backend compatibility, deterministic
replay, and resumable all-task evaluation.

```bash
cd wam/lingbot-va

# Zero constraints on the same rollout path
bash script/run_robotwin_none.sh 1

# Previous-action constraint only
bash script/run_robotwin_rtc.sh 1

# Previous-action plus live state feedback
bash script/run_robotwin_fbfm.sh 1
```

Machine paths and environment variables are documented in
[`wam/lingbot-va/README.md`](wam/lingbot-va/README.md). The FBFM-specific
runtime contract is in
[`wam/lingbot-va/docs/fbfm_runtime_modes.md`](wam/lingbot-va/docs/fbfm_runtime_modes.md).

## DreamZero x LIBERO

This route applies one joint state/action endpoint VJP inside DreamZero's native
joint DiT evaluation. It preserves the released checkpoint's 16-step UniPC
schedule, 8-evaluation cache mask, CFG, VAE, action normalization, and LIBERO
action contract. A localhost protocol isolates the DreamZero Python 3.11 model
environment from the LIBERO Python 3.8 simulator environment.

```bash
cd wam/dreamzero-libero
PYTHONPATH=src python -m pytest
```

Deployment, single-task smoke, full 130-task evaluation, resume behavior, and
result-ledger commands are documented in
[`wam/dreamzero-libero/README.md`](wam/dreamzero-libero/README.md). The exact
mathematical mapping is in
[`wam/dreamzero-libero/docs/IMPLEMENTATION.md`](wam/dreamzero-libero/docs/IMPLEMENTATION.md).

## Constraint modes

| Mode | Previous-action constraint | Live state constraint | Purpose |
| --- | ---: | ---: | --- |
| `NONE` | no | no | zero-guidance control on the matched rollout path |
| `RTC` | yes | no | action-overlap baseline |
| `FBFM` | yes | yes | complete feedback flow matching method |

Pseudo-asynchronous time is defined by simulator steps and released solver
evaluations. Measured wall-clock latency is recorded as a resource metric but
does not change a method schedule.

## Environment policy

Keep each model and simulator in its own environment. Do not install the two
routes into a shared Python environment:

| Route | Model environment | Simulator environment |
| --- | --- | --- |
| LingBot-VA + RoboTwin | Python 3.10 / LingBot-VA stack | independent RoboTwin/SAPIEN stack |
| DreamZero + LIBERO | Python 3.11 / DreamZero stack | Python 3.8 / LIBERO stack |

Checkpoint paths and upstream source roots are deployment configuration, not
repository content. No launcher should download or mutate a checkpoint during
evaluation.

## Reproducibility rules

- Compare modes with the same checkpoint, task/init state, model seed, solver
  budget, execution horizon, and pseudo-clock.
- Treat partial task rows as progress only; a task success rate is final after
  all requested trials complete.
- Keep solver audits and per-episode records with the code revision used for the
  run.
- Changes under one route must not silently change the other route's upstream
  model behavior.
