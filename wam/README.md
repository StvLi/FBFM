# World-Action Model integrations

The `wam` directory contains model-specific FBFM integrations. Each route owns
its model hook, feedback encoder, pseudo-clock bridge, simulator transport,
tests, launchers, and experiment audit format.

## Route index

| Directory | Upstream model | Benchmark | Flow boundary | Runtime |
| --- | --- | --- | --- | --- |
| [`lingbot-va`](lingbot-va/README.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) | RoboTwin | separated/stage-wise video and action flow | LingBot-VA server + RoboTwin client |
| [`dreamzero-libero`](dreamzero-libero/README.md) | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) | LIBERO | joint video-action DiT flow | DreamZero server + LIBERO client |

## Ownership boundary

The integrations are deliberately siblings rather than one shared runtime:

- `lingbot-va` may depend on LingBot-VA and RoboTwin APIs only.
- `dreamzero-libero` may depend on DreamZero, RLinf checkpoint metadata, and
  LIBERO APIs only.
- Method-level ideas can be shared through documentation or small independent
  utilities, but model hooks and environment adapters remain route-local.
- Checkpoints, upstream repositories, environments, and experiment outputs stay
  outside tracked source.

This separation is necessary because the two models expose different flow
states and require incompatible Python/CUDA dependency stacks.

## Shared experiment contract

Both routes provide three matched modes:

| Mode | State feedback | Executed-action overlap |
| --- | ---: | ---: |
| `NONE` | no | no constraint |
| `RTC` | no | constrained |
| `FBFM` | constrained | constrained |

All modes within a route must reuse its native scheduler, model cache behavior,
initial noise/seed rule, action execution contract, and discrete pseudo-clock.
Route-level READMEs contain the authoritative commands and deployment details.
