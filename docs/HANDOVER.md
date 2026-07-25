# DreamZero x FBFM x LIBERO handover

Updated: 2026-07-26

## Repositories and revisions

| Item | Location / revision |
| --- | --- |
| Canonical monorepo route | `/home/oem/tmp_ws/FBFM/wam/dreamzero-libero` |
| Monorepo branch | `fix/dreamzero-binary-state-mask` |
| Imported standalone repository | `/home/oem/tmp_ws/DreamZero-FBFM-LIBERO` |
| A6000 integration repository | `/home/deepcybo-lite/peize/DreamZero-FBFM-LIBERO` |
| Imported standalone branch | `runnable-dreamzero-fbfm-libero` |
| Rolling method revision | `118211b` |
| Canonical audit/metadata revision | `1ee00fa` |
| A6000 rolling method revision | `0d258e4` |
| A6000 audit/metadata revision | `0f04fd3` |
| Paper experiment branch | `/home/oem/tmp_ws/aaai_paper`, branch `experiment` |

The second revision in each repository fixes experiment metadata and audit
grouping; it does not alter the numerical method in `118211b`/`0d258e4`.

## A6000 deployment

| Item | Path |
| --- | --- |
| Base workspace | `/home/deepcybo-lite/fbfm_ws` |
| DreamZero environment | `envs/miniconda3/envs/dreamzero` |
| LIBERO environment | `envs/miniconda3/envs/libero` |
| DreamZero source | `dreamzero` |
| LIBERO source | `LIBERO` |
| Checkpoint | `checkpoints/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000` |
| Tokenizer | `assets/tokenizers/umt5-xxl` |

The checkpoint loads all 1,828 tensors. Model residency is approximately 25.27
GB allocated, and the rolling task-0 pilot peaked at 26.78 GiB allocated on the
RTX A6000. The first model load takes about 2.5 minutes because the six checkpoint
shards are read through host memory before the GPU transfer.

## Method mapping

DreamZero transports video and action in one DiT call. FBFM reconstructs the
joint endpoint at each native DiT evaluation, applies a block-diagonal observed
state/action discrepancy, and computes one joint VJP. Cross-modal Jacobian terms
are therefore retained. The integration preserves these upstream contracts:

- 16 native UniPC scheduler steps and the checkpoint's 8-evaluation DiT cache mask;
- the released checkpoint, CFG, VAE, causal cache, action normalization, and 7D action output;
- one deterministic pseudo-clock grant per executed environment action;
- zero masks and the original numerical path in `NONE` mode.

The action block constrains the committed `8 x 7 = 56` physical coordinates.
The visual block constrains one `48 x 10 x 20 = 9600` latent slot. After every
action, the newest real observation closes a five-frame causal rolling VAE window.
Missing history is left-padded with the measured current-wave anchor; no
unobserved future frame is copied into a hard target. The source window progresses
from `[0,0,0,0,1]` to `[0,2,4,6,8]`; the latter is the complete block. The slot is re-encoded, refreshed and
versioned before each of the eight DiT evaluations. Its default mask weight is
now binary (`1.0`), matching the paper and LingBot-VA. Fractional weights are
explicit ablations. See `docs/IMPLEMENTATION.md` for the solver equations and
hook boundary.

## Experiment state

The intended full experiment covers all standard LIBERO suites in this order:

1. `libero_spatial` (10 tasks)
2. `libero_object` (10 tasks)
3. `libero_goal` (10 tasks)
4. `libero_10` (10 tasks)
5. `libero_90` (90 tasks)

Each task uses official trial IDs 0-19, for 130 tasks and 2,600 episodes. The
full rolling run has not been started. The previous `after_feedback` run was
stopped after 73 episodes because all eight DiT evaluations were delayed until
after action 8; its data are diagnostic only.

| Runtime artifact | Location |
| --- | --- |
| Rolling pilot root | `/home/deepcybo-lite/peize/DreamZero-FBFM-LIBERO/results/rolling_v1_task0_20_0d258e4` |
| Episode outputs | `<pilot root>/tasks/libero_spatial/task_000` |
| Joint solver audit | `<pilot root>/solver.jsonl` |
| Task table | `<pilot root>/task_summary.csv` |
| Trial table | `<pilot root>/trials.csv` |
| Human-readable status | `<pilot root>/live_status.md` |

The rolling pilot is complete at 9/20 successes (45.0%, 95% Wilson interval
25.8%-65.8%). The matched delayed-feedback diagnostic achieved 8/20, while the
historical base task-0 run achieved 19/20. Rolling feedback is implemented and
working, but does not by itself explain or recover the performance gap. These
historical runs used the legacy `56/9600` state weight and must not be combined
with binary-mask results under one protocol label.

## Table management

The rolling pilot tables are stored in the paper repository as:

- `experiments/dreamzero_fbfm_rolling_v1_trials.csv`
- `experiments/dreamzero_fbfm_rolling_v1_task_summary.csv`
- `experiments/dreamzero_fbfm_rolling_v1_live_status.md`
- `experiments/dreamzero_fbfm_rolling_v1_manifest.json`
- `experiments/dreamzero_fbfm_rolling_v1.md`

The old monitor is stopped. Start a new monitor only when a full rolling run is
launched, and use a new result root and filename prefix.

## Validation and caveats

- Canonical CPU regression suite: 23 tests passed.
- A6000 standalone route suite: 17 applicable tests passed. Two monorepo-path
  tests are intentionally inapplicable to the standalone deployment layout.
- The smoke succeeded in 168/480 steps; the formal pilot completed all 20 trials.
- The solver audit contains 6,969 finite evaluations, 0 server errors, context
  versions 1-8 per complete wave, and one slot-0 target refresh per evaluation.
- Mean allocated memory over the first/last 200 solver evaluations was
  24.065/23.970 GiB, so the rolling VAE path does not retain graphs or grow memory.
- The paused base run used eight parallel environments on the RTX PRO 6000; the
  A6000 FBFM run is sequential. Both use official initial states, but batching
  can change floating-point trajectories, so this deployment difference must be
  retained in comparisons.
- Measured wall-clock time is a resource metric only. It does not define the
  pseudo-asynchronous method schedule.
- Before a full benchmark, run matched 20-trial `NONE` and `RTC` pilots on the
  same A6000 path. This separates overlap/action-guidance degradation from the
  effect of rolling state feedback.
