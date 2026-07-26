# DreamZero x FBFM x LIBERO handover

Updated: 2026-07-26

## Repositories and revisions

| Item | Location / revision |
| --- | --- |
| Canonical monorepo route | `/home/oem/tmp_ws/FBFM/wam/dreamzero-libero` |
| Monorepo branch | `fix/dreamzero-binary-state-mask` |
| Imported standalone repository | `/home/oem/tmp_ws/DreamZero-FBFM-LIBERO` |
| A6000 integration repository | `/home/deepcybo-lite/peize/DreamZero-FBFM-LIBERO` |
| Active validation branch | `fix/dreamzero-relinearized-unipc-guidance` |
| Active validation revision | `13de791` |
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
joint endpoint and applies a block-diagonal state/action discrepancy at every
UniPC scheduler index. The full endpoint Jacobian is refreshed at the eight
native DiT evaluations and reused at skipped DiT indices while the current
sample, sigma, endpoint residual, and VJP are recomputed. Native velocities,
not guided velocities, remain in `prev_predictions`. Cross-modal Jacobian terms
are therefore retained without replacing DreamZero's cache schedule. The
integration preserves these upstream contracts:

- 16 native UniPC scheduler steps and the checkpoint's 8-evaluation DiT cache mask;
- the released checkpoint, CFG, VAE, causal cache, action normalization, and 7D action output;
- one deterministic pseudo-clock grant per executed environment action;
- zero masks and the original numerical path in `NONE` mode.

The action block constrains the committed `8 x 7 = 56` physical coordinates.
The visual block constrains one `48 x 10 x 20 = 9600` latent slot. Real feedback
is sampled at the checkpoint's three-action video stride. Missing history is
left-padded with the measured current-wave anchor; no unobserved future frame is
copied into a hard target. The first two windows available in an eight-action
wave are `[0,0,0,0,3]` and `[0,0,0,3,6]`. The action mask remains binary. The
default DreamZero state coefficient is now the Euclidean coordinate-balance
value `sqrt(56/9600)=0.07637626158259733`; `1.0` and `56/9600` are explicit
binary-state and L1-mass ablations. See `docs/IMPLEMENTATION.md` for the solver
equations and hook boundary.

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

The current RMS-balanced, relinearized object-6 validation is complete at
`4/20 = 20.0%` (95% Wilson interval `8.1%-41.6%`), versus the paired native
base at `5/20 = 25.0%` (95% Wilson interval `11.2%-46.9%`). Its A6000 result
root is `results/libero_object6_rms00764_fbfm_10_13de791`; the historical
directory name still contains `10`, but its ledger and summary now contain
trials 0-19. The complete record and negative binary/recurrent diagnostics are
in the paper experiment repository's
`experiments/dreamzero_object6_binary_mask_diagnosis.md`.

The 20-trial extension exposed a rare but severe solver outlier in FBFM trial
18, wave 16. Eight generated actions had norms above 100, with a maximum of
168.37 and maximum absolute coordinate 152.81; native base's maximum action
norm across all 20 trials was 1.57. Scheduler indices 7-9 reused a Jacobian
whose correction grew from 138.15 at index 6 to 11260.61 at index 9. RMS
modality balancing therefore fixes the typical scale mismatch but does not
guarantee that a cached Jacobian remains inside its local linearization domain.
Do not report the current branch as numerically stable until a trust-region or
guarded native-update fallback is implemented and retested.

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

- Canonical CPU regression suite: 24 applicable tests passed.
- A6000 standalone route suite: 24 applicable tests passed. Two monorepo-path
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
