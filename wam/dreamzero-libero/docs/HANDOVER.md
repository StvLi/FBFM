# DreamZero x FBFM x LIBERO handover

Updated: 2026-07-24

## Repositories and revisions

| Item | Location / revision |
| --- | --- |
| Canonical monorepo route | `/home/oem/tmp_ws/FBFM/wam/dreamzero-libero` |
| Monorepo branch | `runnable-fbfm-lingbotva-dreamzero` |
| Imported standalone repository | `/home/oem/tmp_ws/DreamZero-FBFM-LIBERO` |
| A6000 integration repository | `/home/deepcybo-lite/peize/DreamZero-FBFM-LIBERO` |
| Imported standalone branch | `runnable-dreamzero-fbfm-libero` |
| Numerical method revision | `605f76d` |
| Full-benchmark ledger revision | `67c7e80` |
| Ledger synchronization revision | `118c89d` |
| Paper experiment branch | `/home/oem/tmp_ws/aaai_paper`, branch `experiment` |

The ledger and synchronization revisions add sequential-client lifecycle,
resume validation, and result tables. They do not alter the numerical method in
`605f76d`.

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
GB allocated, and an FBFM episode peaks at approximately 27.89 GB on the RTX
A6000. The first model load takes about 2.5 minutes because the six checkpoint
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
The visual block uses the current-wave anchor plus observations at offsets
2/4/6/8 in the frozen causal VAE and constrains one `48 x 10 x 20 = 9600`
latent slot. Its fixed default weight is `56/9600`. See
`docs/IMPLEMENTATION.md` for the solver equations and hook boundary.

## Full experiment protocol

The active experiment covers all standard LIBERO suites in this order:

1. `libero_spatial` (10 tasks)
2. `libero_object` (10 tasks)
3. `libero_goal` (10 tasks)
4. `libero_10` (10 tasks)
5. `libero_90` (90 tasks)

Each task uses official trial IDs 0-19, for 130 tasks and 2,600 episodes. One
model server remains resident while task clients run sequentially. The runner
accepts only a contiguous completed-trial prefix and rejects duplicates before
resuming.

| Runtime artifact | Location |
| --- | --- |
| Experiment root | `/home/deepcybo-lite/peize/DreamZero-FBFM-LIBERO/results/libero_all_fbfm_20_67c7e80` |
| Server PID | `<experiment root>/server.pid` |
| Runner PID | `<experiment root>/runner.pid` |
| Episode outputs | `<experiment root>/tasks/<suite>/task_NNN` |
| Joint solver audit | `<experiment root>/solver.jsonl` |
| Live task table | `<experiment root>/task_summary.csv` |
| Live trial table | `<experiment root>/trials.csv` |
| Human-readable status | `<experiment root>/live_status.md` |

At handover, `libero_spatial/task0` is complete at 14/20 successes. It was run
with method revision `605f76d` and adopted into the `67c7e80` ledger. The
remaining queue starts at `libero_spatial/task1`.

## Table management

The remote runner atomically refreshes its CSV and Markdown tables after every
task. A local detached monitor synchronizes them into the paper repository every
900 seconds:

- `experiments/dreamzero_fbfm_trials.csv`
- `experiments/dreamzero_fbfm_task_summary.csv`
- `experiments/dreamzero_fbfm_live_status.md`
- `experiments/dreamzero_fbfm_manifest.json`

The local monitor command is `scripts/sync_libero_ledger.py`; its deployment log
is `results/monitor/ledger_sync.log`. Only task rows marked `complete` are final
20-trial estimates. Partial rows are operational progress only.

## Validation and caveats

- CPU regression suite: 14 tests passed on the A6000 deployment.
- The first weighted FBFM smoke completed all 220 steps without OOM, NaN, or memory growth.
- The first complete task produced 14/20 successes, confirming nonzero closed-loop performance.
- The paused base run used eight parallel environments on the RTX PRO 6000; the
  A6000 FBFM run is sequential. Both use official initial states, but batching
  can change floating-point trajectories, so this deployment difference must be
  retained in comparisons.
- Measured wall-clock time is a resource metric only. It does not define the
  pseudo-asynchronous method schedule.
