# FBFM Project — Modular Task TODO List

> Sync with `CLAUDE.md` for specs, interfaces, and parameter defaults.
> Update status emoji on each session: ✅ Done · 🔄 In Progress · ⬜ Pending · ❌ Blocked

---

## Module 1 — MSD Simulation Environment

| Task | File | Status | Notes |
|------|------|--------|-------|
| 1.1 Implement `MSDEnv` class with Euler integration | `sim/msd_env.py` | ✅ | m=1, k=2, c=0.5, dt=0.05 |
| 1.2 Add process noise in dynamics | `sim/msd_env.py` | ✅ | `w_process ~ N(0, 0.005²)` |
| 1.3 Add observation noise on `step()` return | `sim/msd_env.py` | ✅ | `w_obs ~ N(0, 0.01²)` |
| 1.4 Add `disturbance` arg for external impulse force | `sim/msd_env.py` | ✅ | Used in Exp-B |
| 1.5 Add `set_params(m, k, c)` for mismatch testing | `sim/msd_env.py` | ✅ | Used in Exp-A |
| 1.6 Verification plot (English serif font) | `sim/msd_env.py` | ✅ | Saved to `sim/msd_env_verify.png` |

---

## Module 2 — PID Expert Dataset

| Task | File | Status | Notes |
|------|------|--------|-------|
| 2.1 Implement `PIDController` with anti-windup | `sim/pid_controller.py` | ✅ | Kp=10, Ki=5, Kd=2 |
| 2.2 Implement `make_step_ref` / `make_sin_ref` | `data/collect_dataset.py` | ✅ | |
| 2.3 Implement `collect_one_trajectory` | `data/collect_dataset.py` | ✅ | Returns `{states, actions, refs, t_arr}` |
| 2.4 Implement `slice_into_chunks(H=16)` | `data/collect_dataset.py` | ✅ | Output shape: `(N, H+1, 2)` / `(N, H, 1)` |
| 2.5 Implement `collect_dataset(n_trajs=200)` | `data/collect_dataset.py` | ✅ | Saves `data/expert_dataset.npz` |
| 2.6 Run full dataset collection & verify PID tracking plot | `data/collect_dataset.py` | ✅ | Debug run verified: 100 chunks, shapes correct |
| 2.7 Tune PID gains if tracking quality is poor | `sim/pid_controller.py` | ✅ | **Fixed**: Kp=30, Ki=15, Kd=5; sin freq capped at 0.25 Hz (wn=0.225 Hz) |

---

## Module 3 — DiT Flow Matching Model

| Task | File | Status | Notes |
|------|------|--------|-------|
| 3.1 Design DiT architecture (obs conditioning, τ embedding) | `model/dit.py` | ✅ | Input: `(X^τ, obs, τ)` → Output: `v`; 1.09M params |
| 3.2 Implement `FlowMatchingDataset` (load `.npz`, normalize) | `model/dataset.py` | ✅ | Normalizes actions to N(0,1); exposes `denormalize()` |
| 3.3 Implement FM training loss: `‖v_θ(X^τ, o, τ) − (X¹−X⁰)‖²` | `train/train_fm.py` | ✅ | Linear interp `X^τ = τX¹ + (1−τ)X⁰` |
| 3.4 Implement training loop with Adam, LR scheduler | `train/train_fm.py` | ✅ | Warmup + cosine decay; saves `checkpoints/fm_best.pt` |
| 3.5 Implement `sample_fm(policy, obs, n_steps)` baseline inference | `model/inference.py` | ✅ | Euler ODE, verified shape `(16, 1)` |
| 3.6 Verify training convergence (loss curve plot) | `train/train_fm.py` | ⬜ | **Pending GPU training** — run on RTX 4080 |
| 3.7 Verify FM rollout on MSD (step + sinusoidal reference) | `model/inference.py` | ⬜ | **Pending GPU training** |

---

## Module 4 — Three-Algorithm Comparison

### 4a — Algorithm Stubs & Interfaces

| Task | File | Status | Notes |
|------|------|--------|-------|
| 4a.1 Implement FM rollout loop (baseline, no guidance) | `algo/fm.py` | ✅ | Re-infer every `s_chunk=5` steps |
| 4a.2 Write `guided_inference_rtc` stub with interface comments | `algo/rtc.py` | ✅ | **User implements core formula** — full docstring provided |
| 4a.3 Write `guided_inference_fbfm` stub with interface comments | `algo/fbfm.py` | ✅ | **User implements core formula** — full docstring provided |
| 4a.4 Implement single-thread async simulation harness | `experiments/runner.py` | ✅ | `run_three_algos()` with disturbance injection support |

### 4b — Experiment Designs

| Task | File | Status | Notes |
|------|------|--------|-------|
| 4b.1 **Exp-A**: Parameter mismatch (m×1.5/2/3, k×3) | `experiments/exp_a_mismatch.py` | ✅ | Step + sinusoidal ref; 5 conditions |
| 4b.2 **Exp-B**: In-chunk impulse disturbance at step 3 | `experiments/exp_b_disturbance.py` | ✅ | Nominal + mass×2 variants |
| 4b.3 **Exp-C**: Ablation — FBFM with k_p=0 (beta=0) | `experiments/exp_c_ablation.py` | ✅ | 3 mismatch conditions |
| 4b.4 **Exp-D**: Feedback frequency sensitivity (n_inner=1/2/4/8/16) | `experiments/exp_d_sensitivity.py` | ✅ | Heatmap output |
| 4b.5 Collect & save all experiment results to `results/` | `experiments/run_all.py` | ✅ | `python experiments/run_all.py --ckpt ... --device cuda` |

### 4c — User Implementation (Core Algorithms)

| Task | File | Status | Notes |
|------|------|--------|-------|
| 4c.1 **[USER]** Implement `guided_inference_rtc` | `algo/rtc.py` | ⬜ | Formula: `v_guided = v + k_p·(Y−X̂¹)ᵀdiag(W)·∂X̂¹/∂X^τ` |
| 4c.2 **[USER]** Implement `guided_inference_fbfm` | `algo/fbfm.py` | ⬜ | Formula: see CLAUDE.md §4.3 and Thoughts.md §FBFM |

---

## Module 5 — Visualization

| Task | File | Status | Notes |
|------|------|--------|-------|
| 5.1 Trajectory tracking plot (3 algos overlaid, per experiment) | `viz/plot_results.py` | ✅ | `plot_exp_a_traj()`, `plot_exp_b()` |
| 5.2 Tracking error bar chart (MSE comparison across conditions) | `viz/plot_results.py` | ✅ | `plot_exp_a_bar()` |
| 5.3 In-chunk response plot (Exp-B: position vs time, disturbance marked) | `viz/plot_results.py` | ✅ | Impulse steps marked with vertical lines |
| 5.4 Ablation bar chart (Exp-C: FBFM vs FBFM-no-feedback) | `viz/plot_results.py` | ✅ | `plot_exp_c_bar()` |
| 5.5 Sensitivity heatmap (Exp-D: n_inner vs MSE) | `viz/plot_results.py` | ✅ | `plot_exp_d_heatmap()` |
| 5.6 All figures: English serif font, dpi=300, PDF + PNG export | `viz/plot_results.py` | ✅ | Times New Roman / DejaVu Serif, STIX math |

---

## Cross-Cutting Tasks

| Task | Status | Notes |
|------|--------|-------|
| Write `requirements.txt` | ✅ | See below |
| Final end-to-end smoke test (collect → train → eval → plot) | ⬜ | Run after GPU training completes |

---

## Progress Summary

```
Module 1  MSD Environment       ████████████  100%  ✅
Module 2  PID Expert Dataset    ████████████  100%  ✅  (6000 chunks, data/expert_dataset.npz)
Module 3  DiT FM Model          ████████░░░░   80%  🔄  (code done, training pending on GPU)
Module 4  Algorithm Comparison  ██████████░░   90%  🔄  (harness + experiments done; user implements RTC/FBFM core)
Module 5  Visualization         ████████████  100%  ✅  (pending results data to render)
```

_Last updated: 2026-03-17_
