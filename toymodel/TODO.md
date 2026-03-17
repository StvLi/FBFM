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

## Module 6 — token_dim=3 统一架构重构（当前进行中）

> **背景**: 用户确认 FBFM 的 X^τ = [state, action]，所有算法共享 token_dim=3 的模型。
> 已完成全部设计决策确认（D/E/F/G/H），计划已创建，**代码修改尚未开始**。

### 6.0 已确认的设计决策

| 代号 | 决策 | 细节 |
|------|------|------|
| D | 归一化 | per-dim 向量: `mean: (3,), std: (3,)`, 分维度独立统计 |
| E | dim_mask | FM: 无引导; RTC: `[0,0,1]` 只引导 action; FBFM: `[1,1,1]` 全维度引导 |
| F | Y 固定 | Y 在 FBFM 推理过程中不变，只有 obs 随 feedback 更新 |
| G | Token 构造 | `Token[h] = [state_{h+1}, action_h]` = state 是执行 action 后的结果 + 该 action |
| | | `tokens = cat(states[:, 1:H+1, :], actions, dim=-1)` → `(N, H, 3)` |
| | | obs = states[:, 0, :] → `(N, 2)` 不变 |
| H | n_steps | FM 和 RTC 的去噪步数也从 16 → 20 |

### 6.1 文件修改清单（按依赖顺序执行）

| # | Task | File | Status | Notes |
|---|------|------|--------|-------|
| 6.1 | `action_dim` → `token_dim=3`, 更新 proj 层和 docstring | `model/dit.py` | ✅ | |
| 6.2 | Token 构造 `cat(states[:,1:H+1,:], actions)`, per-dim 归一化 | `model/dataset.py` | ✅ | |
| 6.3 | CFG `token_dim=3`, checkpoint 存 `token_stats` | `train/train_fm.py` | ✅ | |
| 6.4 | `sample_fm` 适配 token_dim + 向量 denorm + n_steps=20 | `model/inference.py` | ✅ | |
| 6.5 | chunk `(H,3)`, 提取 `action=col2`, n_steps=20, token_stats | `algo/fm.py` | ✅ | |
| 6.6 | token_dim=3, `dim_mask=[0,0,1]`, n_steps=20, token_stats | `algo/rtc.py` | ✅ | |
| 6.7 | **实现 `guided_inference_fbfm`** (5 blocks × 4, interleaved) + 重写 `rollout_fbfm` | `algo/fbfm.py` | ✅ | |
| 6.8 | `load_policy` + `run_three_algos` 适配 token_dim/token_stats/n_steps=20 | `experiments/runner.py` | ✅ | |
| 6.9 | `action_stats` → `token_stats` 变量名 | `exp_a/b/c/d.py` | ✅ | |
| 6.10 | 更新 CLAUDE.md 规范文档 | `CLAUDE.md` | ✅ | |

### 6.2 FBFM 推理节奏详述（n_steps=20, n_inner=4, d=4, s_chunk=5）

```
rollout_fbfm 每个大周期:
  ① Build Y from chunk[chunk_ptr:] → (H, 3), right-padded
  ② Trigger: execute Y[0].action(=col2) → capture obs

  ③ guided_inference_fbfm(obs, Y, W, env_feedback_fn):
     Block 0 (τ steps 0-3):  guidance with Y + obs
       → env_feedback_fn() → execute Y[1].action → update obs
     Block 1 (τ steps 4-7):  guidance with Y + updated obs
       → env_feedback_fn() → execute Y[2].action → update obs
     Block 2 (τ steps 8-11): guidance with Y + updated obs
       → env_feedback_fn() → execute Y[3].action → update obs
     Block 3 (τ steps 12-15): guidance with Y + updated obs
       → env_feedback_fn() → execute Y[4].action → update obs
     Block 4 (τ steps 16-19): final output, NO feedback after

  ④ New chunk → chunk_ptr = d = 4 (skip first 4 tokens)

  Total: 1(trigger) + 4(feedback) = 5 = s_chunk
  Total denoising: 5 blocks × 4 = 20 steps
```

### 6.3 Token 格式约定

```
Token[h] = [x_{h+1}, x_dot_{h+1}, u_h]
            ^^^^^^^^^^^^^^^^^^^^^^^^  ^^
            state AFTER executing     action at
            action_h                  step h

Index 0: position x
Index 1: velocity x_dot
Index 2: action u (force)
ACTION_IDX = 2
STATE_SLICE = slice(0, 2)
```

---

## Cross-Cutting Tasks

| Task | Status | Notes |
|------|--------|-------|
| Write `requirements.txt` | ✅ | See below |
| Final end-to-end smoke test (collect → train → eval → plot) | ⬜ | Run after GPU training completes |
| device 默认值改为优先 CUDA | ✅ | 所有 `toymodel/` 下的 device 默认值已改 |
| 工作范围约束写入 CLAUDE.md | ✅ | 只操作 `toymodel/` 子文件夹 |

---

## Progress Summary

```
Module 1  MSD Environment       ████████████  100%  ✅
Module 2  PID Expert Dataset    ████████████  100%  ✅  (6000 chunks, data/expert_dataset.npz)
Module 3  DiT FM Model          ████████░░░░   80%  🔄  (code done, training pending on GPU)
Module 4  Algorithm Comparison  ██████████░░   90%  🔄  (harness + experiments done; user implements RTC/FBFM core)
Module 5  Visualization         ████████████  100%  ✅  (pending results data to render)
Module 6  token_dim=3 重构     ████████████  100%  ✅  (13 文件全部修改完成)
```

_Last updated: 2026-03-17 — token_dim=3 重构计划已确认, 待执行_
