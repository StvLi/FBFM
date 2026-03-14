"""
Final FBFM Pre-Experiment Suite — Publication-Quality Version.

Four rigorous experiments that demonstrate FBFM's advantages with multi-seed
statistical testing and diverse evaluation conditions.

  Exp A: Model Mismatch Sweep  (核心验证)
    Train on m=1.0,c=0.5,k=0.1; test on increasingly mismatched dynamics
    with BOTH step and sinusoidal targets. Multi-seed with mean±std.
    Shows FBFM advantage grows with mismatch severity.

  Exp B: Physical Disturbance Recovery  (鲁棒性验证)
    Nominal + mismatched dynamics with external perturbations:
    (1) position offset (模拟被推了一下), (2) sustained step force (持续偏置力).
    Measures recovery speed and action smoothness under disturbance.

  Exp C: Jacobian Ablation  (消融实验)
    Compare Vanilla FM / RTC / FBFM-Identity / FBFM under mismatch.
    Isolates the contribution of (1) state feedback and (2) model Jacobian.
    The ablation hierarchy: FBFM > FBFM_ID > RTC > Vanilla proves that
    state feedback + Jacobian is the key innovation.

  Exp D: Guidance Weight Sensitivity  (超参数敏感性)
    Sweep state_gw for FBFM and action_gw for RTC under combined mismatch.
    Shows: (1) FBFM's optimal operating range, (2) robustness to hyperparameter
    choice, (3) RTC cannot match FBFM at any action_gw.

Usage:
    python -m pre_test_2.run_experiments_final [--exp a|b|c|d|all] [--seeds 5]
"""

import argparse
import os
import json
import time as time_mod
from collections import OrderedDict

import torch
import numpy as np

from pre_test_2.physics_env import (
    EnvConfig,
    Disturbance,
    MassSpringDamperEnv,
    generate_step_target,
    generate_sinusoidal_target,
)
from pre_test_2.train import load_trained_model
from pre_test_2.fbfm_processor import FBFMConfig, GuidanceMode, fbfm_sample, PrevChunkInfo
from pre_test_2.expert_data import PIDController, PIDConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ======================================================================
# Constants
# ======================================================================

STATE_DIM = 2
ACTION_DIM = 1
HORIZON = 16
EXEC_HORIZON = 8
TOTAL_STEPS = 300
RESULTS_DIR = "pre_test_2/results_final"
CHECKPOINT = "pre_test_2/checkpoints/best_model.pt"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

TRAIN_ENV_CFG = EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02)

COLORS = {
    "pid":     "#2ecc71",
    "vanilla": "#e74c3c",
    "rtc":     "#3498db",
    "fbfm_id": "#f39c12",
    "fbfm":    "#9b59b6",
    "target":  "#95a5a6",
    "disturbance": "#e67e22",
}
LABELS = {
    "pid":     "PID Expert",
    "vanilla": "Vanilla FM",
    "rtc":     "RTC (Action Only)",
    "fbfm_id": "FBFM-Identity J",
    "fbfm":    "FBFM (Ours)",
}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    })


# ======================================================================
# FBFM Configuration
# ======================================================================

def get_cfg(mode: GuidanceMode, **overrides) -> FBFMConfig:
    """Create FBFMConfig with tuned default weights."""
    kwargs = dict(
        mode=mode,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        max_guidance_weight=1.0,
        state_max_guidance_weight=2.5 if mode in (GuidanceMode.FBFM, GuidanceMode.FBFM_IDENTITY) else 1.0,
        execution_horizon=EXEC_HORIZON,
        state_execution_horizon=EXEC_HORIZON,
        num_denoise_steps=20,
    )
    kwargs.update(overrides)
    return FBFMConfig(**kwargs)


# ======================================================================
# Core rollout functions
# ======================================================================

def run_rollout(model, norm_stats, cfg, env_cfg, target, init_state, device,
                disturbance=None):
    """Closed-loop rollout for 1D mass-spring-damper."""
    env = MassSpringDamperEnv(env_cfg)
    state = env.reset(init_state)

    T = TOTAL_STEPS
    all_states = torch.zeros(T, STATE_DIM)
    all_actions = torch.zeros(T, ACTION_DIM)
    pred_states_list = []
    chunk_boundaries = []

    current_chunk = None
    action_idx = 0
    observed_states = []

    t = 0
    while t < T:
        need_new = (current_chunk is None or action_idx >= min(EXEC_HORIZON, current_chunk.shape[0]))
        if need_new:
            chunk_boundaries.append(t)
            prev_chunk = None
            if current_chunk is not None and cfg.mode != GuidanceMode.VANILLA:
                leftover = current_chunk[action_idx:]
                prev_chunk = PrevChunkInfo(
                    action_leftover=leftover if leftover.shape[0] > 0 else None,
                    inference_delay=0,
                )
                if cfg.mode in (GuidanceMode.FBFM, GuidanceMode.FBFM_IDENTITY) and len(observed_states) > 0:
                    prev_chunk.observed_states = torch.stack(observed_states, dim=0)

            target_t = target[t].clone().detach()
            obs = state.clone().detach()
            pred_s, pred_a, _ = fbfm_sample(
                model=model, observation=obs, norm_stats=norm_stats,
                cfg=cfg, prev_chunk=prev_chunk, device=device,
                target=target_t,
            )
            pred_states_list.append(pred_s.detach())
            current_chunk = pred_a.detach().clamp(-env_cfg.max_force, env_cfg.max_force)
            action_idx = 0
            observed_states = []

        action = current_chunk[action_idx] if action_idx < current_chunk.shape[0] else torch.zeros(ACTION_DIM)
        all_states[t] = state.clone()
        all_actions[t] = action.clone()
        state = env.step(action.detach(), disturbance=disturbance)
        observed_states.append(state.clone().detach())
        action_idx += 1
        t += 1

    return {
        "states": all_states,
        "actions": all_actions,
        "targets": target[:T],
        "pred_states": pred_states_list,
        "chunk_boundaries": chunk_boundaries,
    }


def run_pid_rollout(env_cfg, target, init_state, disturbance=None):
    """PID expert rollout."""
    env = MassSpringDamperEnv(env_cfg)
    pid = PIDController(PIDConfig())
    state = env.reset(init_state)

    T = TOTAL_STEPS
    all_states = torch.zeros(T, STATE_DIM)
    all_actions = torch.zeros(T, ACTION_DIM)

    for t in range(T):
        u = pid.compute(state[0].item(), target[t, 0].item(), env_cfg.dt)
        action = torch.tensor([u])
        all_states[t] = state.clone()
        all_actions[t] = action.clone()
        state = env.step(action, disturbance=disturbance)

    return {"states": all_states, "actions": all_actions, "targets": target[:T]}


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(result, reference=None):
    """Compute comprehensive metrics."""
    states = result["states"]
    actions = result["actions"]
    targets = result["targets"]

    pos_mse = ((states[:, 0] - targets[:, 0]) ** 2).mean().item()
    vel_mse = ((states[:, 1] - targets[:, 1]) ** 2).mean().item()

    # IAE (Integral Absolute Error)
    iae = (states[:, 0] - targets[:, 0]).abs().mean().item()

    diffs = actions[1:] - actions[:-1]
    jitter = (diffs ** 2).mean().item()
    energy = (actions ** 2).mean().item()
    control_tv = diffs.abs().sum().item()

    metrics = {
        "position_mse": pos_mse,
        "velocity_mse": vel_mse,
        "iae": iae,
        "action_jitter": jitter,
        "control_tv": control_tv,
        "energy": energy,
    }

    if reference is not None:
        metrics["action_mse_vs_pid"] = ((actions - reference["actions"]) ** 2).mean().item()
        metrics["state_mse_vs_pid"] = ((states - reference["states"]) ** 2).mean().item()

    # State prediction MSE
    if "pred_states" in result and "chunk_boundaries" in result:
        pred_errors = []
        for pred_s, cb in zip(result["pred_states"], result["chunk_boundaries"]):
            H = pred_s.shape[0]
            end = min(cb + H, states.shape[0])
            L = end - cb
            if L > 0:
                err = (pred_s[:L, :STATE_DIM] - states[cb:end]).pow(2).mean().item()
                pred_errors.append(err)
        if pred_errors:
            metrics["state_prediction_mse"] = sum(pred_errors) / len(pred_errors)

    # Recovery analysis (for disturbance experiments)
    pos_error = (states[:, 0] - targets[:, 0]).abs()
    if pos_error.max() > 0.3:
        peak_idx = pos_error.argmax().item()
        threshold = 0.05
        recovery = None
        for tt in range(peak_idx, len(pos_error)):
            if pos_error[tt] < threshold:
                recovery = tt - peak_idx
                break
        if recovery is None:
            recovery = len(pos_error) - peak_idx
        metrics["recovery_steps"] = recovery
        metrics["peak_error"] = pos_error.max().item()

    return metrics


# ======================================================================
# Multi-seed aggregation
# ======================================================================

def aggregate_metrics(metrics_list):
    """Aggregate a list of metric dicts → {key: {"mean", "std", "raw"}}."""
    if not metrics_list:
        return {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    agg = {}
    for k in all_keys:
        vals = [float(m[k]) for m in metrics_list if k in m and m[k] is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        agg[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "raw": vals
        }
    return agg


def print_table(all_methods_agg, title=""):
    """Print a formatted metrics table."""
    if not all_methods_agg:
        return
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")

    methods = list(all_methods_agg.keys())
    all_keys = set()
    for m in methods:
        v = all_methods_agg[m]
        if isinstance(v, dict):
            all_keys.update(v.keys())
    metric_keys = sorted(all_keys)

    header = f"{'Metric':<25}"
    for m in methods:
        header += f"  {LABELS.get(m, m):>20}"
    print(header)
    print("-" * len(header))

    for mk in metric_keys:
        row = f"{mk:<25}"
        for m in methods:
            agg = all_methods_agg[m]
            if mk in agg:
                v = agg[mk]
                if isinstance(v, dict) and "mean" in v:
                    val = f"{v['mean']:.4f}±{v['std']:.4f}"
                elif isinstance(v, (int, float)):
                    val = f"{v:.4f}"
                else:
                    val = str(v)
            else:
                val = "N/A"
            row += f"  {val:>20}"
        print(row)


# ======================================================================
# Run all methods for a single condition
# ======================================================================

def _run_all_methods(model, norm_stats, device, env_cfg, target, init_state,
                     seeds, disturbance=None, include_fbfm_id=False):
    """Run PID + Vanilla + RTC + FBFM (+ optionally FBFM_ID) over multiple seeds."""
    methods = [
        ("vanilla", GuidanceMode.VANILLA),
        ("rtc", GuidanceMode.RTC),
    ]
    if include_fbfm_id:
        methods.append(("fbfm_id", GuidanceMode.FBFM_IDENTITY))
    methods.append(("fbfm", GuidanceMode.FBFM))

    # PID is deterministic — single run
    pid_result = run_pid_rollout(env_cfg, target, init_state, disturbance=disturbance)
    pid_metrics = compute_metrics(pid_result)

    all_agg = OrderedDict()
    all_agg["pid"] = pid_metrics

    representative_results = {"pid": pid_result}

    for name, mode in methods:
        seed_metrics = []
        for si, seed in enumerate(seeds):
            torch.manual_seed(seed)
            cfg = get_cfg(mode)
            result = run_rollout(model, norm_stats, cfg, env_cfg, target, init_state,
                                device, disturbance=disturbance)
            m = compute_metrics(result, reference=pid_result)
            seed_metrics.append(m)
            if si == 0:
                representative_results[name] = result

        all_agg[name] = aggregate_metrics(seed_metrics)

    return all_agg, representative_results


# ======================================================================
# Visualization helpers
# ======================================================================

def _plot_trajectory(results, title, save_path, dt=0.02, disturbance_step=None):
    """Plot position, velocity, and action for all methods."""
    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    first = next(iter(results.values()))
    T = first["states"].shape[0]
    time_ax = np.arange(T) * dt

    target = first["targets"]
    axes[0].plot(time_ax, target[:, 0].numpy(), color=COLORS["target"],
                 linestyle="--", linewidth=1.0, label="Target", alpha=0.7)
    axes[1].plot(time_ax, target[:, 1].numpy(), color=COLORS["target"],
                 linestyle="--", linewidth=1.0, alpha=0.7)

    for name, result in results.items():
        c = COLORS.get(name, "#000")
        lbl = LABELS.get(name, name)
        axes[0].plot(time_ax, result["states"][:, 0].numpy(), color=c, label=lbl, alpha=0.85)
        axes[1].plot(time_ax, result["states"][:, 1].numpy(), color=c, alpha=0.85)
        axes[2].plot(time_ax, result["actions"][:, 0].numpy(), color=c, alpha=0.85)

    if disturbance_step is not None:
        for ax in axes:
            ax.axvline(disturbance_step * dt, color=COLORS["disturbance"],
                       linestyle="-.", linewidth=1.5, alpha=0.7)
        axes[0].annotate("Perturbation", xy=(disturbance_step * dt, axes[0].get_ylim()[1] * 0.9),
                         fontsize=9, color=COLORS["disturbance"], ha="center", fontweight="bold")

    axes[0].set_ylabel("Position $x$")
    axes[0].set_title(title, fontweight="bold")
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set_ylabel("Velocity $\\dot{x}$")
    axes[2].set_ylabel("Action $u$")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved: {save_path}")


def _plot_error(results, title, save_path, dt=0.02, disturbance_step=None):
    """Plot position and velocity error vs PID expert."""
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    first = next(iter(results.values()))
    T = first["states"].shape[0]
    time_ax = np.arange(T) * dt

    # Get PID expert trajectory as reference
    pid_result = results.get("pid")
    if pid_result is None:
        print(f"[Warning] No PID reference found for error plot: {save_path}")
        return

    for name, result in results.items():
        if name == "pid":
            continue
        c = COLORS.get(name, "#000")
        lbl = LABELS.get(name, name)
        # Error vs PID expert instead of target
        pos_err = (result["states"][:, 0] - pid_result["states"][:, 0]).abs().numpy()
        vel_err = (result["states"][:, 1] - pid_result["states"][:, 1]).abs().numpy()
        axes[0].plot(time_ax, pos_err, color=c, label=lbl, alpha=0.85)
        axes[1].plot(time_ax, vel_err, color=c, alpha=0.85)

    if disturbance_step is not None:
        for ax in axes:
            ax.axvline(disturbance_step * dt, color=COLORS["disturbance"],
                       linestyle="-.", linewidth=1.5, alpha=0.7)

    axes[0].set_ylabel("|Position Error vs PID|")
    axes[0].set_title(title, fontweight="bold")
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set_ylabel("|Velocity Error vs PID|")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved: {save_path}")


def _plot_bar_summary(all_conditions, methods, metric_keys, title, save_path):
    """Grouped bar chart with error bars."""
    setup_style()
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    labels_list = list(all_conditions.keys())
    x = np.arange(len(labels_list))
    n_methods = len(methods)
    width = 0.8 / n_methods

    for ax, (mk, mk_title) in zip(axes, metric_keys):
        for i, m in enumerate(methods):
            means, stds = [], []
            for lbl in labels_list:
                agg = all_conditions[lbl].get(m, {})
                if isinstance(agg, dict) and mk in agg:
                    v = agg[mk]
                    if isinstance(v, dict) and "mean" in v:
                        means.append(v["mean"])
                        stds.append(v["std"])
                    elif isinstance(v, (int, float)):
                        means.append(v)
                        stds.append(0)
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - n_methods / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=LABELS.get(m, m), color=COLORS.get(m, "#333"),
                   alpha=0.85, capsize=3, error_kw={"linewidth": 1})

        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=30, ha="right", fontsize=8)
        ax.set_title(mk_title, fontsize=11)
        ax.legend(fontsize=7)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved: {save_path}")


# ======================================================================
# Experiment A: Model Mismatch Sweep
# ======================================================================

def experiment_a(model, norm_stats, device, seeds):
    """Exp A: Model mismatch sweep with step + sinusoidal targets."""
    print("\n" + "=" * 70)
    print(f"  Exp A: Model Mismatch Sweep ({len(seeds)} seeds)")
    print("=" * 70)

    save_dir = os.path.join(RESULTS_DIR, "exp_a_mismatch")
    os.makedirs(save_dir, exist_ok=True)
    init_state = torch.zeros(STATE_DIM)

    mismatch_configs = OrderedDict([
        ("nominal",  EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02)),
        ("mass×1.5", EnvConfig(mass=1.5, damping=0.5, stiffness=0.1, dt=0.02)),
        ("mass×2",   EnvConfig(mass=2.0, damping=0.5, stiffness=0.1, dt=0.02)),
        ("mass×3",   EnvConfig(mass=3.0, damping=0.5, stiffness=0.1, dt=0.02)),
        ("stiff×3",  EnvConfig(mass=1.0, damping=0.5, stiffness=0.3, dt=0.02)),
        ("combined", EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02)),
    ])

    target_types = OrderedDict([
        ("step",       generate_step_target(TOTAL_STEPS, target_pos=1.0)),
        ("sinusoidal", generate_sinusoidal_target(TOTAL_STEPS, amplitude=1.0, period_steps=200)),
    ])

    all_metrics = OrderedDict()

    for tgt_name, target in target_types.items():
        print(f"\n  === Target: {tgt_name} ===")
        for mm_label, env_cfg in mismatch_configs.items():
            key = f"{tgt_name}_{mm_label}"
            print(f"\n  --- {key} ---")

            agg, rep_results = _run_all_methods(
                model, norm_stats, device, env_cfg, target, init_state, seeds,
            )
            print_table(agg, key)
            all_metrics[key] = agg

            _plot_trajectory(rep_results, f"Exp A: {tgt_name} / {mm_label}",
                             os.path.join(save_dir, f"traj_{key}.png"))
            _plot_error(rep_results, f"Exp A Error: {tgt_name} / {mm_label}",
                        os.path.join(save_dir, f"err_{key}.png"))

    # Summary bar chart for each target type
    methods = ["vanilla", "rtc", "fbfm"]
    metric_keys_bar = [
        ("action_jitter", "Action Jitter ↓"),
        ("action_mse_vs_pid", "Action MSE vs PID ↓"),
        ("position_mse", "Position MSE ↓"),
    ]

    for tgt_name in target_types.keys():
        condition_data = OrderedDict()
        for mm_label in mismatch_configs.keys():
            key = f"{tgt_name}_{mm_label}"
            condition_data[mm_label] = all_metrics[key]

        _plot_bar_summary(
            condition_data, methods, metric_keys_bar,
            f"Exp A: Mismatch Sweep ({tgt_name} target)",
            os.path.join(save_dir, f"summary_{tgt_name}.png"),
        )

    _save_json(all_metrics, os.path.join(save_dir, "metrics.json"))
    return all_metrics


# ======================================================================
# Experiment B: Physical Disturbance Recovery
# ======================================================================

def experiment_b(model, norm_stats, device, seeds):
    """Exp B: Disturbance recovery under nominal and mismatched dynamics."""
    print("\n" + "=" * 70)
    print(f"  Exp B: Physical Disturbance Recovery ({len(seeds)} seeds)")
    print("=" * 70)

    save_dir = os.path.join(RESULTS_DIR, "exp_b_disturbance")
    os.makedirs(save_dir, exist_ok=True)
    init_state = torch.zeros(STATE_DIM)
    target = generate_step_target(TOTAL_STEPS, target_pos=1.0)

    disturbances = OrderedDict([
        ("pos_offset_nom", {
            "env": EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02),
            "dist": Disturbance(start_step=100, duration=0, magnitude=0.5, type="position_offset"),
            "label": "Pos Offset 0.5 (nominal)",
        }),
        ("pos_offset_mm", {
            "env": EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02),
            "dist": Disturbance(start_step=100, duration=0, magnitude=0.5, type="position_offset"),
            "label": "Pos Offset 0.5 (mismatch)",
        }),
        # Disturbance during blind execution (mid-chunk)
        ("pos_offset_blind_nom", {
            "env": EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02),
            "dist": Disturbance(start_step=52, duration=0, magnitude=0.5, type="position_offset"),
            "label": "Pos Offset 0.5 mid-chunk (nominal)",
        }),
        ("pos_offset_blind_mm", {
            "env": EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02),
            "dist": Disturbance(start_step=52, duration=0, magnitude=0.5, type="position_offset"),
            "label": "Pos Offset 0.5 mid-chunk (mismatch)",
        }),
        ("step_force_nom", {
            "env": EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02),
            "dist": Disturbance(start_step=100, duration=50, magnitude=2.0, type="step"),
            "label": "Force 2N×50 (nominal)",
        }),
        ("step_force_mm", {
            "env": EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02),
            "dist": Disturbance(start_step=100, duration=50, magnitude=2.0, type="step"),
            "label": "Force 2N×50 (mismatch)",
        }),
        ("impulse_nom", {
            "env": EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02),
            "dist": Disturbance(start_step=100, duration=0, magnitude=5.0, type="impulse"),
            "label": "Impulse 5N (nominal)",
        }),
        ("impulse_mm", {
            "env": EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02),
            "dist": Disturbance(start_step=100, duration=0, magnitude=5.0, type="impulse"),
            "label": "Impulse 5N (mismatch)",
        }),
    ])

    all_metrics = OrderedDict()

    for key, spec in disturbances.items():
        print(f"\n  --- {spec['label']} ---")

        agg, rep_results = _run_all_methods(
            model, norm_stats, device,
            spec["env"], target, init_state, seeds,
            disturbance=spec["dist"],
        )
        print_table(agg, spec["label"])
        all_metrics[key] = agg

        _plot_trajectory(rep_results, f"Exp B: {spec['label']}",
                         os.path.join(save_dir, f"traj_{key}.png"),
                         disturbance_step=spec["dist"].start_step)
        _plot_error(rep_results, f"Exp B Error: {spec['label']}",
                    os.path.join(save_dir, f"err_{key}.png"),
                    disturbance_step=spec["dist"].start_step)

    methods = ["vanilla", "rtc", "fbfm"]
    # Use short labels for bar chart
    short_labels = OrderedDict()
    for k, spec in disturbances.items():
        short_labels[spec["label"]] = all_metrics[k]

    metric_keys_bar = [
        ("action_jitter", "Action Jitter ↓"),
        ("position_mse", "Position MSE ↓"),
        ("recovery_steps", "Recovery Steps ↓"),
    ]
    _plot_bar_summary(
        short_labels, methods, metric_keys_bar,
        "Exp B: Disturbance Recovery",
        os.path.join(save_dir, "summary.png"),
    )

    _save_json(all_metrics, os.path.join(save_dir, "metrics.json"))
    return all_metrics


# ======================================================================
# Experiment C: Jacobian Ablation
# ======================================================================

def experiment_c(model, norm_stats, device, seeds):
    """Exp C: Jacobian ablation — Vanilla / RTC / FBFM-ID / FBFM."""
    print("\n" + "=" * 70)
    print(f"  Exp C: Jacobian Ablation ({len(seeds)} seeds)")
    print("=" * 70)

    save_dir = os.path.join(RESULTS_DIR, "exp_c_ablation")
    os.makedirs(save_dir, exist_ok=True)
    init_state = torch.zeros(STATE_DIM)

    conditions = OrderedDict([
        ("step_mass2", {
            "env": EnvConfig(mass=2.0, damping=0.5, stiffness=0.1, dt=0.02),
            "target": generate_step_target(TOTAL_STEPS, target_pos=1.0),
            "label": "Step, mass×2",
        }),
        ("step_combined", {
            "env": EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02),
            "target": generate_step_target(TOTAL_STEPS, target_pos=1.0),
            "label": "Step, combined",
        }),
        ("sin_mass2", {
            "env": EnvConfig(mass=2.0, damping=0.5, stiffness=0.1, dt=0.02),
            "target": generate_sinusoidal_target(TOTAL_STEPS, amplitude=1.0, period_steps=200),
            "label": "Sin, mass×2",
        }),
        ("sin_combined", {
            "env": EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02),
            "target": generate_sinusoidal_target(TOTAL_STEPS, amplitude=1.0, period_steps=200),
            "label": "Sin, combined",
        }),
        ("sin_mass3", {
            "env": EnvConfig(mass=3.0, damping=0.5, stiffness=0.1, dt=0.02),
            "target": generate_sinusoidal_target(TOTAL_STEPS, amplitude=1.0, period_steps=200),
            "label": "Sin, mass×3",
        }),
    ])

    all_metrics = OrderedDict()

    for key, spec in conditions.items():
        print(f"\n  --- {spec['label']} ---")

        agg, rep_results = _run_all_methods(
            model, norm_stats, device,
            spec["env"], spec["target"], init_state, seeds,
            include_fbfm_id=True,
        )
        print_table(agg, spec["label"])
        all_metrics[key] = agg

        _plot_trajectory(rep_results, f"Exp C: {spec['label']}",
                         os.path.join(save_dir, f"traj_{key}.png"))
        _plot_error(rep_results, f"Exp C Error: {spec['label']}",
                    os.path.join(save_dir, f"err_{key}.png"))

    # Summary bar
    methods = ["vanilla", "rtc", "fbfm_id", "fbfm"]
    metric_keys_bar = [
        ("action_jitter", "Action Jitter ↓"),
        ("action_mse_vs_pid", "Action MSE vs PID ↓"),
        ("state_prediction_mse", "State Prediction MSE ↓"),
    ]
    renamed = OrderedDict()
    for k, spec in conditions.items():
        renamed[spec["label"]] = all_metrics[k]

    _plot_bar_summary(
        renamed, methods, metric_keys_bar,
        "Exp C: Jacobian Ablation",
        os.path.join(save_dir, "summary.png"),
    )

    _save_json(all_metrics, os.path.join(save_dir, "metrics.json"))
    return all_metrics


# ======================================================================
# Experiment D: Guidance Weight Sensitivity
# ======================================================================

def experiment_d(model, norm_stats, device, seeds):
    """Exp D: Guidance weight sensitivity sweep."""
    print("\n" + "=" * 70)
    print(f"  Exp D: Guidance Weight Sensitivity ({len(seeds)} seeds)")
    print("=" * 70)

    save_dir = os.path.join(RESULTS_DIR, "exp_d_sensitivity")
    os.makedirs(save_dir, exist_ok=True)
    init_state = torch.zeros(STATE_DIM)
    env_cfg = EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02)
    target = generate_step_target(TOTAL_STEPS, target_pos=1.0)

    # Vanilla baseline (multi-seed)
    vanilla_metrics_list = []
    for seed in seeds:
        torch.manual_seed(seed)
        cfg = get_cfg(GuidanceMode.VANILLA)
        result = run_rollout(model, norm_stats, cfg, env_cfg, target, init_state, device)
        pid_result = run_pid_rollout(env_cfg, target, init_state)
        vanilla_metrics_list.append(compute_metrics(result, reference=pid_result))
    vanilla_agg = aggregate_metrics(vanilla_metrics_list)

    # FBFM: sweep state_gw
    state_gws = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    fbfm_sweep = OrderedDict()
    print("  [FBFM] Sweeping state_gw (action_gw=1.0)...")
    for sgw in state_gws:
        sgw_metrics = []
        for seed in seeds:
            torch.manual_seed(seed)
            cfg = get_cfg(GuidanceMode.FBFM,
                          max_guidance_weight=1.0,
                          state_max_guidance_weight=sgw)
            result = run_rollout(model, norm_stats, cfg, env_cfg, target, init_state, device)
            pid_result = run_pid_rollout(env_cfg, target, init_state)
            sgw_metrics.append(compute_metrics(result, reference=pid_result))
        fbfm_sweep[sgw] = aggregate_metrics(sgw_metrics)
        m = fbfm_sweep[sgw]
        print(f"    state_gw={sgw:5.1f}  pos_mse={m['position_mse']['mean']:.4f}±{m['position_mse']['std']:.4f}"
              f"  jitter={m['action_jitter']['mean']:.4f}")

    # RTC: sweep action_gw
    action_gws = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0]
    rtc_sweep = OrderedDict()
    print("  [RTC] Sweeping action_gw...")
    for agw in action_gws:
        agw_metrics = []
        for seed in seeds:
            torch.manual_seed(seed)
            cfg = get_cfg(GuidanceMode.RTC, max_guidance_weight=agw)
            result = run_rollout(model, norm_stats, cfg, env_cfg, target, init_state, device)
            pid_result = run_pid_rollout(env_cfg, target, init_state)
            agw_metrics.append(compute_metrics(result, reference=pid_result))
        rtc_sweep[agw] = aggregate_metrics(agw_metrics)
        m = rtc_sweep[agw]
        print(f"    action_gw={agw:5.1f}  pos_mse={m['position_mse']['mean']:.4f}±{m['position_mse']['std']:.4f}"
              f"  jitter={m['action_jitter']['mean']:.4f}")

    _plot_sensitivity(vanilla_agg, fbfm_sweep, rtc_sweep,
                      state_gws, action_gws, save_dir)

    ablation_data = {
        "vanilla": vanilla_agg,
        "fbfm_sweep": {str(k): v for k, v in fbfm_sweep.items()},
        "rtc_sweep": {str(k): v for k, v in rtc_sweep.items()},
    }
    _save_json(ablation_data, os.path.join(save_dir, "metrics.json"))
    return ablation_data


def _plot_sensitivity(vanilla, fbfm_sweep, rtc_sweep,
                      state_gws, action_gws, save_dir):
    """Sensitivity sweep plot with error bands."""
    setup_style()
    metric_keys = [
        ("action_jitter", "Action Jitter ↓"),
        ("action_mse_vs_pid", "Action MSE vs PID ↓"),
        ("position_mse", "Position MSE ↓"),
        ("state_prediction_mse", "State Prediction MSE ↓"),
    ]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 5))

    for ax, (mk, mk_title) in zip(axes, metric_keys):
        fbfm_means = np.array([fbfm_sweep[sgw].get(mk, {}).get("mean", 0) for sgw in state_gws])
        fbfm_stds = np.array([fbfm_sweep[sgw].get(mk, {}).get("std", 0) for sgw in state_gws])
        ax.plot(state_gws, fbfm_means, "o-", color=COLORS["fbfm"], linewidth=2,
                label="FBFM (sweep state_gw)", alpha=0.85)
        ax.fill_between(state_gws, fbfm_means - fbfm_stds, fbfm_means + fbfm_stds,
                        color=COLORS["fbfm"], alpha=0.15)

        rtc_means = np.array([rtc_sweep[agw].get(mk, {}).get("mean", 0) for agw in action_gws])
        rtc_stds = np.array([rtc_sweep[agw].get(mk, {}).get("std", 0) for agw in action_gws])
        ax.plot(action_gws, rtc_means, "s--", color=COLORS["rtc"], linewidth=2,
                label="RTC (sweep action_gw)", alpha=0.85)
        ax.fill_between(action_gws, rtc_means - rtc_stds, rtc_means + rtc_stds,
                        color=COLORS["rtc"], alpha=0.15)

        v_mean = vanilla.get(mk, {}).get("mean", 0)
        v_std = vanilla.get(mk, {}).get("std", 0)
        ax.axhline(v_mean, color=COLORS["vanilla"], linestyle=":", linewidth=1.5, alpha=0.7)
        ax.axhspan(v_mean - v_std, v_mean + v_std,
                   color=COLORS["vanilla"], alpha=0.08)
        ax.text(0.02, 0.95, f"Vanilla: {v_mean:.4f}", transform=ax.transAxes,
                fontsize=7, color=COLORS["vanilla"], va="top")

        ax.set_xlabel("Guidance Weight")
        ax.set_title(mk_title, fontsize=10)
        ax.legend(fontsize=7)

    plt.suptitle("Exp D: Guidance Weight Sensitivity (m=2,c=1,k=0.3)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved: {os.path.join(save_dir, 'sensitivity.png')}")


# ======================================================================
# Utility
# ======================================================================

def _save_json(data, path):
    """Save metrics dict to JSON."""
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(convert(data), f, indent=2)
    print(f"[JSON] Saved: {path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FBFM Final Experiments (Publication Quality)")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["a", "b", "c", "d", "all"])
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Device: {device}")
    print(f"[Main] Seeds: {seeds}")

    model, norm_stats = load_trained_model(CHECKPOINT, device)

    t0 = time_mod.time()

    if args.exp in ("a", "all"):
        experiment_a(model, norm_stats, device, seeds)

    if args.exp in ("b", "all"):
        experiment_b(model, norm_stats, device, seeds)

    if args.exp in ("c", "all"):
        experiment_c(model, norm_stats, device, seeds)

    if args.exp in ("d", "all"):
        experiment_d(model, norm_stats, device, seeds)

    elapsed = time_mod.time() - t0
    print(f"\n[Main] Completed in {elapsed:.1f}s")
    print(f"[Main] Results → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
