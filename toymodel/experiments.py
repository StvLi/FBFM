"""
Experiment definitions A–D for the FBFM pre-experiment suite.

Each ``experiment_X()`` function is self-contained: it sets up conditions,
runs multi-seed rollouts, prints tables, saves plots and JSON metrics.
"""

import os
import json
from collections import OrderedDict

import torch
import numpy as np

from pre_test_1.config import (
    STATE_DIM, TOTAL_STEPS, RESULTS_DIR, get_cfg,
)
from pre_test_1.physics_env import (
    EnvConfig, Disturbance,
    generate_step_target, generate_sinusoidal_target,
)
from pre_test_1.fbfm_processor import GuidanceMode
from pre_test_1.rollout import run_rollout, run_pid_rollout
from pre_test_1.metrics import compute_metrics, aggregate_metrics, print_table
from pre_test_1.plotting import (
    LABELS, plot_trajectory, plot_error, plot_bar_summary, plot_sensitivity,
)


# ── Helpers ───────────────────────────────────────────────────────────

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
            print_table(agg, key, labels=LABELS)
            all_metrics[key] = agg

            plot_trajectory(rep_results, f"Exp A: {tgt_name} / {mm_label}",
                            os.path.join(save_dir, f"traj_{key}.png"))
            plot_error(rep_results, f"Exp A Error: {tgt_name} / {mm_label}",
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

        plot_bar_summary(
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
        print_table(agg, spec["label"], labels=LABELS)
        all_metrics[key] = agg

        plot_trajectory(rep_results, f"Exp B: {spec['label']}",
                        os.path.join(save_dir, f"traj_{key}.png"),
                        disturbance_step=spec["dist"].start_step)
        plot_error(rep_results, f"Exp B Error: {spec['label']}",
                   os.path.join(save_dir, f"err_{key}.png"),
                   disturbance_step=spec["dist"].start_step)

    methods = ["vanilla", "rtc", "fbfm"]
    short_labels = OrderedDict()
    for k, spec in disturbances.items():
        short_labels[spec["label"]] = all_metrics[k]

    metric_keys_bar = [
        ("action_jitter", "Action Jitter ↓"),
        ("position_mse", "Position MSE ↓"),
        ("recovery_steps", "Recovery Steps ↓"),
    ]
    plot_bar_summary(
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
        print_table(agg, spec["label"], labels=LABELS)
        all_metrics[key] = agg

        plot_trajectory(rep_results, f"Exp C: {spec['label']}",
                        os.path.join(save_dir, f"traj_{key}.png"))
        plot_error(rep_results, f"Exp C Error: {spec['label']}",
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

    plot_bar_summary(
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

    plot_sensitivity(vanilla_agg, fbfm_sweep, rtc_sweep,
                     state_gws, action_gws, save_dir)

    ablation_data = {
        "vanilla": vanilla_agg,
        "fbfm_sweep": {str(k): v for k, v in fbfm_sweep.items()},
        "rtc_sweep": {str(k): v for k, v in rtc_sweep.items()},
    }
    _save_json(ablation_data, os.path.join(save_dir, "metrics.json"))
    return ablation_data
