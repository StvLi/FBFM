"""
Publication-Quality Advanced Visualization for FBFM Pre-Experiment.

Produces 8 high-quality figures targeting top-conference (NeurIPS / ICRA / CoRL) standards:

  Fig 1: fig_method_comparison.pdf/png
      3-panel method overview: trajectory + action + error envelope
      (Step, combined mismatch — the showcase condition)

  Fig 2: fig_mismatch_sweep.pdf/png
      Multi-condition jitter heatmap + bar chart — Exp A step target
      (Shows scaling of advantage with mismatch severity)

  Fig 3: fig_disturbance_recovery.pdf/png
      Exp B: 2×2 subplot grid — impulse + force, nominal + mismatch

  Fig 4: fig_jacobian_ablation.pdf/png
      Exp C: Grouped bar chart with individual seed dots (all 4 methods)

  Fig 5: fig_sensitivity.pdf/png
      Exp D: Dual-axis line chart with std bands

  Fig 6: fig_action_detail.pdf/png
      Zoomed action subplot (step combined, 2–3 s window)
      Clearly shows sawtooth vs smooth patterns

  Fig 7: fig_guidance_profile.pdf/png
      k_p(τ) theoretical schedule for different β values

  Fig 8: fig_overview_grid.pdf/png
      4-panel overview figure for the paper's main figure slot

Usage:
    python -m pre_test_2.advanced_viz
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict

import torch

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "pre_test_2/results_final"
OUT_DIR     = os.path.join(RESULTS_DIR, "advanced_viz")
CHECKPOINT  = "pre_test_2/checkpoints/best_model.pt"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Style – NeurIPS / ICRA conference aesthetic  (v2 – refined)
# ─────────────────────────────────────────────────────────────────────
# High-saturation palette: vivid red / blue / purple tones
PALETTE = {
    "pid":         "#2471A3",   # vivid cobalt blue
    "vanilla":     "#E74C3C",   # vivid red
    "rtc":         "#F39C12",   # vivid amber
    "fbfm_id":     "#1ABC9C",   # vivid teal-green
    "fbfm":        "#8E44AD",   # vivid purple
    "target":      "#95A5A6",   # neutral grey
    "disturbance": "#C0392B",   # deep crimson
    "bg":          "#FAFAFA",
}
LABELS = {
    "pid":     "PID Expert",
    "vanilla": "Vanilla FM",
    "rtc":     "RTC (Baseline)",
    "fbfm_id": "FBFM-Identity J",
    "fbfm":    "FBFM (Ours)",
}
# Per-method line dash patterns for B/W-friendliness
DASHES = {
    "pid":     (4, 2),
    "vanilla": (6, 2, 2, 2),
    "rtc":     (3, 1.5),
    "fbfm_id": (5, 2, 1, 2, 1, 2),
    "fbfm":    "",  # solid
    "target":  (2, 2),
}

def set_style():
    """Apply a clean, publication-ready matplotlib style (v2)."""
    plt.rcParams.update({
        # ── Font: use STIX (matches LaTeX Computer Modern) ──
        "font.family":       "serif",
        "font.serif":        ["STIX", "STIXGeneral", "DejaVu Serif",
                              "Times New Roman", "serif"],
        "mathtext.fontset":  "stix",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "legend.fontsize":   8.5,
        "figure.titlesize":  13,
        # ── Lines ──
        "lines.linewidth":   1.6,
        "lines.markersize":  4.5,
        # ── Grid: very subtle ──
        "axes.grid":         True,
        "grid.alpha":        0.18,
        "grid.linestyle":    "-",
        "grid.linewidth":    0.45,
        "grid.color":        "#CCCCCC",
        # ── Axes / spines ──
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.7,
        "axes.edgecolor":     "#333333",
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        # ── Legend ──
        "legend.frameon":      True,
        "legend.framealpha":   0.85,
        "legend.edgecolor":    "#CCCCCC",
        "legend.fancybox":     True,
        "legend.borderpad":    0.4,
        # ── DPI / save ──
        "figure.dpi":         180,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches":  0.08,
    })

set_style()

# ─────────────────────────────────────────────────────────────────────
# Load pre-computed metrics JSONs
# ─────────────────────────────────────────────────────────────────────
def load_metrics(exp):
    path = os.path.join(RESULTS_DIR, exp, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

metrics_a  = load_metrics("exp_a_mismatch")
metrics_b  = load_metrics("exp_b_disturbance")
metrics_c  = load_metrics("exp_c_ablation")
metrics_d  = load_metrics("exp_d_sensitivity")
metrics_e  = load_metrics("exp_e_delay")

def mean_std(d, key, method):
    """Extract mean±std for a metric/method from an aggregated dict."""
    v = d.get(method, {})
    if not isinstance(v, dict):
        return 0.0, 0.0
    k = v.get(key, {})
    if isinstance(k, dict):
        return float(k.get("mean", 0)), float(k.get("std", 0))
    return float(k), 0.0

# ─────────────────────────────────────────────────────────────────────
# Rollout data (re-run representative seed for qualitative figures)
# ─────────────────────────────────────────────────────────────────────
def load_rollout_data():
    """Re-run representative rollouts for qualitative figures."""
    from pre_test_2.physics_env import EnvConfig, Disturbance, generate_step_target, generate_sinusoidal_target
    from pre_test_2.train import load_trained_model
    from pre_test_2.fbfm_processor import FBFMConfig, GuidanceMode, fbfm_sample, PrevChunkInfo
    from pre_test_2.expert_data import PIDController, PIDConfig
    from pre_test_2.run_experiments_final import (
        run_rollout, run_pid_rollout, STATE_DIM, ACTION_DIM, HORIZON, EXEC_HORIZON,
        TOTAL_STEPS, get_cfg
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, norm_stats = load_trained_model(CHECKPOINT, device)
    torch.manual_seed(0)

    COMBINED_ENV  = EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02)
    NOMINAL_ENV   = EnvConfig(mass=1.0, damping=0.5, stiffness=0.1, dt=0.02)
    MASS3_ENV     = EnvConfig(mass=3.0, damping=0.5, stiffness=0.1, dt=0.02)
    init_state    = torch.zeros(STATE_DIM)
    step_target   = generate_step_target(TOTAL_STEPS, target_pos=1.0)
    sin_target    = generate_sinusoidal_target(TOTAL_STEPS, amplitude=1.0, period_steps=200)

    def rollout_all(env_cfg, target, disturbance=None):
        results = {}
        results["pid"] = run_pid_rollout(env_cfg, target, init_state, disturbance=disturbance)
        for name, mode in [("vanilla", GuidanceMode.VANILLA),
                            ("rtc",     GuidanceMode.RTC),
                            ("fbfm_id", GuidanceMode.FBFM_IDENTITY),
                            ("fbfm",    GuidanceMode.FBFM)]:
            cfg = get_cfg(mode)
            results[name] = run_rollout(model, norm_stats, cfg, env_cfg, target, init_state,
                                        device, disturbance=disturbance)
        return results

    data = {}
    print("  Loading rollout: step / combined …")
    data["step_combined"]  = rollout_all(COMBINED_ENV, step_target)
    print("  Loading rollout: step / nominal …")
    data["step_nominal"]   = rollout_all(NOMINAL_ENV, step_target)
    print("  Loading rollout: step / mass×3 …")
    data["step_mass3"]     = rollout_all(MASS3_ENV, step_target)
    print("  Loading rollout: sin / combined …")
    data["sin_combined"]   = rollout_all(COMBINED_ENV, sin_target)
    print("  Loading rollout: impulse / nominal …")
    imp_dist = Disturbance(start_step=100, duration=0, magnitude=5.0, type="impulse")
    data["impulse_nom"]    = rollout_all(NOMINAL_ENV, step_target, disturbance=imp_dist)
    print("  Loading rollout: force / mismatch …")
    force_dist = Disturbance(start_step=100, duration=50, magnitude=2.0, type="step")
    data["force_mm"]       = rollout_all(COMBINED_ENV, step_target, disturbance=force_dist)
    return data

print("[advanced_viz] Loading rollout data …")
rollout = load_rollout_data()
print("[advanced_viz] Data loaded.\n")

DT = 0.02
T  = 300
time_ax = np.arange(T) * DT


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Method Comparison (flagship figure)
# ═══════════════════════════════════════════════════════════════════════
def fig_method_comparison():
    set_style()
    res = rollout["step_combined"]
    methods_show = ["pid", "vanilla", "rtc", "fbfm"]

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[2.5, 1],
                           hspace=0.08, wspace=0.30)

    ax_pos  = fig.add_subplot(gs[0, 0])
    ax_vel  = fig.add_subplot(gs[1, 0], sharex=ax_pos)
    ax_act  = fig.add_subplot(gs[2, 0], sharex=ax_pos)
    ax_bar  = fig.add_subplot(gs[:, 1])

    # ── Trajectory ──────────────────────────────────────────────────
    tgt = res["pid"]["targets"].numpy()
    ax_pos.plot(time_ax, tgt[:, 0], "--", color=PALETTE["target"],
                lw=1.2, label="Target", alpha=0.7)
    ax_vel.plot(time_ax, tgt[:, 1], "--", color=PALETTE["target"], lw=1.2, alpha=0.7)

    lw_map = {"pid": 1.2, "vanilla": 1.4, "rtc": 1.4, "fbfm": 2.0}
    alpha_map = {"pid": 0.65, "vanilla": 0.80, "rtc": 0.80, "fbfm": 0.95}
    zorder_map = {"pid": 2, "vanilla": 1, "rtc": 1, "fbfm": 3}

    for nm in methods_show:
        r   = res[nm]
        col = PALETTE[nm]
        lbl = LABELS[nm]
        lw  = lw_map.get(nm, 1.4)
        al  = alpha_map.get(nm, 0.80)
        zo  = zorder_map.get(nm, 1)
        kw  = dict(color=col, lw=lw, alpha=al, zorder=zo)
        if DASHES.get(nm):
            kw["dashes"] = DASHES[nm]
        ax_pos.plot(time_ax, r["states"][:, 0].numpy(), label=lbl, **kw)
        ax_vel.plot(time_ax, r["states"][:, 1].numpy(), **kw)
        ax_act.plot(time_ax, r["actions"][:, 0].numpy(), **kw)

    # Chunk boundaries for FBFM
    from pre_test_2.run_experiments_final import EXEC_HORIZON
    for cb in res["fbfm"].get("chunk_boundaries", []):
        ax_act.axvline(cb * DT, color=PALETTE["fbfm"], lw=0.5, alpha=0.25, zorder=0)

    ax_pos.set_ylabel("Position $x$ (m)", labelpad=4)
    ax_vel.set_ylabel("Velocity $\\dot{x}$ (m/s)", labelpad=4)
    ax_act.set_ylabel("Control $u$ (N)", labelpad=4)
    ax_act.set_xlabel("Time (s)")
    ax_pos.legend(loc="lower right", ncol=2, framealpha=0.9)
    ax_pos.set_title(r"$m{=}2,\ c{=}1,\ k{=}0.3$ (combined mismatch, step target)",
                     fontsize=11, pad=6)
    plt.setp(ax_pos.get_xticklabels(), visible=False)
    plt.setp(ax_vel.get_xticklabels(), visible=False)

    # ── Bar chart: MSE-focused metrics ───────────────────────────────
    metric_labels = ["Pos MSE\n↓", "Action\nMSE↓", "State\nMSE↓"]
    metric_keys   = ["position_mse", "action_mse_vs_pid", "state_mse_vs_pid"]
    mlist = ["vanilla", "rtc", "fbfm"]
    cond  = metrics_a["step_combined"]

    x_bar = np.arange(len(metric_labels))
    width = 0.25
    for i, nm in enumerate(mlist):
        means = [mean_std(cond, mk, nm)[0] for mk in metric_keys]
        stds  = [mean_std(cond, mk, nm)[1] for mk in metric_keys]
        ax_bar.bar(x_bar + (i - 1) * width, means, width,
                   yerr=stds, label=LABELS[nm], color=PALETTE[nm],
                   alpha=0.82, capsize=4, edgecolor="#555555", linewidth=0.4,
                   error_kw={"linewidth": 1}, zorder=3)

    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(metric_labels, fontsize=9)
    ax_bar.set_title("MSE Metrics\n(combined mismatch)", fontsize=10)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.set_ylabel("MSE", labelpad=4)
    ax_bar.yaxis.set_major_locator(MaxNLocator(5))

    fig.suptitle("FBFM vs Baselines: Step Response under Model Mismatch",
                 fontsize=13, fontweight="bold", y=1.01)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_method_comparison.{ext}"))
    plt.close(fig)
    print(f"  → fig_method_comparison saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Mismatch Sweep — jitter bar + improvement line  (v2)
# ═══════════════════════════════════════════════════════════════════════
def fig_mismatch_sweep():
    set_style()
    conditions = ["nominal", "mass×1.5", "mass×2", "mass×3", "stiff×3", "combined"]
    x = np.arange(len(conditions))
    methods_show = ["vanilla", "rtc", "fbfm"]
    width = 0.26

    fig, (ax_jitter, ax_improve) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Jitter bar chart (clip large std to keep y-axis readable) ────
    all_means_stds = []
    for nm in methods_show:
        means, stds = [], []
        for cond in conditions:
            m, s = mean_std(metrics_a[f"step_{cond}"], "action_jitter", nm)
            means.append(m)
            stds.append(s)
        all_means_stds.append((nm, means, stds))

    # Determine a sensible y-limit: p95 of (mean+std) values
    all_tops = []
    for _, ms, ss in all_means_stds:
        all_tops.extend([m + s for m, s in zip(ms, ss)])
    y_cap = np.percentile(all_tops, 90) * 1.25  # generous cap
    y_max = max(y_cap, 0.1)

    for i, (nm, means, stds) in enumerate(all_means_stds):
        # Clip error bars so they don't blow out the axis
        clipped_stds = [min(s, y_max - m) if m + s > y_max else s
                        for m, s in zip(means, stds)]
        offset = (i - 1) * width
        bars = ax_jitter.bar(x + offset, means, width, yerr=clipped_stds,
                             label=LABELS[nm], color=PALETTE[nm],
                             alpha=0.82, capsize=3,
                             edgecolor="#555555", linewidth=0.4,
                             error_kw={"linewidth": 1}, zorder=3)
        # If any bar was clipped, add a small arrow indicator
        for j, (m, s, cs) in enumerate(zip(means, stds, clipped_stds)):
            if s > cs + 1e-6:
                ax_jitter.annotate("",
                    xy=(x[j] + offset, y_max * 0.97),
                    xytext=(x[j] + offset, y_max * 0.90),
                    arrowprops=dict(arrowstyle="->", color=PALETTE[nm],
                                    lw=1.2),
                    zorder=5)

    ax_jitter.set_ylim(0, y_max)
    ax_jitter.set_xticks(x)
    ax_jitter.set_xticklabels(conditions, rotation=25, ha="right", fontsize=8)
    ax_jitter.set_ylabel("Action Jitter $\\downarrow$")
    ax_jitter.set_title("Action Jitter across Mismatch Conditions\n(Step Target)",
                        fontsize=10)
    ax_jitter.legend(fontsize=8)

    # ── FBFM improvement % line ──────────────────────────────────────
    improvements = []
    fbfm_means, fbfm_stds = [], []
    van_means = []
    for cond in conditions:
        vm, _ = mean_std(metrics_a[f"step_{cond}"], "action_jitter", "vanilla")
        fm, fs = mean_std(metrics_a[f"step_{cond}"], "action_jitter", "fbfm")
        rm, rs = mean_std(metrics_a[f"step_{cond}"], "action_jitter", "rtc")
        improvements.append((vm - fm) / (vm + 1e-9) * 100)
        fbfm_means.append(fm)
        fbfm_stds.append(fs)
        van_means.append(vm)

    fbfm_means = np.array(fbfm_means)
    fbfm_stds  = np.array(fbfm_stds)
    improvements = np.array(improvements)

    ax2 = ax_improve.twinx()
    ax_improve.fill_between(x, fbfm_means - fbfm_stds, fbfm_means + fbfm_stds,
                             color=PALETTE["fbfm"], alpha=0.15)
    ax_improve.plot(x, fbfm_means, "o-", color=PALETTE["fbfm"],
                    lw=2, label="FBFM Jitter", zorder=3, markersize=5)
    ax_improve.plot(x, van_means, "s--", color=PALETTE["vanilla"],
                    lw=1.5, alpha=0.7, label="Vanilla Jitter", markersize=4)
    ax2.bar(x, improvements, alpha=0.12, color=PALETTE["fbfm"],
            width=0.5, label="Improvement %", edgecolor="none")
    ax2.set_ylabel("FBFM Improvement (%)", color=PALETTE["fbfm"])
    ax2.tick_params(axis="y", colors=PALETTE["fbfm"])
    ax2.set_ylim(0, 120)

    ax_improve.set_xticks(x)
    ax_improve.set_xticklabels(conditions, rotation=25, ha="right", fontsize=8)
    ax_improve.set_ylabel("Action Jitter $\\downarrow$")
    ax_improve.set_title("FBFM Advantage Scales with\nMismatch Severity", fontsize=10)
    lines1, labels1 = ax_improve.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_improve.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper left")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_mismatch_sweep.{ext}"))
    plt.close(fig)
    print(f"  → fig_mismatch_sweep saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Disturbance Recovery — 2×2 grid  (redesigned v2)
# ═══════════════════════════════════════════════════════════════════════
def fig_disturbance_recovery():
    set_style()
    methods_show = ["pid", "vanilla", "rtc", "fbfm"]

    specs = [
        ("impulse_nom",  "Impulse 5 N, Nominal",    rollout["impulse_nom"],  100, 0),
        ("force_mm",     "Force 2 N$\\times$50, Mismatch", rollout["force_mm"],  100, 50),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.28})

    lw_map  = {"pid": 1.1, "vanilla": 1.3, "rtc": 1.3, "fbfm": 1.9}
    zo_map  = {"pid": 1, "vanilla": 2, "rtc": 2, "fbfm": 4}
    al_map  = {"pid": 0.55, "vanilla": 0.75, "rtc": 0.75, "fbfm": 0.95}

    for col_idx, (key, title, res, d_start, d_dur) in enumerate(specs):
        ax_pos = axes[0, col_idx]
        ax_act = axes[1, col_idx]

        tgt = res["pid"]["targets"].numpy()
        ax_pos.plot(time_ax, tgt[:, 0], color=PALETTE["target"],
                    lw=1.0, alpha=0.5, label="Target", dashes=(3, 2))

        for nm in methods_show:
            r = res[nm]
            kw = dict(color=PALETTE[nm], lw=lw_map[nm],
                      alpha=al_map[nm], zorder=zo_map[nm])
            if DASHES.get(nm):
                kw["dashes"] = DASHES[nm]
            ax_pos.plot(time_ax, r["states"][:, 0].numpy(),
                        label=LABELS[nm], **kw)

            # Action: only show 2 key methods in detail, others thinner
            act_lw = lw_map[nm] if nm in ("fbfm", "pid") else lw_map[nm] * 0.8
            act_al = al_map[nm] if nm in ("fbfm", "pid") else al_map[nm] * 0.7
            kw_a = dict(color=PALETTE[nm], lw=act_lw, alpha=act_al,
                        zorder=zo_map[nm])
            if DASHES.get(nm):
                kw_a["dashes"] = DASHES[nm]
            ax_act.plot(time_ax, r["actions"][:, 0].numpy(), **kw_a)

        # Disturbance marker — shaded region for force, vertical line for impulse
        d_time = d_start * DT
        for ax in (ax_pos, ax_act):
            if d_dur > 0:
                ax.axvspan(d_time, (d_start + d_dur) * DT,
                           color=PALETTE["disturbance"], alpha=0.08, zorder=0)
                ax.axvline(d_time, color=PALETTE["disturbance"],
                           lw=0.9, ls="--", alpha=0.5, zorder=0)
            else:
                ax.axvline(d_time, color=PALETTE["disturbance"],
                           lw=1.2, ls="-.", alpha=0.65, zorder=5,
                           label="Disturbance" if ax is ax_pos else "")

        ax_pos.set_title(title, fontsize=10, fontweight="bold")
        ax_act.set_xlabel("Time (s)")
        if col_idx == 0:
            ax_pos.set_ylabel("Position $x$ (m)")
            ax_act.set_ylabel("Control $u$ (N)")
            ax_pos.legend(fontsize=7, ncol=3, loc="lower right",
                          handlelength=2.0, columnspacing=0.6)

        # ── Zoom inset on action around disturbance ──────────────────
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        zoom_half = 25
        zs = max(d_start - 5, 0)
        ze = min(d_start + d_dur + zoom_half, T)
        ax_in = inset_axes(ax_act, width="38%", height="42%",
                           loc="upper right", borderpad=0.6)
        for nm in ["pid", "fbfm"]:
            r = res[nm]
            kw_z = dict(color=PALETTE[nm], alpha=0.9, zorder=zo_map[nm])
            kw_z["lw"] = 1.6 if nm == "fbfm" else 1.0
            if DASHES.get(nm):
                kw_z["dashes"] = DASHES[nm]
            ax_in.plot(time_ax[zs:ze], r["actions"][zs:ze, 0].numpy(), **kw_z)
        if d_dur > 0:
            ax_in.axvspan(d_time, (d_start + d_dur) * DT,
                          color=PALETTE["disturbance"], alpha=0.10)
        else:
            ax_in.axvline(d_time, color=PALETTE["disturbance"],
                          lw=0.8, ls="-.", alpha=0.6)
        ax_in.set_title("PID vs FBFM (zoom)", fontsize=6.5, pad=2)
        ax_in.tick_params(labelsize=5.5)
        ax_in.set_facecolor("#FAFAFA")
        for spine in ax_in.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#999999")

    fig.suptitle("Exp B: Physical Disturbance Recovery",
                 fontsize=12, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_disturbance_recovery.{ext}"))
    plt.close(fig)
    print(f"  → fig_disturbance_recovery saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Jacobian Ablation — grouped bars with scatter dots
# ═══════════════════════════════════════════════════════════════════════
def fig_jacobian_ablation():
    set_style()
    methods_show  = ["vanilla", "rtc", "fbfm_id", "fbfm"]
    conditions    = ["step_mass2", "step_combined", "sin_mass2", "sin_combined"]
    cond_labels   = ["Step\nmass×2", "Step\ncombined", "Sin\nmass×2", "Sin\ncombined"]
    metric_pairs  = [
        ("action_jitter",    "Action Jitter ↓"),
        ("action_mse_vs_pid","Action MSE vs PID ↓"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (mk, mk_title) in zip(axes, metric_pairs):
        x = np.arange(len(conditions))
        width = 0.18

        for i, nm in enumerate(methods_show):
            means, stds, raws = [], [], []
            for cond in conditions:
                d = metrics_c.get(cond, {}).get(nm, {})
                if not isinstance(d, dict):
                    means.append(0); stds.append(0); raws.append([0])
                    continue
                kd = d.get(mk, {})
                if isinstance(kd, dict):
                    means.append(kd.get("mean", 0))
                    stds.append(kd.get("std", 0))
                    raws.append(kd.get("raw", [kd.get("mean", 0)]))
                else:
                    means.append(float(kd)); stds.append(0); raws.append([float(kd)])

            offset = (i - 1.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=LABELS[nm], color=PALETTE[nm],
                   alpha=0.80, capsize=3, edgecolor="#555555", linewidth=0.4,
                   error_kw={"linewidth": 1}, zorder=3)

            # Individual seed dots
            for j, raw_list in enumerate(raws):
                jitter_x = x[j] + offset + np.random.uniform(-0.04, 0.04, len(raw_list))
                ax.scatter(jitter_x, raw_list, color=PALETTE[nm],
                           s=18, alpha=0.6, zorder=4, edgecolors="none")

        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=9)
        ax.set_title(mk_title, fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Exp C: Jacobian Ablation — Full Jacobian is the Key Innovation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_jacobian_ablation.{ext}"))
    plt.close(fig)
    print(f"  → fig_jacobian_ablation saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Sensitivity — dual-axis with std bands
# ═══════════════════════════════════════════════════════════════════════
def fig_sensitivity():
    set_style()
    fbfm_sw  = metrics_d["fbfm_sweep"]
    rtc_sw   = metrics_d["rtc_sweep"]
    van_agg  = metrics_d["vanilla"]

    state_gws  = sorted(float(k) for k in fbfm_sw.keys())
    action_gws = sorted(float(k) for k in rtc_sw.keys())

    metric_pairs = [
        ("action_jitter",    "Action Jitter ↓"),
        ("position_mse",     "Position MSE ↓"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (mk, mk_title) in zip(axes, metric_pairs):
        # FBFM sweep
        fbfm_m = np.array([fbfm_sw[str(k)].get(mk, {}).get("mean", 0) for k in state_gws])
        fbfm_s = np.array([fbfm_sw[str(k)].get(mk, {}).get("std", 0)  for k in state_gws])
        ax.plot(state_gws, fbfm_m, "o-", color=PALETTE["fbfm"],
                lw=2.2, label="FBFM (sweep $\\beta_{\\text{state}}$)", zorder=3)
        ax.fill_between(state_gws, fbfm_m - fbfm_s, fbfm_m + fbfm_s,
                        color=PALETTE["fbfm"], alpha=0.15)

        # RTC sweep
        rtc_m = np.array([rtc_sw[str(k)].get(mk, {}).get("mean", 0) for k in action_gws])
        rtc_s = np.array([rtc_sw[str(k)].get(mk, {}).get("std", 0)  for k in action_gws])
        ax.plot(action_gws, rtc_m, "s--", color=PALETTE["rtc"],
                lw=2.0, label="RTC (sweep $\\beta_{\\text{action}}$)", zorder=3)
        ax.fill_between(action_gws, rtc_m - rtc_s, rtc_m + rtc_s,
                        color=PALETTE["rtc"], alpha=0.15)

        # Vanilla reference
        v_m = van_agg.get(mk, {}).get("mean", 0)
        v_s = van_agg.get(mk, {}).get("std", 0)
        ax.axhline(v_m, color=PALETTE["vanilla"], ls=":", lw=1.5, alpha=0.8)
        ax.axhspan(v_m - v_s, v_m + v_s, color=PALETTE["vanilla"], alpha=0.06)
        ax.text(0.02, 0.92, f"Vanilla: {v_m:.4f}",
                transform=ax.transAxes, fontsize=8, color=PALETTE["vanilla"], va="top")

        ax.set_xlabel("Guidance Weight $\\beta$")
        ax.set_title(mk_title, fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Exp D: FBFM is Robust to Guidance Weight; RTC is Sensitive",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_sensitivity.{ext}"))
    plt.close(fig)
    print(f"  → fig_sensitivity saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Action Detail Zoom — single axes overlay (v2)
# ═══════════════════════════════════════════════════════════════════════
def fig_action_detail():
    set_style()
    res = rollout["step_combined"]
    methods_show = ["vanilla", "rtc", "fbfm"]

    # Show time window around steady-state / chunk transitions (2.0–3.5 s)
    t0_idx, t1_idx = int(2.0 / DT), int(3.5 / DT)
    t_slice = time_ax[t0_idx:t1_idx]

    fig, ax = plt.subplots(figsize=(14, 3.8))

    lw_map = {"vanilla": 1.3, "rtc": 1.3, "fbfm": 2.0}

    for nm in methods_show:
        r = res[nm]
        a_slice = r["actions"][t0_idx:t1_idx, 0].numpy()
        kw = dict(color=PALETTE[nm], lw=lw_map[nm], alpha=0.88, zorder=3 if nm == "fbfm" else 2)
        if DASHES.get(nm):
            kw["dashes"] = DASHES[nm]
        ax.plot(t_slice, a_slice, label=LABELS[nm], **kw)

        # Annotate jitter value next to each curve
        jitter = np.var(np.diff(a_slice))
        # Find a good y-position: use median of the slice
        y_pos = np.median(a_slice)
        ax.annotate(f"Jitter={jitter:.4f}",
                    xy=(t_slice[-1], a_slice[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=7.5, color=PALETTE[nm], va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=PALETTE[nm], alpha=0.7, lw=0.5))

    # Mark chunk boundaries for FBFM (subtle vertical lines)
    for cb in res["fbfm"].get("chunk_boundaries", []):
        if t0_idx <= cb < t1_idx:
            ax.axvline(cb * DT, color=PALETTE["fbfm"], lw=0.5,
                       ls=":", alpha=0.35, zorder=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control $u$ (N)")
    ax.legend(fontsize=8.5, ncol=3, loc="upper left",
              handlelength=2.5, columnspacing=1.2, framealpha=0.9)
    ax.set_title("Zoomed Action Signal (2.0–3.5 s, combined mismatch)  —  "
                 "Sawtooth artifacts in Vanilla FM vs. smooth FBFM output",
                 fontsize=10, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(5))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_action_detail.{ext}"))
    plt.close(fig)
    print(f"  → fig_action_detail saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Guidance Weight Schedule k_p(τ)
# ═══════════════════════════════════════════════════════════════════════
def fig_guidance_profile():
    set_style()
    tau = np.linspace(0.01, 0.99, 500)
    # k_p = (τ² + (1-τ)²) / (τ(1-τ))
    k_unclipped = (tau**2 + (1 - tau)**2) / (tau * (1 - tau))

    betas = [1.0, 3.0, 5.0, 10.0]
    colors_beta = plt.cm.viridis(np.linspace(0.2, 0.85, len(betas)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: k_p(τ) for various β
    ax1.plot(tau, k_unclipped, "--", color="gray", lw=1.5, label="$k_p$ (unclipped)", alpha=0.5)
    for b, c in zip(betas, colors_beta):
        k_clipped = np.minimum(k_unclipped, b)
        ax1.plot(tau, k_clipped, lw=2, color=c, label=f"$\\beta = {b}$")
    ax1.set_xlabel("Flow time $\\tau$")
    ax1.set_ylabel("Guidance weight $k_p$")
    ax1.set_title("PiGDM Adaptive Weight Schedule\n$k_p(\\tau) = \\min\\left(\\beta,\\ "
                  "\\frac{\\tau^2 + (1-\\tau)^2}{\\tau(1-\\tau)}\\right)$", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 12)

    # Right: r²(τ) and 1/r²(τ)
    r2 = (1 - tau)**2 / (tau**2 + (1 - tau)**2)
    inv_r2 = 1.0 / r2
    ax2.plot(tau, r2, color=PALETTE["rtc"], lw=2, label="$r^2_{\\tau}$")
    ax2.plot(tau, inv_r2, color=PALETTE["fbfm"], lw=2, label="$1/r^2_{\\tau}$")
    ax2_r = ax2.twinx()
    # k_p = (1-τ)/τ · 1/r²
    one_over_tau = (1 - tau) / tau
    kp_raw = one_over_tau * inv_r2
    ax2_r.plot(tau, kp_raw, ":", color="gray", lw=1.5, alpha=0.5, label="$(1-\\tau)/\\tau \\cdot 1/r^2$")
    ax2_r.set_ylim(0, 15)
    ax2_r.tick_params(axis="y", colors="gray")
    ax2.set_xlabel("Flow time $\\tau$")
    ax2.set_title("PiGDM Signal Components\n"
                  r"$r^2_\tau = \frac{(1-\tau)^2}{\tau^2+(1-\tau)^2}$", fontsize=10)
    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_guidance_profile.{ext}"))
    plt.close(fig)
    print(f"  → fig_guidance_profile saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 8: Overview Grid (paper main figure)  – enriched v2
# ═══════════════════════════════════════════════════════════════════════
def fig_overview_grid():
    set_style()
    res_comb  = rollout["step_combined"]
    res_mass3 = rollout["step_mass3"]
    res_sin   = rollout["sin_combined"]
    res_imp   = rollout["impulse_nom"]
    methods_traj = ["pid", "vanilla", "rtc", "fbfm"]

    fig = plt.figure(figsize=(17, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.34)

    lw_map = {"pid": 1.2, "vanilla": 1.4, "rtc": 1.4, "fbfm_id": 1.4, "fbfm": 2.0}
    zo_map = {"pid": 1, "vanilla": 2, "rtc": 2, "fbfm_id": 2, "fbfm": 4}
    alpha_map = {"pid": 0.65, "vanilla": 0.80, "rtc": 0.80, "fbfm_id": 0.80, "fbfm": 0.95}

    def _plot_line(ax, t, y, nm, label=True):
        kw = dict(color=PALETTE[nm], lw=lw_map[nm], alpha=alpha_map[nm], zorder=zo_map[nm])
        if DASHES.get(nm):
            kw["dashes"] = DASHES[nm]
        if label:
            kw["label"] = LABELS[nm]
        ax.plot(t, y, **kw)

    # ── (A) Step / combined  — position ──────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    tgt = res_comb["pid"]["targets"].numpy()
    ax_a.plot(time_ax, tgt[:, 0], "--", color=PALETTE["target"],
              lw=1.0, alpha=0.6, label="Target")
    for nm in methods_traj:
        _plot_line(ax_a, time_ax, res_comb[nm]["states"][:, 0].numpy(), nm)
    ax_a.set_ylabel("Position $x$ (m)")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_title("(A) Step response, combined mismatch",
                   fontweight="bold", fontsize=9.5)
    ax_a.legend(fontsize=6.5, ncol=3, loc="lower right",
                handlelength=2.2, columnspacing=0.8)

    # ── (B) Sinusoidal / combined — position ─────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    tgt_sin = res_sin["pid"]["targets"].numpy()
    ax_b.plot(time_ax, tgt_sin[:, 0], "--", color=PALETTE["target"],
              lw=1.0, alpha=0.6)
    for nm in methods_traj:
        _plot_line(ax_b, time_ax, res_sin[nm]["states"][:, 0].numpy(), nm, label=False)
    ax_b.set_ylabel("Position $x$ (m)")
    ax_b.set_xlabel("Time (s)")
    ax_b.set_title("(B) Sinusoidal tracking, combined mismatch",
                   fontweight="bold", fontsize=9.5)

    # ── (C) Mismatch sweep — jitter lines ────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    conditions = ["nominal", "mass×1.5", "mass×2", "mass×3", "stiff×3", "combined"]
    xc = np.arange(len(conditions))
    markers_c = {"vanilla": "s", "rtc": "^", "fbfm": "o"}
    for nm in ["vanilla", "rtc", "fbfm"]:
        means = [mean_std(metrics_a[f"step_{c}"], "action_jitter", nm)[0] for c in conditions]
        stds  = [mean_std(metrics_a[f"step_{c}"], "action_jitter", nm)[1] for c in conditions]
        ax_c.errorbar(xc, means, yerr=stds, fmt=f"{markers_c[nm]}-",
                      color=PALETTE[nm], lw=1.6, capsize=3,
                      label=LABELS[nm], alpha=0.85, markersize=5)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(conditions, rotation=30, ha="right", fontsize=7)
    ax_c.set_ylabel("Action Jitter $\\downarrow$")
    ax_c.set_title("(C) Exp A — Mismatch sweep", fontweight="bold", fontsize=9.5)
    ax_c.legend(fontsize=7, loc="upper left")

    # ── (D) Step / mass×3 — action signal ────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    for nm in methods_traj:
        _plot_line(ax_d, time_ax, res_mass3[nm]["actions"][:, 0].numpy(), nm, label=False)
    ax_d.set_ylabel("Control $u$ (N)")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_title("(D) Action signal, mass$\\times$3",
                   fontweight="bold", fontsize=9.5)

    # ── (E) Tracking error envelope — step combined ──────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    for nm in ["vanilla", "rtc", "fbfm"]:
        err = np.abs(res_comb[nm]["states"][:, 0].numpy() - tgt[:, 0])
        ax_e.plot(time_ax, err, color=PALETTE[nm], lw=lw_map[nm],
                  alpha=alpha_map[nm], zorder=zo_map[nm], label=LABELS[nm])
    ax_e.set_ylabel("$|x - x^*|$ (m)")
    ax_e.set_xlabel("Time (s)")
    ax_e.set_title("(E) Tracking error, combined mismatch",
                   fontweight="bold", fontsize=9.5)
    ax_e.legend(fontsize=7)

    # ── (F) Disturbance recovery — impulse ───────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    tgt_i = res_imp["pid"]["targets"].numpy()
    ax_f.plot(time_ax, tgt_i[:, 0], "--", color=PALETTE["target"],
              lw=1.0, alpha=0.6)
    for nm in methods_traj:
        _plot_line(ax_f, time_ax, res_imp[nm]["states"][:, 0].numpy(), nm, label=False)
    ax_f.axvline(100 * DT, color=PALETTE["disturbance"], ls="-.",
                 lw=1.2, alpha=0.7, label="Impulse 5 N")
    ax_f.set_ylabel("Position $x$ (m)")
    ax_f.set_xlabel("Time (s)")
    ax_f.set_title("(F) Exp B — Impulse recovery, nominal",
                   fontweight="bold", fontsize=9.5)
    ax_f.legend(fontsize=7, loc="lower right")

    # ── (G) Ablation bar ─────────────────────────────────────────────
    ax_g = fig.add_subplot(gs[2, 0])
    abl_methods = ["vanilla", "rtc", "fbfm_id", "fbfm"]
    abl_short   = ["Vanilla", "RTC", "FBFM-ID", "FBFM"]
    abl_cond    = "step_combined"
    means_g = [mean_std(metrics_c[abl_cond], "action_jitter", nm)[0] for nm in abl_methods]
    stds_g  = [mean_std(metrics_c[abl_cond], "action_jitter", nm)[1] for nm in abl_methods]
    colors_g = [PALETTE[nm] for nm in abl_methods]
    bars_g = ax_g.bar(abl_short, means_g, yerr=stds_g, color=colors_g,
                      alpha=0.82, capsize=4, edgecolor="#555555",
                      linewidth=0.5, zorder=3, error_kw={"linewidth": 1})
    for bar, val in zip(bars_g, means_g):
        ax_g.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(stds_g) * 0.15,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax_g.set_ylabel("Action Jitter $\\downarrow$")
    ax_g.set_title("(G) Exp C — Jacobian ablation",
                   fontweight="bold", fontsize=9.5)

    # ── (H) Sensitivity ──────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[2, 1])
    fbfm_sw = metrics_d["fbfm_sweep"]
    rtc_sw  = metrics_d["rtc_sweep"]
    sgws = sorted(float(k) for k in fbfm_sw)
    agws = sorted(float(k) for k in rtc_sw)
    mk = "action_jitter"
    fm_h = np.array([fbfm_sw[str(k)].get(mk, {}).get("mean", 0) for k in sgws])
    fs_h = np.array([fbfm_sw[str(k)].get(mk, {}).get("std", 0)  for k in sgws])
    rm_h = np.array([rtc_sw[str(k)].get(mk, {}).get("mean", 0)  for k in agws])
    rs_h = np.array([rtc_sw[str(k)].get(mk, {}).get("std", 0)   for k in agws])
    ax_h.fill_between(sgws, fm_h - fs_h, fm_h + fs_h,
                      color=PALETTE["fbfm"], alpha=0.15)
    ax_h.plot(sgws, fm_h, "o-", color=PALETTE["fbfm"], lw=1.8,
              label="FBFM ($\\beta_{\\rm state}$)", markersize=4)
    ax_h.fill_between(agws, rm_h - rs_h, rm_h + rs_h,
                      color=PALETTE["rtc"], alpha=0.15)
    ax_h.plot(agws, rm_h, "s--", color=PALETTE["rtc"], lw=1.8,
              label="RTC ($\\beta_{\\rm action}$)", markersize=4)
    ax_h.set_xlabel("Guidance Weight $\\beta$")
    ax_h.set_ylabel("Action Jitter $\\downarrow$")
    ax_h.set_title("(H) Exp D — Weight sensitivity",
                   fontweight="bold", fontsize=9.5)
    ax_h.legend(fontsize=7)

    # ── (I) Summary metrics table ────────────────────────────────────
    ax_i = fig.add_subplot(gs[2, 2])
    ax_i.axis("off")
    cond_tbl = metrics_a["step_combined"]
    mk_list  = ["action_jitter", "position_mse", "action_mse_vs_pid", "control_tv"]
    mk_names = ["Jitter $\\downarrow$", "Pos MSE $\\downarrow$",
                "Act MSE $\\downarrow$", "Ctrl TV $\\downarrow$"]
    tbl_methods = ["vanilla", "rtc", "fbfm"]
    cell_text = []
    for mk_k, mk_n in zip(mk_list, mk_names):
        row = [mk_n]
        vals = []
        for nm in tbl_methods:
            m, s = mean_std(cond_tbl, mk_k, nm)
            vals.append((m, s, nm))
        best_val = min(v[0] for v in vals)
        for m, s, nm in vals:
            txt = f"{m:.4f}"
            if m == best_val:
                txt = f"\\textbf{{{m:.4f}}}"
            row.append(f"{m:.4f}")
        cell_text.append(row)

    col_labels = ["Metric"] + [LABELS[nm] for nm in tbl_methods]
    tbl = ax_i.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.5)
    # Style header row
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(fontweight="bold")
    # Highlight best per row
    for i, row_data in enumerate(cell_text):
        float_vals = []
        for v in row_data[1:]:
            try:
                float_vals.append(float(v))
            except ValueError:
                float_vals.append(1e9)
        best_idx = int(np.argmin(float_vals))
        tbl[i + 1, best_idx + 1].set_facecolor("#D5F5E3")

    ax_i.set_title("(I) Key metrics (step, combined)",
                   fontweight="bold", fontsize=9.5, pad=12)

    fig.suptitle("FBFM: Feedback-Enhanced Flow Matching — Experimental Overview",
                 fontsize=13, fontweight="bold", y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_overview_grid.{ext}"))
    plt.close(fig)
    print(f"  → fig_overview_grid saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 9: Jitter Heatmap across all conditions
# ═══════════════════════════════════════════════════════════════════════
def fig_jitter_heatmap():
    set_style()
    mismatch_conds = ["nominal", "mass×1.5", "mass×2", "mass×3", "stiff×3", "combined"]
    tgt_types      = ["step", "sinusoidal"]
    methods_hm     = ["vanilla", "rtc", "fbfm"]

    fig, axes = plt.subplots(1, len(methods_hm), figsize=(13, 4.5))
    vmax = 1.5

    for ax, nm in zip(axes, methods_hm):
        mat = np.zeros((len(tgt_types), len(mismatch_conds)))
        for ti, tt in enumerate(tgt_types):
            for ci, mc in enumerate(mismatch_conds):
                key = f"{tt}_{mc}"
                m, _ = mean_std(metrics_a.get(key, {}), "action_jitter", nm)
                mat[ti, ci] = m

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(mismatch_conds)))
        ax.set_xticklabels(mismatch_conds, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(tgt_types)))
        ax.set_yticklabels(tgt_types, fontsize=9)
        ax.set_title(f"{LABELS[nm]}", fontsize=10, fontweight="bold")

        # Cell annotations
        for ti in range(len(tgt_types)):
            for ci in range(len(mismatch_conds)):
                val = mat[ti, ci]
                tc  = "white" if val > 0.8 else "black"
                ax.text(ci, ti, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=tc, fontweight="bold")

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(im, cax=cbar_ax, label="Action Jitter ↓")

    fig.suptitle("Jitter Heatmap: All Conditions × All Methods",
                 fontsize=12, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_jitter_heatmap.{ext}"))
    plt.close(fig)
    print(f"  → fig_jitter_heatmap saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 10: MSE Heatmap — action & state MSE across all conditions
# ═══════════════════════════════════════════════════════════════════════
def fig_mse_heatmap():
    """Side-by-side heatmaps: Action MSE vs PID and State MSE vs PID.

    Layout: 2 rows (action MSE / state MSE) × 3 cols (vanilla / rtc / fbfm).
    Cells annotated with numeric values; best method per cell highlighted.
    """
    set_style()
    mismatch_conds = ["nominal", "mass×1.5", "mass×2", "mass×3", "stiff×3", "combined"]
    tgt_types      = ["step", "sinusoidal"]
    methods_hm     = ["vanilla", "rtc", "fbfm"]
    metric_pairs   = [
        ("action_mse_vs_pid", "Action MSE vs PID $\\downarrow$"),
        ("state_mse_vs_pid",  "State MSE vs PID $\\downarrow$"),
    ]

    fig, axes = plt.subplots(len(metric_pairs), len(methods_hm),
                             figsize=(14, 7), constrained_layout=True)

    for row, (mk, mk_title) in enumerate(metric_pairs):
        # Compute per-method matrices and global vmax for this metric
        mats = {}
        for nm in methods_hm:
            mat = np.zeros((len(tgt_types), len(mismatch_conds)))
            for ti, tt in enumerate(tgt_types):
                for ci, mc in enumerate(mismatch_conds):
                    key = f"{tt}_{mc}"
                    m, _ = mean_std(metrics_a.get(key, {}), mk, nm)
                    mat[ti, ci] = m
            mats[nm] = mat

        # Shared vmax = 95th percentile across all methods (avoids outlier blow-up)
        all_vals = np.concatenate([m.ravel() for m in mats.values()])
        vmax = float(np.percentile(all_vals[all_vals > 0], 95)) if all_vals.max() > 0 else 1.0

        for col, nm in enumerate(methods_hm):
            ax = axes[row, col]
            mat = mats[nm]

            # Best-method mask: True where this method is best (lowest) per cell
            best_mask = np.ones_like(mat, dtype=bool)
            for other in methods_hm:
                if other != nm:
                    best_mask &= (mat <= mats[other])

            im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                           vmin=0, vmax=vmax, interpolation="nearest")

            ax.set_xticks(range(len(mismatch_conds)))
            ax.set_xticklabels(mismatch_conds, rotation=35, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(tgt_types)))
            ax.set_yticklabels(tgt_types, fontsize=9)

            title_str = f"{LABELS[nm]}"
            if row == 0:
                ax.set_title(title_str, fontsize=10, fontweight="bold", pad=6)

            if col == 0:
                ax.set_ylabel(mk_title, fontsize=9, labelpad=4)

            # Cell annotations
            for ti in range(len(tgt_types)):
                for ci in range(len(mismatch_conds)):
                    val = mat[ti, ci]
                    tc  = "white" if val > vmax * 0.65 else "black"
                    weight = "bold" if best_mask[ti, ci] else "normal"
                    # Star marker for best cell
                    star = "★" if best_mask[ti, ci] else ""
                    ax.text(ci, ti, f"{val:.3f}{star}",
                            ha="center", va="center",
                            fontsize=7, color=tc, fontweight=weight)

        # Shared colorbar per row
        cbar = fig.colorbar(im, ax=axes[row, :], shrink=0.85, pad=0.02)
        cbar.set_label(mk_title, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("MSE Heatmap: Action & State MSE across All Conditions\n"
                 "(★ = best method per cell)",
                 fontsize=12, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_mse_heatmap.{ext}"))
    plt.close(fig)
    print("  → fig_mse_heatmap saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 11: MSE Sweep — line plots across mismatch severity
# ═══════════════════════════════════════════════════════════════════════
def fig_mse_sweep():
    """4-panel line chart: position MSE, action MSE, state MSE, jitter vs mismatch.

    Each panel shows mean±std bands for vanilla / rtc / fbfm.
    Separate rows for step and sinusoidal targets.
    """
    set_style()
    conditions  = ["nominal", "mass×1.5", "mass×2", "mass×3", "stiff×3", "combined"]
    x           = np.arange(len(conditions))
    methods_show = ["vanilla", "rtc", "fbfm"]
    markers      = {"vanilla": "s", "rtc": "^", "fbfm": "o"}
    tgt_types    = [("step", "Step Target"), ("sinusoidal", "Sinusoidal Target")]
    metric_cols  = [
        ("position_mse",      "Position MSE $\\downarrow$"),
        ("action_mse_vs_pid", "Action MSE vs PID $\\downarrow$"),
        ("state_mse_vs_pid",  "State MSE vs PID $\\downarrow$"),
        ("action_jitter",     "Action Jitter $\\downarrow$"),
    ]

    fig, axes = plt.subplots(len(tgt_types), len(metric_cols),
                             figsize=(18, 8), constrained_layout=True)

    for row, (tgt_key, tgt_label) in enumerate(tgt_types):
        for col, (mk, mk_title) in enumerate(metric_cols):
            ax = axes[row, col]

            for nm in methods_show:
                means, stds = [], []
                for cond in conditions:
                    key = f"{tgt_key}_{cond}"
                    m, s = mean_std(metrics_a.get(key, {}), mk, nm)
                    means.append(m)
                    stds.append(s)
                means = np.array(means)
                stds  = np.array(stds)
                ax.plot(x, means, f"{markers[nm]}-",
                        color=PALETTE[nm], lw=2.0, ms=5,
                        label=LABELS[nm], alpha=0.9, zorder=3)
                ax.fill_between(x, means - stds, means + stds,
                                color=PALETTE[nm], alpha=0.12)

            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=7.5)
            if col == 0:
                ax.set_ylabel(f"{tgt_label}\n{mk_title}", fontsize=9)
            else:
                ax.set_ylabel(mk_title, fontsize=9)
            if row == 0:
                ax.set_title(mk_title, fontsize=10, fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Exp A: MSE & Jitter across Mismatch Conditions (mean ± std, 5 seeds)",
                 fontsize=13, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_mse_sweep.{ext}"))
    plt.close(fig)
    print("  → fig_mse_sweep saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 12: Per-method MSE radar / spider chart
# ═══════════════════════════════════════════════════════════════════════
def fig_mse_radar():
    """Radar chart comparing vanilla / rtc / fbfm across 5 key metrics.

    Uses the step_combined condition (hardest mismatch) for a compact
    single-figure summary suitable for a paper overview.
    """
    set_style()
    cond = metrics_a.get("step_combined", {})
    methods_show = ["vanilla", "rtc", "fbfm"]
    metric_keys  = [
        "position_mse",
        "action_mse_vs_pid",
        "state_mse_vs_pid",
        "action_jitter",
        "control_tv",
    ]
    metric_labels = [
        "Pos MSE",
        "Action MSE",
        "State MSE",
        "Jitter",
        "Control TV",
    ]
    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Normalise each metric to [0, 1] using vanilla as reference
    raw = {}
    for nm in methods_show:
        raw[nm] = np.array([mean_std(cond, mk, nm)[0] for mk in metric_keys])
    ref = raw["vanilla"].copy()
    ref[ref == 0] = 1.0

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))

    for nm in methods_show:
        vals = (raw[nm] / ref).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=PALETTE[nm], lw=2.2,
                label=LABELS[nm], alpha=0.9, markersize=5)
        ax.fill(angles, vals, color=PALETTE[nm], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25])
    ax.set_yticklabels(["0.25×", "0.5×", "0.75×", "1×", "1.25×"], fontsize=7.5)
    ax.yaxis.set_tick_params(labelcolor="#888888")
    ax.axhline(1.0, color="#AAAAAA", lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Method Comparison (step, combined mismatch)\n"
                 "Normalised to Vanilla FM = 1.0",
                 fontsize=11, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_mse_radar.{ext}"))
    plt.close(fig)
    print("  → fig_mse_radar saved")


# ═══════════════════════════════════════════════════════════════════════
# Figure 13: Inference Delay Robustness (Exp E)
# ═══════════════════════════════════════════════════════════════════════
def fig_inference_delay():
    """4-panel figure for Exp E: Inference Delay Robustness.

    Layout (1 row × 4 panels):
      [A] Position MSE vs delay  (line + std band)
      [B] Action Jitter vs delay (line + std band)
      [C] Relative degradation   (normalised to delay=0)
      [D] Trajectory: delay=0 vs delay=4, FBFM vs Vanilla
    """
    if not metrics_e:
        print("  → fig_inference_delay: no exp_e data, skipping.")
        return

    delays_raw = sorted(metrics_e.get("fbfm", metrics_e.get("vanilla", {})).keys(),
                        key=lambda x: float(x))
    delays     = [float(d) for d in delays_raw]

    methods = [
        ("vanilla", "Vanilla FM"),
        ("rtc",     "RTC"),
        ("fbfm",    "FBFM (ours)"),
    ]

    def get_curve(method, metric):
        means, stds = [], []
        for dk in delays_raw:
            cell = metrics_e.get(method, {}).get(str(dk), {}).get(metric, {})
            if isinstance(cell, dict):
                means.append(float(cell.get("mean", 0)))
                stds.append(float(cell.get("std", 0)))
            else:
                means.append(float(cell))
                stds.append(0.0)
        return np.array(means), np.array(stds)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    ax_pos, ax_jit, ax_rel, ax_traj = axes

    method_style = {
        "vanilla": (PALETTE["vanilla"], "o--"),
        "rtc":     (PALETTE["rtc"],     "s-"),
        "fbfm":    (PALETTE["fbfm"],    "^-"),
    }

    # ── A: Position MSE ──────────────────────────────────────────────
    for mname, mlabel in methods:
        col, sty = method_style[mname]
        means, stds = get_curve(mname, "position_mse")
        ax_pos.plot(delays, means, sty, color=col, lw=2, ms=6,
                    label=mlabel, alpha=0.9)
        ax_pos.fill_between(delays, means - stds, means + stds,
                            color=col, alpha=0.13)
    ax_pos.set_xlabel("Inference Delay (steps)")
    ax_pos.set_ylabel("Position MSE")
    ax_pos.set_title("(A)  Position MSE  ↓", fontsize=10)
    ax_pos.set_xticks([int(d) for d in delays])
    ax_pos.legend(fontsize=8)

    # ── B: Action Jitter ─────────────────────────────────────────────
    for mname, mlabel in methods:
        col, sty = method_style[mname]
        means, stds = get_curve(mname, "action_jitter")
        ax_jit.plot(delays, means, sty, color=col, lw=2, ms=6,
                    label=mlabel, alpha=0.9)
        ax_jit.fill_between(delays, means - stds, means + stds,
                            color=col, alpha=0.13)
    ax_jit.set_xlabel("Inference Delay (steps)")
    ax_jit.set_ylabel("Action Jitter")
    ax_jit.set_title("(B)  Action Jitter  ↓", fontsize=10)
    ax_jit.set_xticks([int(d) for d in delays])
    ax_jit.legend(fontsize=8)

    # ── C: Relative Degradation (normalised to delay=0) ──────────────
    for mname, mlabel in methods:
        col, sty = method_style[mname]
        means, _ = get_curve(mname, "position_mse")
        baseline = means[0] if means[0] > 0 else 1.0
        rel = (means / baseline - 1.0) * 100  # % increase
        ax_rel.plot(delays, rel, sty, color=col, lw=2, ms=6,
                    label=mlabel, alpha=0.9)
    ax_rel.axhline(0, color="#888888", lw=0.8, ls=":")
    ax_rel.set_xlabel("Inference Delay (steps)")
    ax_rel.set_ylabel("Position MSE increase (%)")
    ax_rel.set_title("(C)  Relative Degradation", fontsize=10)
    ax_rel.set_xticks([int(d) for d in delays])
    ax_rel.legend(fontsize=8)

    # ── D: Representative trajectory comparison ───────────────────────
    # Re-run two short rollouts on the fly (delay=0 vs delay=4, FBFM vs Vanilla)
    try:
        from pre_test_2.physics_env import EnvConfig, generate_step_target
        from pre_test_2.train import load_trained_model
        from pre_test_2.run_experiments_final import (
            run_rollout, STATE_DIM, TOTAL_STEPS, get_cfg, CHECKPOINT
        )
        from pre_test_2.fbfm_processor import GuidanceMode

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_tmp, ns_tmp = load_trained_model(CHECKPOINT, device)
        env_cfg_e = EnvConfig(mass=2.0, damping=1.0, stiffness=0.3, dt=0.02)
        init_s    = torch.zeros(STATE_DIM)
        tgt       = generate_step_target(TOTAL_STEPS, target_pos=1.0)

        traj_pairs = [
            ("Vanilla  d=0", GuidanceMode.VANILLA, 0, PALETTE["vanilla"], "-"),
            ("Vanilla  d=4", GuidanceMode.VANILLA, 4, PALETTE["vanilla"], "--"),
            ("FBFM  d=0",    GuidanceMode.FBFM,    0, PALETTE["fbfm"],    "-"),
            ("FBFM  d=4",    GuidanceMode.FBFM,    4, PALETTE["fbfm"],    "--"),
        ]
        ts = np.arange(TOTAL_STEPS) * DT
        for lbl, mode, dly, col, ls in traj_pairs:
            torch.manual_seed(0)
            cfg_tmp = get_cfg(mode)
            res_tmp = run_rollout(model_tmp, ns_tmp, cfg_tmp, env_cfg_e,
                                  tgt, init_s, device, inference_delay=dly)
            ax_traj.plot(ts, res_tmp["states"][:, 0].numpy(),
                         color=col, ls=ls, lw=1.5, alpha=0.85, label=lbl)
        ax_traj.plot(ts, tgt[:, 0].numpy(), color=PALETTE["target"],
                     ls=":", lw=1.0, alpha=0.55, label="Target")
        ax_traj.set_xlabel("Time (s)")
        ax_traj.set_ylabel("Position")
        ax_traj.set_title("(D)  Trajectories: delay 0 vs 4", fontsize=10)
        ax_traj.legend(fontsize=7, ncol=2)
    except Exception as exc:
        ax_traj.text(0.5, 0.5, f"Trajectory unavailable\n{exc}",
                     ha="center", va="center", transform=ax_traj.transAxes, fontsize=7)
        ax_traj.set_title("(D)  Trajectories (skipped)", fontsize=10)

    fig.suptitle("Exp E  —  Inference Delay Robustness  (m=2, c=1, k=0.3, step target)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_inference_delay.{ext}"))
    plt.close(fig)
    print("  → fig_inference_delay saved")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[advanced_viz] Generating publication-quality figures …\n")
    fig_method_comparison()
    fig_mismatch_sweep()
    fig_disturbance_recovery()
    fig_jacobian_ablation()
    fig_sensitivity()
    fig_action_detail()
    fig_guidance_profile()
    fig_overview_grid()
    fig_jitter_heatmap()
    fig_mse_heatmap()
    fig_mse_sweep()
    fig_mse_radar()
    fig_inference_delay()
    print(f"\n[advanced_viz] All figures saved to: {OUT_DIR}/")
