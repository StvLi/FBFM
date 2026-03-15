"""
Visualization helpers for the FBFM pre-experiment.

All matplotlib plotting functions live here so that experiment logic
stays free of rendering details.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Visual identity ───────────────────────────────────────────────────
COLORS = {
    "pid":         "#2ecc71",
    "vanilla":     "#e74c3c",
    "rtc":         "#3498db",
    "fbfm_id":     "#f39c12",
    "fbfm":        "#9b59b6",
    "target":      "#95a5a6",
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


# ── Trajectory plot ───────────────────────────────────────────────────

def plot_trajectory(results, title, save_path, dt=0.02, disturbance_step=None):
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


# ── Error plot ────────────────────────────────────────────────────────

def plot_error(results, title, save_path, dt=0.02, disturbance_step=None):
    """Plot position and velocity error vs PID expert."""
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    first = next(iter(results.values()))
    T = first["states"].shape[0]
    time_ax = np.arange(T) * dt

    pid_result = results.get("pid")
    if pid_result is None:
        print(f"[Warning] No PID reference found for error plot: {save_path}")
        return

    for name, result in results.items():
        if name == "pid":
            continue
        c = COLORS.get(name, "#000")
        lbl = LABELS.get(name, name)
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


# ── Bar summary ───────────────────────────────────────────────────────

def plot_bar_summary(all_conditions, methods, metric_keys, title, save_path):
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


# ── Sensitivity sweep ────────────────────────────────────────────────

def plot_sensitivity(vanilla, fbfm_sweep, rtc_sweep,
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
