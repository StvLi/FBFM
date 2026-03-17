"""
viz/plot_results.py — Publication-quality visualization for FBFM experiments

All figures use:
    - English serif font (Times New Roman / DejaVu Serif)
    - STIX math font
    - High-saturation color palette
    - dpi=300 for paper figures, PDF + PNG export

Color scheme (consistent across all figures):
    FM   : #2166AC  (blue)
    RTC  : #D6604D  (red-orange)
    FBFM : #1A9641  (green)
    ref  : #000000  (black dashed)
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -----------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------

COLORS = {
    "fm":             "#2166AC",
    "rtc":            "#D6604D",
    "fbfm":           "#1A9641",
    "ref":            "#000000",
    "fbfm_ablated":   "#888888",
}

LABELS = {
    "fm":             "FM (baseline)",
    "rtc":            "RTC",
    "fbfm":           "FBFM (ours)",
    "fbfm_ablated":   r"FBFM ($k_p=0$, ablated)",
}


def set_style():
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":  "stix",
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def savefig(fig, path_no_ext: str, dpi: int = 300):
    """Save as both PNG and PDF."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    fig.savefig(path_no_ext + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    print(f"  Saved: {path_no_ext}.png / .pdf")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 1 — Exp-A: Tracking error bar chart (mismatch conditions)
# -----------------------------------------------------------------------

def plot_exp_a_bar(
    metrics_path: str = "results/exp_a_mismatch/metrics.json",
    save_dir: str = "results/exp_a_mismatch",
):
    set_style()
    with open(metrics_path) as f:
        metrics = json.load(f)

    ref_names  = list(metrics.keys())
    conditions = list(metrics[ref_names[0]].keys())
    algos      = ["fm", "rtc", "fbfm"]

    fig, axes = plt.subplots(1, len(ref_names),
                             figsize=(5 * len(ref_names), 4.5), sharey=False)
    if len(ref_names) == 1:
        axes = [axes]

    x     = np.arange(len(conditions))
    width = 0.25

    for ax, ref_name in zip(axes, ref_names):
        for i, algo in enumerate(algos):
            rmses = [metrics[ref_name][cond][algo]["rmse"] for cond in conditions]
            ax.bar(x + (i - 1) * width, rmses, width,
                   label=LABELS[algo], color=COLORS[algo],
                   alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel("Tracking RMSE (m)")
        ax.set_title(f"Exp-A: Parameter Mismatch\n({ref_name} reference)")
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, "fig_mismatch_bar"))


# -----------------------------------------------------------------------
# Fig 2 — Exp-A: Trajectory overlay (one mismatch condition)
# -----------------------------------------------------------------------

def plot_exp_a_traj(
    results_dir: str = "results/exp_a_mismatch",
    condition: str = "mass×2",
    save_dir: str = "results/exp_a_mismatch",
):
    set_style()
    ref_names = ["step", "sinusoidal"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    for ax, ref_name in zip(axes, ref_names):
        fname = os.path.join(
            results_dir,
            f"traj_{ref_name}_{condition.replace('x', 'x')}.npz"
        )
        if not os.path.exists(fname):
            ax.set_title(f"{ref_name} — data not found")
            continue

        d = np.load(fname)
        t = np.arange(len(d["ref_seq"])) * 0.05

        ax.plot(t, d["ref_seq"], "--", color=COLORS["ref"],
                label=r"$x_\mathrm{ref}$", linewidth=1.5, zorder=5)
        for algo in ["fm", "rtc", "fbfm"]:
            key = f"{algo}_xs_true"
            if key in d:
                ax.plot(t, d[key], color=COLORS[algo],
                        label=LABELS[algo], linewidth=1.5, alpha=0.9)

        ax.set_ylabel(r"Position $x$ (m)")
        ax.set_xlabel(r"Time (s)")
        ax.set_title(f"Exp-A: {condition} — {ref_name} reference")
        ax.legend(framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, f"fig_traj_{condition.replace('x', 'x')}"))


# -----------------------------------------------------------------------
# Fig 3 — Exp-B: In-chunk disturbance response
# -----------------------------------------------------------------------

def plot_exp_b(
    results_dir: str = "results/exp_b_disturbance",
    save_dir: str = "results/exp_b_disturbance",
):
    set_style()
    conditions = ["nominal", "mass×2"]
    ref_names  = ["step", "sinusoidal"]

    for ref_name in ref_names:
        fig, axes = plt.subplots(1, len(conditions),
                                 figsize=(6 * len(conditions), 4.5))
        if len(conditions) == 1:
            axes = [axes]

        for ax, cond in zip(axes, conditions):
            fname = os.path.join(
                results_dir,
                f"traj_{ref_name}_{cond.replace('x', 'x')}.npz"
            )
            if not os.path.exists(fname):
                ax.set_title(f"{cond} — data not found")
                continue

            d = np.load(fname)
            t = np.arange(len(d["ref_seq"])) * 0.05

            ax.plot(t, d["ref_seq"], "--", color=COLORS["ref"],
                    label=r"$x_\mathrm{ref}$", linewidth=1.5, zorder=5)
            for algo in ["fm", "rtc", "fbfm"]:
                key = f"{algo}_xs_true"
                if key in d:
                    ax.plot(t, d[key], color=COLORS[algo],
                            label=LABELS[algo], linewidth=1.5, alpha=0.9)

            # Mark impulse steps
            if "impulse_steps" in d:
                for step in d["impulse_steps"][:5]:
                    ax.axvline(step * 0.05, color="purple", alpha=0.4,
                               linewidth=1.0, linestyle=":",
                               label="Impulse" if step == d["impulse_steps"][0] else "")

            ax.set_ylabel(r"Position $x$ (m)")
            ax.set_xlabel(r"Time (s)")
            ax.set_title(f"Exp-B: Impulse disturbance\n{cond}, {ref_name} ref")
            ax.legend(framealpha=0.9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        savefig(fig, os.path.join(save_dir, f"fig_disturbance_{ref_name}"))


# -----------------------------------------------------------------------
# Fig 4 — Exp-C: Ablation bar chart
# -----------------------------------------------------------------------

def plot_exp_c_bar(
    metrics_path: str = "results/exp_c_ablation/metrics.json",
    save_dir: str = "results/exp_c_ablation",
):
    set_style()
    with open(metrics_path) as f:
        metrics = json.load(f)

    ref_names  = list(metrics.keys())
    conditions = list(metrics[ref_names[0]].keys())
    variants   = ["fbfm_full", "fbfm_no_feedback", "rtc"]
    var_labels = {
        "fbfm_full":        LABELS["fbfm"],
        "fbfm_no_feedback": LABELS["fbfm_ablated"],
        "rtc":              LABELS["rtc"],
    }
    var_colors = {
        "fbfm_full":        COLORS["fbfm"],
        "fbfm_no_feedback": COLORS["fbfm_ablated"],
        "rtc":              COLORS["rtc"],
    }

    fig, axes = plt.subplots(1, len(ref_names),
                             figsize=(5 * len(ref_names), 4.5))
    if len(ref_names) == 1:
        axes = [axes]

    x     = np.arange(len(conditions))
    width = 0.25

    for ax, ref_name in zip(axes, ref_names):
        for i, var in enumerate(variants):
            rmses = [metrics[ref_name][cond][var]["rmse"] for cond in conditions]
            ax.bar(x + (i - 1) * width, rmses, width,
                   label=var_labels[var], color=var_colors[var],
                   alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel("Tracking RMSE (m)")
        ax.set_title(f"Exp-C: Ablation Study\n({ref_name} reference)")
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, "fig_ablation_bar"))


# -----------------------------------------------------------------------
# Fig 5 — Exp-D: Sensitivity heatmap (n_inner vs condition)
# -----------------------------------------------------------------------

def plot_exp_d_heatmap(
    metrics_path: str = "results/exp_d_sensitivity/metrics.json",
    save_dir: str = "results/exp_d_sensitivity",
):
    set_style()
    with open(metrics_path) as f:
        metrics = json.load(f)

    ref_names  = list(metrics.keys())
    conditions = list(metrics[ref_names[0]].keys())
    n_inners   = [1, 2, 4, 8, 16]

    fig, axes = plt.subplots(1, len(ref_names),
                             figsize=(5 * len(ref_names), 4))
    if len(ref_names) == 1:
        axes = [axes]

    for ax, ref_name in zip(axes, ref_names):
        # rows=conditions, cols=n_inner
        mat = np.zeros((len(conditions), len(n_inners)))
        for ci, cond in enumerate(conditions):
            for ni, n in enumerate(n_inners):
                key = f"n_inner={n}"
                mat[ci, ni] = metrics[ref_name][cond][key]["rmse"]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=mat.max())
        plt.colorbar(im, ax=ax, label="RMSE (m)")

        ax.set_xticks(range(len(n_inners)))
        ax.set_xticklabels([str(n) for n in n_inners])
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.set_xlabel(r"$n_\mathrm{inner}$ (denoising steps per block)")
        ax.set_title(f"Exp-D: Feedback Frequency Sensitivity\n({ref_name} reference)")

        for ci in range(len(conditions)):
            for ni in range(len(n_inners)):
                ax.text(ni, ci, f"{mat[ci, ni]:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if mat[ci, ni] > mat.max() * 0.6 else "black")

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, "fig_sensitivity_heatmap"))


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating all experiment figures...")

    if os.path.exists("results/exp_a_mismatch/metrics.json"):
        print("\n[Exp-A] Mismatch bar chart + trajectory plots")
        plot_exp_a_bar()
        for cond in ["mass×2", "mass×3", "stiff×3"]:
            plot_exp_a_traj(condition=cond)
    else:
        print("[Exp-A] metrics.json not found — run experiments/run_all.py first")

    if os.path.exists("results/exp_b_disturbance/metrics.json"):
        print("\n[Exp-B] Disturbance response plots")
        plot_exp_b()
    else:
        print("[Exp-B] metrics.json not found — run experiments/run_all.py first")

    if os.path.exists("results/exp_c_ablation/metrics.json"):
        print("\n[Exp-C] Ablation bar chart")
        plot_exp_c_bar()
    else:
        print("[Exp-C] metrics.json not found — run experiments/run_all.py first")

    if os.path.exists("results/exp_d_sensitivity/metrics.json"):
        print("\n[Exp-D] Sensitivity heatmap")
        plot_exp_d_heatmap()
    else:
        print("[Exp-D] metrics.json not found — run experiments/run_all.py first")

    print("\nDone.")
