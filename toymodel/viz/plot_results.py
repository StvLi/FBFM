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

TOYMODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOYMODEL_ROOT)

# -----------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------

COLORS = {
    "fm":             "#2166AC",
    "rtc":            "#D6604D",
    "fbfm":           "#1A9641",
    "ref":            "#000000",
    "fbfm_ablated":   "#888888",
    "noskip":         "#7570B3",   # purple — no-delay baseline
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


def condition_to_stem(condition: str) -> str:
    """Convert display condition labels to ASCII-safe filename stems."""
    return condition.replace("×", "x")


def add_chunk_boundaries(ax, T, dt, s_chunk=5):
    """Add vertical dashed lines at chunk boundaries."""
    for step in range(s_chunk, T, s_chunk):
        ax.axvline(step * dt, color="#999999", alpha=0.35,
                   linewidth=0.7, linestyle="--")


# -----------------------------------------------------------------------
# Fig 1 — Exp-A: Tracking error bar chart (mismatch conditions)
# -----------------------------------------------------------------------

def plot_exp_a_bar(
    metrics_path: str = None,
    save_dir: str = None,
):
    if metrics_path is None:
        metrics_path = os.path.join(TOYMODEL_ROOT, "results", "exp_a_mismatch", "metrics.json")
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results", "exp_a_mismatch")
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
    results_dir: str = None,
    condition: str = "mass×2",
    save_dir: str = None,
):
    set_style()
    if results_dir is None:
        results_dir = os.path.join(TOYMODEL_ROOT, "results", "exp_a_mismatch")
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results", "exp_a_mismatch")

    ref_names = ["step", "sinusoidal"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    for ax, ref_name in zip(axes, ref_names):
        fname = os.path.join(
            results_dir,
            f"traj_{ref_name}_{condition_to_stem(condition)}.npz"
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
        add_chunk_boundaries(ax, len(d["ref_seq"]), 0.05)

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, f"fig_traj_{condition_to_stem(condition)}"))


# -----------------------------------------------------------------------
# Fig 3 — Exp-B: In-chunk disturbance response
# -----------------------------------------------------------------------

def plot_exp_b(
    results_dir: str = None,
    save_dir: str = None,
):
    set_style()
    if results_dir is None:
        results_dir = os.path.join(TOYMODEL_ROOT, "results", "exp_b_disturbance")
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results", "exp_b_disturbance")

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
                f"traj_{ref_name}_{condition_to_stem(cond)}.npz"
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
            add_chunk_boundaries(ax, len(d["ref_seq"]), 0.05)

        plt.tight_layout()
        savefig(fig, os.path.join(save_dir, f"fig_disturbance_{ref_name}"))


# -----------------------------------------------------------------------
# Fig 4 — Exp-C: Ablation bar chart
# -----------------------------------------------------------------------

def plot_exp_c_bar(
    metrics_path: str = None,
    save_dir: str = None,
):
    set_style()
    if metrics_path is None:
        metrics_path = os.path.join(TOYMODEL_ROOT, "results", "exp_c_ablation", "metrics.json")
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results", "exp_c_ablation")

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
    metrics_path: str = None,
    save_dir: str = None,
):
    set_style()
    if metrics_path is None:
        metrics_path = os.path.join(TOYMODEL_ROOT, "results", "exp_d_sensitivity", "metrics.json")
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results", "exp_d_sensitivity")

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
# Diagnostic helpers — lazy-loaded model & rollout utilities
# -----------------------------------------------------------------------

_DIAG_CACHE = {}  # singleton cache for policy / token_stats


def _diag_load_policy():
    """Load policy once and cache it."""
    if "policy" not in _DIAG_CACHE:
        import torch
        from experiments.runner import load_policy
        ckpt = os.path.join(TOYMODEL_ROOT, "checkpoints", "fm_best.pt")
        policy, token_stats, cfg = load_policy(ckpt, device="cpu")
        _DIAG_CACHE["policy"] = policy
        _DIAG_CACHE["token_stats"] = token_stats
        _DIAG_CACHE["cfg"] = cfg
    return _DIAG_CACHE["policy"], _DIAG_CACHE["token_stats"]


def _diag_run_noskip(m, ref, s=5):
    """FM rollout without delay (re-infer every s steps from current state)."""
    import torch
    from sim.msd_env import MSDEnv
    from model.inference import sample_fm
    policy, ts = _diag_load_policy()
    T = len(ref)
    env = MSDEnv(m=m, k=2.0, c=0.5, dt=0.05, seed=0); env.reset()
    obs = np.append(env.state.copy(), ref[0]); t = 0; xs = []
    while t < T:
        torch.manual_seed(t)
        chunk = sample_fm(policy, obs, n_steps=20, device="cpu", token_stats=ts)
        for h in range(s):
            if t >= T:
                break
            st, _, _, _ = env.step(float(chunk[h, 2]), add_obs_noise=False)
            xs.append(st[0]); t += 1
        obs = np.append(st, ref[min(t, T - 1)])
    return np.array(xs)


def _diag_run_algo(algo, m, ref, beta=3.0):
    """Run one of fm / rtc / fbfm with skip-d (current implementation)."""
    import torch
    from sim.msd_env import MSDEnv
    from algo.fm import rollout_fm
    from algo.rtc import rollout_rtc
    from algo.fbfm import rollout_fbfm
    policy, ts = _diag_load_policy()
    np.random.seed(0); torch.manual_seed(0)
    env = MSDEnv(m=m, k=2.0, c=0.5, dt=0.05, seed=0); env.reset()
    if algo == "fm":
        return rollout_fm(policy, env, ref, s_chunk=5, n_steps=20,
                          token_stats=ts, device="cpu")
    elif algo == "rtc":
        return rollout_rtc(policy, env, ref, s_chunk=5, n_steps=20, beta=beta,
                           token_stats=ts, device="cpu")
    else:
        return rollout_fbfm(policy, env, ref, n_steps=20, n_inner=4, beta=beta,
                            token_stats=ts, device="cpu")


def _rmse(ref, xs):
    n = min(len(ref), len(xs))
    return np.sqrt(((ref[:n] - xs[:n]) ** 2).mean())


# -----------------------------------------------------------------------
# Diag-Fig 1 — Strategy Comparison (no-skip baseline vs skip-d × 3 algos)
# -----------------------------------------------------------------------

def plot_diag_strategy_comparison(save_dir: str = None):
    """4-panel plot: nominal/mass×2 × step/sin, each with no-skip + 3 algos."""
    set_style()
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results")

    T = 200; dt = 0.05
    t_arr = np.arange(T) * dt
    ref_step = np.ones(T, dtype=np.float32)
    ref_sin = np.sin(2 * np.pi * 0.15 * np.arange(T) * dt).astype(np.float32)

    configs = [
        (1.0, ref_step, "Nominal — Step ref"),
        (1.0, ref_sin,  "Nominal — Sinusoidal ref"),
        (2.0, ref_step, r"mass$\times$2 — Step ref"),
        (2.0, ref_sin,  r"mass$\times$2 — Sinusoidal ref"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for idx, (m, ref, title) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        # no-skip baseline
        xs_ns = _diag_run_noskip(m, ref)
        ax.plot(t_arr[:len(xs_ns)], xs_ns, color=COLORS["noskip"], lw=1.2,
                alpha=0.9, label=f"No-skip (no delay)  RMSE={_rmse(ref, xs_ns):.3f}")
        # three algos with skip-d
        for algo in ["fm", "rtc", "fbfm"]:
            res = _diag_run_algo(algo, m, ref)
            r = _rmse(ref, res["xs_true"])
            ax.plot(t_arr, res["xs_true"], color=COLORS[algo], lw=0.9,
                    alpha=0.8, label=f"{algo.upper()} (skip-d)  RMSE={r:.3f}")
        ax.plot(t_arr, ref, "--", color=COLORS["ref"], lw=1.5, alpha=0.5,
                label=r"$x_{\mathrm{ref}}$")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        ax.grid(True, alpha=0.25); ax.set_ylim(-3, 4.5)
        if idx // 2 == 1:
            ax.set_xlabel("Time (s)")
        if idx % 2 == 0:
            ax.set_ylabel(r"Position $x$")

    fig.suptitle("Rollout Strategy: No-Skip Baseline vs Skip-d "
                 "(FM / RTC / FBFM)", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, os.path.join(save_dir, "diag_fig1_strategy_comparison"))


# -----------------------------------------------------------------------
# Diag-Fig 2 — Chunk Boundary Diagnostics (trajectory + action + mismatch)
# -----------------------------------------------------------------------

def plot_diag_chunk_boundary(save_dir: str = None):
    """4-panel vertical: trajectory, action, action-jump, state-mismatch."""
    import torch
    from sim.msd_env import MSDEnv
    from model.inference import sample_fm
    set_style()
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results")

    policy, ts = _diag_load_policy()
    T = 200; H = 16; d = 4; dt = 0.05
    t_arr = np.arange(T) * dt
    ref = np.ones(T, dtype=np.float32)

    env = MSDEnv(m=1.0, k=2.0, c=0.5, dt=0.05, seed=0); env.reset()
    obs = np.append(env.state.copy(), ref[0])
    torch.manual_seed(0)
    chunk = sample_fm(policy, obs, n_steps=20, device="cpu", token_stats=ts)
    ptr = 0; tg = 0

    xs = []; us = []; a_jumps = []; mm_x = []; mm_xd = []; cy_t = []

    for _ in range(40):
        if tg >= T:
            break
        u = float(chunk[ptr, 2]) if ptr < H else 0.0
        last_u = u; ptr += 1
        st, _, _, _ = env.step(u, add_obs_noise=False)
        xs.append(st[0]); us.append(u); tg += 1
        obs_inf = np.append(st, ref[min(tg, T - 1)])
        torch.manual_seed(tg)
        nc = sample_fm(policy, obs_inf, n_steps=20, device="cpu", token_stats=ts)
        for _ in range(d):
            if tg >= T:
                break
            u = float(chunk[ptr, 2]) if ptr < H else 0.0
            last_u = u; ptr += 1
            st, _, _, _ = env.step(u, add_obs_noise=False)
            xs.append(st[0]); us.append(u); tg += 1
        pred = nc[d - 1, :2]
        a_jumps.append(float(nc[d, 2]) - last_u)
        mm_x.append(st[0] - pred[0])
        mm_xd.append(st[1] - pred[1])
        cy_t.append(tg * dt)
        chunk = nc; ptr = d

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1.2, 1.2]})

    axes[0].plot(t_arr[:len(xs)], xs, COLORS["fm"], lw=1.0)
    axes[0].plot(t_arr, ref, "--", color=COLORS["ref"], lw=1.5, alpha=0.5,
                 label=r"$x_{\mathrm{ref}}$")
    for ct in cy_t:
        axes[0].axvline(ct, color="gray", alpha=0.1, lw=0.4)
    axes[0].set_ylabel(r"Position $x$")
    axes[0].set_title("FM Rollout (Nominal, Step ref)")
    axes[0].legend(loc="upper right"); axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_arr[:len(us)], us, "#9467BD", lw=0.6, alpha=0.8)
    axes[1].set_ylabel("Action $u$ (N)")
    axes[1].set_title("Action Sequence"); axes[1].grid(True, alpha=0.25)

    axes[2].bar(cy_t, a_jumps, width=0.12, color=COLORS["rtc"], alpha=0.75)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel(r"$\Delta u$ (N)")
    axes[2].set_title("Action Discontinuity at Chunk Boundary")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(cy_t, mm_x, "o-", ms=3, lw=0.9, color=COLORS["fm"],
                 label=r"$\Delta x$")
    axes[3].plot(cy_t, mm_xd, "s-", ms=3, lw=0.9, color=COLORS["rtc"],
                 label=r"$\Delta \dot{x}$")
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_ylabel("Mismatch"); axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Blind-End State Mismatch (actual $-$ predicted)")
    axes[3].legend(); axes[3].grid(True, alpha=0.25)

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, "diag_fig2_chunk_boundary"))


# -----------------------------------------------------------------------
# Diag-Fig 3 — Old vs New Chunk Action Mismatch
# -----------------------------------------------------------------------

def plot_diag_action_mismatch(save_dir: str = None):
    """2-panel: mean blind-period action (old vs new) + max |Δu| per cycle."""
    import torch
    from sim.msd_env import MSDEnv
    from model.inference import sample_fm
    set_style()
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results")

    policy, ts = _diag_load_policy()
    T = 200; H = 16; d = 4
    ref = np.ones(T, dtype=np.float32)

    env = MSDEnv(m=1.0, k=2.0, c=0.5, dt=0.05, seed=0); env.reset()
    obs = np.append(env.state.copy(), ref[0])
    torch.manual_seed(0)
    chunk = sample_fm(policy, obs, n_steps=20, device="cpu", token_stats=ts)
    ptr = 0; tg = 0
    old_a = []; new_a = []; diff_a = []; labels = []

    for cyc in range(12):
        if tg >= T:
            break
        u = float(chunk[ptr, 2]); ptr += 1
        st, _, _, _ = env.step(u, add_obs_noise=False); tg += 1
        torch.manual_seed(tg)
        nc = sample_fm(policy, np.append(st, ref[min(tg, T - 1)]),
                        n_steps=20, device="cpu", token_stats=ts)
        oa = chunk[ptr:ptr + d, 2].copy()
        na = nc[0:d, 2].copy()
        old_a.append(oa); new_a.append(na); diff_a.append(oa - na)
        labels.append(f"C{cyc}")
        for _ in range(d):
            if tg >= T:
                break
            u = float(chunk[ptr, 2]); ptr += 1
            st, _, _, _ = env.step(u, add_obs_noise=False); tg += 1
        chunk = nc; ptr = d

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x_pos = np.arange(len(old_a)); w = 0.35
    old_m = [a.mean() for a in old_a]
    new_m = [a.mean() for a in new_a]
    axes[0].bar(x_pos - w / 2, old_m, w, alpha=0.75, color=COLORS["fm"],
                label="Old chunk (executed)")
    axes[0].bar(x_pos + w / 2, new_m, w, alpha=0.75, color=COLORS["rtc"],
                label="New chunk (assumed)")
    axes[0].set_ylabel("Mean action $u$ (N)")
    axes[0].set_title("Blind-Period Mean Action: Old vs New Chunk")
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(labels)
    axes[0].legend(); axes[0].grid(True, alpha=0.25, axis="y")

    max_d = [np.abs(d_).max() for d_ in diff_a]
    axes[1].bar(x_pos, max_d, color=COLORS["fbfm"], alpha=0.75)
    axes[1].set_ylabel(r"Max $|\Delta u|$ (N)"); axes[1].set_xlabel("Cycle")
    axes[1].set_title("Maximum Action Mismatch During Blind Period")
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(labels)
    axes[1].grid(True, alpha=0.25, axis="y")
    for i, v in enumerate(max_d):
        axes[1].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, "diag_fig3_action_mismatch"))


# -----------------------------------------------------------------------
# Diag-Fig 4 — Single Chunk Self-Consistency
# -----------------------------------------------------------------------

def plot_diag_single_chunk(save_dir: str = None):
    """4-panel: model prediction vs real dynamics for various obs / mass."""
    import torch
    from sim.msd_env import MSDEnv
    from model.inference import sample_fm
    set_style()
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results")

    policy, ts = _diag_load_policy()
    cases = [
        ([0.0, 0.0, 1.0], "In-dist: obs=[0, 0, 1]", 1.0),
        ([1.0, 0.0, 1.0], "In-dist: obs=[1, 0, 1]", 1.0),
        ([0.5, 3.0, 1.0], "OOD high vel: obs=[0.5, 3, 1]", 1.0),
        ([0.0, 0.0, 1.0], r"mass$\times$2: obs=[0, 0, 1]", 2.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, (obs_list, title, m_val) in enumerate(cases):
        ax = axes[idx // 2][idx % 2]
        obs = np.array(obs_list, dtype=np.float32)
        torch.manual_seed(0)
        chunk = sample_fm(policy, obs, n_steps=20, device="cpu", token_stats=ts)
        env_c = MSDEnv(m=m_val, k=2.0, c=0.5, dt=0.05)
        env_c.state = np.array([obs[0], obs[1]])
        real_x = [obs[0]]; pred_x = [obs[0]]; errs = []
        for h in range(16):
            st, _, _, _ = env_c.step(float(chunk[h, 2]), add_obs_noise=False)
            real_x.append(st[0]); pred_x.append(chunk[h, 0])
            errs.append(abs(st[0] - chunk[h, 0]))
        h_arr = np.arange(17)
        ax.plot(h_arr, pred_x, "o-", ms=3, lw=1.0, color=COLORS["rtc"],
                label="Model predicted")
        ax.plot(h_arr, real_x, "s-", ms=3, lw=1.0, color=COLORS["fm"],
                label=f"Real (m={m_val})")
        ax.axhline(obs[2], color="k", ls="--", lw=1, alpha=0.5,
                   label=r"$x_{\mathrm{ref}}$")
        ax.fill_between(range(1, 17),
                        [p - e for p, e in zip(pred_x[1:], errs)],
                        [p + e for p, e in zip(pred_x[1:], errs)],
                        alpha=0.15, color="red")
        ax.set_title(f"{title}  (mean err={np.mean(errs):.3f})")
        ax.set_xlabel("Step $h$"); ax.set_ylabel("$x$")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    fig.suptitle("Single Chunk: Model Prediction vs Real Dynamics",
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, os.path.join(save_dir, "diag_fig4_single_chunk"))


# -----------------------------------------------------------------------
# Diag-Fig 5 — Training Data OOD Analysis
# -----------------------------------------------------------------------

def plot_diag_ood_analysis(save_dir: str = None):
    """3-panel: obs scatter, xdot histogram, xdot per chunk position."""
    import torch
    from sim.msd_env import MSDEnv
    from model.inference import sample_fm
    set_style()
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results")

    policy, ts = _diag_load_policy()
    T = 200; H = 16; d = 4
    ref = np.ones(T, dtype=np.float32)

    data = np.load(os.path.join(TOYMODEL_ROOT, "data", "expert_dataset.npz"))
    states = data["states"]  # (N, H+1, 2)
    obs_x = states[:, 0, 0]; obs_xd = states[:, 0, 1]

    # collect rollout trigger obs
    env = MSDEnv(m=1.0, k=2.0, c=0.5, dt=0.05, seed=0); env.reset()
    obs = np.append(env.state.copy(), ref[0])
    torch.manual_seed(0)
    chunk = sample_fm(policy, obs, n_steps=20, device="cpu", token_stats=ts)
    ptr = 0; tg = 0; r_obs = []
    for _ in range(40):
        if tg >= T:
            break
        u = float(chunk[ptr, 2]) if ptr < H else 0.0; ptr += 1
        st, _, _, _ = env.step(u, add_obs_noise=False); tg += 1
        r_obs.append([st[0], st[1]])
        torch.manual_seed(tg)
        nc = sample_fm(policy, np.append(st, ref[min(tg, T - 1)]),
                        n_steps=20, device="cpu", token_stats=ts)
        for _ in range(d):
            if tg >= T:
                break
            u = float(chunk[ptr, 2]) if ptr < H else 0.0; ptr += 1
            st, _, _, _ = env.step(u, add_obs_noise=False); tg += 1
        chunk = nc; ptr = d
    r_obs = np.array(r_obs)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].scatter(obs_x, obs_xd, s=1, alpha=0.15, color=COLORS["fm"],
                    label="Training obs")
    axes[0].scatter(r_obs[:, 0], r_obs[:, 1], s=15, alpha=0.8,
                    color=COLORS["rtc"], marker="x", label="Rollout trigger obs")
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel(r"$\dot{x}$")
    axes[0].set_title("Obs Distribution: Training vs Rollout")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.25)

    axes[1].hist(obs_xd, bins=60, density=True, alpha=0.6, color=COLORS["fm"],
                 label=r"Training $\dot{x}$")
    axes[1].hist(r_obs[:, 1], bins=20, density=True, alpha=0.6,
                 color=COLORS["rtc"], label=r"Rollout $\dot{x}$")
    axes[1].set_xlabel(r"$\dot{x}$"); axes[1].set_ylabel("Density")
    axes[1].set_title("Velocity Distribution")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.25)

    positions = list(range(17))
    means = [states[:, h, 1].mean() for h in positions]
    stds = [states[:, h, 1].std() for h in positions]
    maxs = [np.abs(states[:, h, 1]).max() for h in positions]
    axes[2].fill_between(positions,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.3, color=COLORS["fm"], label=r"$\pm 1\sigma$")
    axes[2].plot(positions, means, "o-", ms=3, color=COLORS["fm"], label="Mean")
    axes[2].plot(positions, maxs, "s--", ms=3, color=COLORS["rtc"], alpha=0.7,
                 label=r"Max $|\dot{x}|$")
    axes[2].set_xlabel("Position $h$"); axes[2].set_ylabel(r"$\dot{x}$")
    axes[2].set_title(r"$\dot{x}$ at Each Chunk Position")
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, "diag_fig5_ood_analysis"))


# -----------------------------------------------------------------------
# Diag-Fig 6 — RMSE Summary Bar Chart
# -----------------------------------------------------------------------

def plot_diag_rmse_summary(save_dir: str = None):
    """Grouped bar: no-skip / FM / RTC / FBFM across 4 scenarios."""
    set_style()
    if save_dir is None:
        save_dir = os.path.join(TOYMODEL_ROOT, "results")

    T = 200
    ref_step = np.ones(T, dtype=np.float32)
    ref_sin = np.sin(2 * np.pi * 0.15 * np.arange(T) * 0.05).astype(np.float32)

    scenarios = ["Nom. Step", "Nom. Sin", r"m$\times$2 Step", r"m$\times$2 Sin"]
    cfgs = [(1.0, ref_step), (1.0, ref_sin), (2.0, ref_step), (2.0, ref_sin)]

    ns_r = []; fm_r = []; rtc_r = []; fbfm_r = []
    for m, ref in cfgs:
        ns_r.append(_rmse(ref, _diag_run_noskip(m, ref)))
        for algo, lst in [("fm", fm_r), ("rtc", rtc_r), ("fbfm", fbfm_r)]:
            lst.append(_rmse(ref, _diag_run_algo(algo, m, ref)["xs_true"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scenarios)); w = 0.2
    for offset, vals, label, color in [
        (-1.5, ns_r,   "No-skip (no delay)", COLORS["noskip"]),
        (-0.5, fm_r,   "FM (skip-d)",        COLORS["fm"]),
        ( 0.5, rtc_r,  "RTC (skip-d)",       COLORS["rtc"]),
        ( 1.5, fbfm_r, "FBFM (skip-d)",      COLORS["fbfm"]),
    ]:
        bars = ax.bar(x + offset * w, vals, w, label=label, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i + offset * w, v + 0.02, f"{v:.2f}", ha="center",
                    fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(scenarios)
    ax.set_ylabel("RMSE"); ax.set_title("RMSE Summary Across Scenarios")
    ax.legend(framealpha=0.9); ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    savefig(fig, os.path.join(save_dir, "diag_fig6_rmse_summary"))


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag-only", action="store_true",
                        help="Only generate diagnostic figures (skip experiment figs)")
    args = parser.parse_args()

    if not args.diag_only:
        print("Generating experiment figures...")

        if os.path.exists(os.path.join(TOYMODEL_ROOT, "results", "exp_a_mismatch", "metrics.json")):
            print("\n[Exp-A] Mismatch bar chart + trajectory plots")
            plot_exp_a_bar()
            for cond in ["mass×2", "mass×3", "stiff×3"]:
                plot_exp_a_traj(condition=cond)
        else:
            print("[Exp-A] metrics.json not found — run experiments/run_all.py first")

        if os.path.exists(os.path.join(TOYMODEL_ROOT, "results", "exp_b_disturbance", "metrics.json")):
            print("\n[Exp-B] Disturbance response plots")
            plot_exp_b()
        else:
            print("[Exp-B] metrics.json not found — run experiments/run_all.py first")

        if os.path.exists(os.path.join(TOYMODEL_ROOT, "results", "exp_c_ablation", "metrics.json")):
            print("\n[Exp-C] Ablation bar chart")
            plot_exp_c_bar()
        else:
            print("[Exp-C] metrics.json not found — run experiments/run_all.py first")

        if os.path.exists(os.path.join(TOYMODEL_ROOT, "results", "exp_d_sensitivity", "metrics.json")):
            print("\n[Exp-D] Sensitivity heatmap")
            plot_exp_d_heatmap()
        else:
            print("[Exp-D] metrics.json not found — run experiments/run_all.py first")

    # Diagnostic figures (require trained checkpoint)
    ckpt_path = os.path.join(TOYMODEL_ROOT, "checkpoints", "fm_best.pt")
    if os.path.exists(ckpt_path):
        print("\n[Diag] Generating diagnostic figures...")
        print("\n  [Diag-1] Strategy comparison (no-skip vs skip-d)")
        plot_diag_strategy_comparison()
        print("\n  [Diag-2] Chunk boundary diagnostics")
        plot_diag_chunk_boundary()
        print("\n  [Diag-3] Action mismatch analysis")
        plot_diag_action_mismatch()
        print("\n  [Diag-4] Single chunk self-consistency")
        plot_diag_single_chunk()
        print("\n  [Diag-5] OOD analysis")
        plot_diag_ood_analysis()
        print("\n  [Diag-6] RMSE summary bar chart")
        plot_diag_rmse_summary()
    else:
        print(f"\n[Diag] Checkpoint not found at {ckpt_path} — skipping diagnostic figures")

    print("\nDone.")
