"""
Metrics computation, multi-seed aggregation, and formatted table output.
"""

import numpy as np

from pre_test_2.config import STATE_DIM


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


def print_table(all_methods_agg, title="", labels=None):
    """Print a formatted metrics table.

    Args:
        all_methods_agg: OrderedDict of {method_name: metrics_dict}.
        title: table title string.
        labels: optional {method_name: display_label} mapping.
    """
    if labels is None:
        labels = {}
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
        header += f"  {labels.get(m, m):>20}"
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
