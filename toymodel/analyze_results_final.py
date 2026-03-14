"""Analyze `pre_test/results_final` and generate publication-friendly summaries.

This script reads the existing metrics JSON files and produces:
1. `pre_test/results_final/summary_report.json`
2. `pre_test/results_final/summary_report.md`

It is intentionally lightweight and does not rerun experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_final"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_mean(entry, key):
    value = entry[key]
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)


def ratio(a: float, b: float) -> float:
    if b == 0:
        return float("inf")
    return a / b


def summarize_exp_a(data):
    step_keys = [k for k in data if k.startswith("step_")]
    sin_keys = [k for k in data if k.startswith("sinusoidal_")]

    step_jitter_gains = []
    step_action_mse_gains = []
    step_energy_gains = []
    sin_pos_gains = []
    sin_jitter_gains = []

    best_step_case = None
    best_step_ratio = -1.0

    for key in step_keys:
        vanilla = data[key]["vanilla"]
        fbfm = data[key]["fbfm"]
        jgain = ratio(metric_mean(vanilla, "action_jitter"), metric_mean(fbfm, "action_jitter"))
        again = ratio(metric_mean(vanilla, "action_mse_vs_pid"), metric_mean(fbfm, "action_mse_vs_pid"))
        egain = ratio(metric_mean(vanilla, "energy"), metric_mean(fbfm, "energy"))
        step_jitter_gains.append(jgain)
        step_action_mse_gains.append(again)
        step_energy_gains.append(egain)
        if jgain > best_step_ratio:
            best_step_ratio = jgain
            best_step_case = key

    for key in sin_keys:
        vanilla = data[key]["vanilla"]
        fbfm = data[key]["fbfm"]
        sin_pos_gains.append(ratio(metric_mean(vanilla, "position_mse"), metric_mean(fbfm, "position_mse")))
        sin_jitter_gains.append(ratio(metric_mean(vanilla, "action_jitter"), metric_mean(fbfm, "action_jitter")))

    return {
        "num_step_conditions": len(step_keys),
        "num_sinusoidal_conditions": len(sin_keys),
        "avg_step_jitter_improvement": mean(step_jitter_gains),
        "avg_step_action_mse_improvement": mean(step_action_mse_gains),
        "avg_step_energy_improvement": mean(step_energy_gains),
        "best_step_jitter_case": best_step_case,
        "best_step_jitter_improvement": best_step_ratio,
        "avg_sinusoidal_position_improvement": mean(sin_pos_gains),
        "avg_sinusoidal_jitter_improvement": mean(sin_jitter_gains),
    }


def summarize_exp_b(data):
    jitter_gains = []
    action_mse_gains = []
    energy_gains = []
    weaker_cases = []

    for key, cond in data.items():
        vanilla = cond["vanilla"]
        fbfm = cond["fbfm"]
        rtc = cond["rtc"]
        jgain = ratio(metric_mean(vanilla, "action_jitter"), metric_mean(fbfm, "action_jitter"))
        again = ratio(metric_mean(vanilla, "action_mse_vs_pid"), metric_mean(fbfm, "action_mse_vs_pid"))
        egain = ratio(metric_mean(vanilla, "energy"), metric_mean(fbfm, "energy"))
        jitter_gains.append(jgain)
        action_mse_gains.append(again)
        energy_gains.append(egain)
        if metric_mean(fbfm, "action_jitter") >= metric_mean(rtc, "action_jitter"):
            weaker_cases.append(key)

    return {
        "num_conditions": len(data),
        "avg_jitter_improvement": mean(jitter_gains),
        "avg_action_mse_improvement": mean(action_mse_gains),
        "avg_energy_improvement": mean(energy_gains),
        "cases_where_fbfm_not_better_than_rtc_in_jitter": weaker_cases,
    }


def summarize_exp_c(data):
    step_cases = [k for k in data if k.startswith("step_")]
    sin_cases = [k for k in data if k.startswith("sin_")]

    step_jacobian_win = 0
    step_total = 0
    rtc_vs_identity = 0
    sin_identity_fail = 0

    for key in step_cases:
        cond = data[key]
        step_total += 1
        if metric_mean(cond["fbfm"], "action_jitter") < metric_mean(cond["fbfm_id"], "action_jitter"):
            step_jacobian_win += 1
        if metric_mean(cond["rtc"], "action_jitter") < metric_mean(cond["fbfm_id"], "action_jitter"):
            rtc_vs_identity += 1

    for key in sin_cases:
        cond = data[key]
        if metric_mean(cond["fbfm_id"], "position_mse") > metric_mean(cond["vanilla"], "position_mse"):
            sin_identity_fail += 1

    step_combined = data["step_combined"]
    return {
        "step_cases_jacobian_beats_identity": step_jacobian_win,
        "step_cases_total": step_total,
        "step_cases_rtc_beats_identity": rtc_vs_identity,
        "sin_cases_identity_worse_than_vanilla_on_position": sin_identity_fail,
        "step_combined_jitter": {
            "vanilla": metric_mean(step_combined["vanilla"], "action_jitter"),
            "rtc": metric_mean(step_combined["rtc"], "action_jitter"),
            "fbfm_id": metric_mean(step_combined["fbfm_id"], "action_jitter"),
            "fbfm": metric_mean(step_combined["fbfm"], "action_jitter"),
        },
    }


def summarize_exp_d(data):
    vanilla_jitter = metric_mean(data["vanilla"], "action_jitter")

    fbfm_sweep = data["fbfm_sweep"]
    rtc_sweep = data["rtc_sweep"]

    best_fbfm_state = min(fbfm_sweep.items(), key=lambda kv: kv[1]["action_jitter"]["mean"])
    best_rtc_action = min(rtc_sweep.items(), key=lambda kv: kv[1]["action_jitter"]["mean"])

    return {
        "best_fbfm_state_weight": float(best_fbfm_state[0]),
        "best_fbfm_jitter": metric_mean(best_fbfm_state[1], "action_jitter"),
        "best_rtc_action_weight": float(best_rtc_action[0]),
        "best_rtc_jitter": metric_mean(best_rtc_action[1], "action_jitter"),
        "vanilla_jitter": vanilla_jitter,
        "fbfm_vs_vanilla_best_jitter_gain": vanilla_jitter / metric_mean(best_fbfm_state[1], "action_jitter"),
        "rtc_vs_vanilla_best_jitter_gain": vanilla_jitter / metric_mean(best_rtc_action[1], "action_jitter"),
        "fbfm_state_sensitivity_range": {
            "min_jitter": min(metric_mean(v, "action_jitter") for v in fbfm_sweep.values()),
            "max_jitter": max(metric_mean(v, "action_jitter") for v in fbfm_sweep.values()),
            "min_position_mse": min(metric_mean(v, "position_mse") for v in fbfm_sweep.values()),
            "max_position_mse": max(metric_mean(v, "position_mse") for v in fbfm_sweep.values()),
        },
    }


def build_report(summary):
    a = summary["exp_a"]
    b = summary["exp_b"]
    c = summary["exp_c"]
    d = summary["exp_d"]

    sufficiency = []
    sufficiency.append("当前结果足以作为一篇顶会论文中的**预实验 / motivating experiment / supplementary preliminary evidence**。")
    sufficiency.append("但如果要作为主结果部分，证据仍不够：系统维度仅 1D，任务类型偏简单，缺少显著性检验、跨参考轨迹泛化、以及更强基线。")
    sufficiency_text = "\n".join(f"- {x}" for x in sufficiency)

    md = f"""# Pre-test Results Final Report

## Overall Assessment

- 结论：**适合作为顶会级工作的预实验部分**，但**还不够单独支撑主论文核心实验部分**。
- 原因：当前结果在 1D mass-spring-damper 上已经形成了清晰且可重复的结论链：
  1. 在 step-type 控制中，FBFM 显著降低动作抖动与动作偏差；
  2. 在外部扰动下，FBFM 通常恢复更稳、能耗更低；
  3. 消融表明真正关键的是 **正确 Jacobian**，不是简单的 state feedback；
  4. 超参数分析表明 1D 场景下 FBFM 对 state guidance 权重不敏感，说明方法较稳定。

## Quantitative Summary

### Exp A — Model mismatch

- Step 条件数：{a['num_step_conditions']}
- Sinusoidal 条件数：{a['num_sinusoidal_conditions']}
- Step 平均动作抖动改善倍数：**{a['avg_step_jitter_improvement']:.2f}×**
- Step 平均动作 MSE 改善倍数：**{a['avg_step_action_mse_improvement']:.2f}×**
- Step 平均能耗改善倍数：**{a['avg_step_energy_improvement']:.2f}×**
- 最强 step 案例：`{a['best_step_jitter_case']}`，动作抖动改善 **{a['best_step_jitter_improvement']:.2f}×**
- Sinusoidal 平均 position MSE 改善倍数：**{a['avg_sinusoidal_position_improvement']:.2f}×**
- Sinusoidal 平均动作抖动改善倍数：**{a['avg_sinusoidal_jitter_improvement']:.2f}×**

解释：step 任务上证据非常强；sinusoidal 任务上仍有增益，但远不如 step 明显，这说明 FBFM 当前更擅长抑制 chunk-boundary induced action inconsistency，而不是解决持续动态跟踪的全部困难。

### Exp B — Disturbance recovery

- 扰动条件数：{b['num_conditions']}
- 平均动作抖动改善倍数：**{b['avg_jitter_improvement']:.2f}×**
- 平均动作 MSE 改善倍数：**{b['avg_action_mse_improvement']:.2f}×**
- 平均能耗改善倍数：**{b['avg_energy_improvement']:.2f}×**
- 在以下案例中，FBFM 的 jitter 未明显优于 RTC：{', '.join(b['cases_where_fbfm_not_better_than_rtc_in_jitter']) if b['cases_where_fbfm_not_better_than_rtc_in_jitter'] else '无'}

解释：FBFM 对 impulsive / force disturbance 的优势很清晰，但对 `pos_offset_mm` 这样的更难分布外偏移，存在方差增大现象，说明闭环反馈建模仍有脆弱点。

### Exp C — Jacobian ablation

- Step 场景中，FBFM 在 **{c['step_cases_jacobian_beats_identity']}/{c['step_cases_total']}** 个案例里优于 FBFM-Identity（按 jitter）
- Step 场景中，RTC 在 **{c['step_cases_rtc_beats_identity']}/{c['step_cases_total']}** 个案例里优于 FBFM-Identity
- Sin 场景中，FBFM-Identity 在 **{c['sin_cases_identity_worse_than_vanilla_on_position']}** 个案例里 position MSE 甚至劣于 Vanilla
- `step_combined` 的动作抖动：Vanilla={c['step_combined_jitter']['vanilla']:.4f}, RTC={c['step_combined_jitter']['rtc']:.4f}, FBFM-ID={c['step_combined_jitter']['fbfm_id']:.4f}, FBFM={c['step_combined_jitter']['fbfm']:.4f}

解释：这个消融是当前证据链里最有论文价值的一部分。它说明“仅靠 action-only guidance 不够；错误 Jacobian 甚至会伤害性能；正确 Jacobian 才能真正释放 state feedback 的价值”。

### Exp D — Sensitivity

- 最优 FBFM state guidance weight：**{d['best_fbfm_state_weight']}**
- 对应 FBFM jitter：**{d['best_fbfm_jitter']:.4f}**
- 最优 RTC action guidance weight：**{d['best_rtc_action_weight']}**
- 对应 RTC jitter：**{d['best_rtc_jitter']:.4f}**
- Vanilla jitter：**{d['vanilla_jitter']:.4f}**
- 最优 FBFM 相对 Vanilla 的 jitter 改善：**{d['fbfm_vs_vanilla_best_jitter_gain']:.2f}×**
- 最优 RTC 相对 Vanilla 的 jitter 改善：**{d['rtc_vs_vanilla_best_jitter_gain']:.2f}×**
- FBFM state-weight 扫描时 jitter 变化范围：[{d['fbfm_state_sensitivity_range']['min_jitter']:.4f}, {d['fbfm_state_sensitivity_range']['max_jitter']:.4f}]

解释：这部分说明方法在 1D 场景并不脆弱；不过它也反过来提示：当前 1D 系统太简单，以至于 state guidance 权重几乎不起作用，因此如果想把“state-feedback modeling”本身讲得更强，还需要更复杂系统来放大差异。

## Is it enough for a top-tier preliminary section?

{sufficiency_text}

## Main Strengths

1. **结论链完整**：主实验、扰动、消融、敏感性四块都齐。
2. **多 seed 统计**：不是单次 lucky run。
3. **贡献点能被消融支撑**：尤其是 Jacobian ablation 很关键。
4. **指标合理**：不仅看 tracking error，也看 action jitter、action MSE、energy。
5. **结果风格有论文感**：已经能写成 motivating figure + ablation table + robustness section。

## Main Weaknesses

1. **系统过于简单**：1D mass-spring-damper 只能说明方法机制，不能说明广泛有效性。
2. **缺少统计显著性报告**：目前只有 mean±std，没有显著性检验和 effect size。
3. **缺少更强基线**：例如 classical MPC / iterative LQR / disturbance observer enhanced controller / receding-horizon behavior cloning baseline 等。
4. **Sinusoidal 叙事较弱**：FBFM 在持续跟踪上优势不总是显著，需要更准确地表述，不宜夸大。
5. **无跨训练分布泛化分析**：目前 mismatch 是手工设定，缺少 systematic OOD curve 或 uncertainty-aware analysis。

## Recommended Paper Framing

建议把这一组结果定位为：

- 一个 **mechanistic preliminary study**；
- 用来证明 **FBFM 的收益主要来自减少 action inconsistency，并依赖正确 Jacobian**；
- 不要声称它已经证明了复杂机器人任务上的全面 superiority；
- 在正文中可作为 “toy closed-loop system validation” 或 “controlled mechanistic study”。

## How to improve the experiment next

### Highest-value additions

1. **加显著性检验和效应量**
   - 对 `action_jitter`, `action_mse_vs_pid`, `energy` 做 paired t-test / Wilcoxon signed-rank。
   - 报 Cohen's d 或 Cliff's delta。
   - 这是最便宜、论文收益最高的增强。

2. **增加跨 reference family 泛化**
   - 除了 step / sinusoidal，再加 ramp、chirp、piecewise-smooth、multi-frequency targets。
   - 这能解释为什么 sinusoidal 上优势变弱，并区分“平滑性收益”与“tracking difficulty”。

3. **把 disturbance 做成强度扫描曲线**
   - impulse amplitude、bias force magnitude、offset magnitude 连续变化。
   - 画出性能-扰动强度曲线，比离散 2~3 个点更像顶会图。

4. **补充 closed-loop stability style metrics**
   - settling time、overshoot、IAE/ISE、control total variation。
   - 这样更符合控制论文读者的习惯。

5. **单独分析失败案例**
   - 重点看 `pos_offset_mm` 和 severe sinusoidal mismatch。
   - 用失败案例强化你的论证：FBFM 不是 everywhere best，但在 chunk consistency 主导的问题上最有效。

### If you want this section to look significantly stronger

- 将同一分析迁移到 `pre_test_2` 两自由度系统；
- 让 1D 实验承担“机制解释”，2-DOF 承担“复杂系统验证”；
- 这会显著提升整篇工作的可信度。

## Suggested Writing Style for Paper

可直接采用如下叙述逻辑：

1. 在受控 1D 动力系统中，我们首先隔离 action sequence consistency 问题；
2. 随着模型失配增强，Vanilla FM 的 action jitter 快速上升；
3. RTC 可部分缓解，但无法稳定利用 state feedback；
4. 使用错误 Jacobian 的 FBFM-Identity 不仅无益，甚至可能恶化性能；
5. 完整 FBFM 在 step 与 disturbance 任务中稳定地降低 jitter、action deviation 和 energy；
6. 在持续正弦跟踪中，其优势较弱，说明该方法主要解决的是 feedback-conditioned chunk inconsistency，而不是所有 tracking error 来源。

## Bottom-line Recommendation

- **如果你的目标是“预实验部分”**：现在这套结果已经够用了，而且质量不错。
- **如果你的目标是“主实验核心之一”**：还需要补更复杂系统、显著性检验、以及更强基线。

"""
    return md


def main():
    exp_a = load_json(RESULTS_DIR / "exp_a_mismatch" / "metrics.json")
    exp_b = load_json(RESULTS_DIR / "exp_b_disturbance" / "metrics.json")
    exp_c = load_json(RESULTS_DIR / "exp_c_ablation" / "metrics.json")
    exp_d = load_json(RESULTS_DIR / "exp_d_sensitivity" / "metrics.json")

    summary = {
        "exp_a": summarize_exp_a(exp_a),
        "exp_b": summarize_exp_b(exp_b),
        "exp_c": summarize_exp_c(exp_c),
        "exp_d": summarize_exp_d(exp_d),
    }

    md = build_report(summary)

    with (RESULTS_DIR / "summary_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (RESULTS_DIR / "summary_report.md").open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"[OK] Wrote {(RESULTS_DIR / 'summary_report.json')}")
    print(f"[OK] Wrote {(RESULTS_DIR / 'summary_report.md')}")


if __name__ == "__main__":
    main()
