# Pre-test Results Final Report

## Overall Assessment

- 结论：**适合作为顶会级工作的预实验部分**，但**还不够单独支撑主论文核心实验部分**。
- 原因：当前结果在 1D mass-spring-damper 上已经形成了清晰且可重复的结论链：
  1. 在 step-type 控制中，FBFM 显著降低动作抖动与动作偏差；
  2. 在外部扰动下，FBFM 通常恢复更稳、能耗更低；
  3. 消融表明真正关键的是 **正确 Jacobian**，不是简单的 state feedback；
  4. 超参数分析表明 1D 场景下 FBFM 对 state guidance 权重不敏感，说明方法较稳定。

## Quantitative Summary

### Exp A — Model mismatch

- Step 条件数：6
- Sinusoidal 条件数：6
- Step 平均动作抖动改善倍数：**12.07×**
- Step 平均动作 MSE 改善倍数：**6.95×**
- Step 平均能耗改善倍数：**1.14×**
- 最强 step 案例：`step_stiff×3`，动作抖动改善 **26.74×**
- Sinusoidal 平均 position MSE 改善倍数：**1.00×**
- Sinusoidal 平均动作抖动改善倍数：**1.00×**

解释：step 任务上证据非常强；sinusoidal 任务上仍有增益，但远不如 step 明显，这说明 FBFM 当前更擅长抑制 chunk-boundary induced action inconsistency，而不是解决持续动态跟踪的全部困难。

### Exp B — Disturbance recovery

- 扰动条件数：6
- 平均动作抖动改善倍数：**12.67×**
- 平均动作 MSE 改善倍数：**6.33×**
- 平均能耗改善倍数：**1.15×**
- 在以下案例中，FBFM 的 jitter 未明显优于 RTC：pos_offset_mm

解释：FBFM 对 impulsive / force disturbance 的优势很清晰，但对 `pos_offset_mm` 这样的更难分布外偏移，存在方差增大现象，说明闭环反馈建模仍有脆弱点。

### Exp C — Jacobian ablation

- Step 场景中，FBFM 在 **2/2** 个案例里优于 FBFM-Identity（按 jitter）
- Step 场景中，RTC 在 **2/2** 个案例里优于 FBFM-Identity
- Sin 场景中，FBFM-Identity 在 **3** 个案例里 position MSE 甚至劣于 Vanilla
- `step_combined` 的动作抖动：Vanilla=0.6482, RTC=0.2301, FBFM-ID=0.3734, FBFM=0.0410

解释：这个消融是当前证据链里最有论文价值的一部分。它说明“仅靠 action-only guidance 不够；错误 Jacobian 甚至会伤害性能；正确 Jacobian 才能真正释放 state feedback 的价值”。

### Exp D — Sensitivity

- 最优 FBFM state guidance weight：**10.0**
- 对应 FBFM jitter：**0.0433**
- 最优 RTC action guidance weight：**1.0**
- 对应 RTC jitter：**0.3521**
- Vanilla jitter：**1.0548**
- 最优 FBFM 相对 Vanilla 的 jitter 改善：**24.34×**
- 最优 RTC 相对 Vanilla 的 jitter 改善：**3.00×**
- FBFM state-weight 扫描时 jitter 变化范围：[0.0433, 0.0436]

解释：这部分说明方法在 1D 场景并不脆弱；不过它也反过来提示：当前 1D 系统太简单，以至于 state guidance 权重几乎不起作用，因此如果想把“state-feedback modeling”本身讲得更强，还需要更复杂系统来放大差异。

## Is it enough for a top-tier preliminary section?

- 当前结果足以作为一篇顶会论文中的**预实验 / motivating experiment / supplementary preliminary evidence**。
- 但如果要作为主结果部分，证据仍不够：系统维度仅 1D，任务类型偏简单，缺少显著性检验、跨参考轨迹泛化、以及更强基线。

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

