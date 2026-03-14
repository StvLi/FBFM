# `pre_test/results_final` 实验评估与优化报告

## 目的

本文档针对 `pre_test/results_final` 中已经完成的四组实验结果进行系统评估，回答两个问题：

1. 当前结果是否足够作为顶会论文中的预实验部分；
2. 应如何进一步优化实验设计、结果呈现和论文叙事，以提升其学术说服力。

本文档严格基于以下结果文件：

- `pre_test/results_final/exp_a_mismatch/metrics.json`
- `pre_test/results_final/exp_b_disturbance/metrics.json`
- `pre_test/results_final/exp_c_ablation/metrics.json`
- `pre_test/results_final/exp_d_sensitivity/metrics.json`
- `pre_test/results_final/summary_report.json`
- `pre_test/results_final/summary_report.md`

---

## 一、总体判断

### 1.1 结论

**当前这套结果，已经足够作为顶会论文中的“预实验 / motivating experiment / controlled toy validation”部分。**

但需要明确区分：

- **作为预实验：够，而且质量较好。**
- **作为主实验核心：还不够。**

原因不在于结果不显著，而在于实验载体仍然过于简单。当前系统是一个可控的 1D mass-spring-damper 闭环控制问题，它非常适合做机制验证，却不足以单独支撑“方法具有广泛复杂场景优势”的强结论。

### 1.2 为什么说它“够做预实验”

因为它已经具备了一套比较完整的科学证据链：

1. **主实验（Exp A）**：验证方法在模型失配下的趋势；
2. **鲁棒性实验（Exp B）**：验证外部扰动下的恢复表现；
3. **机制消融（Exp C）**：验证 Jacobian 的必要性；
4. **敏感性分析（Exp D）**：验证超参数稳定性。

这四块组合起来，已经有明显的“论文式结构”，而不是简单的 demo。

---

## 二、当前实验最有价值的科学结论

## 2.1 FBFM 的主要优势不是 tracking error 本身，而是 action consistency

从 `exp_a_mismatch` 可以看到，FBFM 在 step 目标上的优势非常突出，尤其是：

- `action_jitter`
- `action_mse_vs_pid`
- `energy`

这说明 FBFM 的核心改进更像是在**减少闭环 chunk-based 生成过程中的动作不一致性**，而不是直接“神奇地降低所有 tracking error”。

这是一个非常重要的认识，因为它决定了论文叙事的方向：

> 你的方法最擅长解决的是 **feedback-conditioned sequence generation 的稳定性问题**，而不是全能型最优跟踪控制器问题。

这个定位是准确且学术上更稳妥的。

## 2.2 Step 任务是当前最强证据

在 step 条件下，FBFM 的优势非常稳定，且改善倍数很可观。

例如：

- 在 `step_combined` 中，FBFM 相比 Vanilla 的 `action_jitter` 改善约为 **15.8×**；
- 在多个 step mismatch 条件下，FBFM 在动作抖动和动作偏差上都明显优于 Vanilla 和 RTC；
- 同时，FBFM 经常还能带来更低的能耗与更低的方差。

这类结果非常适合做成论文中的核心可视化图：

- 一张代表性轨迹图；
- 一张动作序列图；
- 一张 jitter 柱状图；
- 一张 mismatch severity 曲线图。

## 2.3 Disturbance recovery 进一步证明了方法的闭环价值

在 `exp_b_disturbance` 中，FBFM 在 impulse 和持续外力扰动下表现很好。

例如：

- `impulse_nom` 下，FBFM 相对 Vanilla 的 `action_jitter` 改善接近 **19×**；
- `step_force_mm` 下，FBFM 相对 Vanilla 也有 **15×** 量级的 jitter 优势；
- 在不少条件下，FBFM 同时具有更低能耗和更低 seed 方差。

这说明 FBFM 不只是离线拟合得好，而是在闭环受扰环境中仍然能维持平稳动作。

这对“feedback-based flow matching”这一方法命名和动机是加分的。

## 2.4 Jacobian ablation 是目前最关键的结果之一

`exp_c_ablation` 是这套结果里最有理论说服力的一组。

因为它回答了一个最重要的问题：

> FBFM 的提升到底来自 state feedback 本身，还是来自正确的 Jacobian 建模？

实验结果已经清楚表明：

- RTC（只有 action guidance）通常优于 Vanilla；
- 但 **FBFM-Identity**（错误 Jacobian）往往不如 RTC，甚至会恶化表现；
- **完整 FBFM** 才是最优。

这说明：

1. 单纯加入 state feedback 并不够；
2. Jacobian 若建模错误，会把 state feedback 变成“错误方向的修正”；
3. 正确 Jacobian 才是 FBFM 成立的关键。

这个结论对论文来说非常宝贵，因为它能让你的方法不只是“有效”，而且“为什么有效”也被实验支持了。

## 2.5 Sensitivity 分析说明 1D 场景下方法不脆弱，但也暴露了任务太简单

`exp_d_sensitivity` 显示：

- FBFM 对 state guidance weight 几乎不敏感；
- RTC 对 action guidance weight 明显敏感，约在 `1.0` 左右最优；
- 最优 FBFM 的 jitter 仍显著好于最优 RTC。

这带来两个信息：

### 正面信息

- 方法不是靠精调超参数才有效；
- 结果具有一定稳定性。

### 负面信息

- 当前 1D 系统太简单，state guidance 的权重几乎不改变结果；
- 因此这部分很难强有力地支撑“state-feedback modeling 在复杂系统中至关重要”的更广义命题。

也就是说，它适合做方法稳定性说明，但不适合过度拔高。

---

## 三、目前还不够顶会“主实验”的原因

## 3.1 系统复杂度不足

1D 质量-弹簧-阻尼系统是一个非常好的受控实验平台，但它的问题也很明显：

- 状态维度太低；
- 非线性程度有限；
- 动作空间过于简单；
- 真实机器人系统中的耦合、延迟、约束、非共址扰动等因素都不存在。

所以它更像是：

- **机制展示系统**，而不是
- **大规模方法有效性验证系统**。

## 3.2 缺少统计显著性与效应量

目前结果已经有 `mean ± std`，这很好，但顶会读者仍会自然问：

- 差异是否显著？
- 改善的 effect size 有多大？
- 结果是否稳健到足以排除 seed 波动？

目前缺失：

- paired t-test 或 Wilcoxon signed-rank test；
- Cohen's d 或 Cliff's delta；
- 明确指出哪些比较是 statistically significant。

## 3.3 缺少更强 baseline

当前 baseline 主要是：

- Vanilla FM
- RTC
- FBFM-ID
- PID only reference

这对于方法机制验证够了，但如果要说服更广泛的 reviewer，通常还需要更强基线，例如：

- receding-horizon imitation / behavior cloning baseline；
- classical MPC；
- disturbance observer enhanced baseline；
- 更强的 model-based receding controller。

## 3.4 Sinusoidal 结果不够“漂亮”

这是当前文稿中最需要诚实处理的地方。

在 sinusoidal tracking 上：

- FBFM 有时优于 Vanilla；
- 但改善没有 step 那么稳定和显著；
- 在更 severe mismatch 下，所有方法都明显吃力。

这并不意味着方法失败，而意味着：

- FBFM 主要解决的是 chunk inconsistency 和 action smoothness；
- 对持续动态跟踪，误差来源更复杂，方法优势未必总是直接体现为 position MSE 最优。

因此，论文表述应避免写成“FBFM 在所有 tracking 任务上显著优于基线”，而应写成更精确的机制性陈述。

---

## 四、如何写进论文更稳

推荐将这组实验定位为：

> A controlled 1D closed-loop dynamical system study for isolating the effect of feedback-aware flow matching on sequence consistency under model mismatch and disturbances.

中文可以表述为：

> 为了在可控环境中隔离并研究 feedback-aware flow matching 对闭环序列一致性的影响，我们首先在一维质量-弹簧-阻尼系统上开展机制性预实验。

接下来建议按如下逻辑写：

1. 先解释为什么选 1D 系统：可控、可解释、能分离机制；
2. 再说明 mismatch / disturbance / ablation / sensitivity 四类实验分别验证什么；
3. 再给核心结论：
   - 随失配增强，Vanilla 动作抖动显著上升；
   - RTC 可部分缓解；
   - 错误 Jacobian 会破坏反馈修正；
   - 完整 FBFM 在 step 和 disturbance 场景中最稳定；
   - 在连续正弦跟踪上收益较弱，说明其主作用更偏向 sequence consistency 而非万能 tracking 优化。

这种写法，理论上更不容易被 reviewer 抓住“夸大结论”的问题。

---

## 五、最值得优先补强的内容

如果你只想投入少量额外工作，但想显著提高论文质量，我建议优先做下面 5 件事。

## 5.1 增加显著性检验与效应量

这是**投入最小、收益最大**的一步。

建议对以下指标做 method pairwise comparison：

- `action_jitter`
- `action_mse_vs_pid`
- `energy`
- `position_mse`

比较组建议至少包含：

- FBFM vs Vanilla
- FBFM vs RTC
- RTC vs Vanilla
- FBFM vs FBFM-ID

统计方法建议：

- 样本量小（5 seeds）时，优先加 **Wilcoxon signed-rank**；
- 同时可给出 **paired t-test** 作为补充；
- 报告 **Cohen's d** 或 **Cliff's delta**。

## 5.2 做 disturbance strength sweep

当前 disturbance 是几个离散案例，这已经不错。

但更论文化的方式是：

- impulse amplitude: 1N, 2N, 5N, 8N, 10N
- bias force magnitude: 0.5N, 1N, 2N, 4N
- position offset: 0.1, 0.2, 0.5, 1.0

然后画出：

- `action_jitter` vs disturbance magnitude
- `recovery_steps` vs disturbance magnitude
- `position_mse` vs disturbance magnitude

这种曲线比单点柱状图更有说服力。

## 5.3 增加更多 reference trajectory families

目前只有：

- step
- sinusoidal

建议继续加入：

- ramp
- chirp
- piecewise constant with random dwell time
- multi-frequency sinusoid
- trapezoidal profile

这样可以更精确地说明：

- 什么时候 FBFM 优势最大；
- 优势是源于参考轨迹不连续，还是源于系统 mismatch；
- 方法是否对更复杂 time-varying target 同样稳定。

## 5.4 增加控制领域更熟悉的指标

除了当前指标，建议增补：

- overshoot
- settling time
- IAE: integral absolute error
- ISE: integral squared error
- total variation of control
- peak control effort

这些指标会让控制背景 reviewer 更容易接受。

## 5.5 重点分析失败案例

目前最值得单独讨论的弱点案例是：

- `pos_offset_mm`
- 严重 mismatch 下的 sinusoidal tracking

不要回避它们，反而应该拿出来分析：

- 为什么方差变大；
- 为什么 RTC 有时更稳；
- 为什么 identity Jacobian 会把问题放大；
- 是否是因为本地线性近似失真、状态估计误差放大、或 guidance 方向不再可靠。

这种“解释失败”的能力，通常会让论文更成熟。

---

## 六、对实验本身的进一步优化建议

## 6.1 把 1D 实验升级成一个更完整的 benchmark section

建议形成如下结构：

### Section A: Controlled Mechanistic Validation
- 1D MSD
- mismatch sweep
- disturbance sweep
- Jacobian ablation
- sensitivity

### Section B: Higher-dimensional Validation
- 2-DOF arm (`pre_test_2`)
- payload shift
- friction mismatch
- external disturbance
- trajectory tracking with more realistic references

这样，1D 负责“证明机制”，2-DOF 负责“证明复杂系统有效”。

## 6.2 建议统一所有图表叙事主轴

你现在的图很多，但论文里不应该平铺直叙。

最好统一成一个主问题：

> Does feedback-aware flow matching reduce closed-loop action inconsistency under model mismatch and disturbances?

然后所有图都围绕这个问题展开。

具体而言：

- 主图：mismatch 下的 jitter 曲线；
- 副图：代表性动作轨迹；
- 表格：ablation；
- 附录：敏感性与更多扰动案例。

## 6.3 强化可复现性包装

建议后续在结果目录中统一补：

- 每个实验的配置快照；
- 使用的 seeds；
- 运行时间；
- 模型 checkpoint 信息；
- 自动生成表格的脚本。

这样在投稿时更容易整理 supplementary material。

---

## 七、建议你在正文里怎么下结论

下面是一种更稳健的结论方式：

> In a controlled 1D closed-loop dynamical system, FBFM consistently reduces action jitter, action deviation from the expert controller, and often control energy under dynamics mismatch and external disturbances. Ablation results further show that this gain does not arise from feedback alone: replacing the Jacobian with an identity mapping substantially weakens or even reverses the benefit, indicating that accurate Jacobian-mediated feedback propagation is critical. While the gains are strongest on step-like references and more modest on continuous sinusoidal tracking, the results provide clear mechanistic evidence that FBFM primarily improves closed-loop sequence consistency rather than universally minimizing all tracking errors.

这比直接说“我们在所有实验上都优于基线”要强，因为它更精确，也更诚实。

---

## 八、最终建议

### 如果你现在就要写论文

你可以把这套 `results_final` 直接作为：

- 预实验部分；
- toy validation section；
- supplementary mechanistic analysis。

### 如果你还有时间继续做

优先顺序建议如下：

1. **补统计显著性 + 效应量**
2. **补 disturbance strength sweep**
3. **补更多 reference family**
4. **把同样逻辑迁移到 2-DOF 系统**
5. **补更强 baseline**

---

## 九、当前仓库中新增的分析产物

本轮已新增：

- `pre_test/analyze_results_final.py`：自动分析现有结果并生成总结
- `pre_test/results_final/summary_report.json`：机器可读汇总
- `pre_test/results_final/summary_report.md`：英文风格摘要报告
- `pre_test/RESULTS_FINAL_REPORT_ZH.md`：本中文详报

这些文件可以直接作为后续文稿撰写和实验扩展的基础。

---

## 十、一句话结论

**目前这套 `results_final` 已经足够好，完全可以作为顶会论文中的高质量预实验部分；但若想把它抬升为主实验核心，还需要更复杂系统、更强基线与更严格统计支撑。**
