# CLAUDE.md — FBFM 项目总览与规范文档

> 每次对话开始时读取此文件，从中了解项目全部进度与规范。

---

## 一、项目目标

验证 **FBFM (Feed-Back Flow Matching)** 算法在模型推理预测的 state 准确性上，相较于传统 **FM** 和 **RTC** 有显著提升。核心创新点：在推理链与执行链的异步过程中，通过实时状态反馈避免"盲目执行"问题。

---

## 二、项目规范

### 2.1 张量 Shape 标注要求

所有重要张量必须在定义处或首次使用处以注释标注 shape，格式如下：

```python
# state:  (batch, state_dim)   = (B, 2)   — [x, x_dot]
# action: (batch, action_dim)  = (B, 1)   — [u]
# chunk:  (batch, H, action_dim) = (B, 16, 1)
```

- `B`：batch size
- `H`：prediction horizon（默认 16）
- `state_dim = 2`（位置 x，速度 x_dot）
- `action_dim = 1`（施加力 u）
- `obs_dim = state_dim = 2`（本项目 observation = state）

### 2.2 单线程模拟异步逻辑

本项目用**单线程交替调用**模拟推理链与执行链的异步双线程：

**RTC 节奏（每个大周期）：**
```
执行链: 执行 5 步 action（前 1 步读取 state 作为 FM 输入，后 4 步盲目执行）
推理链: 完整带反馈的 Flow Matching 推理（4 个去噪步）
→ 下一大周期
```

**FBFM 节奏（每个大周期，细粒度交替）：**
```
推理链: 去噪 n_inner 步（默认 4 步）
执行链: 执行 1 步 action，读取 (state, action) 反馈给推理链
→ 循环直到去噪完成（共 n_denoising=16 步，穿插 16 次执行）
→ 下一大周期
```

### 2.3 扰动噪声添加位置（已确定）

| 扰动类型 | 添加位置 | 说明 |
|---------|---------|------|
| 观测噪声 | `env.step()` 返回的 state | 模拟传感器噪声 |
| 过程噪声 | `env.step()` 内部动力学积分 | 模拟建模误差 |
| 外部扰动力 | `env.step()` 的 action 输入端 | 模拟突发外力 |
| 参数失配 | 测试时使用不同 (m, k, c) | 模拟 sim-to-real gap |

---

## 三、模块化进度表

| # | 模块 | 文件 | 状态 |
|---|------|------|------|
| 1 | MSD 仿真环境 | `sim/msd_env.py` | ✅ 完成 |
| 2 | PID 专家数据集 | `sim/pid_controller.py`, `data/collect_dataset.py` | ✅ 完成（待全量采集） |
| 3 | DiT Flow Matching 模型训练 | `model/dit.py`, `train/train_fm.py` | ⬜ 待开始 |
| 4 | 三算法对比实验 | `algo/fm.py`, `algo/rtc.py`, `algo/fbfm.py`, `experiments/run_exp.py` | ⬜ 待开始 |
| 5 | 可视化 | `viz/plot_results.py` | ⬜ 待开始 |

---

## 四、核心数学接口约定

### 4.1 MSD 环境接口（已实现）

```python
class MSDEnv:
    def reset() -> np.ndarray          # state: (2,) = [x, x_dot]
    def step(u, disturbance=0.0)       # u: scalar force
        -> (state, reward, done, info) # state: (2,)
    def get_params() -> dict           # 返回 {m, k, c, dt}
```

### 4.2 RTC 核心函数位（待实现，由用户编写）

```python
# 文件: algo/rtc.py
def guided_inference_rtc(
    policy,          # Flow Matching 神经网络
    obs,             # (obs_dim,) 当前观测
    A_prev,          # (H, action_dim) 上一周期未执行的动作序列（右补零至 H）
    W,               # (H,) 掩码权重向量
    n_steps,         # int 去噪步数
    beta,            # float 最大引导权重
) -> np.ndarray:     # (H, action_dim) 新动作序列
    """
    RTC 带反馈的 Flow Matching 推理。
    核心公式：
        v_guided = v(X^τ, o, τ) + k_p * (Y - X̂¹)ᵀ diag(W) * ∂X̂¹/∂X^τ
    参考 Thoughts.md §对照二 及 RTC Paper.pdf
    """
    raise NotImplementedError("由用户实现")
```

### 4.3 FBFM 核心函数位（待实现，由用户编写）

```python
# 文件: algo/fbfm.py
def guided_inference_fbfm(
    policy,          # Flow Matching 神经网络
    obs,             # (obs_dim,) 当前观测
    X_prev,          # (H, state_dim + action_dim) 上一周期状态-动作序列（右补零至 H）
    W,               # (H,) 掩码权重向量
    n_steps,         # int 总去噪步数
    n_inner,         # int 每次推理链执行的去噪步数（交替粒度）
    beta,            # float 最大引导权重
    env_feedback_fn, # Callable: (action) -> (state, action_executed) 执行链回调
) -> np.ndarray:     # (H, action_dim) 新动作序列
    """
    FBFM 带实时反馈的 Flow Matching 推理。
    核心公式：
        X̂¹_t = X^τ_t + (1-τ) * v(X^τ_t, o_t, τ)
        v_FBFM = v(X^τ, o, τ) + k_p * (Y - X̂¹)ᵀ diag(W) * ∂X̂¹/∂X^τ
        k_p = min(β, (1-τ)/τ * 1/r²_τ),  r²_τ = (1-τ)²/(τ²+(1-τ)²)
    参考 Thoughts.md §实验组 及 Algorithm: FBFM 伪代码
    """
    raise NotImplementedError("由用户实现")
```

---

## 五、实验设计细节

### 5.1 FBFM 优势场景（针对性扰动实验）

FBFM 的核心优势：**chunk 执行期间实时感知环境变化并修正流场**。

| 实验 | 扰动方式 | 预期结论 |
|------|---------|---------|
| Exp-A: 参数失配 | 测试时 m×1.5/2/3，k×3 | FBFM 因实时纠偏，跟踪误差更小 |
| Exp-B: 执行中突发力 | chunk 执行第 3 步施加冲击力 | FBFM 能在同一 chunk 内响应，FM/RTC 需等下一周期 |
| Exp-C: 消融实验 | 关闭 FBFM 反馈项（k_p=0） | 验证反馈项的贡献 |
| Exp-D: 反馈频率敏感性 | n_inner = 1/2/4/8 | 找到最优交替粒度 |

### 5.2 参考轨迹

- Step 参考：$x_{ref}(t) = 1.0$（阶跃到 1m）
- Sinusoidal 参考：$x_{ref}(t) = \sin(2\pi \cdot 0.15 \cdot t)$（0.15 Hz，PID RMSE < 3%）
  - 注意：原设计 0.5 Hz 远超系统固有频率 0.225 Hz，已修正为 0.15 Hz

---

## 六、关键参数默认值

```python
# MSD 环境
m   = 1.0   # kg
k   = 2.0   # N/m
c   = 0.5   # N·s/m
dt  = 0.05  # s
# 系统固有频率 wn = sqrt(k/m) = 1.414 rad/s = 0.225 Hz
# 阻尼比 zeta = c/(2*sqrt(k*m)) = 0.177 (欠阻尼)

# PID 控制器（已调参验证）
Kp = 30.0   # 阶跃 SS_err < 0.001 m，正弦 RMSE < 5% @ 0.25 Hz
Ki = 15.0
Kd = 5.0
u_max = 20.0

# 数据集参考轨迹频率范围（物理约束）
sin_freq_range = [0.05, 0.25]  # Hz，不超过系统固有频率 0.225 Hz

# Flow Matching
H           = 16    # prediction horizon
n_denoising = 16    # 去噪步数
n_inner     = 4     # FBFM 每轮推理链去噪步数

# RTC/FBFM
beta        = 10.0  # 最大引导权重
s_chunk     = 5     # 每大周期执行步数（RTC）

# 噪声
obs_noise_std     = 0.01   # 观测噪声标准差
process_noise_std = 0.005  # 过程噪声标准差
```

---

## 七、绘图规范（全局）

所有输出图片统一使用英文衬线字体，禁止中文字符出现在图表中：

```python
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})
```

- 标签、标题、图例全部使用英文
- 数学符号使用 LaTeX 语法，如 `r"$x_{\mathrm{true}}$"`
- 保存分辨率 `dpi=150`（验证图）/ `dpi=300`（论文图）

---

## 八、对话历史关键结论

- 2026-03-17：项目初始化，Thoughts.md 解析完成，CLAUDE.md 创建，MSD 环境实现完成。
- 2026-03-17：PID 控制器 (`sim/pid_controller.py`) 和数据采集脚本 (`data/collect_dataset.py`) 实现完成。
- 2026-03-17：全局绘图规范确定——英文衬线字体 (Times New Roman / DejaVu Serif)，STIX 数学字体，`msd_env.py` 验证图已更新。
- 2026-03-17：创建 `TODO.md` 作为模块化任务追踪文件。
- 2026-03-17：**数据质量问题修复**：原 PID 增益 (Kp=10, Ki=5, Kd=2) 在正弦跟踪上 RMSE 高达 88%（参考幅值）。根因：MSD 固有频率 wn=0.225 Hz，原频率范围 [0.3, 1.0] Hz 远超系统带宽。修复：Kp=30/Ki=15/Kd=5，正弦频率范围收窄至 [0.05, 0.25] Hz，修复后 RMSE < 5%。
- 2026-03-17：**数据集设计决策**：500 条轨迹（250 阶跃 + 250 正弦）× 12 chunks = 6000 条；初始状态随机化 x_init~U(-0.5,0.5)；obs = chunk 首帧 [x, x_dot]；测试正弦频率改为 0.15 Hz。完整数据集已生成至 `data/expert_dataset.npz`。
- 2026-03-17：**数据集质量确认**：6000 chunks 中 4.5% (269条) RMSE > 0.3m，100% 来自每条轨迹的第0个chunk（PID瞬态响应期，初始误差均值1.06m）。第1-11个chunk质量完美（RMSE < 0.06m）。决策：保留全部数据（含瞬态），瞬态chunk代表"从偏离状态恢复"行为，对Exp-B扰动实验有价值。动作饱和率仅1.3%，状态覆盖良好。
- 2026-03-17：**训练将在 GPU 设备（RTX 4080）上进行**，本机不运行训练。Module 3 代码（`model/dit.py`, `model/dataset.py`, `model/inference.py`, `train/train_fm.py`）已完成，前向传播验证通过（1.09M 参数）。
- 2026-03-17：**Module 4 完成**：`algo/fm.py`（FM rollout）、`algo/rtc.py`（RTC harness + 用户实现位）、`algo/fbfm.py`（FBFM harness + 用户实现位）、`experiments/runner.py`（三算法统一 runner）、四个实验脚本（exp_a/b/c/d）、`experiments/run_all.py` 全部完成，17 个文件全部通过语法检查。
- 2026-03-17：**Module 5 完成**：`viz/plot_results.py` 实现五类图表（mismatch bar、trajectory overlay、disturbance response、ablation bar、sensitivity heatmap），英文衬线字体，dpi=300，PDF+PNG 双格式输出。
