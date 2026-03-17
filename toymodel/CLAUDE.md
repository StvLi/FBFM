# CLAUDE.md — FBFM 项目总览与规范文档

> 每次对话开始时读取此文件，从中了解项目全部进度与规范。

---

## 一、项目目标

验证 **FBFM (Feed-Back Flow Matching)** 算法在模型推理预测的 state 准确性上，相较于传统 **FM** 和 **RTC** 有显著提升。核心创新点：在推理链与执行链的异步过程中，通过实时状态反馈避免"盲目执行"问题。

---

## 二、项目规范

### 2.0 工作范围约束

**所有代码检索、编辑、新建操作的范围永远只锁定在 `toymodel/` 子文件夹内，不得触碰仓库中其他任何文件夹。**

### 2.1 张量 Shape 标注要求

所有重要张量必须在定义处或首次使用处以注释标注 shape，格式如下：

```python
# token:  (batch, H, token_dim)  = (B, 16, 3) — [x, x_dot, u]
# obs:    (batch, obs_dim)       = (B, 2)      — [x, x_dot]
# chunk:  (H, token_dim)         = (16, 3)     — 单条 token 序列
```

- `B`：batch size
- `H`：prediction horizon（默认 16）
- `token_dim = 3`（每个 token = [position x, velocity x_dot, action u]）
- `obs_dim = 2`（conditioning observation = [x, x_dot]）
- `ACTION_IDX = 2`：token 中 action 所在的列索引
- `STATE_SLICE = slice(0, 2)`：token 中 state 所在的列切片

### 2.1.1 Token 构造约定（已确认 2026-03-17）

```
Token[h] = [state_{h+1}, action_h]
         = [x_{h+1}, x_dot_{h+1}, u_h]
```
含义：执行 action_h 后得到的 state，与该 action 配对。

数据集构造：`tokens = cat(states[:, 1:H+1, :], actions, dim=-1)` → `(N, H, 3)`
obs 不变：`obs = states[:, 0, :]` → `(N, 2)`

### 2.1.2 归一化格式（已确认 2026-03-17）

per-dim 独立统计，向量格式：
```python
token_stats = {
    'mean': np.array([mean_x, mean_xdot, mean_u]),  # (3,)
    'std':  np.array([std_x, std_xdot, std_u]),      # (3,)
}
```

### 2.1.3 W 掩码约定（per-element mask，已更新 2026-03-17）

`dim_mask` 已废弃。改用 **逐元素掩码 `W: (H, token_dim)`**，state 维度和 action 维度**独立管理**：

| 算法 | W 形状 | state 维度 (col 0-1) | action 维度 (col 2) |
|------|--------|---------------------|---------------------|
| FM   | 无引导 | — | — |
| RTC  | `(H, 3)` | 始终 = 0 | 非零填充位 = 1 |
| FBFM | `(H, 3)` | 仅真实观测位 = 1（随 feedback 递增） | 非零填充位 = 1（同 RTC） |

**关键原则**：预测的 state 不做数——只有真实执行后获取的 state 才参与引导。
Action 掩码与 RTC 一致，遮掩后补的零位。

构造函数：`build_rtc_W(n_remaining, H, token_dim)` / `build_fbfm_W(n_remaining, n_real_states, H, token_dim)`

FBFM 推理中 W_state 随 feedback 逐步扩展：
```
trigger 后:    W_state = [1, 0, 0, ..., 0]   (1 real)
feedback 1 后: W_state = [1, 1, 0, ..., 0]   (2 real)
feedback 2 后: W_state = [1, 1, 1, ..., 0]   (3 real)
feedback 3 后: W_state = [1, 1, 1, 1, 0..]   (4 real)
feedback 4 后: W_state = [1, 1, 1, 1, 1, ..]  (5 real)
```

### 2.1.4 去噪步数（已确认 2026-03-17）

所有算法统一 `n_steps = 20`（原为 16）

### 2.2 单线程模拟异步逻辑

本项目用**单线程交替调用**模拟推理链与执行链的异步双线程：

**RTC 节奏（每个大周期）：**
```
执行链: 执行 5 步 action（前 1 步读取 state 作为 FM 输入，后 4 步盲目执行）
推理链: 完整带反馈的 Flow Matching 推理（4 个去噪步）
→ 下一大周期
```

**FBFM 节奏（每个大周期，细粒度交替，n_steps=20, n_inner=4, d=4）：**
```
rollout 每个大周期:
  Build Y from chunk[chunk_ptr:] → (H, 3) right-padded
  Trigger: 执行 Y[0].action → capture obs (1 步)

  guided_inference_fbfm(obs, Y, W, env_feedback_fn):
    Block 0 (τ=0-3):  guidance with Y + obs + W(state=1)
      → feedback: 执行 Y[1].action → update obs → W_state grows to 2
    Block 1 (τ=4-7):  Y(不变) + 新 obs + W(state=2)
      → feedback → W_state grows to 3
    Block 2 (τ=8-11):  → W_state grows to 4
    Block 3 (τ=12-15): → W_state grows to 5
    Block 4 (τ=16-19): 最终输出, 无 feedback

  New chunk → chunk_ptr = d = 4 (跳过前 4 个 token)
  Total: 1(trigger) + 4(feedback) = 5 = s_chunk
  Y 在整个推理过程中固定不变, obs 和 W_state 随 feedback 更新
  W_action 与 RTC 一致（非零填充位=1），W_state 仅真实观测位=1
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
| 2 | PID 专家数据集 | `sim/pid_controller.py`, `data/collect_dataset.py` | ✅ 完成 |
| 3 | DiT FM 模型 (token_dim=3) | `model/dit.py`, `model/dataset.py`, `train/train_fm.py` | ✅ 代码完成（待 GPU 训练） |
| 4 | 三算法 (token_dim=3, n_steps=20) | `algo/fm.py`, `algo/rtc.py`, `algo/fbfm.py` | ✅ 全部实现 |
| 5 | 实验脚本 | `experiments/runner.py`, `exp_a/b/c/d.py`, `run_all.py` | ✅ 完成 |
| 6 | 可视化 | `viz/plot_results.py` | ✅ 完成（待结果数据渲染） |

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

### 4.2 RTC 核心函数（已实现）

```python
# 文件: algo/rtc.py
def guided_inference_rtc(
    policy,          # FlowMatchingDiT
    obs,             # (obs_dim,) = (2,)
    A_prev,          # (H, token_dim) = (16, 3) 归一化空间, 右补零至 H
    W,               # (H, token_dim) per-element mask (state=0, action=non-padded)
    n_steps=20,      # 去噪步数
    beta=10.0,       # 最大引导权重
    device=...,
) -> np.ndarray:     # (H, token_dim) 归一化空间
```

### 4.3 FBFM 核心函数（已实现）

```python
# 文件: algo/fbfm.py
def guided_inference_fbfm(
    policy,          # FlowMatchingDiT
    obs,             # (obs_dim,) = (2,) trigger 后的观测
    X_prev,          # (H, token_dim) = (16, 3) 归一化空间, 右补零至 H
    W,               # (H, token_dim) per-element mask (state 随 feedback 递增)
    n_steps=20,      # 总去噪步数
    n_inner=4,       # 每 block 去噪步数
    beta=10.0,       # 最大引导权重
    env_feedback_fn=None,  # () -> (obs_new, u_executed)
    device=...,
) -> np.ndarray:     # (H, token_dim) 归一化空间
# W 在函数内部随 feedback 动态扩展 state 维度的有效范围
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
token_dim   = 3     # 每个 token = [x, x_dot, u]
n_denoising = 20    # 去噪步数（所有算法统一, 原为 16）
n_inner     = 4     # FBFM 每轮推理链去噪步数

# RTC/FBFM
beta        = 10.0  # 最大引导权重
s_chunk     = 5     # 每大周期执行步数（RTC/FBFM）
                    # = 1(trigger) + d(blind/feedback), d = n_denoising/n_inner - 1 = 4

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
- 2026-03-17：**device 默认值**：所有 `toymodel/` 下的 device 默认值已改为 `"cuda" if torch.cuda.is_available() else "cpu"`。
- 2026-03-17：**工作范围约束**：已写入 CLAUDE.md §2.0，所有操作锁定 `toymodel/` 子文件夹。
- 2026-03-17：**RTC guided_inference_rtc 已实现**：完整 7 步算法（VJP 引导），rollout_rtc 已重写为正确的异步双线程模拟节奏。
- 2026-03-17：**token_dim=3 统一架构重构完成**：设计决策 D/E/F/G/H 全部确认并实施。13 个文件全部修改完成：
  - `dit.py`: action_dim → token_dim=3
  - `dataset.py`: Token[h]=[state_{h+1}, action_h], per-dim 归一化
  - `train_fm.py`: CFG token_dim=3, checkpoint 存 token_stats
  - `inference.py`: token_dim + 向量 denorm + n_steps=20
  - `fm.py`: chunk(H,3), action=col2, n_steps=20, token_stats
  - `rtc.py`: W=(H,token_dim) per-element mask (state=0,action=non-padded), n_steps=20, token_stats
  - `fbfm.py`: **guided_inference_fbfm 完整实现**（5 blocks × 4, interleaved feedback, W_state 递增）+ rollout_fbfm 重写
- 2026-03-17：**W 掩码从 (H,) 升级为 (H, token_dim) per-element mask**：废弃 dim_mask 参数。State 维度和 action 维度独立管理——action 同 RTC（遮掩零填充），state 仅真实观测有效（FBFM 中随 feedback 递增）。参考 `fbfm/policies/fbfm/modeling_rtc.py` 的设计思路。
  - `runner.py`: load_policy/run_three_algos 适配 token_dim/token_stats/n_steps=20
  - `exp_a/b/c/d.py`: action_stats → token_stats
