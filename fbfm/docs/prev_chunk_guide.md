# FBFM.PrevChunk 类详解

## 1. 概述

`PrevChunk` 类是 **FBFM (Flow-Based Feedback Model)** 策略中的核心状态管理器。它主要用于在多 Chunk（分块）生成过程中，维护上一轮生成的动作序列（Actions）和观测到的状态潜变量（State Latents），以便为当前 Chunk 的生成提供连续性约束和反馈校正。

该类实现了“反馈即约束”的设计理念，通过将历史数据封装为带权重的张量，指导扩散模型（Diffusion Model）或流匹配模型（Flow Matching Model）在去噪过程中向期望的状态轨迹收敛。

## 2. 核心设计意图

在实时控制（Real-Time Control）场景中，模型通常以固定长度的 Chunk 进行滚动预测。为了保证动作的平滑性和状态的一致性，`PrevChunk` 承担以下职责：

1.  **状态缓存**：存储上一轮未执行完的动作残差（Action Leftovers）和执行过程中观测到的真实状态（Observed States）。
2.  **约束注入**：根据配置的 `constrain_mode`，动态生成权重掩码（Weights Mask），指示调度器（Scheduler）哪些时间步需要强约束，哪些可以自由生成。
3.  **维度对齐**：确保输入的状态和动作张量符合模型预期的固定尺寸 `(T, D)`，处理动态追加时的形状匹配问题。

## 3. 类结构与属性

### 3.1 初始化参数 (`__init__`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `constrain_mode` | str | `"Feedback"` | 约束模式。可选值：<br>- `"Feedback"`: 同时约束动作前缀和状态前缀。<br>- `"RTC"`: 仅约束动作前缀（标准 RTC 模式）。<br>- `"None"`: 无约束，全自由生成。 |
| `actions` | Tensor | `None` | 初始动作张量，形状应为 `(T_a, D_a)`。若提供，将填充至内部缓冲区。 |
| `action_constrained_num` | int | `0` | 初始需要约束的动作步数。 |
| `action_num` | int | `16` | 动作缓冲区的最大时间步长 ($T_a$)。 |
| `action_dim` | int | `16` | 动作空间的维度 ($D_a$)。 |
| `states` | Tensor | `None` | 初始状态张量，形状应为 `(T_s, D_s)`。 |
| `state_constrained_num` | int | `0` | 初始需要约束的状态步数。 |
| `state_num` | int | `4` | 状态缓冲区的最大时间步长 ($T_s$)。 |
| `state_dim` | int | `128` | 状态潜变量的维度 ($D_s$)，通常由 VAE 编码器输出决定。 |
| `inference_delay` | int | `0` | 推理延迟步数，用于时间轴对齐补偿。 |

### 3.2 核心属性

-   `self.actions`: 形状为 `(action_num, action_dim)` 的固定大小张量，存储动作历史。未填充部分为零。
-   `self.action_constrained_num`: 整数，表示 `self.actions` 中前多少步是有效的约束数据。
-   `self.states`: 形状为 `(state_num, state_dim)` 的固定大小张量，存储状态历史。
-   `self.state_constrained_num`: 整数，表示 `self.states` 中前多少步是有效的观测数据。
-   `self.constrain_mode`: 当前的约束策略标识。

## 4. 关键方法详解

### 4.1 `append_new_state(new_state: Tensor)`
**功能**：将新观测到的状态追加到状态缓冲区。
**逻辑**：
1.  **维度规范化**：自动将 1D 输入 `(D,)` 转换为 2D `(1, D)`。
2.  **维度检查**：验证输入维度是否与初始化时的 `state_dim` 一致。
3.  **顺序填充**：
    -   若缓冲区未满 (`state_constrained_num < state_num`)，将新状态写入索引 `state_constrained_num` 处，并递增计数器。
    -   若缓冲区已满，当前实现选择**丢弃**新状态以保持历史顺序性（也可根据需求修改为环形覆盖）。

### 4.2 `get_action_prefix_weights() -> Tensor`
**功能**：生成动作前缀的权重掩码。
**返回**：形状为 `(action_num,)` 的张量。
**规则**：
-   若 `constrain_mode` 为 `"RTC"` 或 `"Feedback"`：前 `action_constrained_num` 个元素为 `1.0`，其余为 `0.0`。
-   若 `constrain_mode` 为 `"None"`：全 `0.0`。

### 4.3 `get_state_prefix_weights() -> Tensor`
**功能**：生成状态前缀的权重掩码。
**返回**：形状为 `(state_num,)` 的张量。
**规则**：
-   仅当 `constrain_mode` 为 `"Feedback"` 时：前 `state_constrained_num` 个元素为 `1.0`，其余为 `0.0`。
-   其他模式（`"RTC"`, `"None"`）：全 `0.0`（因为标准 RTC 不直接约束状态潜变量，只约束动作）。

### 4.4 `get_prefix_weights() -> Tensor`
**功能**：获取完整的组合权重向量，供 Scheduler 直接使用。
**返回**：形状为 `(action_num + state_num,)` 的张量。
**结构**：`[动作权重部分，状态权重部分]`。
**用途**：该张量直接传递给 `WrapperedFlowMatchScheduler`，用于计算梯度校正项 $(y_{target} - x_1) \cdot weights$ 中的 $weights$。

### 4.5 数据访问方法
-   `get_constrained_actions()`: 返回当前的 `self.actions` 张量（包含零填充部分）。
-   `get_constrained_states()`: 返回当前的 `self.states` 张量。
-   `get_constrain_mode()`: 返回当前的模式字符串。

## 5. 工作流程图解

下图展示了 `PrevChunk` 类在多 Chunk 滚动生成过程中的数据流与控制流。核心逻辑在于利用上一轮的真实观测状态更新内部缓冲区，并生成权重掩码以指导当前轮的扩散去噪过程。
┌─────────────────────────────────────────────────────────────────┐
│                    FBFM.PrevChunk 状态管理                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Chunk N-1                    Chunk N                           │
│  ┌──────────┐                ┌──────────┐                       │
│  │ 生成动作  │ ──────────────▶│ 接收反馈  │                      │
│  └──────────┘   (actions)    └──────────┘                       │
│       │                          │                              │
│       ▼                          ▼                              │
│  ┌──────────┐                ┌──────────┐                       │
│  │ 状态观测  │ ──────────────▶│ 约束注入  │                      │
│  └──────────┘   (obs/latent) └──────────┘                       │
│                                  │                              │
│                                  ▼                              │
│                          ┌──────────────┐                       │
│                          │ Scheduler.step│                      │
│                          │ constrained_y │                      │
│                          │ weights       │                      │
│                          └──────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
