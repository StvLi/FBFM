# FBFM 1D Pre-experiment (State Feedback Flow-Matching)

## 概要

本预实验使用 1D 质量-弹簧-阻尼器系统验证 **FBFM (State Feedback Flow-Matching)** 算法的有效性。

### 核心思路

在 Flow-matching 生成 action chunk 的去噪过程中，将执行过程中**实时获得的物理状态**
作为已知反馈信号，通过 PiGDM 梯度引导约束状态预测，从而影响动作序列的生成。

### 算法公式

$$v_{\text{FBFM}}(X^\tau, o, \tau) = v(X^\tau, o, \tau) + k_p \cdot (Y - \hat{X}^1)^\top \text{diag}(W) \cdot \frac{\partial \hat{X}^1}{\partial X^\tau}$$

其中 $k_p = \min\left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right)$，
$Y$ 是包含状态反馈和动作反馈的目标值，$W$ 是动态掩码。

## 文件结构

```
pre_test_2/
├── README.md                 # 本文件
├── __init__.py
├── physics_env.py            # 1D 物理仿真环境 (m·ẍ + c·ẋ + kx = u + Δ)
├── expert_data.py            # PID 专家数据生成 & 数据集管理
├── flow_model.py             # 轻量级 DiT 流匹配模型
├── train.py                  # 训练脚本
├── fbfm_processor.py         # FBFM 核心推理引擎 (对标 modeling_rtc.py)
├── inference_pipeline.py     # 闭环 Rollout 推理管线
├── visualize.py              # 可视化工具
└── run_experiments.py        # 主实验脚本 (3个实验场景)
```

## 快速开始

```bash
# 从项目根目录运行 (rtc/)
cd /path/to/rtc

# 1. 安装依赖
pip install torch numpy matplotlib

# 2. 训练模型 (约2-5分钟)
python -m pre_test_2.train --epochs 200 --batch_size 256

# 3. 运行全部实验
python -m pre_test_2.run_experiments

# 或单独运行某个实验
python -m pre_test_2.run_experiments --exp static
python -m pre_test_2.run_experiments --exp disturbance
python -m pre_test_2.run_experiments --exp delay
```

## 实验说明

### 实验 1: 静态追踪 (Static Tracking)

- **描述**: 无干扰下，Vanilla FM / RTC / FBFM 的对比
- **目的**: 验证 `max_guidance_weight` 不会破坏原始性能
- **指标**: State MSE, Position MSE, Action Jitter

### 实验 2: 瞬时干扰 (Instantaneous Disturbance) ⭐核心

- **描述**: 在 chunk 执行中途给位移加一个突然偏移
- **目的**: 验证 FBFM 的状态反馈能实时修正动作
- **指标**: MSE, Recovery Steps, Peak Error

### 实验 3: 延迟补偿 (Delay Compensation)

- **描述**: 模拟不同的 inference_delay (0, 1, 2, 4, 6 步)
- **目的**: 验证 RTC/FBFM 框架对系统延迟的鲁棒性
- **指标**: 各 delay 下的 State MSE, Action Jitter

## 设计说明

### 与 `modeling_rtc.py` 的对应关系

| pre_test_2 模块 | 原始代码 | 说明 |
|---|---|---|
| `FBFMConfig` | `RTCConfig` | 配置参数完全对应 |
| `PrevChunkInfo` | `RTCPrevChunk` | 携带动作残余 + 状态反馈 |
| `FBFMProcessor._guided_denoise_step` | `RTCProcessor.denoise_step` | 核心算法 1:1 复现 |
| `fbfm_sample` | — | 高层推理入口，处理归一化 |

### 为什么不直接 import `modeling_rtc.py`?

原代码依赖 `lerobot` 包的类型和配置系统（如 `RTCAttentionSchedule`），
在预实验环境中不便安装。因此我们 **忠实复现** 了核心算法逻辑，确保数学公式
和代码执行路径与原版一致。

### 数据布局

严格遵循 `configuration_rtc.py` 中的约定：

```
chunk X[t] = [state(t), action(t)] = [x, ẋ, u]
         dim:  [  2   ,   1    ]
```

每个 chunk 形状为 `(H=16, 3)`，先状态后动作。

## 预期结果

1. **实验 1**: 三种方法性能接近，FBFM 不显著劣化
2. **实验 2**: FBFM 在干扰后显著快于 Vanilla/RTC 恢复目标追踪
3. **实验 3**: FBFM/RTC 在中等 delay 下优于 Vanilla，大 delay 下优势减弱

## 输出

结果保存在 `pre_test_2/results/` 下：
```
results/
├── exp1_static/
│   ├── trajectories.png
│   ├── errors.png
│   └── metrics.png
├── exp2_disturbance/
│   ├── trajectories.png
│   ├── errors.png
│   ├── recovery.png
│   └── metrics.png
└── exp3_delay/
    └── delay_sweep.png
```
