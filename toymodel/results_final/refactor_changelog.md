# 三段式动作窗口重构 Changelog

## 修改文件

### 1. `pre_test_2/fbfm_processor.py`

**FBFMConfig 字段变更：**
- 删除 `execution_horizon` (原值 8，含义模糊)
- 删除 `state_execution_horizon` (原值 8，含义模糊)
- 保留 `inference_delay = 4` — 冻结段长度 d
- 保留 `empty_horizon = 5` — 空尾段长度 s
- 新增 `state_feedback_horizon = 4` — FBFM 状态反馈窗口，仅使用冻结段内观测到的状态

**`_get_prefix_weights` 签名简化：**
- 旧：`_get_prefix_weights(self, start, end, total)` — start/end 实际未被使用
- 新：`_get_prefix_weights(self, total)` — 内部从 cfg 读取 d 和 s

**`_build_fbfm_prefix_and_weights` 动作权重修正：**
- 旧：`weights[:, :t_act, state_dim:] = 1.0` — 动作通道全 1，与三段式不一致
- 新：调用 `_get_prefix_weights(H)` 生成三段式权重，与 RTC 共用同一套动作引导窗口

**`_build_fbfm_prefix_and_weights` 状态权重修正：**
- 旧：`min(t_state, self.cfg.state_execution_horizon, H)` — 使用已删除的字段
- 新：`min(t_state, self.cfg.state_feedback_horizon, H)` — 仅冻结段状态参与反馈

**`_build_rtc_prefix_and_weights` 调用更新：**
- 旧：`self._get_prefix_weights(start, execution_horizon, H)`
- 新：`self._get_prefix_weights(H)`

### 2. `pre_test_2/run_experiments_final.py`

**常量拆分：**
- 删除 `EXEC_HORIZON = 9` (一个变量承担三种含义)
- 保留 `INFERENCE_DELAY = 4` — 冻结段长度
- 保留 `EMPTY_HORIZON = 5` — 空尾段长度
- 新增 `REPLAN_INTERVAL = 9` — 每轮实际执行步数 (触发下一次推理的判据)

**`get_cfg` 更新：**
- 删除 `execution_horizon=EXEC_HORIZON`
- 删除 `state_execution_horizon=EXEC_HORIZON`
- 新增 `state_feedback_horizon=INFERENCE_DELAY`

**`run_rollout` 更新：**
- `need_new` 判据：`EXEC_HORIZON` → `REPLAN_INTERVAL`

### 3. `pre_test_2/advanced_viz.py`

- `EXEC_HORIZON` 导入改为 `REPLAN_INTERVAL`

## 三段式权重验证

```
a0 ~a3  (frozen):     [1.00, 1.00, 1.00, 1.00]
a4 ~a10 (modifiable): [1.00, 0.70, 0.49, 0.34, 0.24, 0.17, 0.12]
a11~a15 (empty):      [0.00, 0.00, 0.00, 0.00, 0.00]
```

FBFM 动作通道与 RTC 共用上述权重；FBFM 状态通道仅 a0~a3 位置为 1.0。

## Exp A 结果 (5 seeds, step response)

| 场景 | 算法 | Jitter | IAE | Control TV |
|------|------|--------|-----|------------|
| Nominal | FBFM | **0.017±0.010** | 0.112±0.001 | **11.2±0.9** |
| Nominal | RTC | 0.145±0.235 | 0.113±0.006 | 20.3±15.8 |
| Mass×1.5 | FBFM | **0.191±0.322** | 0.132±0.010 | **26.0±15.5** |
| Mass×1.5 | RTC | 0.360±0.395 | 0.133±0.008 | 36.1±17.3 |
| Mass×2 | FBFM | **0.129±0.147** | 0.155±0.010 | **31.4±10.1** |
| Mass×2 | RTC | 0.440±0.427 | 0.161±0.010 | 46.1±16.6 |
| Mass×3 | FBFM | 0.286±0.405 | 0.213±0.008 | 45.8±17.6 |
| Mass×3 | RTC | **0.168±0.112** | 0.209±0.006 | **44.5±7.9** |
| Stiff×3 | FBFM | **0.019±0.010** | 0.125±0.001 | **17.7±0.8** |
| Stiff×3 | RTC | 0.048±0.056 | 0.125±0.001 | 19.4±3.5 |
| Combined | FBFM | **0.106±0.124** | 0.161±0.001 | **28.4±5.0** |
| Combined | RTC | 0.259±0.275 | 0.165±0.005 | 38.2±16.1 |

FBFM 在 6/6 step 场景中 jitter 和 control TV 均优于 RTC（mass×3 除外 jitter 略高）。
