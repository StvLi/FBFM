# Lingbot-VA RTC Integration Log

1. **调研与对照**
  - 阅读 `lerobot-main/src/lerobot/policies/rtc/modeling_rtc.py` 和 `lerobot-main/tests/policies/pi0_pi05/test_pi0_rtc.py`。
  - 记录在 PI0 中的集成方式：
    - `RTCProcessor` 在 policy 初始化阶段创建，并注入到模型里。
    - 推理阶段 `sample_actions` 使用 RTC 包装 `denoise_step`，当 `prev_chunk_left_over` 存在时应用前缀引导。
2. **新增 RTC 模块（Lingbot-VA）**
  - 新增 `wan_va/modules/rtc.py`：包含 `RTCConfig`、`RTCAttentionSchedule`、`RTCProcessor`，结构与 PI0 对齐，便于复用。
  - 保持实现自包含，减少对外部依赖。
3. **配置层引入 RTC 参数**
  - 在 `wan_va/configs/shared_config.py` 添加 RTC 默认配置（关闭状态）。
  - 在 `wan_va/configs/va_robotwin_cfg.py` 添加可覆盖的 RTC 配置项，方便实验切换。
4. **推理阶段接入 RTC 逻辑**
  - 在 `wan_va/wan_va_server.py` 中：
    - 初始化 `RTCProcessor`（根据 config 选择是否启用）。
    - `_infer` 增加 `prev_action_chunk` 入参，用于上一段动作前缀。
    - Action 生成循环中，使用 `RTCProcessor.denoise_step` 包裹原有 `transformer` 的动作去噪过程。
    - 推理结束缓存上一段动作，供下一次调用时作为 RTC 前缀引导。
5. **注释与可读性**
  - 在关键 RTC 相关代码处添加说明性注释（初始化、数据形状转换、前缀引导位置）。
6. **后续可选验证点**
  - 启动时在 config 中将 `rtc_enabled=True` 以开启对照组。
  - 验证 RTC 开启前后 action 输出是否有差异。

