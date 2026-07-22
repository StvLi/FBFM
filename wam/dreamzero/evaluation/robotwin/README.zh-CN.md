# DreamZero + FBFM 接入 RoboTwin：中文说明

本文说明 `feature/dreamzero-fbfm-robotwin` 分支中的实现、运行方式、数据与权重门槛，以及三模式公平评测协议。

当前状态：代码接入、原生 RoboTwin EEF 数据转换和 CPU 验证已经完成；DreamZero-RoboTwin post-training、GPU smoke 和正式评测尚未完成。远端已下载 DreamZero-AgiBot、UMT5 与 `adjust_bottle` clean 50 条演示，Wan2.1 和独立 DreamZero 环境仍在完成下载/安装。禁止用 FastWAM、LingBot 或仅含 `config.json` 的目录冒充 DreamZero-RoboTwin 权重。

## 1. 接入目标

DreamZero 使用同一个因果 DiT 联合预测视频 flow 和动作 flow：

- 视频 latent 沿时间自回归；
- 动作在每个控制周期重新预测；
- action chunk 执行后，真实 RGB 观测由 DreamZero 自身 VAE 编码；
- 真实 observation latent 写回或替换因果视频 KV context；
- FBFM 只在推理时修正交给原始 UniPC scheduler 的联合 flow，不重新实现 UniPC 多步公式。

LingBot 实现只用于参考三模式语义、反馈缓存和评测公平性。DreamZero 的 VAE 时间压缩率、latent slot 数、action horizon 和 KV 更新时序全部由 DreamZero 自身张量推导，没有硬编码 LingBot 的“4 帧 RGB 对应 1 个 latent”。

## 2. 三种模式

统一开关为：

```bash
FBFM_CONSTRAINT_MODE=None|RTC|Feedback
```

| 模式 | 动作约束 | 视频/状态约束 | 额外 VJP | DiT velocity cache |
| --- | --- | --- | --- | --- |
| `None` | 无 | 无 | 无 | 保持官方路径 |
| `RTC` | 上一 horizon 尚未执行的 normalized action tail | 无 | 有 | 引导 step 强制重算 |
| `Feedback` | 与 RTC 相同 | 真实观测经 DreamZero VAE 得到的 latent | 有 | 引导 step 强制重算 |

三种模式必须加载同一个 DreamZero-RoboTwin checkpoint、配置、CFG、推理步数、action horizon 和模型噪声。切换模式不会修改或加载另一套权重。

## 3. FBFM 数学与实现

DreamZero 的 `FlowUniPCMultistepScheduler` 使用 flow prediction。对当前 scheduler sample `x_sigma` 和 DiT flow `v`，clean prediction 直接按 scheduler 的 sigma 约定计算：

```text
x_clean = x_sigma - sigma * v
```

联合状态写作：

```text
x = (z_video, z_action)
```

实现顺序如下：

1. 在调用 DiT 之前，对当前 noisy video/action 执行 `clone().detach().requires_grad_(True)`；
2. 运行联合 DiT，获得 video/action flow；
3. 根据 action tail 和真实 video latent 构造 masked target error；
4. 对 noisy video/action 做联合 VJP；
5. 修正 video/action flow；
6. 把修正后的 flow 交给原始 `FlowUniPCMultistepScheduler.step`；
7. scheduler 的 model history、corrector 和 solver order 仍由原实现维护。

主要代码：

- `groot/vla/model/dreamzero/modules/fbfm_guidance.py`
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

原分支顶部不可运行的 `WrapperedFlowUniPCMultistepScheduler` 已删除。主循环仍实例化原始 `FlowUniPCMultistepScheduler`。

### 3.1 `inference_mode` 与参数梯度

`torch.inference_mode()` 无法在内部用 `torch.enable_grad()` 逃逸，因此：

- `None` 继续使用官方 `torch.inference_mode()`；
- `RTC/Feedback` 外层改用 `torch.no_grad()`；
- 只在 DiT 输入 VJP 的局部范围打开梯度；
- 所有模型参数保持 `requires_grad=False`；
- TensorRT 路径不支持输入 VJP，引导模式会直接拒绝 TensorRT。

### 3.2 两 rank CFG

DreamZero 的双 rank CFG 中：

- rank 0 持有可微 conditional branch；
- rank 1 持有可微 unconditional branch；
- `send/recv` 得到的对端 flow 只用于组装全局数值 CFG，不假装它仍有 autograd 图；
- 每个 rank 对自己持有的 CFG flow 分量计算局部 VJP；
- identity Jacobian 在两个 rank 间按 `1/world_size` 分配；
- video/action correction 经 `all_reduce(SUM)` 聚合。

单测会把两个 rank 的局部 VJP 相加，并与单 rank 完整联合 VJP 比较。

### 3.3 DiT cache 与 KV cache

旧 DiT velocity 与当前 `x_t` 没有 Jacobian，因此有约束的 denoising step 必须重新运行 DiT。因果 KV context cache 可以继续使用。

每次 episode reset 会清理：

- action tail；
- feedback video latent；
- 当前语言；
- `current_start_frame`；
- conditional/unconditional KV cache；
- cross-attention cache。

UniPC scheduler 在每次 action-head forward 内重新创建，因此 scheduler history 不会跨 episode 保留。

## 4. Action chunk 与反馈对齐

RoboTwin schema 同时声明：

```json
{
  "action_horizon": 24,
  "execute_steps": 8,
  "frames_per_chunk": 4
}
```

以上示例表示：模型预测 24 步动作，客户端本周期只执行前 8 步；余下 16 步 normalized action 成为下一周期的 RTC/Feedback action prefix。`None` 也只返回相同的前 8 步，保证三模式执行协议一致，但不会使用那 16 步进行约束。

Feedback 模式不会提前用额外 VAE 预热。客户端发送的真实 RGB 先进入 feedback buffer；下一次 joint causal forward 时，再由 DreamZero 原有 VAE 编码路径产生真实 latent，并由原有代码更新 KV context。

## 5. 原生 RoboTwin embodiment

新增 embodiment：

```text
robotwin
```

默认原生布局为三相机、双臂 14 维 EEF/gripper：

- `video.cam_high`
- `video.cam_left_wrist`
- `video.cam_right_wrist`
- 左臂：position 3 + Euler rotation 3 + gripper 1
- 右臂：position 3 + Euler rotation 3 + gripper 1

相关配置：

- `groot/vla/configs/data/dreamzero/robotwin_relative.yaml`
- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
- `evaluation/robotwin/robotwin_schema.example.json`

服务启动时必须加载 checkpoint 自带的 `robotwin_schema.json`。Schema 会强制检查：

- 相机顺序必须是 high、left wrist、right wrist；
- state/action slice 必须无遗漏、无重叠地覆盖所有维度；
- `execute_steps <= action_horizon`；
- action representation 必须显式标记为 `robotwin_native*`；
- normalization metadata 文件存在且 SHA-256 一致；
- embodiment 不能是 `agibot`、`oxe_droid` 或 `droid`。

这可以防止把 AgiBot/DROID normalization 直接用于 RoboTwin。

## 6. 数据门槛

原生 RoboTwin LeRobot 数据至少需要：

```text
<dataset>/meta/embodiment.json
<dataset>/meta/modality.json
<dataset>/meta/stats.json
<dataset>/meta/relative_stats_dreamzero.json
<dataset>/meta/tasks.jsonl
<dataset>/meta/episodes.jsonl
```

其中 `meta/embodiment.json` 必须包含：

```json
{"embodiment_tag": "robotwin"}
```

训练前执行：

```bash
cd /mnt/project_eai_hs/zrm/FBFM-DreamZero-FBFM/wam/dreamzero
python -m evaluation.robotwin.validate_dataset "$ROBOTWIN_DATA_ROOT"
```

校验器会检查三相机、六组 state/action modality，以及 native relative stats。

## 7. 权重和环境

所需资产：

- `GEAR-Dreams/DreamZero-AgiBot`；
- `Wan-AI/Wan2.1-I2V-14B-480P`；
- `google/umt5-xxl`；
- post-training 后的 DreamZero-RoboTwin LoRA/checkpoint。

网络恢复后可运行：

```bash
cd /mnt/project_eai_hs/zrm/FBFM-DreamZero-FBFM/wam/dreamzero
CHECKPOINT_ROOT=/mnt/project_eai_hs/zrm/checkpoints/dreamzero \
  bash scripts/robotwin/download_assets.sh
```

不要使用以下内容代替 DreamZero 权重：

- FastWAM 的 `robotwin_uncond_3cam_384.pt`；
- LingBot checkpoint；
- 只包含 `config.json`、没有完整模型权重的目录。

## 8. Post-training

确认当前 4 张 GPU 没有他人任务后再启动：

```bash
cd /mnt/project_eai_hs/zrm/FBFM-DreamZero-FBFM/wam/dreamzero

export ROBOTWIN_DATA_ROOT=/path/to/robotwin_lerobot
export PRETRAINED_MODEL_PATH=/mnt/project_eai_hs/zrm/checkpoints/dreamzero/DreamZero-AgiBot
export WAN_CKPT_DIR=/mnt/project_eai_hs/zrm/checkpoints/dreamzero/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=/mnt/project_eai_hs/zrm/checkpoints/dreamzero/umt5-xxl
export OUTPUT_DIR=/mnt/project_eai_hs/zrm/checkpoints/dreamzero/dreamzero_robotwin_lora
export NUM_GPUS=4

bash scripts/train/robotwin_training.sh
```

脚本默认：

- LoRA post-training；
- action horizon 24；
- 三相机；
- 33 个训练视频帧；
- `num_frame_per_block=2`；
- bf16 + TF32；
- DeepSpeed ZeRO-2。

先用较小 `MAX_STEPS` 做训练 smoke，确认 loss、数据维度和 checkpoint 保存正常后再启动正式训练。

## 9. 生成 checkpoint manifest

三种模式必须引用同一个内容哈希：

```bash
python -m evaluation.robotwin.checkpoint_manifest "$MODEL_PATH" \
  --output "$MODEL_PATH/checkpoint_manifest.json"
```

该 manifest 会记录 checkpoint 内所有文件的大小、SHA-256 和整棵目录的内容哈希。评测聚合器会拒绝混用不同 checkpoint hash 的结果。

## 10. 启动服务

每个 DreamZero 实例默认使用两个 GPU：

```bash
cd /mnt/project_eai_hs/zrm/FBFM-DreamZero-FBFM/wam/dreamzero

export MODEL_PATH=/path/to/dreamzero_robotwin_checkpoint
export ROBOTWIN_SCHEMA="$MODEL_PATH/robotwin_schema.json"
export CHECKPOINT_MANIFEST="$MODEL_PATH/checkpoint_manifest.json"
export CUDA_VISIBLE_DEVICES=0,1
export PORT=29500

FBFM_CONSTRAINT_MODE=None bash scripts/robotwin/launch_server.sh
```

RTC 和 Feedback 只更改模式：

```bash
FBFM_CONSTRAINT_MODE=RTC bash scripts/robotwin/launch_server.sh
FBFM_CONSTRAINT_MODE=Feedback bash scripts/robotwin/launch_server.sh
```

不要同时用不同模式覆盖相同端口或相同输出目录。

## 11. RoboTwin websocket 协议

服务兼容以下请求：

### Reset

```python
{"reset": True, "prompt": instruction}
```

清理当前 episode 的反馈、action tail 和所有因果 cache；分布式 worker 同步 reset。

### 正常推理

```python
{
    "obs": {
        "observation.images.cam_high": high_rgb,
        "observation.images.cam_left_wrist": left_rgb,
        "observation.images.cam_right_wrist": right_rgb,
        "observation.state": state_14d,
        "task": instruction,
    }
}
```

返回：

```python
{
    "action": action,  # (action_dim, 1, execute_steps)
    "mode": "None|RTC|Feedback",
}
```

### 真实观测反馈

```python
{"obs": observation, "feedback": True}
```

该请求只缓冲真实观测，不单独运行 DiT。

### Chunk 结束通知

```python
{"obs": key_frame_list, "compute_kv_cache": True}
```

如果此前已逐帧发送 feedback，服务不会重复插入同一批 keyframe。真正的 VAE 编码和 KV 替换发生在下一次 joint causal forward。

## 12. Canonical episode manifest

先用 RoboTwin expert 初始化检查生成 candidate JSONL。每条记录至少包含：

```json
{
  "accepted": true,
  "task": "adjust_bottle",
  "config": "demo_randomized",
  "seed": 10000,
  "instruction": "adjust the bottle",
  "instruction_index": 0,
  "randomization": {},
  "background_texture": "/absolute/path/texture.png",
  "background_texture_sha256": "..."
}
```

冻结 17 tasks × 2 configs × 20 accepted episodes：

```bash
python -m evaluation.robotwin.experiment freeze \
  --candidates accepted_candidates.jsonl \
  --output canonical_episodes.jsonl
```

冻结过程会：

- 检查每个 task/config 恰好选取 20 个 accepted seed；
- 固定 instruction 与 instruction index；
- 强制 randomized episode 记录背景纹理路径和 checksum；
- 为每个 episode 生成确定性的模型噪声 seed base；
- 输出 `canonical_episodes.jsonl.sha256`。

三种模式必须复用同一份 manifest。

## 13. 断点续跑

`run_manifest.py` 每个 episode 单独保存结果，已完成项会自动跳过：

```bash
python -m evaluation.robotwin.run_manifest \
  --manifest canonical_episodes.jsonl \
  --checkpoint-manifest "$MODEL_PATH/checkpoint_manifest.json" \
  --mode Feedback \
  --output-dir results \
  -- <单episode评测命令>
```

它会向单 episode evaluator 注入：

- `FBFM_CONSTRAINT_MODE`；
- `ROBOTWIN_EPISODE_JSON`；
- `ROBOTWIN_RESULT_PATH`；
- `DREAMZERO_MODEL_NOISE_SEED_BASE`；
- `DREAMZERO_FIRST_CHUNK_NOISE_SEED`；
- `DREAMZERO_CHECKPOINT_SHA256`。

单 episode evaluator 必须在 `ROBOTWIN_RESULT_PATH` 写入至少：

```json
{"success": true}
```

runner 会补齐 mode、episode ID 和 checkpoint hash。

## 14. 聚合论文表格

将每个 episode 结果合并成一个 JSONL 后运行：

```bash
python -m evaluation.robotwin.experiment aggregate \
  --manifest canonical_episodes.jsonl \
  --results all_results.jsonl \
  --output dreamzero_fbfm_robotwin.md
```

输出列顺序为：

```text
Task | Baseline Clean | RTC Clean | Ours Clean |
       Baseline Random | RTC Random | Ours Random
```

每格显示 `success/20` 和 SR%。同时生成 JSON summary，包含 Clean、Random 和 Overall 的 success、total、rate 与 bootstrap 95% CI。

## 15. 验收顺序

严格按以下顺序推进：

1. `py_compile`、shell `bash -n` 和 CPU 单测；
2. None 与原始 UniPC 逐 step 等价；
3. finite/nonzero VJP、target sensitivity、RTC 无 state target、双 rank 聚合、reset 无泄漏；
4. 同一 seed 的两 chunk GPU smoke；
5. 单任务单 clean seed 的 None/RTC/Feedback A/B；
6. 同任务 clean/random 各 20 次；
7. 全部 17 个任务；
8. 三模式共 2040 个有效 episode 后再生成最终表格。

当前已经完成第 1–3 项的 CPU 部分，共 27 个测试通过。第 4 项及以后必须等待 post-training checkpoint、完整环境和可用 GPU。

## 16. 测试命令

```bash
cd /mnt/project_eai_hs/zrm/FBFM-DreamZero-FBFM
export PYTHONDONTWRITEBYTECODE=1

/mnt/project_eai_hs/zrm/miniconda3/envs/fastwam/bin/python -m pytest -q \
  -p no:cacheprovider wam/dreamzero/tests
```

当前结果：

```text
27 passed
```

## 17. 当前未解决问题

截至当前提交：

- 远端仍没有 post-training 后的 DreamZero-RoboTwin checkpoint；
- 原生数据已转换为 50 episodes、7188 帧的 EEF14 LeRobot 数据并通过 metadata 校验；
- 独立 Python 3.11 DreamZero 环境正在安装，Wan2.1 大文件仍在断点续传；
- 4 张 H100 仍有其他 LingBot 任务，不允许停止；
- Feedback 把真实观测 latent 对齐到下一 future video block 的时序语义，必须在真实两 chunk GPU smoke 中确认，不能只凭 CPU mask 测试判定正确；
- 因此尚无 baseline success、三模式 A/B 或完整 SR 数据。

服务成功启动、通过 health check 或完成一次 websocket 通信都不能替代策略成功率。只有在相同 checkpoint、canonical manifest 和 RoboTwin evaluator 下完成的有效 episode 才能进入最终结果。
