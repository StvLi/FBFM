# pre_test_2 代码重构记录

## 重构前（1个969行巨型文件 + 3个废弃文件）

```
run_experiments_final.py  969行  混合：常量/配置/rollout/metrics/可视化/实验/CLI
inference_pipeline.py     311行  废弃，无人导入
analyze_results_final.py  416行  废弃，无人导入
advanced_viz.py          1700行  废弃，功能被 run_experiments_final 覆盖
```

## 重构后（按职责拆分为6个模块）

```
config.py        50行   全局常量 + get_cfg() 配置工厂
rollout.py       95行   run_rollout() + run_pid_rollout()
metrics.py      120行   compute_metrics() + aggregate_metrics() + print_table()
plotting.py     220行   COLORS/LABELS + 4个绘图函数
experiments.py  310行   experiment_a/b/c/d + _run_all_methods + _save_json
run.py           55行   CLI入口 main()
```

## 保留不动的基础模块

```
physics_env.py      237行  物理仿真环境（无内部依赖）
flow_model.py       318行  DiT神经网络模型（无内部依赖）
fbfm_processor.py   649行  FBFM/RTC推理引擎（无内部依赖）
expert_data.py      272行  PID专家数据生成（依赖 physics_env）
train.py            242行  模型训练（依赖 expert_data, flow_model, physics_env）
```

## 删除的文件

- `run_experiments_final.py` — 已拆分为 config/rollout/metrics/plotting/experiments/run
- `inference_pipeline.py` — 早期rollout封装，功能已被 rollout.py 替代
- `analyze_results_final.py` — JSON摘要脚本，功能已被 metrics.py 覆盖
- `advanced_viz.py` — 高级可视化，功能已被 plotting.py 覆盖

## 依赖关系

```
physics_env  ←── expert_data ←── train
     ↑                              ↑
flow_model ─────────────────────────┘

fbfm_processor ←── config ←── rollout ←── experiments ←── run
                      ↑          ↑            ↑
                 physics_env  expert_data   metrics
                                           plotting
```

## 运行方式

```bash
# 旧方式（已删除）
python -m pre_test_2.run_experiments_final --exp a --seeds 5

# 新方式
python -m pre_test_2.run --exp a --seeds 5
```

## 验证

- `python -m pre_test_2.run --exp a --seeds 1` 通过，输出与重构前一致
