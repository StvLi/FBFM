# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_a2d_cfg import va_a2d_cfg 
import os

# 统一定名为 va_a2d_train_cfg
va_a2d_train_cfg = EasyDict(__name__='Config: VA a2d train')
va_a2d_train_cfg.update(va_a2d_cfg)

# 你的真机数据集根目录
# va_a2d_train_cfg.dataset_path = '/share/project/caomingyu/WAM_baseline/data/real_all_lite'
va_a2d_train_cfg.dataset_path = '/share/project/caomingyu/WAM_baseline/data/real_shortest_task_lite'
va_a2d_train_cfg.empty_emb_path = os.path.join(va_a2d_train_cfg.dataset_path, 'empty_emb.pt')

va_a2d_train_cfg.enable_wandb = False
va_a2d_train_cfg.load_worker = 16  # H100 多开 worker 加速读取
va_a2d_train_cfg.save_interval = 50
va_a2d_train_cfg.gc_interval = 50
va_a2d_train_cfg.cfg_prob = 0.1

# 训练超参数
# va_a2d_train_cfg.learning_rate = 1e-4
va_a2d_train_cfg.learning_rate = 1e-5
va_a2d_train_cfg.beta1 = 0.9
va_a2d_train_cfg.beta2 = 0.95
va_a2d_train_cfg.weight_decay = 1e-1
va_a2d_train_cfg.warmup_steps = 10

va_a2d_train_cfg.batch_size = 1 
va_a2d_train_cfg.gradient_accumulation_steps = 4
va_a2d_train_cfg.num_steps = 3000