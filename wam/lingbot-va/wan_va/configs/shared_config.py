# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = True

# FBFM / RTC-style inference-time guidance
va_shared_cfg.fbfm_enabled = False
va_shared_cfg.fbfm_execution_horizon = 4
va_shared_cfg.fbfm_inference_delay = 0
va_shared_cfg.fbfm_max_guidance_weight = 10.0
va_shared_cfg.fbfm_debug = False
va_shared_cfg.fbfm_debug_maxlen = 100
