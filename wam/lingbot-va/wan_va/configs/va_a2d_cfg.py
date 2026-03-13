# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import va_shared_cfg

# 统一定名为 va_a2d_cfg
va_a2d_cfg = EasyDict(__name__='Config: VA a2d')
va_a2d_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_a2d_cfg.wan22_pretrained_model_name_or_path = "/share/project/caomingyu/WAM_baseline/model/checkpoints/lingbot-va"

va_a2d_cfg.attn_window = 30
va_a2d_cfg.frame_chunk_size = 4

va_a2d_cfg.env_type = 'real_robot'

va_a2d_cfg.height = 256
va_a2d_cfg.width = 256
va_a2d_cfg.action_dim = 30
va_a2d_cfg.action_per_frame = 8

va_a2d_cfg.obs_cam_keys = [
    'head_color', 'hand_left_color', 'hand_right_color'
]

va_a2d_cfg.guidance_scale = 5
va_a2d_cfg.action_guidance_scale = 1

va_a2d_cfg.num_inference_steps = 5
va_a2d_cfg.video_exec_step = -1
va_a2d_cfg.action_num_inference_steps = 10

va_a2d_cfg.snr_shift = 5.0
va_a2d_cfg.action_snr_shift = 1.0

# 16维到30维的精准映射: 左臂(0-6), 左夹爪(28), 右臂(8-14), 右夹爪(29)
va_a2d_cfg.used_action_channel_ids = [
    0, 1, 2, 3, 4, 5, 6,
    28,
    8, 9, 10, 11, 12, 13, 14,
    29
]
inverse_used_action_channel_ids = [len(va_a2d_cfg.used_action_channel_ids)] * va_a2d_cfg.action_dim
for i, j in enumerate(va_a2d_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_a2d_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_a2d_cfg.action_norm_method = 'quantiles'

va_a2d_cfg.norm_stat = {
    "q01": [-0.02209573984146118, -0.042001657485961914, -0.19282095193862914, -0.2832447585535504, -0.2806652430022417, -0.10453604771867664, 0.9072210104900286, -0.03277058362960815, -0.1514815580844879, -0.2248403549194336, -0.46330273011721657, -0.1388276881065935, -0.6105396542015172, 0.7501109606087916, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "q99": [0.1900769495964043, 0.0991011166572568, 0.026838674545287766, 0.036238033956384814, 0.035749064383159335, 0.1522854724422378, 1.0, 0.2213547611236567, 0.1901729601621619, 0.07668605804443351, 0.12786288149908312, 0.3578509543597846, 0.1539564604923019, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
}
# ##盖章norm+小noise
# va_a2d_cfg.norm_stat = {
#     "q01": [-0.009996781349182129, -0.01000348687171936, -0.010002384185791016, -0.00999525540884283, -0.010002240111629852, -0.010001439660458184, 0.9899999996754417, -0.003511091470718384, -0.06108502238988876, -0.20208461284637452, -0.3444112026902207, -0.08840293308280084, -0.053250774912084416, 0.9307591338093032, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 0.99, 0.0],
#     "q99": [0.010003218650817871, 0.00999651312828064, 0.009997615814208985, 0.010004744591157171, 0.009997759888370148, 0.009998560339541817, 1.0099999996754416, 0.17984349846839875, 0.2076052293181415, -2.4914145469670176e-05, -0.0005203389520710167, 0.200661577143925, 0.10118524519035117, 0.9999888761854049, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.01, 1.0],
# }
