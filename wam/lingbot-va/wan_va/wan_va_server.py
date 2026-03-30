# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)

# FBFM
import fbfm.policies.fbfm.modeling_rtc_fbfm as FBFM
from fbfm.policies.fbfm.configuration_rtc import RTCConfig
from torch import Tensor

class VA_PrevChunkAdapter(FBFM.PrevChunk):
    """
    Adapter that makes FBFM.PrevChunk constraints compatible with VA_server's
    (actions, latents) tensor formats.

    - VA_server actions are returned in "used channels" space; PrevChunk expects
      full `action_dim`.
    - VA_server schedulers require constrained_y/weights tensors with the same 5D
      shape as `x_t` (video: B,C,F,H,W; action: B,D,F,N,1), while PrevChunk's default
      getters return flattened 2D/1D tensors.
    """

    def __init__(self,
                    *,
                    constrain_mode: str,
                    prev_actions: Tensor | np.ndarray | None,
                    used_action_channel_ids: list[int],
                    action_num: int,
                    action_dim: int,
                    frame_chunk_size: int,
                    action_per_frame: int,
                    state_num: int,
                    latent_channel: int,
                    latent_height: int,
                    latent_width: int,
                    state_dim: int,
                    device: torch.device,
                    dtype: torch.dtype,
                    inference_delay: int = 0):
        """
        初始化约束模块，配置动作与状态的空间维度、设备类型及历史动作数据。

        该构造函数负责解析输入的历史动作数据（如有），将其转换为内部所需的二维格式，
        并计算受约束的动作数量。随后调用父类初始化方法建立基础约束结构，最后将
        生成的动作和状态张量迁移至指定的计算设备和数据类型。

        参数:
            constrain_mode (str): 约束模式字符串，定义具体的约束策略。
            prev_actions (Tensor | np.ndarray | None): 上一帧或历史动作数据，若为 None 则无历史约束。
            used_action_channel_ids (list[int]): 被启用的动作通道索引列表，用于筛选有效动作。
            action_num (int): 动作的总数量上限。
            action_dim (int): 单个动作向量的维度。
            frame_chunk_size (int): 时间帧块的大小，用于处理时序数据切片。
            action_per_frame (int): 每一帧包含的动作数量。
            state_num (int): 状态的总数量上限。
            latent_channel (int): 潜在空间（latent space）的通道数。
            latent_height (int): 潜在空间的高度。
            latent_width (int): 潜在空间的宽度。
            state_dim (int): 单个状态向量的维度。
            device (torch.device): 指定模型运行所在的计算设备（如 CPU 或 CUDA）。
            dtype (torch.dtype): 指定张量运算的数据类型（如 float32）。
            inference_delay (int): 推理延迟步数，默认为 0。

        返回:
            None
        """
        self._frame_chunk_size = frame_chunk_size
        self._action_per_frame = action_per_frame
        self._action_dim = action_dim
        self._state_num = state_num
        self._latent_channel = latent_channel
        self._latent_height = latent_height
        self._latent_width = latent_width
        self._state_dim = state_dim
        self._device = device
        self._dtype = dtype
        self._used_action_channel_ids = used_action_channel_ids

        # 初始化历史动作转换变量，若存在历史动作则进行格式转换并计算受约束数量
        actions_2d = None
        action_constrained_num = 0
        if prev_actions is not None:
            actions_2d, action_constrained_num = self._va_prev_actions_to_prev_actions_2d(
                prev_actions,
                used_action_channel_ids=used_action_channel_ids,
                action_dim=action_dim,
                frame_chunk_size=frame_chunk_size,
                action_per_frame=action_per_frame,
            )
            action_constrained_num = min(action_constrained_num, action_num)

        # 调用父类构造函数，传入处理后的动作数据、状态占位符及各项配置参数以完成基础初始化
        super().__init__(
            constrain_mode=constrain_mode,
            actions=actions_2d,
            action_constrained_num=action_constrained_num,
            action_num=action_num,
            action_dim=action_dim,
            states=None,
            state_constrained_num=0,
            state_num=state_num,
            state_dim=state_dim,
            inference_delay=inference_delay,
        )

        # 将初始化生成的动作和状态张量迁移至指定的计算设备并转换为设定的数据类型
        self.actions = self.actions.to(device=self._device, dtype=self._dtype)
        self.states = self.states.to(device=self._device, dtype=self._dtype)

    @staticmethod
    def _to_torch(x: Tensor | np.ndarray) -> Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if torch.is_tensor(x):
            return x
        raise TypeError(f"Unsupported tensor type: {type(x)}")

    def _va_prev_actions_to_prev_actions_2d(
        self,
        prev_actions: Tensor | np.ndarray,
        *,
        used_action_channel_ids: list[int],
        action_dim: int,
        frame_chunk_size: int,
        action_per_frame: int,
    ) -> tuple[Tensor, int]:
        """
        将 VA_server 输出的历史动作数据转换为 PrevChunk 内部使用的二维动作张量。

        该函数负责处理来自 VA_server 的 `prev_actions`，将其从可能的多种维度排列
        (如 (C_used, F, N) 或 (F, N, C)) 统一标准化，并根据提供的通道映射关系
        还原为完整的动作维度，最后展平为二维张量供后续模型使用。

        参数:
            prev_actions (Tensor | np.ndarray): 输入的历史动作数据，通常形状为 (C, F, N) 或其变体。
            used_action_channel_ids (list[int]): 实际被使用的动作通道 ID 列表，用于将稀疏动作映射回完整维度。
            action_dim (int): 动作空间的完整维度大小 (D)。
            frame_chunk_size (int): 时间帧块的大小 (F)。
            action_per_frame (int): 每帧包含的动作数量 (N)。

        返回:
            tuple[Tensor, int]: 包含两个元素：
                - actions_2d (Tensor): 展平后的二维动作张量，形状为 (F*N, D)。
                - action_num (int): 动作总数，即 F * N。

        异常:
            ValueError: 当输入张量维度不为 3，或形状与预期的 (C, F, N)/(F, N, C) 不匹配，
                        或通道维度既不是完整维度也不是已使用维度时抛出。
        """
        pa = self._to_torch(prev_actions)
        if pa.dim() != 3:
            raise ValueError(
                f"prev_actions must be 3D, got shape={tuple(pa.shape)}")

        # 标准化张量维度顺序至 (C, F, N)，支持两种常见的输入排列方式
        if pa.shape[1] == frame_chunk_size and pa.shape[2] == action_per_frame:
            pa_cfn = pa
        elif pa.shape[0] == frame_chunk_size and pa.shape[1] == action_per_frame:
            pa_cfn = pa.permute(2, 0, 1)
        else:
            raise ValueError(
                "prev_actions shape mismatch. "
                f"Got {tuple(pa.shape)}, expected (C, {frame_chunk_size}, {action_per_frame}) "
                f"or ({frame_chunk_size}, {action_per_frame}, C).")

        c = pa_cfn.shape[0]
        c_used = len(used_action_channel_ids)

        # 根据通道维度判断是完整动作空间还是稀疏动作空间，并构建统一的 (F, N, D) 三维张量
        if c == action_dim:
            # 输入已是完整维度，直接调整轴顺序为 (F, N, D)
            actions_3d = pa_cfn.permute(1, 2, 0).contiguous()
        elif c == c_used:
            # 输入为稀疏维度，初始化全零张量并将有效通道映射到对应位置
            actions_3d = torch.zeros(frame_chunk_size,
                                        action_per_frame,
                                        action_dim,
                                        dtype=pa_cfn.dtype,
                                        device=pa_cfn.device)
            for used_idx, full_ch_id in enumerate(used_action_channel_ids):
                actions_3d[:, :, full_ch_id] = pa_cfn[used_idx]
        else:
            raise ValueError(
                f"prev_actions channel dim mismatch: got C={c}, expected "
                f"C={action_dim} (full) or C={c_used} (used).")

        # 将三维动作张量展平为二维 (ActionNum, ActionDim)，并确保设备和数据类型一致
        actions_2d = actions_3d.reshape(-1, action_dim).to(device=self._device,
                                                            dtype=self._dtype)
        return actions_2d, actions_2d.shape[0]

    def append_new_state(self, new_state: Tensor) -> None:
        """
        接收并处理新的状态张量，支持多种维度格式以适配 VA_server 的潜在反馈或常规状态更新。
        
        该函数重载了父类方法，专门用于处理形状为 (B=1, C, F, H, W) 的五维潜在反馈张量，
        将其按帧拆解并逐个追加到状态队列中。同时也兼容二维 (T, state_dim) 和一维 (state_dim) 
        的输入格式。在追加过程中会检查状态数量约束，若达到上限则停止追加。

        参数:
            new_state (Tensor): 输入的新状态数据。
                - 若为 5 维张量：形状应为 (1, C, F, H, W)，其中 B 必须为 1，C 需匹配内部配置的潜在通道数；
                - 若为 2 维张量：形状应为 (T_new, state_dim)，表示多个时间步的状态向量；
                - 若为 1 维张量：形状应为 (state_dim)，表示单个状态向量；
                - 若为 None：直接返回，不执行任何操作。
        
        返回:
            None: 无返回值，直接修改内部状态队列。
        
        异常:
            ValueError: 当输入张量维度不支持，或 5 维输入的批次大小不为 1、通道数不匹配时抛出。
        """
        if new_state is None:
            return

        # 确保输入转换为 torch 张量格式
        ns = new_state if torch.is_tensor(new_state) else torch.as_tensor(new_state)

        # 处理五维潜在反馈张量 (B=1, C, F, H, W)
        if ns.dim() == 5:
            if ns.shape[0] != 1:
                raise ValueError(f"Expected latent feedback B=1, got B={ns.shape[0]}")
            if ns.shape[1] != self._latent_channel:
                raise ValueError(
                    f"latent channel mismatch: got C={ns.shape[1]}, expected {self._latent_channel}"
                )
            ns_cfHW = ns[0]  # (C, F, H, W)
            f_new = ns_cfHW.shape[1]
            # 按帧遍历并逐个追加状态向量
            for ft in range(f_new):
                if self.state_constrained_num >= self.state_num:
                    break
                vec = ns_cfHW[:, ft, :, :].reshape(self._state_dim)
                super().append_new_state(vec.to(device=self._device, dtype=self._dtype))
            return

        # 处理二维状态张量 (T_new, state_dim)
        if ns.dim() == 2:
            # (T_new, state_dim)
            for i in range(ns.shape[0]):
                if self.state_constrained_num >= self.state_num:
                    break
                vec = ns[i].reshape(self._state_dim)
                super().append_new_state(vec.to(device=self._device, dtype=self._dtype))
            return

        # 处理一维状态向量 (state_dim)
        if ns.dim() == 1:
            super().append_new_state(ns.to(device=self._device, dtype=self._dtype))
            return

        # 抛出 unsupported shape 错误
        raise ValueError(f"Unsupported latent shape for append_new_state: {tuple(ns.shape)}")


    def get_constrained_states(self) -> Tensor:
        # (B=1, C, F, H, W)
        states_4d = self.states.reshape(self._state_num,
                                         self._latent_channel,
                                         self._latent_height,
                                         self._latent_width)  # (F, C, H, W)
        return states_4d.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    def get_state_prefix_weights(self) -> Tensor:
        # (B=1, 1, F, 1, 1)
        w_1d = super().get_state_prefix_weights().to(device=self._device,
                                                     dtype=self._dtype)
        return w_1d[None, None, :, None, None].contiguous()

    def get_constrained_actions(self) -> Tensor:
        # (B=1, D, F, N, 1)
        actions_3d = self.actions.reshape(self._frame_chunk_size,
                                            self._action_per_frame,
                                            self._action_dim)  # (F, N, D)
        return actions_3d.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).contiguous()

    def get_action_prefix_weights(self) -> Tensor:
        # (B=1, 1, F, N, 1)
        w_1d = super().get_action_prefix_weights().to(device=self._device,
                                                     dtype=self._dtype)
        w_2d = w_1d.reshape(self._frame_chunk_size, self._action_per_frame)
        return w_2d[None, None, :, :, None].contiguous()

class WrapperedFlowMatchScheduler(FlowMatchScheduler):
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
        rtc_config: RTCConfig = None,
        ):
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps,
            extra_one_step=extra_one_step,
            reverse_sigmas=reverse_sigmas,
            exponential_shift=exponential_shift,
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
            )
        self.rtc_config = rtc_config
    @torch.enable_grad()
    def step(self,
             original_denoise_step_partial,
             x_t,
             timestep, 
             sample, 
             to_final=False, 
             constrained_y : Tensor | None = None,
             weights : Tensor | None = None, 
             device = None, 
             **kwargs
             ):
        # 原step函数
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        
        # 去噪时间方案
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps
                           or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]

        # 转换为RTC中 由1到0的虚时间轴 由tau表示
        tau = 1 - sigma

        if constrained_y is not None and weights is not None:
            x_t = x_t.clone().detach()

            # 疑似没有用 weights 已经自动过了
            # batch_size = x_t.shape[0] # B
            # chunk_size = x_t.shape[1] # T
            # action_dim = x_t.shape[2] # N

            with torch.enable_grad():
                v_t = original_denoise_step_partial(x_t)
                x_t.requires_grad_(True)

                x1_t = x_t - sigma * v_t  # noqa: N806
                err = (constrained_y - x1_t) * weights
                grad_outputs = err.clone().detach()
                correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

            max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
            tau_tensor = torch.as_tensor(tau)
            squared_one_minus_tau = (1 - tau_tensor) ** 2
            inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
            c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
            guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
            guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

            v_result_t = v_t - guidance_weight * correction

        else:
            # First step, no guidance - return v_t
            v_result_t = original_denoise_step_partial(x_t)

        
        prev_sample = sample + v_result_t * (sigma_ - sigma)

        return prev_sample

class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.rtc_config = RTCConfig(
            # TODO: 暂时用这个，后面改成从job_config中读取
        )

        self.last_action = None # 上次推理得到的动作组

        # FBFM装饰后得Scheduler
        self.scheduler = WrapperedFlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True,
                                            rtc_config=self.rtc_config)
        self.action_scheduler = WrapperedFlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
            rtc_config=self.rtc_config)
        
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        
        # # VAE 的 3D 卷积需要至少 2 帧，如果不足则重复最后一帧
        # min_frames = 2
        # if len(images) < min_frames:
        #     original_len = len(images)
        #     last_image = images[-1]
        #     images = images + [last_image] * (min_frames - len(images))
        #     logger.warning(f"Input has only {original_len} frames, padding to {min_frames} frames by repeating the last frame")
        
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        self.last_action = None
        if hasattr(self, 'prev_chunk_left_over'):
            delattr(self, 'prev_chunk_left_over')
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)
    
        # 1. Video Generation Loop
        for i, t in enumerate(tqdm(timesteps)):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
            
            input_dict = self._prepare_latent_input(
                latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id
            )
            
            # 使用通用去噪函数
            denoise_fn = self.get_denoise_fn(
                input_dict, 
                last_step, 
                frame_chunk_size, 
                mode='video',
                guidance_scale=self.job_config.guidance_scale,
                # `need_patch` is kept for backward compatibility, but denoise_fn always
                # converts transformer outputs back to x_t's 5D shape.
                need_patch=True
            )
            
            latents = self.scheduler.step(
                original_denoise_step_partial=denoise_fn,
                x_t=latents,
                timestep=t,
                sample=latents,
                to_final=last_step,
                constrained_y=self.prev_chunk_left_over.get_constrained_states() if hasattr(self, 'prev_chunk_left_over') else None,
                weights=self.prev_chunk_left_over.get_state_prefix_weights() if hasattr(self, 'prev_chunk_left_over') else None,
                device=self.device
            )
            
            latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
        
        # 2. Action Generation Loop
        for i, t in enumerate(tqdm(action_timesteps)):
            last_step = i == len(action_timesteps) - 1
            action_cond = torch.zeros(
                [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
                device=self.device, dtype=self.dtype
            ) if frame_st_id == 0 else None

            input_dict = self._prepare_latent_input(
                None, actions, t, t, None, action_cond, frame_st_id=frame_st_id
            )
            
            # 复用通用去噪函数
            denoise_fn = self.get_denoise_fn(
                input_dict, 
                last_step, 
                frame_chunk_size, 
                mode='action',
                guidance_scale=self.job_config.action_guidance_scale,
                need_patch=False
            )
            
            actions = self.action_scheduler.step(
                original_denoise_step_partial=denoise_fn,
                x_t=actions,
                timestep=t,
                sample=actions,
                to_final=last_step,
                constrained_y=self.prev_chunk_left_over.get_constrained_actions(),
                weights=self.prev_chunk_left_over.get_action_prefix_weights(),
                device=self.device
            )

            actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _feedback(self, obs):
        # 1. 将obs转换成latent
        latent_model_input = self._encode_obs(obs)
        # 2. 将latent输入加入反馈队列（权重自主维护）
        self.prev_chunk_left_over.append_new_state(latent_model_input)
        
    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)
        feedback = obs.get('feedback', False)    # 状态反馈标志

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif feedback:
            # FBFM 处理中间帧逻辑
            # 第4帧获取到之后才会进入这个循环
            logger.info(f"################# Feedback #################")
            self._feedback(obs=obs)
            return dict()
        elif compute_kv_cache:
            logger.info(f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")

            frame_chunk_size = self.job_config.frame_chunk_size
            action_per_frame = self.action_per_frame
            action_num = frame_chunk_size * action_per_frame

            latent_channel = getattr(self.transformer.config, 'in_channels', 48)

            state_num = frame_chunk_size
            state_dim = latent_channel * self.latent_height * self.latent_width

            # Build PrevChunk adapter so FBFM constraints can work with VA outputs.
            self.prev_chunk_left_over = VA_PrevChunkAdapter(
                constrain_mode="Feedback",
                prev_actions=self.last_action,
                used_action_channel_ids=self.job_config.used_action_channel_ids,
                action_num=action_num,
                action_dim=self.job_config.action_dim,
                frame_chunk_size=frame_chunk_size,
                action_per_frame=action_per_frame,
                state_num=state_num,
                latent_channel=latent_channel,
                latent_height=self.latent_height,
                latent_width=self.latent_width,
                state_dim=state_dim,
                device=self.device,
                dtype=self.dtype,
                inference_delay=16,
            )

            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            self.last_action = action
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs

    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)
    
    def get_denoise_fn(self, input_dict, last_step, frame_chunk_size, 
                    mode='video', guidance_scale=1.0, need_patch=False):
        """
        返回一个可求导的去噪函数，支持 video 和 action 两种模式
        
        Args:
            input_dict: 输入字典，包含 latent_res_lst 或 action_res_lst
            last_step: 是否为最后一步
            frame_chunk_size: 帧块大小
            mode: 'video' 或 'action'
            guidance_scale: CFG 引导系数
            need_patch: 保留参数（当前实现中不再影响逻辑）；函数会始终把 transformer 输出还原为与 `x_t` 相同的 5D 形状。
        """
        assert mode in ['video', 'action'], "mode must be 'video' or 'action'"
        
        def denoise_fn(x_t):
            # 根据模式选择对应的 key
            res_key = 'latent_res_lst' if mode == 'video' else 'action_res_lst'
            input_dict[res_key]['noisy_latents'] = x_t
            
            # 调用 transformer
            noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict[res_key]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=(mode == 'action')
            )

            # 后处理：无论 last_step 与否，都要把 transformer 输出还原为与 x_t 相同的 5D 形状，
            # 否则 scheduler.step 内的 sample + v*(...) 会出现维度不匹配。
            if mode == 'video':
                noise_pred = data_seq_to_patch(
                    self.job_config.patch_size,
                    noise_pred,
                    frame_chunk_size,
                    self.latent_height,
                    self.latent_width,
                    batch_size=2 if self.use_cfg else 1,
                )
            elif mode == 'action':
                noise_pred = rearrange(noise_pred,
                                       'b (f n) c -> b c f n 1',
                                       f=frame_chunk_size)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # CFG 处理（统一逻辑）
            if guidance_scale > 1:
                noise_pred = noise_pred[1:] + guidance_scale * (
                    noise_pred[:1] - noise_pred[1:])
            else:
                noise_pred = noise_pred[:1]

            return noise_pred
        
        return denoise_fn

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
