import torch
from torch import Tensor
import numpy as np
import fbfm.policies.fbfm.modeling_rtc_fbfm as FBFM
from fbfm.policies.fbfm.configuration_rtc import RTCConfig
from utils import FlowMatchScheduler

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
