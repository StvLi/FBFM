# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import logging
import os

import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel

logger = logging.getLogger(__name__)


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
    )
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def _register_debug_hooks(self):
        handles = []

        def should_trace(name, module):
            lname = name.lower()
            return (
                "shortcut" in lname
                or "downsample" in lname
                or "time_conv" in lname
                or module.__class__.__name__ == "WanCausalConv3d"
            )

        def hook(name):
            def _hook(module, inputs, output):
                in_shape = None
                if inputs and isinstance(inputs[0], torch.Tensor):
                    in_shape = tuple(inputs[0].shape)
                out_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
                logger.info("VAE hook %s: in=%s out=%s", name, in_shape, out_shape)
            return _hook

        for name, module in self.encoder.named_modules():
            if should_trace(name, module):
                handles.append(module.register_forward_hook(hook(name)))
        return handles

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        logger.info("encode_chunk input shape=%s", tuple(x_chunk.shape))
        feat_idx = [0]
        debug_shapes = os.getenv("WAN_VAE_DEBUG_SHAPES", "0") == "1"
        handles = self._register_debug_hooks() if debug_shapes else []
        try:
            out = self.encoder(x_chunk,
                               feat_cache=self.feat_cache,
                               feat_idx=feat_idx)
        finally:
            for handle in handles:
                handle.remove()
        enc = self.quant_conv(out)
        return enc
