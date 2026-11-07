# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni Thinker inputs at verl's Megatron BSHD model-call boundary."""

from copy import copy, deepcopy
from typing import Callable

import torch
from verl.models.mcore.util import build_vlm_attn_mask_bshd, postprocess_bshd_engine, preprocess_bshd_engine
from verl.utils.megatron_utils import unwrap_model

_MULTIMODAL_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "input_features",
    "feature_attention_mask",
    "audio_feature_lengths",
)


def prepare_qwen3_omni_megatron_config(model_config, engine_config):
    """Validate Thinker support and return an isolated Megatron config view."""
    if model_config.hf_config.model_type != "qwen3_omni_moe" or model_config.model_stage != "thinker":
        raise ValueError("The Omni Megatron engine currently supports Qwen3-Omni Thinker only.")
    if getattr(getattr(model_config, "mtp", None), "enable", False):
        raise ValueError("Qwen3-Omni Megatron does not support MTP because the Thinker constructs M-RoPE.")
    if engine_config.use_remove_padding or engine_config.use_fused_kernels:
        raise ValueError("Qwen3-Omni Megatron requires use_remove_padding=false and use_fused_kernels=false.")
    if engine_config.pipeline_model_parallel_size != 1 or engine_config.context_parallel_size != 1:
        raise ValueError("Qwen3-Omni Megatron BSHD forward currently requires PP=CP=1.")
    if getattr(engine_config, "dynamic_context_parallel", False):
        raise ValueError("Qwen3-Omni Megatron BSHD forward does not support dynamic CP.")
    if getattr(getattr(engine_config, "router_replay", None), "mode", "disabled") != "disabled":
        raise ValueError("Qwen3-Omni Megatron BSHD forward does not support router replay.")
    # Upstream module construction reads text_config.hidden_size even for
    # policy models. Keep this compatibility view private to the engine;
    # the worker/rollout retain the original nested Omni configuration.
    model_config = copy(model_config)
    model_config.hf_config = deepcopy(model_config.hf_config)
    model_config.hf_config.text_config = model_config.hf_config.thinker_config.text_config
    return model_config


def qwen3_omni_forward_model_engine(
    model,
    input_ids: torch.Tensor,
    multi_modal_inputs: dict,
    *,
    logits_processor: Callable | None = None,
    logits_processor_args: dict | None = None,
    vision_model: bool = False,
    pad_token_id: int | None = None,
    forced_max_seqlen: int | None = None,
):
    """Follow verl's BSHD forward, passing audio directly and letting Thinker build M-RoPE.

    MTP, fused kernels, THD, PP and CP are rejected by ``OmniMegatronEngine``.
    Logits processing and BSHD postprocessing match pinned verl's model forward.
    """
    unwrapped_model = unwrap_model(model)
    post_process = unwrapped_model.post_process
    use_fp8_padding = unwrapped_model.config.fp8 in ("e4m3", "hybrid")
    input_ids_bshd, attention_mask_bshd, _ = preprocess_bshd_engine(
        input_ids,
        pre_process=unwrapped_model.pre_process,
        use_fp8_padding=use_fp8_padding,
        forced_max_seqlen=forced_max_seqlen,
    )
    if vision_model:
        input_ids_bshd, attention_mask = build_vlm_attn_mask_bshd(
            input_ids, input_ids.shape[0], pad_token_id, forced_max_seqlen=forced_max_seqlen
        )
    else:
        attention_mask = attention_mask_bshd

    model_kwargs = {
        key: multi_modal_inputs[key].to(input_ids.device)
        for key in _MULTIMODAL_KEYS
        if key in multi_modal_inputs and multi_modal_inputs[key] is not None
    }
    output_orig = model(
        input_ids=input_ids_bshd,
        attention_mask=attention_mask,
        position_ids=None,
        **model_kwargs,
    )

    if post_process and logits_processor is not None:
        processor_args = {
            key: preprocess_bshd_engine(
                value,
                pre_process=True,
                need_roll=(key == "label"),
                use_fp8_padding=use_fp8_padding,
                forced_max_seqlen=forced_max_seqlen,
            )[0]
            for key, value in (logits_processor_args or {}).items()
            if key not in ("loss_mask", "response_attention_mask")
        }
        output_dict = logits_processor(output_orig, **processor_args)
        return {
            key: postprocess_bshd_engine(value, attention_mask_bshd, post_process=True)
            for key, value in output_dict.items()
        }
    return postprocess_bshd_engine(output_orig, attention_mask_bshd, post_process=post_process)
