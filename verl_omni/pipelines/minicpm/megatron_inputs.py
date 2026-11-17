# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniCPM-o understanding inputs for the shared Omni Megatron engine."""

from copy import copy, deepcopy
from typing import Callable

import torch
from verl.models.mcore.util import postprocess_bshd_engine, preprocess_bshd_engine
from verl.utils.megatron_utils import unwrap_model


def prepare_minicpm_megatron_config(model_config, engine_config):
    """Return an isolated flat-text config view for dense MiniCPM-o 4.5 training."""
    hf_config = model_config.hf_config
    if hf_config.model_type != "minicpmo" or str(hf_config.version) != "4.5" or model_config.model_stage != "thinker":
        raise ValueError("MiniCPM Megatron currently supports MiniCPM-o 4.5 Thinker only.")
    if not model_config.trust_remote_code:
        raise ValueError("MiniCPM Megatron requires explicit trust_remote_code=True.")
    if engine_config.use_remove_padding or engine_config.use_fused_kernels:
        raise ValueError("MiniCPM Megatron requires use_remove_padding=false and use_fused_kernels=false.")
    if engine_config.pipeline_model_parallel_size != 1 or engine_config.context_parallel_size != 1:
        raise ValueError("MiniCPM Megatron requires PP=CP=1.")
    if engine_config.sequence_parallel or getattr(engine_config, "dynamic_context_parallel", False):
        raise ValueError("MiniCPM Megatron does not support sequence parallelism or dynamic CP.")
    if getattr(getattr(model_config, "mtp", None), "enable", False):
        raise ValueError("MiniCPM Megatron does not support MTP.")
    if getattr(getattr(engine_config, "router_replay", None), "mode", "disabled") != "disabled":
        raise ValueError("MiniCPM Megatron does not support router replay.")
    if getattr(engine_config, "vanilla_mbridge", False):
        raise ValueError("MiniCPM requires the Megatron-Bridge MiniCPMO builder, not vanilla_mbridge.")
    model_config = copy(model_config)
    model_config.hf_config = deepcopy(hf_config)
    model_config.hf_config.text_config = deepcopy(hf_config)
    return model_config


def minicpm_forward_model_engine(
    model,
    input_ids: torch.Tensor,
    multi_modal_inputs: dict,
    *,
    model_config,
    logits_processor: Callable | None = None,
    logits_processor_args: dict | None = None,
    vision_model: bool = False,
    pad_token_id: int | None = None,
    forced_max_seqlen: int | None = None,
):
    """Preserve dense positions, per-sample media, next-token labels and autograd."""
    from .thinker_training_adapter import MiniCPMThinkerAdapter

    unwrapped = unwrap_model(model)
    if unwrapped.config.fp8 is not None:
        raise ValueError("MiniCPM Megatron quantized forwards have not been validated.")
    ids, valid_tokens, positions = preprocess_bshd_engine(
        input_ids, pre_process=unwrapped.pre_process, forced_max_seqlen=forced_max_seqlen
    )
    if not valid_tokens.any(dim=-1).all():
        raise ValueError("MiniCPM Megatron does not accept empty token sequences.")
    if pad_token_id is not None:
        ids = ids.masked_fill(~valid_tokens, pad_token_id)
    sequence = ids.shape[1]
    causal = torch.ones((sequence, sequence), dtype=torch.bool, device=ids.device).triu(1)
    attention_mask = causal[None, None] | ~valid_tokens[:, None, None, :]
    prepared = MiniCPMThinkerAdapter.prepare_model_inputs(
        {**multi_modal_inputs, "input_ids": ids, "position_ids": positions}, {}, model_config
    )["data"]
    # Bounds are reconstructed after padding; each row must retain its own media.
    for key in ("image_bound", "audio_bounds"):
        prepared[key] = [
            torch.as_tensor(row, dtype=torch.long, device=ids.device).reshape(-1, 2) for row in prepared[key]
        ]
    for key in ("pixel_values", "tgt_sizes"):
        if len(prepared[key]) != ids.shape[0]:
            raise ValueError(f"MiniCPM {key} lost a sample during collation.")
    prepared["pixel_values"] = [[part.to(ids.device) for part in row] for row in prepared["pixel_values"]]
    if isinstance(prepared["audio_features"], torch.Tensor):
        prepared["audio_features"] = prepared["audio_features"].to(ids.device)
    output = model(input_ids=ids, position_ids=positions, attention_mask=attention_mask, media=prepared)
    if unwrapped.post_process and logits_processor is not None:
        args = {
            key: preprocess_bshd_engine(value, need_roll=(key == "label"), forced_max_seqlen=forced_max_seqlen)[0]
            for key, value in (logits_processor_args or {}).items()
            if key not in ("loss_mask", "response_attention_mask")
        }
        return {
            key: postprocess_bshd_engine(value, valid_tokens, post_process=True)
            for key, value in logits_processor(output, **args).items()
        }
    return postprocess_bshd_engine(output, valid_tokens, post_process=unwrapped.post_process)
