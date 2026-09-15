# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Model-scoped input adaptation for the upstream Megatron BSHD forward."""

from contextlib import contextmanager

import torch

_AUDIO_KEYS = ("input_features", "feature_attention_mask", "audio_feature_lengths")


@contextmanager
def qwen3_omni_megatron_inputs(model: torch.nn.Module, multi_modal_inputs: dict):
    """Forward audio tensors and let the Thinker construct multimodal M-RoPE.

    The pinned verl forward only copies image/video inputs and otherwise supplies
    ordinary text position IDs. A temporary module hook keeps the upstream loss,
    padding and pipeline schedule intact; it never patches another model's code.
    """

    def prepare_inputs(_module, args, kwargs):
        if kwargs.get("packed_seq_params") is not None:
            raise ValueError("Qwen3-Omni Megatron requires BSHD (use_remove_padding=false).")
        kwargs = {**kwargs, "position_ids": None}
        device = kwargs["input_ids"].device
        for key in _AUDIO_KEYS:
            value = multi_modal_inputs.get(key)
            if value is not None:
                kwargs[key] = value.to(device)
        return args, kwargs

    handle = model.register_forward_pre_hook(prepare_inputs, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
