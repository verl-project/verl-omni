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

"""Qwen-Image Dual-GRPO AR training adapter for the composite text encoder.

``CompositeARFSDPEngineWithLMHead`` dispatches here via ``DiffusionModelARBase``
after verl builds the language-model module and before FSDP wrap. The DiT
adapter remains ``diffusers_training_adapter.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from verl_omni.pipelines.model_base import DiffusionModelARBase

logger = logging.getLogger(__name__)

__all__ = ["QwenImageDualGRPOAR"]


def _strip_vision_tower(module: torch.nn.Module) -> bool:
    """Drop the unused Qwen2.5-VL vision tower before FSDP wrap.

    Qwen-Image trains text-only AR on the pipeline text encoder. The vision tower
    is unused and must not remain trainable under FSDP2 ``ignored_params``.
    Deleting it mirrors vLLM-Omni ``QwenImagePipeline``.

    Handles both Transformers layouts: ``module.model.visual`` (newer) and
    ``module.visual`` (older).
    """
    visual_owner = None
    if hasattr(module, "model") and hasattr(module.model, "visual"):
        visual_owner = module.model
    elif hasattr(module, "visual"):
        visual_owner = module
    if visual_owner is None:
        logger.warning("Qwen-VL vision tower not found on Qwen-Image AR module; skipping strip")
        return False
    del visual_owner.visual

    # verl still runs a dummy visual forward pass
    visual_owner.visual = lambda *args, **kwargs: torch.tensor([0.0], device=module.device)

    logger.info("Stripped Qwen-VL vision tower from Qwen-Image AR module")
    return True


def _transformers_needs_qwen2_vl_rotary_patch() -> bool:
    """True when transformers dropped ``apply_multimodal_rotary_pos_emb`` (5.17+)."""
    import transformers.models.qwen2_vl.modeling_qwen2_vl as modeling

    return not hasattr(modeling, "apply_multimodal_rotary_pos_emb")


def qwen2_vl_attn_forward_tf5(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """verl ``qwen2_vl_attn_forward`` with transformers >=5.17.0 rotary API.

    Adapted from verl ``qwen2_vl_attn_forward`` with transformers <5.17.0 rotary API.

    Transformers folded ``mrope_section`` into ``Qwen2VLRotaryEmbedding`` /
    ``Qwen2_5_VLRotaryEmbedding``, so attention must call ``apply_rotary_pos_emb``
    on the already-recomposed cos/sin — matching upstream
    ``Qwen2VLAttention.forward``. Kept as a drop-in for verl's flash-attn path.

    TODO: remove once verl's ``qwen2_vl_attn_forward`` adopts the same rotary call.
    """
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb, repeat_kv
    from verl.models.transformers.qwen2_vl import _custom_flash_attention_forward, _flash_use_top_left_mask
    from verl.utils.transformers_compat import is_transformers_version_in_range

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    sliding_window = None
    if (
        getattr(self.config, "use_sliding_window", False)
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    q_len = query_states.shape[2]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if position_ids is not None and position_ids.ndim == 3:
        position_ids = position_ids[0]

    attn_output = _custom_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=q_len,
        is_causal=getattr(self, "is_causal", True),
        dropout=dropout_rate,
        sliding_window=sliding_window,
        use_top_left_mask=_flash_use_top_left_mask,
        position_ids=position_ids,
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    if is_transformers_version_in_range(min_version="4.54.0"):
        return attn_output, None
    return attn_output, None, None


def _patch_qwen2_vl_attn_rotary() -> bool:
    """Swap verl's Qwen2-VL attention rotary onto the transformers >=5.17.0 API.

    ``apply_monkey_patch`` installs ``verl.models.transformers.qwen2_vl.qwen2_vl_attn_forward``
    on the attention classes before this adapter runs. Replace that function and
    rebind any class that still points at the pre-5.17.0 import of
    ``apply_multimodal_rotary_pos_emb``.
    """
    if not _transformers_needs_qwen2_vl_rotary_patch():
        return False

    import verl.models.transformers.qwen2_vl as verl_qwen2_vl
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention

    previous = verl_qwen2_vl.qwen2_vl_attn_forward
    patched = qwen2_vl_attn_forward_tf5
    verl_qwen2_vl.qwen2_vl_attn_forward = patched
    logger.info("Patched verl qwen2_vl_attn_forward for transformers>=5.17.0 rotary API")

    rebound = False
    for attn_cls in (Qwen2VLAttention, Qwen2_5_VLAttention):
        current = attn_cls.forward
        if current is previous or current is patched or getattr(current, "__name__", "") == "qwen2_vl_attn_forward":
            if current is not patched:
                attn_cls.forward = patched
                rebound = True
    if rebound:
        logger.info("Rebound Qwen2-VL attention.forward to transformers>=5.17.0 rotary path")
    return True


@DiffusionModelARBase.register("QwenImagePipeline", algorithm="dual_grpo")
class QwenImageDualGRPOAR(DiffusionModelARBase):
    """AR training adapter for the Qwen-Image text encoder under Dual-GRPO."""

    @classmethod
    def configure_module(cls, module: torch.nn.Module, model_config) -> None:
        """Strip the unused vision tower and fix verl's Qwen2-VL rotary patch.

        verl's ``apply_monkey_patch`` installs ``qwen2_vl_base_forward`` /
        ``qwen2_vl_attn_forward`` on the model class inside ``_build_module``.
        The dummy visual path unpacks the plain tensor the stripped tower returns.
        On transformers >=5.17.0, rebind attention to ``apply_rotary_pos_emb``.
        """
        _patch_qwen2_vl_attn_rotary()
        _strip_vision_tower(module)

    @classmethod
    def get_fsdp_ignored_module_names(cls, model_config) -> list[str]:
        # Vision tower is deleted in ``configure_module``; do not ignore a missing subtree.
        return []
