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

"""Shared helpers for verl-omni training engines."""

from __future__ import annotations

import functools
import types
from typing import TYPE_CHECKING, Optional

import torch
from verl.utils.transformers_compat import unpack_visual_output
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from verl_omni.workers.config import DiffusionModelConfig

if TYPE_CHECKING:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration


def patch_composite_ar_engine_fsdp_build(
    ar_engine: FSDPEngineWithLMHead,
    diffusion_model_config: DiffusionModelConfig,
) -> None:
    """Use ``DiffusersFSDPEngine._build_fsdp_module`` for the composite AR backend.

    verl's ``FSDPEngine._build_fsdp_module`` (``transformer_impl.py``) calls upstream
    ``verl.utils.fsdp_utils.apply_fsdp2`` without ``ignored_names``. verl-omni's diffusers
    engine passes adapter-declared subtrees into ``verl_omni.utils.fsdp_utils.apply_fsdp2``
    so frozen towers skipped on some micro-batches stay unsharded under FSDP2.
    """
    ar_engine._verl_omni_diffusion_model_config = diffusion_model_config

    def _build_fsdp_module(self, module):
        from verl.utils.activation_offload import enable_activation_offloading
        from verl.utils.torch_dtypes import PrecisionType

        from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine

        hf_model_config = self.model_config
        diffusion_cfg = self._verl_omni_diffusion_model_config

        # verl diff part
        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
        else:
            param_dtype = torch.bfloat16

        self._autocast_dtype = param_dtype
        if param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # apply verl-omni _build_fsdp_module
        # temporarily sets self.model_config to DiffusionModelConfig
        # so DiffusionModelBase.get_class(...).get_fsdp_ignored_module_names(...) resolves (e.g. DualGRPO ["visual"]).
        saved_model_config = self.model_config
        self.model_config = diffusion_cfg
        try:
            module = DiffusersFSDPEngine._build_fsdp_module(self, module)
        finally:
            self.model_config = saved_model_config # restore model config

        # verl diff part:
        if hf_model_config.enable_activation_offload:
            enable_activation_offloading(
                module,
                self.engine_config.strategy,
                hf_model_config.enable_gradient_checkpointing,
            )
        return module

    ar_engine._build_fsdp_module = types.MethodType(_build_fsdp_module, ar_engine)


# verl monkey patch use
# TODO: (susan) delete after fixing the bug in verl
def _get_input_embeds(
    model: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds, _ = unpack_visual_output(model.visual(pixel_values, grid_thw=image_grid_thw))
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds, _ = unpack_visual_output(model.visual(pixel_values_videos, grid_thw=video_grid_thw))
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return inputs_embeds, attention_mask


def qwen2_vl_base_forward(
    self: "Qwen2VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    kwargs["inputs_embeds"], kwargs["attention_mask"] = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    return self.language_model(input_ids=None, **kwargs)


# Bug in pytorch FSDP2
# TODO: (susan) remove after the verl fix PR got merged and released or new torch version released:
# https://github.com/verl-project/verl/pull/7475
# https://github.com/pytorch/pytorch/pull/194058
def _guard_fsdp2_accumulated_grad() -> None:
    """Work around an AttributeError in torch's FSDP2 gradient accumulation.

    `FSDPParam.to_accumulated_grad_if_needed` reads `self._unsharded_param`
    without checking that it exists. That attribute is created by
    `init_unsharded_param` (which guards its own access with `hasattr`) and
    dropped by `free_unsharded_param`, so a parameter that never took part in the
    forward pass does not have it, and training dies with

        AttributeError: 'FSDPParam' object has no attribute '_unsharded_param'

    Seen on a Qwen3.5 VL model under text-only batches, where the vision tower is
    never gathered. Such a parameter has no unsharded gradient to upcast, which is
    the case the method already returns early for, so returning is what it means
    to do.

    Fixed upstream in pytorch/pytorch#194058. This shim keeps verl working on the
    torch releases that carry the bug and becomes a no-op once the fix lands: it
    only inserts an early return for the case that would otherwise raise.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except ImportError:
        return

    original = getattr(FSDPParam, "to_accumulated_grad_if_needed", None)
    if original is None or getattr(original, "_verl_guarded", False):
        return

    @functools.wraps(original)
    def to_accumulated_grad_if_needed(self):
        if getattr(self, "_unsharded_param", None) is None:
            return
        return original(self)

    to_accumulated_grad_if_needed._verl_guarded = True
    FSDPParam.to_accumulated_grad_if_needed = to_accumulated_grad_if_needed


# _guard_fsdp2_accumulated_grad()
