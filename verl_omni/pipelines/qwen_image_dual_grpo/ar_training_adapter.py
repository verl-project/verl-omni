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
from typing import TYPE_CHECKING, Optional

import torch
from verl.utils.transformers_compat import unpack_visual_output

from verl_omni.pipelines.model_base import DiffusionModelARBase

if TYPE_CHECKING:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2_5_VLModel

logger = logging.getLogger(__name__)

__all__ = ["QwenImageDualGRPOAR", "qwen2_vl_base_forward"]


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


# TODO(susan): delete this copied ``_get_input_embeds``, ``qwen2_vl_base_forward``, and the
# instance ``module.model.forward`` assignment in ``configure_module`` *together* once the
# pinned verl includes https://github.com/verl-project/verl/pull/8035. That PR unpacks the
# dummy visual path.
def _get_input_embeds(
    model: Qwen2_5_VLModel,
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

    if pixel_values is None and pixel_values_videos is None:  # handle mixed text-image data
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds, _ = unpack_visual_output(model.visual(pixel_values, grid_thw=image_grid_thw))
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return inputs_embeds, attention_mask


def qwen2_vl_base_forward(
    self: Qwen2_5_VLModel,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Qwen2-VL forward that builds embeds before the language-model call.

    Local copy of verl's base forward that routes the dummy visual path through
    ``unpack_visual_output`` (verl#8035). Delete with ``_get_input_embeds`` and the
    instance patch in ``configure_module`` when that PR is in the pinned verl.
    """
    kwargs["inputs_embeds"], kwargs["attention_mask"] = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    return self.language_model(input_ids=None, **kwargs)


@DiffusionModelARBase.register("QwenImagePipeline", algorithm="dual_grpo")
class QwenImageDualGRPOAR(DiffusionModelARBase):
    """AR training adapter for the Qwen-Image text encoder under Dual-GRPO."""

    @classmethod
    def configure_module(cls, module: torch.nn.Module, model_config) -> None:
        """Strip the unused vision tower and patch the Qwen2-VL text forward.

        Delete the instance forward assignment together with the copied
        ``_get_input_embeds`` / ``qwen2_vl_base_forward`` once verl#8035 is in the
        pinned verl; keep ``_strip_vision_tower``.
        """
        module.model.forward = qwen2_vl_base_forward.__get__(module.model)
        logger.info("Patched Qwen2-VL base forward on Dual-GRPO AR module")
        _strip_vision_tower(module)

    @classmethod
    def get_fsdp_ignored_module_names(cls, model_config) -> list[str]:
        # Vision tower is deleted in ``configure_module``; do not ignore a missing subtree.
        return []
