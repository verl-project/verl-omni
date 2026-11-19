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


@DiffusionModelARBase.register("QwenImagePipeline", algorithm="dual_grpo")
class QwenImageDualGRPOAR(DiffusionModelARBase):
    """AR training adapter for the Qwen-Image text encoder under Dual-GRPO."""

    @classmethod
    def configure_module(cls, module: torch.nn.Module, model_config) -> None:
        """Strip the unused vision tower; the text forward stays verl's class patch.

        verl's ``apply_monkey_patch`` installs ``qwen2_vl_base_forward`` on the
        model class inside ``_build_module``, and its dummy visual path unpacks
        the plain tensor the stripped tower returns.
        """
        _strip_vision_tower(module)

    @classmethod
    def get_fsdp_ignored_module_names(cls, model_config) -> list[str]:
        # Vision tower is deleted in ``configure_module``; do not ignore a missing subtree.
        return []
