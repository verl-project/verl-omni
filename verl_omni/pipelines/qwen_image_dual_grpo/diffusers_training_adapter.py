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

"""
Qwen-Image training-side adapter for DualGRPO algorithm.
Inherits model-specific forward/sampling behavior from FlowGRPO.
"""

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage

__all__ = ["QwenImageDualGRPO", "QwenImageDualGRPOFSDP"]


@DiffusionModelBase.register("QwenImagePipeline", algorithm="dual_grpo")
class QwenImageDualGRPO(QwenImage):
    """Training adapter for Qwen-Image with the DualGRPO algorithm.

    The composite AR text encoder is Qwen2.5-VL. Dual-GRPO uses text-only data, so
    the vision tower is deleted before FSDP wrap (see
    ``strip_qwen_image_vision_tower`` in ``verl_omni.workers.engine.utils``) — same as vLLM-Omni ``QwenImagePipeline``.
    Do not list ``visual`` in ``get_fsdp_ignored_module_names``: ignored params
    must be frozen, and leaving a trainable tower trips FSDP2's fail-closed check.
    """


@DiffusionModelBase.register("QwenImagePipeline", algorithm="dual_grpo_fsdp")
class QwenImageDualGRPOFSDP(QwenImage):
    """Training adapter for Qwen-Image with the DualGRPO algorithm."""
