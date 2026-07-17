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
"""Qwen3-Omni pipeline adapters (Thinker training + rollout pipeline topology)."""

from .omni_rollout_adapter import Qwen3OmniRolloutAdapter
from .thinker_training_adapter import Qwen3OmniThinkerAdapter

__all__ = [
    "Qwen3OmniThinkerAdapter",
    "Qwen3OmniRolloutAdapter",
]


# TODO (mike): remove after upstream vllm-omni fix lands.
# Qwen3OmniMoeThinkerForConditionalGeneration is missing ``is_3d_moe_weight=True``,
# so vLLM routes fused-MoE LoRA through the 2-D ``FusedMoEWithLoRA`` instead of
# the 3-D ``FusedMoE3DWithLoRA``.  PEFT ``ParamWrapper`` produces expert-stacked
# tensors; only the 3-D wrapper's ``_stack_moe_lora_weights`` reshapes them into
# the per-expert list that ``set_lora`` expects.  Without the flag, ``set_lora``
# hits ``assert isinstance(lora_a, list)``.
def _patch_qwen3_omni_moe_is_3d_moe_weight() -> None:
    try:
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
    except ImportError:
        return
    if not getattr(Qwen3OmniMoeThinkerForConditionalGeneration, "is_3d_moe_weight", False):
        Qwen3OmniMoeThinkerForConditionalGeneration.is_3d_moe_weight = True


_patch_qwen3_omni_moe_is_3d_moe_weight()
