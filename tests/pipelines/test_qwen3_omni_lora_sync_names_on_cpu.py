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
"""CPU pin for Qwen3-Omni Thinker LoRA adapter-delta key alignment.

The actor injects LoRA into the full ``Qwen3OmniMoeForConditionalGeneration``, so its
keys carry ``thinker.model.*``. The thinker-only rollout stage builds
``Qwen3OmniMoeThinkerForConditionalGeneration``, whose text backbone is
``language_model.model.*`` and whose ``q/k/v_proj`` are packed into ``qkv_proj``. The
keys must reach that shape through vLLM's parser and the Thinker's own mapper.
"""

import re

from peft import LoraConfig, inject_adapter_in_model
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from verl.utils.model import convert_weight_keys
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from tests.special_e2e.build_qwen3_omni_tiny_random import _build_tiny_config

# The GPU recipes' filter keeps LoRA off the frozen towers, which carry q_proj too.
_EXCLUDE_MODULES = ".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"
_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
_ROLLOUT_ATTENTION_MODULE = re.compile(r"language_model\.model\.layers\.\d+\.self_attn\.(?:q|k|v|o)_proj")


def test_actor_adapter_keys_resolve_onto_thinker_rollout_modules():
    actor = Qwen3OmniMoeForConditionalGeneration(_build_tiny_config(vocab_size=64))
    inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=8, target_modules=_TARGET_MODULES, exclude_modules=_EXCLUDE_MODULES), actor
    )
    adapter_keys = convert_weight_keys(get_peft_model_state_dict(actor, adapter_name="default"), actor)
    mapper = Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper()

    assert adapter_keys, "no LoRA tensors were collected"
    module_names = {parse_fine_tuned_lora_name(key, mapper)[0] for key in adapter_keys}
    off_shape = [name for name in module_names if not _ROLLOUT_ATTENTION_MODULE.fullmatch(name)]
    assert not off_shape, f"adapter modules off the rollout attention shape: {off_shape}"
    assert {name.rsplit(".", 1)[-1] for name in module_names} == set(_TARGET_MODULES)
    assert set(Qwen3OmniMoeThinkerForConditionalGeneration.packed_modules_mapping["qkv_proj"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
    }
