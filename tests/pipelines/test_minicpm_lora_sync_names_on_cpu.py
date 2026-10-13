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
"""CPU pin for MiniCPM-o LoRA adapter-delta key alignment on separate-async sync.

The recipe ships ``lora.merge=True``; this pins the planned ``merge=False``
flip, whose peft adapter tensors must bind on
``MiniCPMO45OmniLLMForConditionalGeneration``: its Qwen3 backbone is registered
under the ``llm.`` prefix, matching the actor's ``MiniCPMO.llm.*`` tree, and
MiniCPM-o defines no ``_checkpoint_conversion_mapping`` — so the keys resolve
through vLLM's ``parse_fine_tuned_lora_name`` without a remap. The rollout side
is pinned through the recorded module-name shape of that registration (the
``llm.``-prefixed Qwen3 attention projections), not by constructing the engine
class. If either side's naming drifts, the fix is a remap at the MiniCPM sync
surface, not an engine change.
"""

import re

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM
from verl.utils.model import convert_weight_keys
from vllm.lora.utils import parse_fine_tuned_lora_name

# The recipe's regex keeps LoRA off the frozen towers (they carry q_proj too).
_EXCLUDE_MODULES = (
    ".*vpm.*|.*apm.*|.*talker.*|.*code2wav.*|.*code_predictor.*|.*codec.*|"
    ".*audio_decoder.*|.*audio_generator.*|.*audio_head.*|.*tts.*|.*vocoder.*"
)
_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class _TowerLayerWithQProj(torch.nn.Module):
    """Minimal tower block exposing a q_proj, like SigLIP's self-attention."""

    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(4, 4)})


class _MiniCPMOShapedWrapper(torch.nn.Module):
    """The naming-relevant slice of the actor's ``MiniCPMO`` module tree."""

    def __init__(self):
        super().__init__()
        # The real wrapper's text backbone: self.llm = Qwen3ForCausalLM.
        self.llm = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            )
        )
        self.vpm = torch.nn.ModuleList([_TowerLayerWithQProj()])
        self.apm = torch.nn.ModuleList([_TowerLayerWithQProj()])


# Recorded rollout-side module shape of MiniCPMO45OmniLLMForConditionalGeneration:
# the Qwen3 backbone registered under maybe_prefix(prefix, "llm").
_ROLLOUT_BACKBONE_MODULE = re.compile(r"llm\.model\.layers\.\d+\.self_attn\.(?:q|k|v|o)_proj")


def test_adapter_keys_resolve_onto_rollout_side_names():
    wrapper = _MiniCPMOShapedWrapper()
    # The actor path injects LoRA into the wrapper itself
    # (non_diffusers_model_base.add_adapter), so keys keep the llm.* prefix.
    inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=8, target_modules=_TARGET_MODULES, exclude_modules=_EXCLUDE_MODULES),
        wrapper,
    )
    adapter_sd = get_peft_model_state_dict(wrapper, adapter_name="default")

    assert adapter_sd, "no LoRA tensors were collected"

    resolved_full_names = set()
    for key in adapter_sd:
        assert key.endswith((".lora_A.weight", ".lora_B.weight")), key
        module_name, _ = parse_fine_tuned_lora_name(key)
        assert _ROLLOUT_BACKBONE_MODULE.fullmatch(module_name), (
            f"{key!r} resolves to {module_name!r}, off the recorded rollout-side shape"
        )
        resolved_full_names.add(module_name)

    # All four LoRA targets landed on the recorded shape; the frozen towers
    # (vpm/apm carry their own q_proj) stay outside it.
    assert {name.rsplit(".", 1)[-1] for name in resolved_full_names} == set(_TARGET_MODULES)


def test_convert_weight_keys_is_identity_for_the_minicpm_wrapper():
    # No _checkpoint_conversion_mapping (unlike Qwen3-Omni), so the conversion must be identity.
    wrapper = _MiniCPMOShapedWrapper()
    inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=8, target_modules=_TARGET_MODULES, exclude_modules=_EXCLUDE_MODULES),
        wrapper,
    )
    adapter_sd = get_peft_model_state_dict(wrapper, adapter_name="default")

    assert not hasattr(wrapper, "_checkpoint_conversion_mapping")
    converted = convert_weight_keys(dict(adapter_sd), wrapper)
    assert converted.keys() == adapter_sd.keys()
    for key, value in adapter_sd.items():
        assert torch.equal(converted[key], value)
