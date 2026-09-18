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

The standalone weight sync (``lora.merge=False``) ships the actor's peft
adapter tensors; the rollout binds them through vLLM's
``parse_fine_tuned_lora_name`` against the HF-side module naming of
``MiniCPMO45OmniLLMForConditionalGeneration``, which registers the Qwen3
backbone under the ``llm.`` prefix (``maybe_prefix(prefix, "llm")`` +
``init_vllm_registered_model``; vLLM's own qkv packing happens after name
resolution, through the same machinery every verl LoRA run uses). The actor's
``MiniCPMO`` wrapper carries the same ``self.llm`` subtree and defines no
``_checkpoint_conversion_mapping``, so verl's ``convert_weight_keys`` is an
identity pass — both sides meet without a remap. The separate-async recipe
currently ships ``lora.merge=True`` (accuracy parity with its colocated
reference); this contract governs the planned performance flip to
``merge=False`` and stays valid for any run that sets it. This test pins that
contract with a tiny CPU model; if either side's naming drifts, the sanctioned
fix is a remap at the MiniCPM sync surface, not an engine change.
"""

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM
from verl.utils.model import convert_weight_keys
from vllm.lora.utils import parse_fine_tuned_lora_name

# The recipe's exclude regex keeps LoRA off the frozen towers, which also carry
# q_proj modules (SigLIP vision / Whisper audio).
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


def _rollout_side_module_names(wrapper: _MiniCPMOShapedWrapper) -> set[str]:
    """Module names the vLLM rollout class exposes for LoRA name resolution.

    Mirrors ``MiniCPMO45OmniLLMForConditionalGeneration``: the Qwen3 backbone
    is instantiated under the ``llm.`` prefix and keeps HF naming inside it
    (``llm.model.layers.N.self_attn.q_proj``); the towers hang off the root.
    """
    llm_names = {f"llm.{name}" for name, _ in wrapper.llm.named_modules() if name}
    tower_names = {name for name, _ in wrapper.named_modules() if name.startswith(("vpm.", "apm."))}
    return llm_names | tower_names


def test_adapter_keys_resolve_onto_rollout_side_names():
    wrapper = _MiniCPMOShapedWrapper()
    # The actor path injects LoRA into the wrapper itself (non_diffusers_model_base
    # .add_adapter), so peft keys carry the wrapper's llm.* prefix — no PeftModel
    # re-rooting, no base_model.model. prefix.
    inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=8, target_modules=_TARGET_MODULES, exclude_modules=_EXCLUDE_MODULES),
        wrapper,
    )
    adapter_sd = get_peft_model_state_dict(wrapper, adapter_name="default")

    assert adapter_sd, "no LoRA tensors were collected"
    rollout_names = _rollout_side_module_names(wrapper)

    resolved_full_names = set()
    for key in adapter_sd:
        assert key.endswith((".lora_A.weight", ".lora_B.weight")), key
        assert key.startswith("llm.model.layers."), key
        module_name, _ = parse_fine_tuned_lora_name(key)
        assert module_name in rollout_names, f"{key!r} resolves to {module_name!r}, absent rollout-side"
        resolved_full_names.add(module_name)

    # All four LoRA targets landed, and the frozen towers were excluded.
    assert {name.rsplit(".", 1)[-1] for name in resolved_full_names} == set(_TARGET_MODULES)
    assert not any(name.startswith(("vpm.", "apm.")) for name in resolved_full_names)


def test_convert_weight_keys_is_identity_for_the_minicpm_wrapper():
    # No _checkpoint_conversion_mapping on the MiniCPMO wrapper (unlike
    # Qwen3-Omni, whose thinker prefix strip is what aligns that model), so the
    # adapter keys must survive verl's checkpoint-name conversion unchanged.
    wrapper = _MiniCPMOShapedWrapper()
    inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=8, target_modules=_TARGET_MODULES, exclude_modules=_EXCLUDE_MODULES),
        wrapper,
    )
    adapter_sd = get_peft_model_state_dict(wrapper, adapter_name="default")

    assert not hasattr(wrapper, "_checkpoint_conversion_mapping")
    assert convert_weight_keys(dict(adapter_sd), wrapper) == dict(adapter_sd)
