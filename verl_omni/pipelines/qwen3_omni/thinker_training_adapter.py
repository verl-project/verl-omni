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
"""Qwen3-Omni thinker-only training adapter."""

import json
import logging
import os

import torch.nn as nn

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)

__all__ = ["Qwen3OmniThinkerAdapter"]


def _register_qwen3_omni_automodel() -> None:
    """Register the full Qwen3-Omni class as a thinker-only causal LM."""
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeConfig,
            Qwen3OmniMoeForConditionalGeneration,
        )
    except ImportError:
        return

    from verl.utils.model import _architecture_to_auto_class

    _architecture_to_auto_class.setdefault("Qwen3OmniMoeForConditionalGeneration", AutoModelForCausalLM)

    def _qwen3_omni_get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def _qwen3_omni_set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)

    def _qwen3_omni_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    Qwen3OmniMoeForConditionalGeneration.forward = _qwen3_omni_forward
    Qwen3OmniMoeForConditionalGeneration.get_input_embeddings = _qwen3_omni_get_input_embeddings
    Qwen3OmniMoeForConditionalGeneration.set_input_embeddings = _qwen3_omni_set_input_embeddings
    Qwen3OmniMoeForConditionalGeneration._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    Qwen3OmniMoeForConditionalGeneration._verl_strip_modules = Qwen3OmniThinkerAdapter.get_strip_modules(None)

    logger.warning(
        "verl_omni: forcing tie_word_embeddings=False on Qwen3OmniMoeConfig because tied embeddings "
        "disable the FSDP meta-tensor init path for Qwen3-Omni."
    )

    class _FalseTieDescriptor:
        def __get__(self, obj, objtype=None):
            return False

        def __set__(self, obj, value):
            pass

    Qwen3OmniMoeConfig.tie_word_embeddings = _FalseTieDescriptor()
    AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)


def _load_chat_template_from_json(tokenizer, model_path: str):
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer

    chat_template_path = os.path.join(model_path, "chat_template.json")
    if not os.path.exists(chat_template_path):
        return tokenizer

    try:
        with open(chat_template_path) as f:
            chat_template = json.load(f).get("chat_template")
    except (OSError, json.JSONDecodeError):
        return tokenizer

    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def _configure_qwen3_omni_processor(processor, model_path: str, **kwargs):
    try:
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    except ImportError:
        return processor

    import types

    config = AutoConfig.from_pretrained(model_path, **kwargs)
    processor.config = config.thinker_config
    processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
    processor.config.vision_start_token_id = config.talker_config.vision_start_token_id
    processor.get_rope_index = types.MethodType(Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index, processor)
    processor.get_llm_pos_ids_for_vision = types.MethodType(
        Qwen3OmniMoeThinkerForConditionalGeneration.get_llm_pos_ids_for_vision, processor
    )
    return processor


def patch_hf_processor_for_qwen3_omni() -> None:
    """Wrap verl's hf_processor to recognize Qwen3OmniMoeProcessor."""
    try:
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration  # noqa: F401
    except ImportError:
        return

    import sys

    import verl.utils.tokenizer as _vt

    _original_hf_processor = _vt.hf_processor

    def _patched_hf_processor(name_or_path, **kwargs):
        result = _original_hf_processor(name_or_path, **kwargs)
        if result is not None:
            return result

        try:
            from transformers import AutoProcessor, PreTrainedTokenizerBase

            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
            if isinstance(processor, PreTrainedTokenizerBase):
                return None
            if processor.__class__.__name__ != "Qwen3OmniMoeProcessor":
                return None
            return _configure_qwen3_omni_processor(processor, name_or_path, **kwargs)
        except Exception:
            return None

    _vt.hf_processor = _patched_hf_processor
    for mod_name in ("verl.utils", "verl.workers.config.model"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "hf_processor"):
            mod.hf_processor = _patched_hf_processor


_EXPERTS_UNFUSE_APPLIED = False


def _patch_unfuse_qwen3_omni_thinker_experts() -> None:
    """Hook PEFT to unfuse Qwen3-Omni thinker MoE experts before LoRA."""
    global _EXPERTS_UNFUSE_APPLIED
    if _EXPERTS_UNFUSE_APPLIED:
        return

    try:
        import peft as _peft
        import transformers.integrations.moe  # noqa: F401
    except ImportError:
        return

    import sys

    import torch
    import torch.nn.functional as F

    class _Expert(nn.Module):
        def __init__(self, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    class _Qwen3OmniMoeThinkerTextExpertsUnfused(nn.Module):
        def __init__(self, n: int, hidden: int, intermediate: int, act_fn) -> None:
            super().__init__()
            self.num_experts = n
            self.act_fn = act_fn
            self.experts = nn.ModuleList([_Expert(hidden, intermediate) for _ in range(n)])

        def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, self.num_experts).permute(2, 1, 0)
                hits = mask.sum(dim=(-1, -2)).gt(0).nonzero()
            for row in hits:
                i = row[0].item()
                if i >= self.num_experts:
                    continue
                top_k_pos, tok_idx = torch.where(mask[i])
                x = hidden_states[tok_idx]
                expert = self.experts[i]
                out = expert.down_proj(expert.act_fn(expert.gate_proj(x)) * expert.up_proj(x))
                out = out * top_k_weights[tok_idx, top_k_pos, None]
                final.index_add_(0, tok_idx, out.to(final.dtype))
            return final

    def _convert_model_experts(model) -> None:
        for path, module in list(model.named_modules()):
            if type(module).__name__ != "Qwen3OmniMoeThinkerTextExperts":
                continue
            gate_up = module.gate_up_proj.data
            down = module.down_proj.data
            n = gate_up.shape[0]
            intermediate = gate_up.shape[1] // 2
            hidden = gate_up.shape[2]

            new_mod = _Qwen3OmniMoeThinkerTextExpertsUnfused(n, hidden, intermediate, module.act_fn)
            for i, expert in enumerate(new_mod.experts):
                expert.gate_proj.weight = nn.Parameter(gate_up[i, :intermediate, :].clone())
                expert.up_proj.weight = nn.Parameter(gate_up[i, intermediate:, :].clone())
                expert.down_proj.weight = nn.Parameter(down[i].clone())

            parent_path, _, child_name = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, child_name, new_mod)

    _orig_get_peft_model = _peft.get_peft_model

    try:
        import peft.utils.transformers_weight_conversion as _twc

        _orig_get_mapping = _twc.get_model_conversion_mapping
        _orig_convert = _twc.convert_peft_config_for_transformers

        def _patched_get_mapping(model):
            if type(model).__name__ == "Qwen3OmniMoeForConditionalGeneration":
                return []
            return _orig_get_mapping(model)

        def _patched_convert(peft_config, model=None, conversions=None):
            if model is not None and type(model).__name__ == "Qwen3OmniMoeForConditionalGeneration":
                return
            return _orig_convert(peft_config, model=model, conversions=conversions)

        _twc.get_model_conversion_mapping = _patched_get_mapping
        _twc.convert_peft_config_for_transformers = _patched_convert
    except (ImportError, AttributeError) as exc:
        logger.warning("verl_omni: could not patch PEFT name remapping: %s", exc)

    def _patched_get_peft_model(model, peft_config, **kwargs):
        if type(model).__name__ == "Qwen3OmniMoeForConditionalGeneration":
            _convert_model_experts(model)
            if isinstance(peft_config.target_modules, str) and "," in peft_config.target_modules:
                peft_config.target_modules = set(peft_config.target_modules.split(","))
        return _orig_get_peft_model(model, peft_config, **kwargs)

    _peft.get_peft_model = _patched_get_peft_model
    transformer_impl = sys.modules.get("verl.workers.engine.fsdp.transformer_impl")
    if transformer_impl is not None:
        transformer_impl.get_peft_model = _patched_get_peft_model
    _EXPERTS_UNFUSE_APPLIED = True
    logger.info("verl_omni: installed Qwen3-Omni thinker expert unfuse hook")


def patch_hf_tokenizer_for_qwen3_omni() -> None:
    """Wrap verl's hf_tokenizer to auto-load chat_template.json."""
    import functools
    import sys

    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    _original_hf_tokenizer = _vt.hf_tokenizer

    @functools.wraps(_original_hf_tokenizer)
    def _patched_hf_tokenizer(name_or_path, *args, **kwargs):
        tokenizer = _original_hf_tokenizer(name_or_path, *args, **kwargs)
        if isinstance(name_or_path, str):
            _load_chat_template_from_json(tokenizer, name_or_path)
        return tokenizer

    _vt.hf_tokenizer = _patched_hf_tokenizer
    for mod_name in list(sys.modules.keys()):
        if not mod_name.startswith("verl"):
            continue
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "hf_tokenizer", None) is _original_hf_tokenizer:
            mod.hf_tokenizer = _patched_hf_tokenizer


@OmniModelBase.register("Qwen3OmniMoeForConditionalGeneration", stage="thinker")
class Qwen3OmniThinkerAdapter(OmniModelBase):
    """Training adapter for Qwen3-Omni thinker-only RL/DPO."""

    @classmethod
    def get_model_architecture(cls) -> type:
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

        return Qwen3OmniMoeForConditionalGeneration

    @classmethod
    def get_model_loading_kwargs(cls, model_config) -> dict:
        return {
            "hf_config_name": "thinker_config",
            "no_split_modules": ["Qwen3OmniMoeThinkerTextDecoderLayer"],
            "tie_word_embeddings_override": False,
        }

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return ["talker", "code2wav", "code_predictor"]

    @classmethod
    def configure_model(cls, module, model_config):
        module = super().configure_model(module, model_config)
        if hasattr(module, "thinker"):
            module.forward = module.thinker.forward
            module.get_input_embeddings = module.thinker.get_input_embeddings
            module.set_input_embeddings = module.thinker.set_input_embeddings
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=model_config.get("trust_remote_code", True)
        )
        return _configure_qwen3_omni_processor(
            processor,
            model_path,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=model_config.get("trust_remote_code", True)
        )
        return _load_chat_template_from_json(tokenizer, model_path)

    @classmethod
    def apply_model_patches(cls) -> None:
        _register_qwen3_omni_automodel()
        patch_hf_processor_for_qwen3_omni()
        _patch_unfuse_qwen3_omni_thinker_experts()
        patch_hf_tokenizer_for_qwen3_omni()
