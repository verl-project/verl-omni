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
"""Qwen3-Omni Thinker patches: route via AutoModelForMultimodalLM, fix FSDP-init
blockers, unfuse MoE experts for PEFT LoRA (tf5+), and extend verl's hf_processor
for Qwen3-Omni."""

import logging

logger = logging.getLogger(__name__)


def _register_qwen3_omni_automodel() -> None:
    """Point Omni at AutoModelForMultimodalLM and patch FSDP-init blockers."""
    try:
        from transformers import AutoModelForMultimodalLM
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeConfig,
            Qwen3OmniMoeForConditionalGeneration,
        )
    except ImportError:
        return

    from verl.utils.model import _architecture_to_auto_class

    # Official HF mapping for qwen3_omni_moe (not AutoModelForCausalLM).
    _architecture_to_auto_class["Qwen3OmniMoeForConditionalGeneration"] = AutoModelForMultimodalLM

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
    # Upstream lists Qwen3OmniMoeDecoderLayer which does not exist; fix to the real class.
    Qwen3OmniMoeForConditionalGeneration._no_split_modules = [
        "Qwen3OmniMoeThinkerTextDecoderLayer",
        "Qwen3OmniMoeVisionBlock",
    ]
    # _verl_strip_modules: verl's FSDPEngine drops these sub-modules for Thinker-only training.
    Qwen3OmniMoeForConditionalGeneration._verl_strip_modules = [
        "talker",
        "code2wav",
        "code_predictor",
    ]

    # tie_word_embeddings=True disables FSDP meta-tensor init and OOMs on 30B-A3B.
    logger.warning(
        "verl_omni: forcing tie_word_embeddings=False on Qwen3OmniMoeConfig — tied "
        "embeddings disable the FSDP meta-tensor init path and OOM on 30B-A3B."
    )

    class _FalseTieDescriptor:
        def __get__(self, obj, objtype=None):
            return False

        def __set__(self, obj, value):
            pass

    Qwen3OmniMoeConfig.tie_word_embeddings = _FalseTieDescriptor()


def patch_hf_processor_for_qwen3_omni() -> None:
    """Wrap verl.utils.tokenizer.hf_processor to recognize Qwen3OmniMoeProcessor.
    Installs a fallback that handles Qwen3-Omni only when the original returns None."""
    try:
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    except ImportError:
        return

    import types

    import verl.utils.tokenizer as _vt

    _original_hf_processor = _vt.hf_processor

    def _patched_hf_processor(name_or_path, **kwargs):
        result = _original_hf_processor(name_or_path, **kwargs)
        if result is not None:
            return result

        try:
            from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
            if isinstance(processor, PreTrainedTokenizerBase):
                return None
            if processor.__class__.__name__ != "Qwen3OmniMoeProcessor":
                return None

            config = AutoConfig.from_pretrained(name_or_path, **kwargs)
            # Token IDs / spatial_merge_size live on thinker_config, not the top-level config.
            processor.config = config.thinker_config
            processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
            processor.config.vision_start_token_id = config.talker_config.vision_start_token_id
            model_class = Qwen3OmniMoeThinkerForConditionalGeneration
            # upstream get_rope_index returns float32. cast to int64 so agent_loop's
            # cat(text_position_ids=int64, vision_position_ids) yields int64 and FSDP
            # root-input casting (floating dtypes only) leaves positions exact -
            # avoids BF16 large-int rounding (e.g. 2879 -> 2880) that breaks SDPA
            # packed-sequence detection under use_remove_padding.
            _ori_get_rope_index = types.MethodType(model_class.get_rope_index, processor)

            def _get_rope_index_long(*args, **kwargs):
                vision_position_ids, deltas = _ori_get_rope_index(*args, **kwargs)
                return vision_position_ids.long(), deltas

            processor.get_rope_index = _get_rope_index_long
            processor.get_llm_pos_ids_for_vision = types.MethodType(model_class.get_llm_pos_ids_for_vision, processor)
            return processor
        except Exception:
            return None

    _vt.hf_processor = _patched_hf_processor
    # Also refresh verl.utils's stale re-export (callers use `from verl.utils import hf_processor`).
    import sys as _sys

    for _mod_name in ("verl.utils", "verl.workers.config.model"):
        _mod = _sys.modules.get(_mod_name)
        if _mod is not None and hasattr(_mod, "hf_processor"):
            _mod.hf_processor = _patched_hf_processor


_EXPERTS_UNFUSE_APPLIED = False


def _patch_unfuse_qwen3_omni_thinker_experts() -> None:
    """Hook peft.get_peft_model to unfuse tf5 fused MoE experts before LoRA (tf5+ only).
    Converts Qwen3OmniMoeThinkerTextExperts (fused 3D params) to per-expert nn.Linear."""
    global _EXPERTS_UNFUSE_APPLIED
    if _EXPERTS_UNFUSE_APPLIED:
        return

    # tf5 sentinel: transformers.integrations.moe only exists in transformers >= 5.x
    try:
        import transformers.integrations.moe  # noqa
        import peft as _peft
    except ImportError:
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _Expert(nn.Module):
        def __init__(self, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    class _Qwen3OmniMoeThinkerTextExpertsUnfused(nn.Module):
        """Per-expert nn.Linear replacement for the tf5 fused Qwen3OmniMoeThinkerTextExperts.
        Weights are cloned from fused params at conversion time so the original can be GC'd."""

        def __init__(self, n: int, hidden: int, intermediate: int, act_fn) -> None:
            super().__init__()
            self.num_experts = n
            self.act_fn = act_fn
            self.experts = nn.ModuleList([_Expert(hidden, intermediate) for _ in range(n)])

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
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
                e = self.experts[i]
                out = e.down_proj(self.act_fn(e.gate_proj(x)) * e.up_proj(x))
                out = out * top_k_weights[tok_idx, top_k_pos, None]
                final.index_add_(0, tok_idx, out.to(final.dtype))
            return final

    def _convert_model_experts(model) -> None:
        """Replace all fused Thinker expert modules with unfused per-expert nn.Linear."""
        for path, module in list(model.named_modules()):
            if type(module).__name__ != "Qwen3OmniMoeThinkerTextExperts":
                continue
            gate_up = module.gate_up_proj.data  # (n, 2*intermediate, hidden)
            down = module.down_proj.data  # (n, hidden, intermediate)
            n = gate_up.shape[0]
            di = gate_up.shape[1] // 2
            h = gate_up.shape[2]

            new_mod = _Qwen3OmniMoeThinkerTextExpertsUnfused(n, h, di, module.act_fn)
            for i, e in enumerate(new_mod.experts):
                e.gate_proj.weight = nn.Parameter(gate_up[i, :di, :].clone())
                e.up_proj.weight = nn.Parameter(gate_up[i, di:, :].clone())
                e.down_proj.weight = nn.Parameter(down[i].clone())

            parent_path, _, child_name = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, child_name, new_mod)

    _orig_get_peft_model = _peft.get_peft_model

    # No-op PEFT's gate_proj/up_proj → gate_up_proj remap for Qwen3-Omni, else expert LoRA won't attach.
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
    except (ImportError, AttributeError) as e:
        logger.warning("verl_omni: could not patch PEFT tf5 name remapping (%s); MoE expert LoRA may not attach", e)

    def _patched_get_peft_model(model, peft_config, **kwargs):
        if type(model).__name__ == "Qwen3OmniMoeForConditionalGeneration":
            _convert_model_experts(model)
            # verl passes target_modules as a comma-separated string; PEFT treats it as regex — split to set.
            if isinstance(peft_config.target_modules, str) and "," in peft_config.target_modules:
                peft_config.target_modules = set(peft_config.target_modules.split(","))
        return _orig_get_peft_model(model, peft_config, **kwargs)

    _peft.get_peft_model = _patched_get_peft_model
    # Also update verl's module-level binding if it was already imported before us.
    import sys as _sys

    _vi = _sys.modules.get("verl.workers.engine.fsdp.transformer_impl")
    if _vi is not None:
        _vi.get_peft_model = _patched_get_peft_model
    _EXPERTS_UNFUSE_APPLIED = True
    logger.info("verl_omni: installed get_peft_model hook for Qwen3-Omni MoE expert unfusing (tf5+)")


def patch_hf_tokenizer_for_qwen3_omni() -> None:
    """Wrap ``verl.utils.tokenizer.hf_tokenizer`` to auto-load chat_template from chat_template.json.

    Some models (e.g., Qwen3-Omni) store chat_template in a separate file
    instead of tokenizer_config.json. This patch ensures the tokenizer
    has a valid chat_template before returning it.
    """
    import functools
    import json
    import os

    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    _original_hf_tokenizer = _vt.hf_tokenizer

    @functools.wraps(_original_hf_tokenizer)
    def _patched_hf_tokenizer(name_or_path, *args, **kwargs):
        tokenizer = _original_hf_tokenizer(name_or_path, *args, **kwargs)

        if getattr(tokenizer, "chat_template", None) is None and isinstance(name_or_path, str):
            chat_template_path = os.path.join(name_or_path, "chat_template.json")
            if os.path.exists(chat_template_path):
                try:
                    with open(chat_template_path) as f:
                        data = json.load(f)
                        chat_template = data.get("chat_template")
                        if chat_template:
                            tokenizer.chat_template = chat_template
                except (OSError, json.JSONDecodeError):
                    pass

        return tokenizer

    _vt.hf_tokenizer = _patched_hf_tokenizer

    # Patch sys.modules entries that already imported hf_tokenizer
    import sys

    for mod_name in list(sys.modules.keys()):
        if not mod_name.startswith("verl"):
            continue
        mod = sys.modules.get(mod_name)
        if (
            mod is not None
            and hasattr(mod, "hf_tokenizer")
            and mod.__dict__.get("hf_tokenizer") is _original_hf_tokenizer
        ):
            mod.hf_tokenizer = _patched_hf_tokenizer


def patch_register_vllm_moe_model_weight_loader() -> None:
    """Register Qwen3-Omni Thinker with verl's ``SUPPORTED_MOE_MODELS`` whitelist.

    verl's ``patch_vllm_moe_model_weight_loader`` re-attaches ``weight_loader``
    on FusedMoE ``w13_weight``/``w2_weight`` before IPC weight sync (the attr
    that vllm-ascend's ``process_weights_after_loading`` drops when rebuilding
    the params). But it early-returns unless the model class is in
    ``SUPPORTED_MOE_MODELS``. Qwen3-Omni Thinker isn't there by default, so we
    append it.
    """
    try:
        from verl.utils.vllm import patch as _vp
    except ImportError:
        return

    added = []
    for mod_path, cls_name in [
        (
            "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker",
            "Qwen3OmniMoeThinkerForConditionalGeneration",
        ),
    ]:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is not None and cls not in _vp.SUPPORTED_MOE_MODELS:
                _vp.SUPPORTED_MOE_MODELS.append(cls)
                added.append(f"{mod_path}.{cls_name}")
        except ImportError:
            continue


def apply_qwen3_omni_thinker_patches() -> None:
    """Apply all Qwen3-Omni Thinker patches (idempotent registrations)."""
    _register_qwen3_omni_automodel()
    patch_hf_processor_for_qwen3_omni()
    _patch_unfuse_qwen3_omni_thinker_experts()
    patch_hf_tokenizer_for_qwen3_omni()
    patch_register_vllm_moe_model_weight_loader()


# Apply on import so this module works as a verl ``external_lib`` target.
apply_qwen3_omni_thinker_patches()
