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
"""Qwen3-Omni Thinker model-specific patches for upstream transformers / verl.

These patches make verl's FSDP engine treat the Qwen3-Omni Thinker (a
decoder-only LM despite its ``ForConditionalGeneration`` suffix) as a causal LM:
  * register it with ``AutoModelForCausalLM`` + verl's architecture lookup,
  * delegate ``forward`` / embeddings to ``self.thinker``,
  * fix ``_no_split_modules`` and declare ``_verl_strip_modules`` so the FSDP
    engine drops talker / code2wav / code_predictor,
  * force ``tie_word_embeddings=False`` (tied embeddings OOM at FSDP init), and
  * extend ``verl.utils.tokenizer.hf_processor`` to recognize the Qwen3-Omni
    multimodal processor.

This is a *model-specific* patch (cf. ``verl_omni/models/diffusers`` for the
diffusion equivalents). It is applied as an import side effect and is NOT
imported by ``verl_omni.models.__init__`` — load it on demand (e.g. via verl's
``actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker``)
so it only takes effect when a Qwen3-Omni Thinker is actually trained.
"""

import logging

logger = logging.getLogger(__name__)


def _register_qwen3_omni_automodel() -> None:
    """Register the Thinker with AutoModelForCausalLM and patch FSDP-init blockers."""
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeConfig,
            Qwen3OmniMoeForConditionalGeneration,
        )
    except ImportError:
        return

    from verl.utils.model import _architecture_to_auto_class

    # The thinker is decoder-only despite the "ForConditionalGeneration"
    # suffix; tell verl's architecture lookup to dispatch to AutoModelForCausalLM.
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
    # Upstream lists Qwen3OmniMoeDecoderLayer which does not exist; fix to the real class.
    Qwen3OmniMoeForConditionalGeneration._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    # _verl_strip_modules is read by verl's FSDPEngine to delete unused sub-modules
    # (talker / code2wav / code_predictor are not needed for Thinker-only training).
    Qwen3OmniMoeForConditionalGeneration._verl_strip_modules = [
        "talker",
        "code2wav",
        "code_predictor",
    ]

    # tie_word_embeddings=True forces use_meta_tensor=False during FSDP init
    # which OOMs on 30B-A3B. Override at the config-class level via a no-op
    # descriptor so config __init__ assignments are tolerated.
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
    AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)


def _patch_hf_processor_for_qwen3_omni() -> None:
    """Wrap ``verl.utils.tokenizer.hf_processor`` to recognize Qwen3OmniMoeProcessor.

    The original uses a ``match`` block that cannot be extended at runtime; we
    install a wrapper that handles the Qwen3-Omni case only when the original
    returns ``None`` (so other models are unaffected).
    """
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

        # Original returned None — either it's a tokenizer-only model (fine)
        # or it failed because of an unsupported processor (maybe Qwen3-Omni).
        try:
            from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
            if isinstance(processor, PreTrainedTokenizerBase):
                return None
            if processor.__class__.__name__ != "Qwen3OmniMoeProcessor":
                return None

            config = AutoConfig.from_pretrained(name_or_path, **kwargs)
            # Token IDs / spatial_merge_size live on thinker_config, not the
            # top-level Qwen3OmniMoeConfig that AutoConfig returns.
            processor.config = config.thinker_config
            processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
            model_class = Qwen3OmniMoeThinkerForConditionalGeneration
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            processor.get_llm_pos_ids_for_vision = types.MethodType(model_class.get_llm_pos_ids_for_vision, processor)
            return processor
        except Exception:
            return None

    _vt.hf_processor = _patched_hf_processor


def apply_qwen3_omni_thinker_patches() -> None:
    """Apply all Qwen3-Omni Thinker patches (idempotent registrations)."""
    _register_qwen3_omni_automodel()
    _patch_hf_processor_for_qwen3_omni()


# Apply on import so this module works as a verl ``external_lib`` target.
apply_qwen3_omni_thinker_patches()
