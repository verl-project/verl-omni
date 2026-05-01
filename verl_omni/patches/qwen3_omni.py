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
Qwen3-Omni Thinker patches for upstream veRL.

Patches applied:
1. Register Qwen3OmniMoe in AutoModelForCausalLM (verl/utils/model.py)
2. Fix text_config lookup in apply_monkey_patch (verl/models/transformers/monkey_patch.py)
3. Add _verl_strip_modules support to FSDPEngine (verl/workers/engine/fsdp/transformer_impl.py)
4. Fix VLLMHijack duck-typing for OmniTensorLoRARequest (verl/utils/vllm/utils.py)
5. Add ForConditionalGeneration → AutoModelForCausalLM mapping (verl/utils/model.py)
"""

import logging

logger = logging.getLogger(__name__)


def apply_all():
    _register_qwen3_omni_model()
    _patch_model_architecture_mapping()
    _patch_monkey_patch_text_config()
    _patch_fsdp_strip_modules()
    _patch_vllm_lora_duck_typing()


def _register_qwen3_omni_model():
    """Register Qwen3-Omni Thinker in AutoModelForCausalLM.

    Qwen3OmniMoe uses "ForConditionalGeneration" suffix but the Thinker
    is a decoder-only causal LM.  We redirect forward/embeddings to the
    thinker sub-module and strip unused stages (talker, code2wav) to
    avoid OOM during FSDP init.
    """
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeConfig,
            Qwen3OmniMoeForConditionalGeneration,
        )

        def _get_input_embeddings(self):
            return self.thinker.get_input_embeddings()

        def _set_input_embeddings(self, value):
            self.thinker.set_input_embeddings(value)

        def _forward(
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

        Qwen3OmniMoeForConditionalGeneration.forward = _forward
        Qwen3OmniMoeForConditionalGeneration.get_input_embeddings = _get_input_embeddings
        Qwen3OmniMoeForConditionalGeneration.set_input_embeddings = _set_input_embeddings
        Qwen3OmniMoeForConditionalGeneration._no_split_modules = [
            "Qwen3OmniMoeThinkerTextDecoderLayer"
        ]
        Qwen3OmniMoeForConditionalGeneration._verl_strip_modules = [
            "talker",
            "code2wav",
            "code_predictor",
        ]

        # Force tie_word_embeddings=False so veRL uses meta-tensor loading
        # (avoids full CPU load + OOM during FSDP init).
        class _FalseTieDescriptor:
            def __get__(self, obj, objtype=None):
                return False

            def __set__(self, obj, value):
                pass

        Qwen3OmniMoeConfig.tie_word_embeddings = _FalseTieDescriptor()
        AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)
        logger.info("Registered Qwen3OmniMoe in AutoModelForCausalLM")
    except Exception:
        pass


def _patch_model_architecture_mapping():
    """Add 'ForConditionalGeneration' to veRL's architecture→AutoClass mapping."""
    try:
        from transformers import AutoModelForCausalLM

        import verl.utils.model as model_mod

        if "ForConditionalGeneration" not in model_mod._architecture_to_auto_class:
            model_mod._architecture_to_auto_class["ForConditionalGeneration"] = AutoModelForCausalLM
    except Exception:
        pass


def _patch_monkey_patch_text_config():
    """Fix text_config lookup for multimodal models whose config lacks
    num_attention_heads at the top level (e.g. Qwen3-Omni).

    The upstream code does ``model.config.text_config.num_attention_heads``
    which fails when ``text_config`` is None.  We pre-populate it via
    ``get_text_config()`` before the original function runs.
    """
    try:
        import verl.models.transformers.monkey_patch as mp_mod

        _original_apply = mp_mod.apply_monkey_patch

        def _patched_apply(model, *args, **kwargs):
            if not hasattr(model.config, "num_attention_heads"):
                if not hasattr(model.config, "text_config") or model.config.text_config is None:
                    try:
                        model.config.text_config = model.config.get_text_config()
                    except Exception:
                        pass
            return _original_apply(model, *args, **kwargs)

        mp_mod.apply_monkey_patch = _patched_apply
    except Exception:
        pass


def _patch_fsdp_strip_modules():
    """Add _verl_strip_modules support to FSDPEngine.

    After the model is loaded via from_pretrained, delete sub-modules
    listed in ``_verl_strip_modules`` (e.g. talker, code2wav) to free
    memory before FSDP wrapping.

    We wrap ``from_pretrained`` on Qwen3OmniMoeForConditionalGeneration
    rather than patching FSDPEngine.__init__ (which would be fragile).
    """
    try:
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
        )

        _original_from_pretrained = Qwen3OmniMoeForConditionalGeneration.from_pretrained

        @classmethod  # type: ignore[misc]
        def _from_pretrained_with_strip(cls, *args, **kwargs):
            module = _original_from_pretrained.__func__(cls, *args, **kwargs)
            for attr in getattr(module, "_verl_strip_modules", []):
                if hasattr(module, attr):
                    delattr(module, attr)
                    logger.info("Stripped unused sub-module '%s' from %s", attr, type(module).__name__)
            return module

        Qwen3OmniMoeForConditionalGeneration.from_pretrained = _from_pretrained_with_strip
    except Exception:
        pass


def _patch_vllm_lora_duck_typing():
    """Fix VLLMHijack to accept OmniTensorLoRARequest via duck-typing.

    Upstream VLLMHijack uses ``isinstance(req, TensorLoRARequest)`` which
    fails for OmniTensorLoRARequest (different base class).  We re-apply
    the hijack with duck-typing checks (``hasattr`` for peft_config/lora_tensors).
    """
    try:
        from vllm.lora.peft_helper import PEFTHelper
        from vllm.lora.utils import get_adapter_absolute_path
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

        try:
            from vllm.lora.lora_model import LoRAModel
        except ImportError:
            from vllm.lora.models import LoRAModel

        def hijack__load_adapter(self, lora_request):
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: list[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)
                expected_lora_modules = list(set(expected_lora_modules))

                lora_tensors = None

                _has_tensor_lora = hasattr(lora_request, "peft_config") and hasattr(
                    lora_request, "lora_tensors"
                )
                if _has_tensor_lora:
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)
                    peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)

                peft_helper.validate_legal(self.lora_config)

                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper

                lora_request_kwargs = {
                    "peft_helper": peft_helper,
                    "lora_model_id": lora_request.lora_int_id,
                    "device": "cpu",
                    "dtype": self.lora_config.lora_dtype,
                    "weights_mapper": hf_to_vllm_mapper,
                }
                if hasattr(self, "embedding_padding_modules"):
                    lora_request_kwargs["embedding_modules"] = self.embedding_modules
                    lora_request_kwargs["embedding_padding_modules"] = self.embedding_padding_modules
                else:
                    lora_request_kwargs["model_vocab_size"] = self.vocab_size
                if hasattr(self.lora_config, "lora_extra_vocab_size"):
                    lora_request_kwargs["target_embedding_padding"] = (
                        self.vocab_size + self.lora_config.lora_extra_vocab_size
                    )
                if _has_tensor_lora:
                    lora = self._lora_model_cls.from_lora_tensors(
                        tensors=lora_tensors,
                        **lora_request_kwargs,
                    )
                else:
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        **lora_request_kwargs,
                    )
            except Exception:
                raise

            if getattr(lora, "extra_vocab_size", 0) > getattr(self.lora_config, "lora_extra_vocab_size", 0):
                raise ValueError(
                    f"LoRA added vocab size {lora.extra_vocab_size} is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
                )
            return lora

        LRUCacheWorkerLoRAManager._load_adapter = hijack__load_adapter
    except Exception:
        pass
