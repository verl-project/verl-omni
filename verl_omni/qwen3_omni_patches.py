"""
Monkey-patches for upstream veRL to support Qwen3-Omni Thinker RL training.

verl-omni depends on veRL via pip, so we can't modify veRL's source directly.
Instead, we override specific functions/classes at import time.

These should eventually be upstreamed as PRs to veRL.
"""

import logging

logger = logging.getLogger(__name__)


def apply_all():
    _register_qwen3_omni_model()
    _patch_model_architecture_mapping()
    _patch_monkey_patch_text_config()
    _patch_fsdp_strip_modules()
    _patch_fsdp_lora_thinker_prefixes()
    _patch_vllm_lora_duck_typing()


# ---------------------------------------------------------------------------
# 1. Register Qwen3OmniMoe in AutoModelForCausalLM
#
#    Qwen3OmniMoe uses "ForConditionalGeneration" suffix but the Thinker
#    is a decoder-only causal LM.  We redirect forward/embeddings to the
#    thinker sub-module, force tie_word_embeddings=False (so veRL uses
#    meta-tensor loading to avoid OOM), and mark talker/code2wav/code_predictor
#    for stripping.
# ---------------------------------------------------------------------------

def _register_qwen3_omni_model():
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


# ---------------------------------------------------------------------------
# 2. Add "ForConditionalGeneration" → AutoModelForCausalLM mapping
# ---------------------------------------------------------------------------

def _patch_model_architecture_mapping():
    try:
        from transformers import AutoModelForCausalLM

        import verl.utils.model as model_mod

        if "ForConditionalGeneration" not in model_mod._architecture_to_auto_class:
            model_mod._architecture_to_auto_class["ForConditionalGeneration"] = AutoModelForCausalLM
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 3. Fix text_config lookup in apply_monkey_patch
#
#    Upstream does model.config.text_config.num_attention_heads which fails
#    when text_config is None.  We pre-populate it via get_text_config().
# ---------------------------------------------------------------------------

def _patch_monkey_patch_text_config():
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


# ---------------------------------------------------------------------------
# 4. Strip unused sub-modules after from_pretrained
#
#    Wraps from_pretrained on Qwen3OmniMoeForConditionalGeneration to delete
#    talker/code2wav/code_predictor before FSDP wrapping — avoids OOM.
# ---------------------------------------------------------------------------

def _patch_fsdp_strip_modules():
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


# ---------------------------------------------------------------------------
# 5. Add Qwen3-Omni thinker prefixes to layered_summon_lora_params
#
#    Upstream only has prefixes for standard LLMs (model.layers.) and
#    vision-language models (language_model.layers.).  Qwen3-Omni puts
#    layers at thinker.model.layers. — without these prefixes, layered
#    summon finds nothing and LoRA sync fails.
# ---------------------------------------------------------------------------

def _patch_fsdp_lora_thinker_prefixes():
    try:
        import verl.utils.fsdp_utils as fsdp_mod

        _original_layered_summon = fsdp_mod.layered_summon_lora_params

        def _patched_layered_summon(fsdp_module, is_diffusers=False):
            if not is_diffusers:
                # Inject thinker prefixes by temporarily replacing the function
                # with one that has the extended prefix list.
                from collections import OrderedDict

                from peft.utils.save_and_load import get_peft_model_state_dict
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                from verl.utils.device import get_torch_device

                def _prefix_submodules(module, prefix):
                    for name, submodule in module.named_modules():
                        if name.startswith(prefix) and "." not in name[len(prefix):]:
                            yield name, submodule

                prefix_list = [
                    # fsdp
                    "_fsdp_wrapped_module.base_model.model.",
                    "_fsdp_wrapped_module.base_model.model.model.",
                    "_fsdp_wrapped_module.base_model.model.model.layers.",
                    "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
                    "_fsdp_wrapped_module.base_model.model.thinker.model.layers.",
                    # fsdp2
                    "base_model.model.",
                    "base_model.model.model.",
                    "base_model.model.model.layers.",
                    "base_model.model.model.language_model.layers.",
                    "base_model.model.thinker.model.layers.",
                ]

                lora_params = OrderedDict()
                peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
                for prefix in prefix_list:
                    for name, submodule in _prefix_submodules(fsdp_module, prefix):
                        key_prefix = name.replace(
                            "_fsdp_wrapped_module.base_model.model.", "base_model.model."
                        )
                        if name.endswith(".model") or name.endswith(".layers"):
                            continue
                        if fsdp_mod.fsdp_version(submodule) > 0:
                            with FSDP.summon_full_params(submodule, writeback=False):
                                sub_lora_params = get_peft_model_state_dict(
                                    peft_model, state_dict=submodule.state_dict()
                                )
                                sub_lora_params = {
                                    f"{key_prefix}.{n}": (
                                        p.full_tensor().detach().cpu()
                                        if hasattr(p, "full_tensor")
                                        else p.detach().cpu()
                                    )
                                    for n, p in sub_lora_params.items()
                                }
                                lora_params.update(sub_lora_params)
                                submodule._is_root = False
                            get_torch_device().empty_cache()
                return lora_params

            return _original_layered_summon(fsdp_module, is_diffusers=is_diffusers)

        fsdp_mod.layered_summon_lora_params = _patched_layered_summon
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 6. Fix VLLMHijack to accept OmniTensorLoRARequest via duck-typing
#
#    Upstream uses isinstance(req, TensorLoRARequest) which fails for
#    OmniTensorLoRARequest (different base class).  We re-apply the hijack
#    with hasattr checks for peft_config/lora_tensors.
# ---------------------------------------------------------------------------

def _patch_vllm_lora_duck_typing():
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


apply_all()
