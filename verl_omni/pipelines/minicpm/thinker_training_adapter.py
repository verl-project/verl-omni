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
"""MiniCPM thinker training adapter.

MiniCPM-V/o checkpoints are loaded through their Hugging Face remote-code
``AutoModel`` entrypoint.  For offline DPO we keep only the multimodal
understanding path and remove inference-only audio generation modules before
FSDP wrapping.
"""

from __future__ import annotations

import logging
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)

_MINICPM_ARCHITECTURES = (
    "MiniCPMV4_6ForConditionalGeneration",
    "MiniCPMV4ForConditionalGeneration",
    "MiniCPMVForConditionalGeneration",
    "MiniCPMOForConditionalGeneration",
    "MiniCPMOmniForConditionalGeneration",
)


def _register_minicpm_architectures(cls):
    for architecture in _MINICPM_ARCHITECTURES:
        OmniModelBase.register(architecture, stage="thinker")(cls)
    return cls


def _override_config(model_config) -> dict[str, Any]:
    override = getattr(model_config, "override_config", None) or {}
    try:
        return dict(override)
    except TypeError:
        return {}


def _configured_list(model_config, key: str, default: list[str]) -> list[str]:
    value = _override_config(model_config).get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _first_existing_attr(module, names: list[str]):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


@_register_minicpm_architectures
class MiniCPMThinkerAdapter(OmniModelBase):
    """Training adapter for MiniCPM multimodal understanding-only DPO."""

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return _configured_list(
            model_config,
            "minicpm_strip_modules",
            [
                "talker",
                "tts",
                "audio_decoder",
                "audio_generator",
                "audio_head",
                "audio_detokenizer",
                "codec",
                "code2wav",
                "code_predictor",
                "snac",
                "vocoder",
            ],
        )

    @classmethod
    def build_module(cls, model_config, torch_dtype):
        from transformers import AutoModel

        from_pretrained_kwargs = dict(_override_config(model_config).get("minicpm_from_pretrained_kwargs", {}))
        return AutoModel.from_pretrained(
            model_config.local_path,
            torch_dtype=torch_dtype,
            config=model_config.hf_config,
            trust_remote_code=model_config.trust_remote_code,
            **from_pretrained_kwargs,
        )

    @classmethod
    def configure_model(cls, module, model_config):
        module = super().configure_model(module, model_config)

        trainable_component = _first_existing_attr(
            module,
            _configured_list(
                model_config,
                "minicpm_understanding_module_names",
                ["llm", "language_model", "model", "base_model", "text_model"],
            ),
        )
        if trainable_component is not None:
            if hasattr(trainable_component, "forward"):
                module.forward = trainable_component.forward
            if hasattr(trainable_component, "get_input_embeddings"):
                module.get_input_embeddings = trainable_component.get_input_embeddings
            if hasattr(trainable_component, "set_input_embeddings"):
                module.set_input_embeddings = trainable_component.set_input_embeddings

        no_split_modules = _configured_list(model_config, "minicpm_no_split_modules", [])
        if no_split_modules:
            module._no_split_modules = no_split_modules
        elif not getattr(module, "_no_split_modules", None):
            logger.warning(
                "MiniCPM adapter did not set _no_split_modules. Set "
                "`+actor_rollout_ref.model.override_config.minicpm_no_split_modules=[...]` "
                "if FSDP wraps too coarsely."
            )
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        from transformers import AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        except Exception as exc:
            logger.warning("Falling back to MiniCPM tokenizer as processor after AutoProcessor failed: %s", exc)
            processor = cls.configure_tokenizer(model_path, model_config)
        if not hasattr(processor, "tokenizer"):
            try:
                processor.tokenizer = cls.configure_tokenizer(model_path, model_config)
            except Exception:
                logger.debug("MiniCPM processor does not expose a tokenizer attribute.", exc_info=True)
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
