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

_MINICPM_ARCHITECTURES = ("MiniCPMO",)
_MINICPM_NO_SPLIT_MODULES = ["Qwen3DecoderLayer", "MiniCPMODecoderLayer"]


def _register_minicpm_architectures(cls):
    for architecture in _MINICPM_ARCHITECTURES:
        OmniModelBase.register(architecture, stage="thinker")(cls)
    return cls


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
        return [
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
        ]

    @classmethod
    def build_module(cls, model_config, torch_dtype):
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            model_config.local_path,
            torch_dtype=torch_dtype,
            config=model_config.hf_config,
            trust_remote_code=model_config.trust_remote_code,
        )

    @classmethod
    def configure_model(cls, module, model_config):
        module = super().configure_model(module, model_config)

        trainable_component = _first_existing_attr(
            module,
            ["llm", "language_model", "model", "base_model", "text_model"],
        )
        if trainable_component is not None:
            if hasattr(trainable_component, "forward"):
                module.forward = trainable_component.forward
            if hasattr(trainable_component, "get_input_embeddings"):
                module.get_input_embeddings = trainable_component.get_input_embeddings
            if hasattr(trainable_component, "set_input_embeddings"):
                module.set_input_embeddings = trainable_component.set_input_embeddings
            if hasattr(trainable_component, "prepare_inputs_for_generation"):
                module.prepare_inputs_for_generation = trainable_component.prepare_inputs_for_generation

        module._no_split_modules = _MINICPM_NO_SPLIT_MODULES
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
