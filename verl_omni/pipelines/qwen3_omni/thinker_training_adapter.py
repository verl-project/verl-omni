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
"""Qwen3-Omni Thinker training adapter.

Implements ``OmniModelBase`` for thinker-stage training of
Qwen3-Omni: sub-module stripping, forward redirection, and
processor/tokenizer configuration.
"""

import json
import logging
import os
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


@OmniModelBase.register("Qwen3OmniMoeForConditionalGeneration", stage="thinker")
class Qwen3OmniThinkerAdapter(OmniModelBase):
    """Thinker-stage training adapter for Qwen3-Omni.

    Handles model setup that is required before verl's FSDP engine
    loads and wraps the model: sub-module stripping, forward redirection
    to the thinker component, and processor/tokenizer configuration.
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return ["talker", "code2wav", "code_predictor"]

    @classmethod
    def configure_model(cls, module, model_config):
        """Strip non-training stages and redirect forward to thinker."""
        for submod_name in cls.get_strip_modules(model_config):
            if hasattr(module, submod_name):
                delattr(module, submod_name)

        module.forward = module.thinker.forward
        module.get_input_embeddings = module.thinker.get_input_embeddings
        module.set_input_embeddings = module.thinker.set_input_embeddings
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the Qwen3-Omni multimodal processor with RoPE helpers.

        Swaps ``processor.config`` to ``thinker_config`` (Qwen3-Omni
        nests multimodal settings under sub-configs).  Binds
        ``get_rope_index`` and ``get_llm_pos_ids_for_vision`` to the
        processor — the omni agent loop calls these on the processor,
        but they are model methods.
        """
        import types

        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        processor.config = config.thinker_config
        processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
        processor.config.vision_start_token_id = config.talker_config.vision_start_token_id

        model_cls = Qwen3OmniMoeThinkerForConditionalGeneration
        processor.get_rope_index = types.MethodType(model_cls.get_rope_index, processor)
        processor.get_llm_pos_ids_for_vision = types.MethodType(model_cls.get_llm_pos_ids_for_vision, processor)
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Load the tokenizer. Qwen3-Omni stores its chat template in
        ``chat_template.json`` instead of ``tokenizer_config.json``."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        chat_template_path = os.path.join(model_path, "chat_template.json")
        if not os.path.isfile(chat_template_path):
            raise FileNotFoundError(
                f"Qwen3-Omni chat template not found at {chat_template_path}. "
                f"Ensure the model checkpoint includes chat_template.json."
            )
        with open(chat_template_path) as f:
            tokenizer.chat_template = json.load(f)["chat_template"]
        return tokenizer
