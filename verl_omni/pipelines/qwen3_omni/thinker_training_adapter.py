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
Qwen3-Omni: sub-module stripping, forward redirection,
processor/tokenizer configuration, and LoRA key normalization for
vLLM-Omni weight sync.
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
        """Strip non-training stages and redirect forward to thinker.

        Args:
            module: The loaded Qwen3-Omni model before FSDP wrapping.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured module with talker/codec stripped and
            forward/embedding accessors redirected to thinker.
        """
        module = super().configure_model(module, model_config)
        module.forward = module.thinker.forward
        module.get_input_embeddings = module.thinker.get_input_embeddings
        module.set_input_embeddings = module.thinker.set_input_embeddings
        module._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the Qwen3-Omni multimodal processor with RoPE helpers.

        Swaps ``processor.config`` to ``thinker_config`` (Qwen3-Omni
        nests multimodal settings under sub-configs).  Binds
        ``get_rope_index`` and ``get_llm_pos_ids_for_vision`` to the
        processor — the omni agent loop calls these on the processor,
        but they are model methods.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured processor with RoPE helpers bound.
        """
        import types

        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)

        processor.config = config.thinker_config
        processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
        processor.config.vision_start_token_id = config.talker_config.vision_start_token_id

        model_cls = Qwen3OmniMoeThinkerForConditionalGeneration

        # Cast to int64: HF returns float32, FSDP would otherwise bf16-round positions.
        _ori_get_rope_index = types.MethodType(model_cls.get_rope_index, processor)

        def _get_rope_index_long(*args, **kwargs):
            vision_position_ids, deltas = _ori_get_rope_index(*args, **kwargs)
            return vision_position_ids.long(), deltas

        processor.get_rope_index = _get_rope_index_long
        processor.get_llm_pos_ids_for_vision = types.MethodType(model_cls.get_llm_pos_ids_for_vision, processor)
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Load the tokenizer with chat template from ``chat_template.json``.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured tokenizer with ``chat_template`` loaded from
            ``chat_template.json``.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        chat_template_path = os.path.join(model_path, "chat_template.json")
        if not os.path.isfile(chat_template_path):
            raise FileNotFoundError(
                f"Qwen3-Omni chat template not found at {chat_template_path}. "
                f"Ensure the model checkpoint includes chat_template.json."
            )
        with open(chat_template_path) as f:
            tokenizer.chat_template = json.load(f)["chat_template"]
        return tokenizer
