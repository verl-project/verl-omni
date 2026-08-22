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

"""MiniMax H3 agent loop for raw-text tokenization."""

from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.utils.ray_utils import get_event_loop
from verl.utils.tokenizer import normalize_token_ids

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.agent_loop.utils import messages_to_text


@register("minimax_h3_diffusion_single_turn_agent")
class MiniMaxH3DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """Tokenize H3 prompts without applying a chat template."""

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        extra_tokenizer_map: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        # H3 uses its text encoder tokenizer directly. Avoid probing an optional
        # chat template before the model-specific raw-text path runs.
        del kwargs
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        self.extra_tokenizer_map = extra_tokenizer_map or {}
        self.system_prompt = []
        self.loop = get_event_loop()

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        sampling_params = {
            **sampling_params,
            "verl_raw_prompt": messages_to_text(kwargs["raw_prompt"]),
        }
        return await super().run(sampling_params, **kwargs)

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        """Match H3's raw T2VA presentation with no special tokens."""
        del tools, images, videos, audios, mm_processor_kwargs, remove_system_prompt
        text = messages_to_text(messages)
        prompt_length = self.rollout_config.prompt_length
        tokenized = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=prompt_length,
                add_special_tokens=False,
            )["input_ids"],
        )
        return normalize_token_ids(tokenized)
