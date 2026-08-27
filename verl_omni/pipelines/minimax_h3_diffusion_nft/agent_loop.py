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
"""MiniMax H3 agent loop for token-id-native raw-text prompts."""

from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.utils.tokenizer import normalize_token_ids

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop

from .common import MINIMAX_H3_TOKEN_ID_NATIVE_KEY, messages_to_text

__all__ = ["MiniMaxH3DiffusionSingleTurnAgentLoop"]


@register("minimax_h3_diffusion_single_turn_agent")
class MiniMaxH3DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """Tokenize H3 prompt text verbatim without applying a chat template."""

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        """Mark IDs so the H3 rollout can reject generic chat-template tokens."""
        sampling_params = {**sampling_params, MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True}
        return await super().run(sampling_params, **kwargs)

    async def _tokenize_raw_text(self, messages: list[dict]) -> list[int]:
        """Return raw H3 text IDs without applying a chat template."""
        text = messages_to_text(messages)
        if not text:
            raise ValueError("MiniMax H3 requires a non-empty text prompt.")
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

    async def ct_build_initial_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        """Override verl's Continuous Token entry point with H3 raw-text IDs."""
        del tools, images, videos, audios
        return await self._tokenize_raw_text(messages)

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
        """Keep the legacy entry point aligned with Continuous Token behavior."""
        del tools, images, videos, audios, mm_processor_kwargs, remove_system_prompt
        return await self._tokenize_raw_text(messages)
