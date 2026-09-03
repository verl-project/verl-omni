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
from verl.utils.tokenizer import normalize_token_ids

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.agent_loop.utils import messages_to_text


@register("minimax_h3_diffusion_single_turn_agent")
class MiniMaxH3DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """Tokenize H3 prompts without applying a chat template."""

    async def ct_build_initial_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        """Match H3's verbatim T2VA presentation with no special tokens."""
        del tools, images, videos, audios
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
