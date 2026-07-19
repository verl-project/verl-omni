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

"""LTX-2.3 agent loop that matches the upstream raw-text tokenizer path."""

from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.ray_utils import get_event_loop
from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop


def _messages_to_text(messages: Any) -> str:
    """Extract textual message content without applying a chat template."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        messages = [messages]

    parts = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        for item in content or []:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
    return "\n".join(part for part in parts if part).strip()


@register("ltx2_diffusion_single_turn_agent")
class LTX2DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """Tokenize raw LTX prompts exactly as ``LTX2Pipeline.encode_prompt`` does."""

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        **kwargs,
    ) -> None:
        # LTX-2 uses its text encoder tokenizer as a raw-text tokenizer. Calling
        # AgentLoopBase.__init__ would probe its optional chat template with two
        # consecutive user messages, which strict templates reject before the
        # LTX-specific raw-text path gets a chance to run.
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
        self.system_prompt = []
        self.loop = get_event_loop()
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
        """Encode raw text with special tokens and right-side truncation."""
        del tools, images, videos, audios, mm_processor_kwargs, remove_system_prompt
        text = _messages_to_text(messages)
        prompt_length = self.rollout_config.prompt_length
        tokenized = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=prompt_length,
                add_special_tokens=True,
            )["input_ids"],
        )
        return normalize_token_ids(tokenized)
