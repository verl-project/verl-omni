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
import logging
import os
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

_LIGHTWEIGHT_IMPORT = os.getenv("VERL_OMNI_LIGHTWEIGHT_IMPORT") == "1"

if _LIGHTWEIGHT_IMPORT:
    _AGENT_LOOP_IMPORT_ERROR = RuntimeError("VERL_OMNI_LIGHTWEIGHT_IMPORT=1")

    class AgentLoopBase:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Diffusion single-turn agent loop requires verl agent-loop runtime") from (
                _AGENT_LOOP_IMPORT_ERROR
            )

    def register(_):
        return lambda cls: cls

else:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register

if _LIGHTWEIGHT_IMPORT:

    @contextmanager
    def simple_timer(name: str, timing_raw: dict[str, float]):
        del name, timing_raw
        yield

else:
    from verl.utils.profiler import simple_timer

if _LIGHTWEIGHT_IMPORT:

    class DiffusionAgentLoopOutput:
        def __init__(
            self,
            *,
            prompt_ids,
            response_diffusion_output,
            response_logprobs=None,
            reward_score=None,
            num_turns=0,
            metrics=None,
            extra_fields=None,
        ):
            self.prompt_ids = prompt_ids
            self.response_diffusion_output = response_diffusion_output
            self.response_logprobs = response_logprobs
            self.reward_score = reward_score
            self.num_turns = num_turns
            self.metrics = metrics or {}
            self.extra_fields = extra_fields or {}

else:
    from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_list(value: Any) -> list | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return None


def _content_to_text_parts(content: Any) -> list[str]:
    if hasattr(content, "tolist"):
        content = content.tolist()
    if isinstance(content, str):
        return [content]

    items = _as_list(content)
    if items is None:
        return []

    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return parts


def _messages_to_text(messages: Any) -> str | None:
    """Extract text content for diffusion pipelines that run their own text encoders."""
    if messages is None:
        return None
    if isinstance(messages, str):
        return messages

    if isinstance(messages, dict):
        messages = [messages]
    else:
        messages = _as_list(messages)
        if messages is None:
            return None

    user_parts: list[str] = []
    fallback_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        content_parts = _content_to_text_parts(content)
        fallback_parts.extend(content_parts)
        if role == "user":
            user_parts.extend(content_parts)

    parts = user_parts or fallback_parts
    text = "\n".join(part for part in parts if part)
    return text or None


@register("diffusion_single_turn_agent")
class DiffusionSingleTurnAgentLoop(AgentLoopBase):
    """Agent loop for diffusion model serving."""

    async def _encode_prompt_ids(self, raw_prompt, *, images=None, videos=None) -> list[int]:
        if getattr(self.tokenizer, "chat_template", None) is not None:
            return await self.apply_chat_template(raw_prompt, images=images, videos=videos)

        text = _messages_to_text(raw_prompt)
        if text is None:
            return await self.apply_chat_template(raw_prompt, images=images, videos=videos)
        return self.tokenizer.encode(text, add_special_tokens=True)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> DiffusionAgentLoopOutput:
        """Run one diffusion generation turn and package agent-loop output.

        Args:
            sampling_params: Generation parameters forwarded to the server manager.
            **kwargs: Per-sample fields from the dataset, including ``raw_prompt``
                and optional ``raw_negative_prompt``.

        Returns:
            DiffusionAgentLoopOutput: Prompt ids, generated diffusion output,
            optional logprobs, runtime metrics, and extra fields.
        """
        raw_prompt = kwargs["raw_prompt"]
        raw_negative_prompt = kwargs.get("raw_negative_prompt")

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(raw_prompt)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self._encode_prompt_ids(raw_prompt, images=images, videos=videos)

        if raw_negative_prompt is not None:
            negative_prompt_ids = await self._encode_prompt_ids(raw_negative_prompt, images=images, videos=videos)
        else:
            negative_prompt_ids = None

        # 3. generate sequences
        metrics = {}
        request_sampling_params = sampling_params.copy()
        raw_prompt_text = _messages_to_text(raw_prompt)
        raw_negative_prompt_text = _messages_to_text(raw_negative_prompt)
        if raw_prompt_text is not None:
            request_sampling_params["_raw_prompt"] = raw_prompt_text
        if raw_negative_prompt_text is not None:
            request_sampling_params["_raw_negative_prompt"] = raw_negative_prompt_text

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=request_sampling_params,
                image_data=images,
                video_data=videos,
                negative_prompt_ids=negative_prompt_ids,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        output = DiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=output.diffusion_output,
            response_logprobs=output.log_probs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        return output
