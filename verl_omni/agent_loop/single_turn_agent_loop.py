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
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
from verl.utils.profiler import simple_timer

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    return ""


def _messages_to_model_prompt(messages: Any) -> str | None:
    if messages is None or isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return None

    user_parts = []
    fallback_parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        text = _content_to_text(message.get("content"))
        if not text:
            continue
        fallback_parts.append(text)
        if message.get("role") == "user":
            user_parts.append(text)

    parts = user_parts or fallback_parts
    return "\n".join(parts) or None


@register("diffusion_single_turn_agent")
class DiffusionSingleTurnAgentLoop(AgentLoopBase):
    """Agent loop for diffusion model serving."""

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
        prompt_ids = await self.apply_chat_template(raw_prompt, images=images, videos=videos)

        if raw_negative_prompt is not None:
            negative_prompt_ids = await self.apply_chat_template(raw_negative_prompt, images=images, videos=videos)
        else:
            negative_prompt_ids = None

        if getattr(self.rollout_config.agent, "pass_model_prompt", False):
            sampling_params = sampling_params.copy()
            model_prompt = _messages_to_model_prompt(raw_prompt)
            model_negative_prompt = _messages_to_model_prompt(raw_negative_prompt)
            if model_prompt is not None:
                sampling_params["_model_prompt"] = model_prompt
            if model_negative_prompt is not None:
                sampling_params["_model_negative_prompt"] = model_negative_prompt

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
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
