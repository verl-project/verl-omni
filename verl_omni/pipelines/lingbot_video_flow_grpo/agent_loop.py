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

"""Single-turn agent loop for structured LingBot Dense T2V captions."""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
from verl.utils.profiler import simple_timer

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput

from .common import DEFAULT_NEGATIVE_PROMPT, apply_prompt_template, caption_to_json

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("lingbot_dense_t2v_agent")
class LingBotDenseT2VAgentLoop(AgentLoopBase):
    """Prepare LingBot structured captions and submit one T2V rollout request."""

    def _tokenize_caption(self, caption: str, max_length: int) -> list[int]:
        processor = self.processor if self.processor is not None else self.tokenizer
        inputs = processor(
            text=apply_prompt_template(caption),
            images=None,
            videos=None,
            do_resize=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        return input_ids[0].tolist()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> DiffusionAgentLoopOutput:
        raw_caption = caption_to_json(kwargs["raw_prompt"])
        raw_negative_caption = kwargs.get("raw_negative_prompt")
        negative_caption = (
            caption_to_json(raw_negative_caption) if raw_negative_caption is not None else DEFAULT_NEGATIVE_PROMPT
        )
        max_length = int(sampling_params.get("max_sequence_length", 37698))
        prompt_ids = self._tokenize_caption(raw_caption, max_length)
        negative_prompt_ids = self._tokenize_caption(negative_caption, max_length)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                negative_prompt_ids=negative_prompt_ids,
                sampling_params=sampling_params,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        extra_fields = dict(output.extra_fields or {})
        # Keep the compact caption available to rewards/loggers without
        # replacing the original dataset field added by the worker postprocess.
        extra_fields["lingbot_caption"] = raw_caption
        return DiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=output.diffusion_output,
            response_logprobs=output.log_probs,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
