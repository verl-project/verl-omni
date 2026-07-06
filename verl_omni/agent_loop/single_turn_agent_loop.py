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
from verl.utils.chat_template import apply_chat_template as _apply_chat_template
from verl.utils.profiler import simple_timer

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("diffusion_single_turn_agent")
class DiffusionSingleTurnAgentLoop(AgentLoopBase):
    """Agent loop for diffusion model serving.

    Models with multiple text encoders (e.g. SD3.5) configure
    ``actor_rollout_ref.model.extra_tokenizers`` so the prompt is tokenized once
    per encoder here and the rollout pipeline consumes token ids directly,
    instead of decoding and re-encoding text inside the pipeline.
    """

    def __init__(self, *args, extra_tokenizer_map: dict[str, dict[str, Any]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_tokenizer_map = extra_tokenizer_map or {}

    async def _tokenize_per_encoder(self, messages: list[dict]) -> dict[str, list[int]]:
        """Tokenize the rendered prompt once per configured text-encoder tokenizer.

        Returns unpadded token ids (with special tokens) per encoder name; the
        rollout pipeline pads each sequence with its own tokenizer's pad token.
        """

        def _tokenize() -> dict[str, list[int]]:
            processing_class = self.processor if self.processor is not None else self.tokenizer
            text = _apply_chat_template(
                processing_class,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.apply_chat_template_kwargs,
            )
            encoder_prompt_ids = {}
            for name, spec in self.extra_tokenizer_map.items():
                tokenizer = spec["tokenizer"]
                max_length = spec.get("max_length")
                encode_kwargs: dict[str, Any] = {"add_special_tokens": True}
                if max_length is not None:
                    encode_kwargs.update(truncation=True, max_length=max_length)
                encoder_prompt_ids[name] = tokenizer(text, **encode_kwargs)["input_ids"]
            return encoder_prompt_ids

        return await self.loop.run_in_executor(None, _tokenize)

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

        # 3. tokenize once per extra text-encoder tokenizer (multi-encoder models)
        extra_prompt_ids = None
        negative_extra_prompt_ids = None
        if self.extra_tokenizer_map:
            extra_prompt_ids = await self._tokenize_per_encoder(raw_prompt)
            if raw_negative_prompt is not None:
                negative_extra_prompt_ids = await self._tokenize_per_encoder(raw_negative_prompt)

        # 4. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                negative_prompt_ids=negative_prompt_ids,
                extra_prompt_ids=extra_prompt_ids,
                negative_extra_prompt_ids=negative_extra_prompt_ids,
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
