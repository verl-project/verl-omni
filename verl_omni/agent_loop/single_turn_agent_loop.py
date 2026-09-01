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

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer.chat_template import apply_chat_template as _apply_chat_template

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("omni_single_turn_agent")
class OmniSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Single-turn loop for an omni model's autoregressive Talker policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_adapter = self._resolve_rollout_adapter(self.rollout_config)

    @staticmethod
    def _resolve_rollout_adapter(rollout_config):
        try:
            pipeline_name = rollout_config.engine_kwargs["vllm_omni"]["pipeline_name"]
        except (KeyError, TypeError) as error:
            raise ValueError("omni_single_turn_agent requires engine_kwargs.vllm_omni.pipeline_name.") from error

        adapter = OmniRolloutPipelineBase.get_class(pipeline_name)
        if adapter is None:
            raise ValueError(
                "omni_single_turn_agent requires a registered "
                f"engine_kwargs.vllm_omni.pipeline_name, got {pipeline_name!r}."
            )
        return adapter

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the standard flow, then map and validate the model's policy sequence."""
        output = await super().run(sampling_params, **kwargs)
        output = self.rollout_adapter.postprocess_agent_loop_output(
            output,
            tokenizer=self.tokenizer,
            response_length=self.response_length,
        )
        if not isinstance(output, AgentLoopOutput):
            raise TypeError(
                "OmniRolloutPipelineBase.postprocess_agent_loop_output must return an AgentLoopOutput, "
                f"got {type(output).__name__}."
            )

        policy_length = len(output.response_ids)
        if policy_length > self.response_length:
            raise ValueError(
                f"Omni policy output has {policy_length} tokens, exceeding response_length={self.response_length}."
            )
        if len(output.response_mask) != policy_length:
            raise ValueError(
                "Omni policy response_mask must align one-to-one with response_ids, "
                f"got {len(output.response_mask)} and {policy_length}."
            )
        if output.response_logprobs is not None and len(output.response_logprobs) != policy_length:
            raise ValueError(
                "Omni policy response_logprobs must align one-to-one with response_ids, "
                f"got {len(output.response_logprobs)} and {policy_length}."
            )
        if not isinstance(output.extra_fields, dict):
            raise TypeError(f"Omni policy extra_fields must be a dict, got {type(output.extra_fields).__name__}.")
        return output


@register("diffusion_single_turn_agent")
class DiffusionSingleTurnAgentLoop(AgentLoopBase):
    """Agent loop for diffusion model serving."""

    def __init__(self, *args, extra_tokenizer_map: dict[str, dict[str, Any]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_tokenizer_map = extra_tokenizer_map or {}

    def _get_routing_request_id(self, sample_uid: Any | None) -> str:
        use_prompt_cache_affinity = bool(
            self.rollout_config.enable_prompt_embed_cache
            and self.rollout_config.enable_prompt_embed_cache_routing_affinity
            and sample_uid is not None
        )
        return str(sample_uid) if use_prompt_cache_affinity else uuid4().hex

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
        multi_modal_data = await self.process_multi_modal_info(raw_prompt)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")

        # 2. build the initial prompt with Continuous Token
        self._assert_mm_supported(bool(multi_modal_data))
        prompt_ids = await self.ct_build_initial_tokens(
            raw_prompt,
            images=images,
            videos=videos,
            audios=audios,
        )

        if raw_negative_prompt is not None:
            negative_prompt_ids = await self.ct_build_initial_tokens(
                raw_negative_prompt,
                images=images,
                videos=videos,
                audios=audios,
            )
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
                request_id=self._get_routing_request_id(kwargs.get("uid")),
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
