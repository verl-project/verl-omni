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
"""MiniCPM-o bounded collection with immutable processor outputs for replay."""

from uuid import uuid4

import ray
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.utils.profiler import simple_timer

from .omni_rollout_adapter import MINICPM_PROMPT_KEY
from .processor import (
    _load_audio,
    clone_minicpmo_actor_inputs,
    prepare_minicpmo_inputs,
    render_minicpmo_messages,
    split_minicpmo_actor_inputs,
)


@ray.remote
class MiniCPMAgentLoopWorker(AgentLoopWorkerTQ.__ray_metadata__.modified_class):
    """Keep native media tensors instead of reconstructing them from decoded text."""

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        if (
            not validate
            and self.distillation_enabled
            and not self.config.distillation.distillation_loss.use_task_rewards
        ):
            for item in output if isinstance(output, list) else [output]:
                item.reward_score = 0.0
                item.extra_fields["reward_extra_info"] = {}
        return await super()._agent_loop_postprocess(output, validate, **kwargs)

    def _compute_multi_modal_inputs(self, output, input_ids):
        inputs = getattr(output, "_minicpm_actor_inputs", None)
        if inputs is None:
            raise ValueError("MiniCPM-o replay requires minicpm_simplex_agent processor outputs.")
        return clone_minicpmo_actor_inputs(inputs)

    async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
        await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
        if self.distillation_enabled and not validate:
            teacher_ids = output.extra_fields["teacher_ids"]
            if teacher_ids.shape[-1] != 1:
                raise ValueError("MiniCPM-o simplex OPD currently supports selected-token reverse KL only.")
            # verl aligns each teacher score with the next-token label and appends a dummy final row.
            expected = torch.tensor((prompt_ids + response_ids)[1:], device=teacher_ids.device)
            if not torch.equal(teacher_ids[:-1, 0], expected):
                raise ValueError("MiniCPM-o teacher scoring changed the student's token sequence.")
            if not torch.isfinite(output.extra_fields["teacher_logprobs"]).all():
                raise ValueError("MiniCPM-o teacher returned non-finite log probabilities.")


class MiniCPMAgentLoopManager(AgentLoopManagerTQ):
    """Install the native replay worker through the existing V1 manager hook."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = MiniCPMAgentLoopWorker


@register("minicpm_simplex_agent")
class MiniCPMSimplexAgentLoop(SingleTurnAgentLoop):
    """Generate thinker tokens from a complete text/image/audio turn."""

    async def run(self, sampling_params, **kwargs):
        messages = kwargs["raw_prompt"]
        media = await self.process_multi_modal_info(messages)
        if media.get("videos"):
            raise NotImplementedError("MiniCPM-o simplex video replay is not yet supported.")
        if len(media.get("audios") or []) > 1:
            raise NotImplementedError("MiniCPM-o simplex replay currently accepts at most one audio clip per prompt.")
        if media.get("audios"):
            media["audios"] = await self.loop.run_in_executor(None, lambda: [_load_audio(a) for a in media["audios"]])
        processor_kwargs = self._get_mm_processor_kwargs(media.get("audios"))
        rendered = render_minicpmo_messages(
            self.processor, messages, add_generation_prompt=True, **self.apply_chat_template_kwargs
        )
        model_inputs = await self.loop.run_in_executor(
            None,
            lambda: prepare_minicpmo_inputs(
                self.processor,
                rendered,
                images=media.get("images"),
                audios=media.get("audios"),
                mm_processor_kwargs=processor_kwargs,
            ),
        )
        prompt_ids, actor_inputs = split_minicpmo_actor_inputs(model_inputs)
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError(f"MiniCPM-o prompt length {len(prompt_ids)} exceeds rollout.prompt_length.")
        processor_kwargs[MINICPM_PROMPT_KEY] = {
            "source_ids": self.tokenizer.encode(rendered, add_special_tokens=False),
            "expanded_ids": prompt_ids,
        }
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=media.get("images"),
                audio_data=media.get("audios"),
                mm_processor_kwargs=processor_kwargs,
            )
        if output.extra_fields.get("rollout_prompt_ids") != prompt_ids:
            raise ValueError("MiniCPM-o rollout's actual prompt IDs differ from actor replay.")
        if len(output.token_ids) > self.response_length:
            raise ValueError("MiniCPM-o rollout exceeded response_length; refusing to truncate the policy sequence.")
        metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        extra_fields = dict(output.extra_fields)
        extra_fields.pop("rollout_prompt_ids")
        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids,
            response_mask=[1] * len(output.token_ids),
            response_logprobs=output.log_probs,
            multi_modal_data=media,
            mm_processor_kwargs=processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        object.__setattr__(result, "_minicpm_actor_inputs", clone_minicpmo_actor_inputs(actor_inputs))
        return result
