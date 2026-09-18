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
"""Agent-loop plumbing for full-vocabulary (hidden-state) on-policy distillation.

When ``distillation.distillation_loss.loss_mode`` is a nitrobrew mode the
teacher signal is per-position hidden states instead of logprobs.  The worker
below is a copy of verl's ``AgentLoopWorkerTQ`` (Ray forbids subclassing actor
classes) with two additions:

* ``_compute_teacher_logprobs`` fetches ``teacher_hidden_states`` [S, D] plus
  the routing ``teacher_key`` through the vllm-omni hidden-states
  channel (``verl_omni.workers.rollout.vllm_rollout.process_hidden_states``);
* ``_agent_loop_postprocess`` lifts the hidden states from
  ``extra_fields`` onto the TransferQueue field dict so they ride the batch
  as a first-class (jagged) tensor next to ``teacher_logprobs``.

Wire-up (recipe / hydra override)::

    actor_rollout_ref.rollout.agent.agent_loop_manager_class=\
verl_omni.agent_loop.nitrobrew_opd_agent_loop.NitrobrewOPDAgentLoopManagerTQ
"""

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict
from verl.experimental.agent_loop import AgentLoopOutput, AgentLoopWorker, get_trajectory_info
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, _settle_session_tasks, apply_greedy_sampling_params
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from verl_omni.workers.config.omni.distillation import HIDDEN_STATE_LOSS_MODES

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

__all__ = [
    "HIDDEN_STATE_LOSS_MODES",
    "use_hidden_states_from_config",
    "lift_teacher_hidden_states",
    "AsyncTeacherHiddenStatesManager",
    "NitrobrewOPDAgentLoopWorker",
    "NitrobrewOPDAgentLoopManagerTQ",
]


def use_hidden_states_from_config(config) -> bool:
    """True when the distillation loss consumes teacher hidden states."""
    distillation = config.get("distillation", None)
    if distillation is None:
        return False
    loss_mode = distillation.get("distillation_loss", {}).get("loss_mode", None)
    return loss_mode in HIDDEN_STATE_LOSS_MODES


def lift_teacher_hidden_states(field: dict[str, Any]) -> dict[str, Any]:
    """Move teacher hidden states from ``extra_fields`` onto the field dict.

    ``AgentLoopOutput.as_dict`` only promotes ``teacher_ids`` /
    ``teacher_logprobs`` to top-level keys; anything left in ``extra_fields``
    is stored as a non-tensor payload and loses its tensor identity. The hidden
    states must reach the TransferQueue as a top-level key so the batch collate
    turns them into a jagged nested tensor.
    """
    extra = field.get("extra_fields")
    if not isinstance(extra, dict):
        return field
    hidden = extra.pop("teacher_hidden_states", None)
    if hidden is not None:
        field["teacher_hidden_states"] = hidden
        field["teacher_key"] = extra.pop("teacher_key", None)
    return field


class AsyncTeacherHiddenStatesManager:
    """Teacher client that fetches hidden states instead of logprobs."""

    def __init__(self, config, teacher_client: dict):
        from omegaconf import OmegaConf
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import DistillationConfig

        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.teacher_key: str = self.distillation_config.teacher_key
        self.teacher_model_configs = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client = teacher_client
        _ = OmegaConf  # noqa: F401  (import kept for config compat elsewhere)

    def _resolve_teacher_key(self, routing_key: str | None) -> str:
        if len(self.teacher_model_configs) == 1:
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_hidden_states_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: dict | None = None,
        mm_processor_kwargs: dict | None = None,
        routing_key: str | None = None,
    ) -> torch.Tensor:
        """Fetch teacher hidden states [S, D] for one unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params={
                "max_tokens": 1,
                "temperature": 1.0,
                "return_hidden_states": True,
            },
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        hidden = teacher_output.extra_fields.get("teacher_hidden_states", None)
        if hidden is None:
            raise RuntimeError(
                "Teacher returned no teacher_hidden_states; check that the teacher engine has the "
                "hidden-states patch applied and return_hidden_states was requested."
            )
        if hidden.shape[0] != len(sequence_ids):
            raise RuntimeError(
                f"Teacher hidden states rows ({hidden.shape[0]}) != sequence length ({len(sequence_ids)})."
            )
        return hidden


class NitrobrewOPDAgentLoopWorker(AgentLoopWorker):
    """Agent-loop worker (TransferQueue flavour) with hidden-state teacher signal.

    Copied from verl's ``AgentLoopWorkerTQ`` — Ray does not allow subclassing
    actor classes, so the TQ fire-and-forget loop is replicated here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        self._use_hidden_states = use_hidden_states_from_config(self.config)
        if self.distillation_enabled and self._use_hidden_states:
            self.teacher_server_manager = AsyncTeacherHiddenStatesManager(
                config=self.config, teacher_client=self.teacher_client
            )

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        for i in range(len(batch)):
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=False)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)

            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _compute_teacher_logprobs(
        self,
        output,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: dict | None = None,
    ) -> None:
        """Compute the teacher signal: hidden states for nitrobrew modes."""
        if not (self.distillation_enabled and not validate):
            return
        if not self._use_hidden_states:
            await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
            return

        routing_key = None
        if sample_kwargs is not None:
            routing_value = sample_kwargs.get(self.teacher_key)
            if routing_value is not None:
                routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
        hidden = await self.teacher_server_manager.compute_teacher_hidden_states_single(
            sequence_ids=prompt_ids + response_ids,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            routing_key=routing_key,
        )
        output.extra_fields["teacher_hidden_states"] = hidden
        output.extra_fields["teacher_key"] = routing_key if routing_key is not None else "default"

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            if self._use_hidden_states:
                lift_teacher_hidden_states(field)
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    "global_steps": kwargs["global_steps"],
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class NitrobrewOPDAgentLoopManagerTQ(AgentLoopManagerTQ):
    """TQ agent-loop manager spawning NitrobrewOPDAgentLoopWorker."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(NitrobrewOPDAgentLoopWorker)
