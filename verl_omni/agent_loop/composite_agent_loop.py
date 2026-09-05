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
"""

Composite agent framework for multi-stage visual generation by
AR (LLM/MLLM) + DiT composite architecture, as well as for agentic RL.

- CompositeAgentLoopWorker extends DiffusionAgentLoopWorker with:
  - a reward handle for AR part
  - extra returns from AR generation, e.g., reward score and token-level log-probs.

"""

import asyncio
import random
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from verl.base_config import BaseConfig
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    DictConfigWrap,
    _agent_loop_registry,
    auto_await,
)
from verl.protocol import DataProto
from verl.utils.profiler import simple_timer
from verl.utils.skip import SkipManager
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.agent_loop.diffusion_agent_loop import (
    DiffusionAgentLoopOutput,
    DiffusionAgentLoopWorker,
    _InternalDiffusionAgentLoopOutput,
)
from verl_omni.agent_loop.utils import maybe_per_rollout_seeds


def _config_to_sampling_dict(config: Optional[BaseConfig]) -> dict:
    if config is None:
        return {}
    return {k: v for k, v in config.items() if not k.startswith("_")}


class ARAgentLoopOutput(BaseModel):
    """Agent loop output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    """Input ids of raw input prompt"""
    response_ids: Any
    """Full response AR tokens output (torch.Tensor)."""
    refined_prompt: Any
    """Refined rewritten prompt in chat-message form for diffusion."""

    ar_response_logprobs: Optional[Any] = None
    """Log probabilities for the response tokens."""
    ar_reward_score: Optional[float] = None
    """Reward score for the semantic reward."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalARAgentLoopOutput(ARAgentLoopOutput):
    """Internal agent loop output with padded sequences.
    Supplement additional fields for AR part.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded AR response token ids."""
    ar_response_logprobs: Optional[torch.Tensor] = None
    """Log probabilities for the response tokens."""


class CompositeAgentLoopWorker(DiffusionAgentLoopWorker):
    """Composite Agent loop worker takes a batch of messages and run each message in an agent loop.
    Two-stage rollout: LLM CoT+rewrite → prompt embeds → DiT sampling.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server replicas, produced by
            ``LLMServerManager.get_client()`` in the trainer.
        teacher_client (dict[str, LLMServerClient]): Not used by diffusion training; accepted to
            keep the constructor signature compatible with verl's ``AgentLoopManager.create()``,
            which positionally forwards a teacher client argument to each worker.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation.
        ar_reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation. Optional if reward_loop_worker_handles provides reward computation for both LLM and DiT.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        assert reward_loop_worker_handles is None or len(reward_loop_worker_handles) == 2
        if reward_loop_worker_handles is None:
            self.ar_reward_loop_worker_handles = None
        else:
            self.ar_reward_loop_worker_handles = reward_loop_worker_handles[1:]  # second for llm
            reward_loop_worker_handles = reward_loop_worker_handles[:1]  # first for dit

        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)

    async def generate_sequences(self, batch: DataProto) -> tuple[DataProto, DataProto]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            tuple[DataProto, DataProto]: AR and diffusion outputs separately, since they
            may have different batch sizes (``n`` vs ``n * m``).

            AR ``DataProto`` batch fields (optional keys omitted when disabled):

            - ``prompts``: ``[ar_bsz, prompt_length]`` original prompt token ids.
            - ``response_ids``: ``[ar_bsz, ar_max_new_tokens]`` generated AR tokens.
            - ``rollout_ar_log_probs`` (optional): AR token log-probs.
            - ``rm_scores`` (optional): ``[ar_bsz, 1]`` AR reward scores.

            Diffusion ``DataProto`` batch fields match :class:`DiffusionAgentLoopWorker`.

            - ``prompts``: ``[dit_bsz, prompt_length]`` refined prompt token ids.
            - ``responses``: ``[dit_bsz, C, H, W]`` or ``[dit_bsz, T, C, H, W]``.
            - ``rm_scores`` (optional): ``[dit_bsz, 1]`` diffusion reward scores.
        """
        config = self.rollout_config

        # composite sampling configs for LLM and DiT
        sampling_params = {
            **_config_to_sampling_dict(config.pipeline),
            **_config_to_sampling_dict(config.algo),
            **_config_to_sampling_dict(config.ar),
            "logprobs": config.calculate_log_probs,
            "ar_logprobs": config.ar_calculate_log_probs,
        }

        is_validate = batch.meta_info.get("validate", False)
        per_rollout_seeds: Optional[list[int]] = None

        if is_validate:
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.pipeline))
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.algo))
            sampling_params["seed"] = config.val_kwargs.seed
            sampling_params["logprobs"] = False

            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
            sampling_params["ar_logprobs"] = False
        else:
            sampling_params["global_steps"] = batch.meta_info["global_steps"]
            # Prefer trainer-assigned global indices so chunked workers derive the
            # same per-row seed regardless of local batch position / pack order.
            global_indices = batch.non_tensor_batch.get("_rollout_seed_global_idx")
            if global_indices is not None:
                global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
            per_rollout_seeds = maybe_per_rollout_seeds(batch.meta_info, len(batch), global_indices)

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        # two-stage generation: one batch row -> one AR output -> diffusion_n images
        diffusion_n = config.n
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["diffusion_n"] = diffusion_n
            if per_rollout_seeds is not None:
                kwargs["per_rollout_seeds"] = per_rollout_seeds[i * diffusion_n : (i + 1) * diffusion_n]
            else:
                kwargs["per_rollout_seeds"] = None
            task_sampling_params = sampling_params.copy()
            tasks.append(
                asyncio.create_task(self._run_agent_loop(task_sampling_params, validate=is_validate, **kwargs))
            )
        outputs = await asyncio.gather(*tasks)

        ar_inputs: list[_InternalARAgentLoopOutput] = []
        diffusion_inputs: list[_InternalDiffusionAgentLoopOutput] = []
        for ar_output, diffusion_outputs in outputs:
            ar_inputs.append(ar_output)
            diffusion_inputs.extend(diffusion_outputs)

        ar_non_tensor_batch = None
        if batch.non_tensor_batch:
            ar_non_tensor_batch = {k: v[: len(ar_inputs)] for k, v in batch.non_tensor_batch.items()}

        ar_output = self._postprocess_ar(ar_inputs, input_non_tensor_batch=ar_non_tensor_batch)
        diffusion_output = super()._postprocess(diffusion_inputs)

        return ar_output, diffusion_output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        validate: bool = False,
        **kwargs,
    ) -> tuple[_InternalARAgentLoopOutput, list[_InternalDiffusionAgentLoopOutput]]:
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )

        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            extra_tokenizer_map=self.model_config.extra_tokenizer_map,
        )
        output: tuple[ARAgentLoopOutput, list[DiffusionAgentLoopOutput]] = await agent_loop.run(
            sampling_params, **kwargs
        )
        return await self._agent_loop_postprocess(output, validate=validate, **kwargs)

    async def _agent_loop_postprocess(
        self,
        output: tuple[ARAgentLoopOutput, list[DiffusionAgentLoopOutput]],
        validate: bool = False,
        **kwargs,
    ) -> tuple[_InternalARAgentLoopOutput, list[_InternalDiffusionAgentLoopOutput]]:
        """Perform post-processing operations on the output of each individual agent loop."""
        ar_output, diffusion_outputs = output

        # AR part post-processing
        ar_internal = await self._agent_loop_ar_postprocess(ar_output, diffusion_outputs, validate=validate, **kwargs)

        # Diffusion part post-processing
        diffusion_internals: list[_InternalDiffusionAgentLoopOutput] = []
        for diffusion_output in diffusion_outputs:
            diffusion_output.prompt_ids = ar_output.prompt_ids  # use original prompt for reward
            diffusion_internal = await super()._agent_loop_postprocess(
                diffusion_output,
                validate=validate,
                **kwargs,
            )
            diffusion_internals.append(diffusion_internal)

        return ar_internal, diffusion_internals

    async def _agent_loop_ar_postprocess(
        self,
        output: ARAgentLoopOutput,
        diffusion_outputs: list[DiffusionAgentLoopOutput],
        validate: bool = False,
        **kwargs,
    ) -> _InternalARAgentLoopOutput:
        """Post-process a single AR agent-loop output."""
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        response_ids = output.response_ids
        if not isinstance(response_ids, torch.Tensor):
            response_ids = torch.as_tensor(response_ids)
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)

        ar_response_logprobs = None
        if output.ar_response_logprobs is not None:
            ar_response_logprobs = output.ar_response_logprobs
            if ar_response_logprobs.dim() == 2:
                ar_response_logprobs = ar_response_logprobs.unsqueeze(0)

        prompt_ids = prompt_output["input_ids"]
        extra_fields = dict(output.extra_fields)
        extra_fields["raw_prompt"] = kwargs["raw_prompt"]
        extra_fields["attention_mask"] = prompt_output["attention_mask"]
        if "text_encoder_responses" not in extra_fields:
            extra_fields["text_encoder_responses"] = output.refined_prompt

        await self._compute_ar_score(
            output,
            prompt_ids,
            diffusion_outputs,
            kwargs=kwargs,
            validate=validate,
        )

        if "reward_extra_info" in output.extra_fields:
            extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]

        return _InternalARAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            refined_prompt=output.refined_prompt,
            ar_response_logprobs=ar_response_logprobs,
            ar_reward_score=output.ar_reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )

    async def _compute_ar_score(
        self,
        output: ARAgentLoopOutput,
        prompts: torch.Tensor,
        diffusion_outputs: list[DiffusionAgentLoopOutput],
        kwargs,
        validate: bool = False,
    ):
        """Compute image-grounded AR reward scores using diffusion prompts and images.

        AR and DiT rewards share the same inputs: padded input ``prompts`` and
        ``response_diffusion_output``. When multiple diffusion samples exist per AR
        rollout, scores are averaged into one AR reward.
        """
        ar_enable_async_reward = self.ar_reward_loop_worker_handles is not None
        if output.ar_reward_score is not None or not ar_enable_async_reward or not diffusion_outputs:
            return

        timing = {}
        ar_reward_scores: list[float] = []
        with simple_timer("compute_score", timing):
            for diffusion_output in diffusion_outputs:
                batch = TensorDict(
                    {
                        "prompts": prompts,  # [1, prompt_length], padded input raw prompt ids
                        "responses": diffusion_output.response_diffusion_output.unsqueeze(
                            0
                        ),  # [1, C, H, W] or [1, T, C, H, W]
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([diffusion_output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                    meta_info={"validate": validate},
                )
                selected_ar_reward_loop_worker_handle = random.choice(self.ar_reward_loop_worker_handles)
                result = await selected_ar_reward_loop_worker_handle.compute_score.remote(data)
                ar_reward_scores.append(result["reward_score"])
                if output.extra_fields.get("reward_extra_info") is not None:
                    output.extra_fields["reward_extra_info"].update(result["reward_extra_info"])
                else:
                    output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

        output.ar_reward_score = float(np.mean(ar_reward_scores))  # average score per prompt
        if "reward_extra_info" in output.extra_fields:
            output.extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]
        output.metrics.compute_score = timing["compute_score"]

    def _postprocess_ar(
        self,
        inputs: list[_InternalARAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
    ) -> DataProto:
        """Process padded AR outputs and combine them into a batch."""
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        ar_response_ids = torch.cat([input.response_ids for input in inputs], dim=0)

        batch_dict: dict[str, torch.Tensor] = {
            "prompts": prompt_ids,
            "ar_response_ids": ar_response_ids,
        }
        if inputs[0].ar_response_logprobs is not None:
            batch_dict["rollout_ar_log_probs"] = torch.cat([input.ar_response_logprobs for input in inputs], dim=0)

        ar_scores = [input.ar_reward_score for input in inputs]
        if all(score is not None for score in ar_scores):
            batch_dict["rm_scores"] = torch.tensor(ar_scores, dtype=torch.float32).unsqueeze(-1)

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = sorted(set.intersection(*(set(info) for info in reward_extra_infos)))
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        text_encoder_responses = np.empty(len(inputs), dtype=object)
        text_encoder_responses[:] = [input.extra_fields.get("text_encoder_responses") for input in inputs]
        non_tensor_batch["text_encoder_responses"] = text_encoder_responses

        metrics = [input.metrics.model_dump() for input in inputs]
        if "rm_scores" in batch_dict:
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=TensorDict(batch_dict, batch_size=len(inputs)),
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


class CompositeAgentLoopManager(AgentLoopManager):
    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
    ):
        self.agent_loop_workers_class = ray.remote(CompositeAgentLoopWorker)
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)

    @auto_await
    @SkipManager.annotate(role="rollout")
    async def generate_sequences(self, prompts: DataProto) -> tuple[DataProto, DataProto]:
        """Gather ``(ar_batch, diffusion_batch)`` tuples from composite workers."""

        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
            ]
        )

        # concatenate outputs
        ar_outputs = [output[0] for output in outputs]
        diffusion_outputs = [output[1] for output in outputs]

        ar_batch = DataProto.concat(ar_outputs)
        diffusion_batch = DataProto.concat(diffusion_outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in ar_outputs]
        ar_timing = self._performance_metrics(metrics, ar_batch)
        old_keys = list(ar_timing.keys())
        for key in old_keys:
            new_key = key.replace("agent_loop/", "agent_loop/ar/")
            ar_timing[new_key] = ar_timing.pop(key)
        ar_batch.meta_info = {"timing": ar_timing, **ar_outputs[0].meta_info}

        metrics = [output.meta_info.pop("metrics") for output in diffusion_outputs]
        diffusion_timing = self._performance_metrics(metrics, ar_batch)
        diffusion_batch.meta_info = {"timing": diffusion_timing, **diffusion_outputs[0].meta_info}

        return ar_batch, diffusion_batch
